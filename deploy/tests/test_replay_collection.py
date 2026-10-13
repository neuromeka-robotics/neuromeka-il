import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import h5py
import numpy as np

from replay_collection import EpisodeReplay, joint_values, load_episode


class EpisodeLoaderTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "episode.h5"

    def write(self, measured, commanded=None):
        with h5py.File(self.path, "w") as episode:
            episode["q_0"] = measured
            if commanded is not None:
                episode["joint_abs_control_0"] = commanded

    def test_preserves_every_sample_joint_and_alternating_command(self):
        measured = np.arange(88, dtype=np.float32).reshape(4, 22) / 7.
        commanded = measured.copy()
        commanded[:, 4] = [1., -1., 1., -1.]
        self.write(measured, commanded)
        actual, sent = load_episode(self.path)
        np.testing.assert_array_equal(actual, measured)
        np.testing.assert_array_equal(sent, commanded)

    def test_rejects_missing_command_instead_of_substituting_state(self):
        self.write(np.zeros((3, 22)))
        with self.assertRaisesRegex(ValueError, "Missing dataset"):
            load_episode(self.path)

    def test_rejects_misaligned_partial_empty_or_invalid_arrays(self):
        for q, cmd in (
            (np.zeros((3, 22)), np.zeros((2, 22))),
            (np.zeros((3, 18)), np.zeros((3, 18))),
            (np.zeros((0, 22)), np.zeros((0, 22))),
            (np.zeros((3, 22)), np.full((3, 22), np.nan)),
        ):
            with self.subTest(q=q.shape, cmd=cmd.shape):
                self.write(q, cmd)
                with self.assertRaises(ValueError):
                    load_episode(self.path)


    def test_copied_script_loads_data_without_repository_imports_or_working_directory(self):
        self.write(np.zeros((3, 22)), np.ones((3, 22)))
        folder = Path(self.folder.name)
        (folder / "data").mkdir()
        shutil.copy2(self.path, folder / "data/3.h5")
        script = folder / "replay_collection.py"
        shutil.copy2(Path(__file__).resolve().parents[1] / "replay_collection.py", script)
        code = """
import importlib.abc
import runpy
import sys
class BlockRepositoryImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args, **kwargs):
        if fullname.split('.')[0] in ('data_collector', 'helper', 'neuromeka', 'pink', 'pinocchio'):
            raise AssertionError('Unexpected repository/robot dependency: ' + fullname)
sys.meta_path.insert(0, BlockRepositoryImports())
module = runpy.run_path(sys.argv[1], run_name='portable_replay')
q, command = module['load_episode'](module['bundle_path']('data/3.h5'))
assert q.shape == command.shape == (3, 22)
assert (q == 0).all() and (command == 1).all()
"""
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, str(script)],
            cwd=folder.parent, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_standalone_mapping_preserves_dcp_arm_order_and_degree_units(self):
        values = joint_values(np.arange(22, dtype=float) * 10.)
        self.assertEqual(len(values), 18)
        self.assertAlmostEqual(values["Joint_L2_L"], np.deg2rad(40.))
        self.assertAlmostEqual(values["Joint_L2_R"], np.deg2rad(110.))
        self.assertAlmostEqual(values["Joint_L8_R"], np.deg2rad(170.))


class EpisodePlaybackTest(unittest.TestCase):
    def setUp(self):
        self.server = MagicMock()
        gui = self.server.gui
        gui.add_checkbox.side_effect = lambda label, value, **kw: MagicMock(value=value)
        gui.add_dropdown.side_effect = lambda label, options, initial_value: MagicMock(value=initial_value)
        gui.add_slider.side_effect = lambda label, **kw: MagicMock(value=kw["initial_value"])
        gui.add_uplot.side_effect = lambda data, *args, **kw: MagicMock(data=data)
        self.robots = [MagicMock(), MagicMock()]
        measured = np.zeros((4, 22))
        commanded = np.zeros((4, 22))
        commanded[:, 4] = [1., -1., 1., -1.]
        self.replay = EpisodeReplay(
            self.server, self.robots, [MagicMock(), MagicMock()],
            measured, commanded, dt=0.05, name="synthetic.h5")

    def test_slow_render_advances_one_frame_and_never_skips_commands(self):
        replay = self.replay
        replay.playing.value = True
        replay.tick(0.)
        for now, expected in [(0.05, 1), (1., 2), (5., 3)]:
            replay.tick(now)
            self.assertEqual(replay.frame.value, expected)
        rendered = [call.args[0]["Joint_L2_L"]
                    for call in self.robots[1].update_cfg.call_args_list]
        # First frame is also rendered when switching from paused to playing.
        np.testing.assert_allclose(rendered[-3:], np.deg2rad([-1., 1., -1.]))
        np.testing.assert_allclose(replay.position_plot.data[2], [1., -1., 1., -1.])
        np.testing.assert_allclose(replay.step_plot.data[1], [-2., 2., -2.])

    def test_speed_pause_step_and_end_of_recording(self):
        replay = self.replay
        replay.speed.value = "0.5x"
        replay.playing.value = True
        replay.tick(0.)
        replay.tick(0.06)
        self.assertEqual(replay.frame.value, 0)
        replay.tick(0.101)
        self.assertEqual(replay.frame.value, 1)
        replay.seek(100)
        replay.tick(1.)
        self.assertFalse(replay.playing.value)
        self.assertEqual(replay.frame.value, 3)
        replay.playing.value = True
        replay.tick(2.)
        replay.tick(2.2)
        self.assertFalse(replay.playing.value)
        self.assertEqual(replay.frame.value, 3)
        replay.loop.value = True
        replay.tick(3.)
        replay.playing.value = True
        replay.tick(4.)
        replay.tick(4.2)
        self.assertEqual(replay.frame.value, 0)

    def test_joint_selection_and_single_frame_episode(self):
        replay = self.replay
        replay.joint.value = "Joint_L2_R"
        replay.seek(3)
        replay.tick(1.)
        np.testing.assert_array_equal(replay.position_plot.data[2], np.zeros(4))
        single = EpisodeReplay(
            self.server, self.robots, [MagicMock(), MagicMock()],
            np.zeros((1, 22)), np.ones((1, 22)), dt=0.05, name="one.h5")
        single.seek(1)
        single.tick(1.)
        self.assertEqual(single.frame.value, 0)
        self.assertEqual(len(single.step_plot.data[1]), 0)


if __name__ == "__main__":
    unittest.main()
