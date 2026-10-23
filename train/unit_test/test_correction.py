"""Nominal-command preprocessing, masking, config paths and ACT training."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

TRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TRAIN), str(TRAIN/'nrmk_il/src')]
import h5py
import numpy as np
import torch
import yaml
from correction_config import DEFAULT_CONFIG, DEFAULT_DATA_CONFIG, load_settings, load_data_settings
from nrmk_il.dataset.correction_dataset import (STATE_KEYS, ACTION_KEYS, RAW_TO_KEY,
    CorrectionDataset, CorrectionNormalizer, policy_features, preprocess_corrections, select_normalization)
from nrmk_il.policies import ACTConfig, ACTPolicy
from nrmk_il.policies.selection import ControlMode


class CorrectionTests(unittest.TestCase):
    def settings(self):
        return yaml.safe_load(DEFAULT_CONFIG.read_text())

    def write_raw(self, root, episode, shift):
        values = np.arange(12).reshape(4, 3)*.1+shift
        with h5py.File(root/f'{episode}.h5', 'w') as file:
            for key in RAW_TO_KEY:
                file[key] = values
            file['unused_historical_field'] = np.full_like(values, np.nan)
            file.attrs.update(physics_dt_s=.005, nominal_command_timing='next_step_immediate')
            file.attrs['joint_names'] = ['locked', 'a', 'b']
            file.attrs['active_joints'] = [False, True, True]
            file.attrs['recorded_ticks'] = [0, 2, 4]
        return values[:, 1:]

    def test_normalization_uses_train_only_and_masks_command_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root/'raw'; raw.mkdir()
            values = [self.write_raw(raw, i, i*100.) for i in range(2)]
            split, stats = preprocess_corrections(raw, root/'processed', .5, seed=0,
                                                  policy_config=self.settings()['policy'])
            self.assertEqual(set(stats['features']), set(STATE_KEYS+ACTION_KEYS))
            self.assertEqual(stats['nominal_command_timing'], 'next_step_immediate')
            index = int(Path(split['episodes'][split['train'][0]]).stem)
            np.testing.assert_allclose(stats['features'][STATE_KEYS[0]]['mean'], values[index].mean(axis=0))
            normalizer = CorrectionNormalizer(stats)
            for key in stats['features']:
                np.testing.assert_allclose(normalizer.denormalize(key, normalizer.normalize(key, values[index])), values[index])
            data = CorrectionDataset(root/'processed', chunk_size=3)
            self.assertEqual(data[0]['is_pad'].tolist(), [False, False, True])
            self.assertEqual(data[1]['is_pad'].tolist(), [False, True, True])
            self.assertEqual(data[3]['is_pad'].tolist(), [False, True, True])
            with h5py.File(root/'processed/0.h5') as file:
                self.assertEqual(set(file), set(STATE_KEYS+ACTION_KEYS))
                self.assertEqual(file.attrs['nominal_command_timing'], 'next_step_immediate')
            # The second action from tick 0 predicts the boundary at tick 2;
            # tick 1's second action would need the new input and is masked.
            for key in ACTION_KEYS:
                torch.testing.assert_close(data[0][key][1], data[1][key][0])

    def test_yaml_paths_and_fixed_design(self):
        settings, paths = load_settings(DEFAULT_CONFIG)
        self.assertEqual(set(settings['data']), {'processed_dir'})
        self.assertEqual(set(paths), {'processed', 'weights'})
        self.assertTrue(paths['processed'].is_absolute())
        _, preparation = load_data_settings(DEFAULT_DATA_CONFIG)
        self.assertEqual(preparation['tasks'][0].name, '0.h5')
        self.assertEqual(preparation['tasks'][1].name, '1.h5')
        self.assertEqual(preparation['processed'], paths['processed'])
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/'train.yaml'
            settings['data'] = {'processed_dir': 'already_processed'}
            settings['train']['output_dir'] = 'weights'
            config.write_text(yaml.safe_dump(settings))
            saved, resolved = load_settings(config)
            self.assertEqual(resolved['processed'], config.parent/'already_processed')
            self.assertEqual(resolved['weights'].parent, config.parent/'weights')
            relocated = config.parent/'subdir/config.yaml'
            relocated.parent.mkdir()
            relocated.write_text(yaml.safe_dump(saved))
            self.assertEqual(load_settings(relocated)[1], resolved)
        settings['policy']['control_mode'] = 'direct_command'
        with self.assertRaises(ValueError):
            policy_features(settings['policy'])
        self.assertEqual(ControlMode.NOMINAL_COMMAND.value, 7)
        self.assertEqual(ControlMode.get_candidate('task'), [ControlMode.TASK_SPACE, ControlMode.RELATIVE_DELTA_TASK_SPACE])
        with self.assertRaises(ValueError):
            load_settings(DEFAULT_CONFIG, '../outside')

    def test_three_token_act_masked_loss_backward_and_action_queue(self):
        cfg = self.settings()['policy']
        cfg.update(dim_model=32, dim_feedforward=64, n_encoder_layers=1, dropout=0., n_action_steps=2, n_obs_steps=1)
        policy = ACTPolicy(ACTConfig(**cfg))
        batch = {key: torch.randn(2, 14) for key in STATE_KEYS}
        batch.update({key: torch.randn(2, 2, 14) for key in ACTION_KEYS})
        batch['is_pad'] = torch.tensor([[False, True], [False, True]])
        losses, _ = policy(batch)
        changed = copy.deepcopy(batch)
        for key in ACTION_KEYS:
            changed[key][:, 1] = 10000.
        again, _ = policy(changed)
        torch.testing.assert_close(losses['loss'], again['loss'])
        losses['loss'].backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters()))
        policy.eval(); policy.reset()
        with torch.inference_mode():
            first = policy.select_action(batch)['actions']
            second = policy.select_action(batch)['actions']
        self.assertEqual(tuple(first.shape), (2, 28))
        self.assertEqual(tuple(second.shape), (2, 28))

    def test_history_windows_and_online_chunks_match_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root/'raw'; raw.mkdir()
            for episode in range(2):
                self.write_raw(raw, episode, episode*100.)
            preprocess_corrections(raw, root/'processed', 0.)
            data = CorrectionDataset(root/'processed', chunk_size=2, n_obs_steps=3)
            frames = CorrectionDataset(root/'processed', chunk_size=2)
            self.assertEqual(len(data), 8)
            for start in range(8):
                episode_start = start//4*4
                expected_indices = [max(episode_start, start-2), max(episode_start, start-1), start]
                for key in STATE_KEYS:
                    expected = torch.stack([frames[i][key] for i in expected_indices])
                    torch.testing.assert_close(data[start][key], expected, rtol=0, atol=0)
                for key in [*ACTION_KEYS, 'is_pad']:
                    torch.testing.assert_close(data[start][key], frames[start][key], rtol=0, atol=0)

            cfg = self.settings()['policy']
            cfg.update(dim_model=32, dim_feedforward=64, n_encoder_layers=1, dropout=0.,
                       n_action_steps=2, n_obs_steps=3,
                       input_shapes={key: [2] for key in STATE_KEYS},
                       output_shapes={key: [2] for key in ACTION_KEYS})
            torch.manual_seed(0)
            policy = ACTPolicy(ACTConfig(**cfg))
            batch = {key: torch.stack([data[i][key] for i in [0, 3, 4, 7]]) for key in data[0]}
            losses, _ = policy(batch)
            losses['loss'].backward()
            grad = policy.model.encoder_time_pos_embed.weight.grad
            self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum() > 0)
            self.assertEqual(policy.model.num_state_tokens, 9)
            policy.eval()
            with torch.inference_mode():
                old = policy.model(batch)[0]['actions']
                altered = {key: value.clone() for key, value in batch.items()}
                altered[STATE_KEYS[0]][:, 0] += 1.  # Current input stays the same.
                self.assertFalse(torch.allclose(old, policy.model(altered)[0]['actions']))
                reversed_history = {key: value.flip(1) if key in STATE_KEYS else value
                                    for key, value in batch.items()}
                self.assertFalse(torch.allclose(old, policy.model(reversed_history)[0]['actions']))
                for episode in range(2):
                    policy.reset()
                    for step in range(4):
                        index = episode*4+step
                        if step % 2 == 0:
                            offline = {key: data[index][key].unsqueeze(0) for key in STATE_KEYS}
                            expected_chunk = policy.model(offline)[0]['actions']
                        current = {key: frames[index][key].unsqueeze(0).clone() for key in STATE_KEYS}
                        action = policy.select_action(current)['actions']
                        torch.testing.assert_close(action, expected_chunk[:, step % 2])
                        # In-place caller updates must not modify buffered observations.
                        for value in current.values():
                            value.fill_(999.)
                        for key in STATE_KEYS:
                            torch.testing.assert_close(torch.stack(list(policy._observation_history[key]), 1),
                                                       data[index][key].unsqueeze(0), rtol=0, atol=0)

    def test_invalid_history_configs_fail_clearly(self):
        for steps in (0, -1, 1.5, True):
            cfg = self.settings()['policy']
            cfg['n_obs_steps'] = steps
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                ACTConfig(**cfg)
        cfg = self.settings()['policy']
        cfg.update(n_obs_steps=3, use_vae=True)
        with self.assertRaisesRegex(ValueError, 'use_vae: false'):
            ACTConfig(**cfg)


if __name__ == '__main__':
    unittest.main()
