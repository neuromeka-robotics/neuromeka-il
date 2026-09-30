import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import compare_box_lift_results as comparison
from data_collector.pink_ik import DCP_ACTIVE_JOINT_NAMES


class CompareBoxLiftResultsTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "result.csv"
        self.header = ["sample_index", "elapsed_time_s", "ik_mode"]
        self.header += [f"q_{name}_deg" for name in DCP_ACTIVE_JOINT_NAMES]
        self.header += [f"command_{name}_deg" for name in DCP_ACTIVE_JOINT_NAMES]

    def write_rows(self, rows, header=None):
        with self.path.open("w", newline="") as source:
            writer = csv.DictWriter(source, fieldnames=header or self.header)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, index=0):
        row = {name: 0. for name in self.header}
        row.update(sample_index=index, elapsed_time_s=index * .02, ik_mode="pink")
        return row

    def test_last_row_and_named_joint_columns_determine_pose_and_arm_error(self):
        first, last = self.row(), self.row(17)
        # Large non-arm error is excluded; the final arm error is [3, 4].
        last[f"command_{DCP_ACTIVE_JOINT_NAMES[0]}_deg"] = 999.
        last[f"q_{DCP_ACTIVE_JOINT_NAMES[4]}_deg"] = 10.
        last[f"command_{DCP_ACTIVE_JOINT_NAMES[4]}_deg"] = 13.
        last[f"command_{DCP_ACTIVE_JOINT_NAMES[5]}_deg"] = 4.
        self.write_rows([first, last], list(reversed(self.header)))
        with self.path.open("a") as source:
            source.write("\n\n")
        with patch.object(comparison, "RESULT_DIR", self.path.parent):
            sample = comparison.load_final_sample("result.csv")
        self.assertEqual(sample.index, 17)
        self.assertAlmostEqual(sample.elapsed, .34)
        self.assertEqual(sample.measured[4], 10.)
        self.assertEqual(sample.commanded[4], 13.)
        self.assertEqual(sample.arm_error_norm, 5.)
        self.assertEqual(len(sample.measured), 22)
        self.assertEqual(sample.commanded[-4:], [0.] * 4)

    def test_empty_log_is_rejected(self):
        self.write_rows([])
        with self.assertRaisesRegex(ValueError, "no recorded samples"):
            comparison.load_final_sample(self.path)

    def test_missing_sent_command_columns_are_rejected(self):
        self.write_rows([], self.header[:21])
        with self.assertRaisesRegex(ValueError, "missing result columns"):
            comparison.load_final_sample(self.path)

    def test_invalid_final_sample_is_not_silently_replaced_by_an_earlier_row(self):
        for value in ("nan", "inf", "bad", ""):
            with self.subTest(value=value):
                last = self.row(1)
                last[f"command_{DCP_ACTIVE_JOINT_NAMES[4]}_deg"] = value
                self.write_rows([self.row(), last])
                with self.assertRaisesRegex(ValueError, "invalid final sample"):
                    comparison.load_final_sample(self.path)


if __name__ == "__main__":
    unittest.main()
