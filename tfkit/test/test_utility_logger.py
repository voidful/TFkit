import csv
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import unittest
import tfkit


class TestLogger(unittest.TestCase):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../../"))
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'tfkit/test/cache/')

    def test_write_log(self):
        logger = tfkit.Logger(savedir=self.MODEL_SAVE_PATH)
        logger.write_log("test")
        with open(logger.logfilepath, 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1]
            print(last_line)
            self.assertEqual(last_line, "test")

    def test_write_metric(self):
        logger = tfkit.Logger(savedir=self.MODEL_SAVE_PATH)
        logger.write_metric("test", 1, 0)
        with open(logger.metricfilepath, 'r') as f:
            last_row = list(csv.reader(f))[-1]
            self.assertEqual(last_row, ["test", '1', '0'])
