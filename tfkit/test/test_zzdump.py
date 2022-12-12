import unittest
from tfkit.test import *
import os

import tfkit


class TestEval(unittest.TestCase):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../../"))
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'tfkit/test/cache/')

    def testHelp(self):
        result = os.system('tfkit-dump -h')
        assert (result == 0)

    def test_parser(self):
        parser = tfkit.dump.parse_dump_args(['--model', 'a', '--dumpdir', 'b'])
        self.assertTrue(parser.get('model') == 'a')
        self.assertTrue(parser.get('dumpdir') == 'b')

    def testDump(self):
        dump_dir = './cache/dump'
        tfkit.dump.main(["--model", ADDTOKFILE_MODEL_PATH, '--dumpdir', dump_dir])
        result = os.system(
            'tfkit-dump --model ' + ADDTOKFILE_MODEL_PATH + ' --dumpdir ' + dump_dir)
        self.assertTrue(result == 0)
