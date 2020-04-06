import unittest

import os


class TestTrain(unittest.TestCase):

    def testHelp(self):
        result = os.system('tfkit-train -h')
        assert (result == 0)

    def testTrain(self):
        result = os.system(
            'tfkit-train --train ../demo_data/generate.csv --valid ../demo_data/generate.csv --model onebyone --config voidful/albert_chinese_tiny  --savedir ./cache/ --maxlen 20')
        print(result == 0)
