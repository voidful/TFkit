import unittest

import os


class TestTrain(unittest.TestCase):

    def testHelp(self):
        result = os.system('tfkit-train -h')
        assert (result == 0)

    def testGenOneByOne(self):
        result = os.system(
            'tfkit-train --batch 2 --train ../demo_data/generate.csv --valid ../demo_data/generate.csv --model onebyone --config voidful/albert_chinese_tiny  --savedir ./cache/ --maxlen 50')
        self.assertTrue(result == 0)

    def testGenWithSentLoss(self):
        result = os.system(
            'tfkit-train --train ../demo_data/generate.csv --valid ../demo_data/generate.csv --model onebyone-pos --config voidful/albert_chinese_tiny  --savedir ./cache/ --maxlen 50')
        self.assertTrue(result == 0)

    def testClassify(self):
        result = os.system(
            'tfkit-train --lr 1e-4  --train ../demo_data/classification.csv --valid ../demo_data/classification.csv --model classify --config voidful/albert_chinese_tiny  --savedir ./cache/ --maxlen 50')
        self.assertTrue(result == 0)

    def testMultiClass(self):
        result = os.system(
            'tfkit-train --batch 3 --train ../demo_data/classification.csv ../demo_data/generate.csv --lr 5e-5 --valid ../demo_data/classification.csv ../demo_data/generate.csv --model classify onebyone --config voidful/albert_chinese_tiny  --savedir ./cache/ --maxlen 50')
        self.assertTrue(result == 0)
