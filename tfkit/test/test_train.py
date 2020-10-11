import unittest

import os

import pytest


@pytest.mark.skip()
class TestTrain(unittest.TestCase):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../"))
    DATASET_DIR = os.path.join(ROOT_DIR, 'demo_data')

    def testHelp(self):
        result = os.system('tfkit-train -h')
        assert (result == 0)

    def testMultiClass(self):
        result = os.system(
            'tfkit-train --batch 2 --epoch 2  --train ' + os.path.join(self.DATASET_DIR,
                                                                       'classification.csv') + ' ' + os.path.join(
                self.DATASET_DIR, 'generate.csv') + ' --lr 5e-5 --test ' + os.path.join(self.DATASET_DIR,
                                                                                        'classification.csv') + ' ' + os.path.join(
                self.DATASET_DIR,
                'generate.csv') + ' --model clas onebyone --config voidful/albert_chinese_tiny  --savedir ./cache/ --maxlen 50')
        self.assertTrue(result == 0)
        result = os.system(
            'tfkit-train --batch 2 --epoch 2  --train ' + os.path.join(self.DATASET_DIR,
                                                                       'classification.csv') + ' ' + os.path.join(
                self.DATASET_DIR, 'generate.csv') + ' --lr 5e-5 --test ' + os.path.join(self.DATASET_DIR,
                                                                                        'classification.csv') + ' ' + os.path.join(
                self.DATASET_DIR,
                'generate.csv') + ' --model clas onebyone-pos --config voidful/albert_chinese_tiny  --savedir ./cache/ --maxlen 50')
        self.assertTrue(result == 0)
        result = os.system(
            'tfkit-train --batch 2 --epoch 2  --train ' + os.path.join(self.DATASET_DIR,
                                                                       'classification.csv') + ' ' + os.path.join(
                self.DATASET_DIR, 'generate.csv') + ' --lr 5e-5 --test ' + os.path.join(self.DATASET_DIR,
                                                                                        'classification.csv') + ' ' + os.path.join(
                self.DATASET_DIR,
                'generate.csv') + ' --model clas onebyone-neg --config voidful/albert_chinese_tiny  --savedir ./cache/ --maxlen 50')
        self.assertTrue(result == 0)
        result = os.system(
            'tfkit-train --batch 2 --epoch 2  --train ' + os.path.join(self.DATASET_DIR,
                                                                       'classification.csv') + ' ' + os.path.join(
                self.DATASET_DIR, 'generate.csv') + ' --lr 5e-5 --test ' + os.path.join(self.DATASET_DIR,
                                                                                        'classification.csv') + ' ' + os.path.join(
                self.DATASET_DIR,
                'generate.csv') + ' --model clas onebyone-both --config voidful/albert_chinese_tiny  --savedir ./cache/ --maxlen 50')
        self.assertTrue(result == 0)

    def testGenOneByOne(self):
        result = os.system(
            'tfkit-train --batch 2 --epoch 2  --train ' + os.path.join(self.DATASET_DIR,
                                                                       'generate.csv') + ' --test ' + os.path.join(
                self.DATASET_DIR,
                'generate.csv') + ' --model onebyone --config voidful/albert_chinese_tiny  --savedir ./cache/ --maxlen 50')
        self.assertTrue(result == 0)

    def testGenOnce(self):
        result = os.system(
            'tfkit-train --batch 2 --epoch 2  --train ' + os.path.join(self.DATASET_DIR,
                                                                       'generate.csv') + ' --test ' + os.path.join(
                self.DATASET_DIR,
                'generate.csv') + ' --model once --config voidful/albert_chinese_tiny  --savedir ./cache/ --maxlen 50')
        self.assertTrue(result == 0)

    def testGenMask(self):
        result = os.system(
            'tfkit-train --batch 2 --epoch 2  --train ' + os.path.join(self.DATASET_DIR,
                                                                       'mask.csv') + ' --test ' + os.path.join(
                self.DATASET_DIR,
                'mask.csv') + ' --model mask --config voidful/albert_chinese_tiny  --savedir ./cache/ --maxlen 50')
        self.assertTrue(result == 0)

    def testGenWithSentLoss(self):
        result = os.system(
            'tfkit-train --batch 2 --epoch 2  --train ' + os.path.join(self.DATASET_DIR,
                                                                       'generate.csv') + ' --test ' + os.path.join(
                self.DATASET_DIR,
                'generate.csv') + ' --model onebyone-pos --config voidful/albert_chinese_tiny  --savedir ./cache/ --maxlen 50')
        self.assertTrue(result == 0)

    def testClassify(self):
        result = os.system(
            'tfkit-train --lr 1e-4 --grad_accum 2 --batch 2 --epoch 10 --train ' + os.path.join(self.DATASET_DIR,
                                                                                               'classification.csv') + ' --test ' + os.path.join(
                self.DATASET_DIR,
                'classification.csv') + ' --model clas --config voidful/albert_chinese_tiny  --savedir ./cache/ --maxlen 50')
        self.assertTrue(result == 0)

    def testAddToken(self):
        result = os.system(
            'tfkit-train --batch 2 --add_tokens 0 --epoch 1  --train ' + os.path.join(self.DATASET_DIR,
                                                                                      'generate.csv') + ' --test ' + os.path.join(
                self.DATASET_DIR,
                'generate.csv') + ' --model onebyone --config voidful/albert_chinese_tiny  --savedir ./cache/ --maxlen 50')
        self.assertTrue(result == 0)
