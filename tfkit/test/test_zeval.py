import unittest

import os

import pytest


@pytest.mark.skip()
class TestEval(unittest.TestCase):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../"))
    MODEL_PATH = os.path.join(ROOT_DIR, 'test/cache/1.pt')
    ADDED_TOK_MODEL = os.path.join(ROOT_DIR, 'test/cache/voidful/albert_chinese_tiny_added_tok')
    DATASET_DIR = os.path.join(ROOT_DIR, 'demo_data')

    def testHelp(self):
        result = os.system('tfkit-eval -h')
        self.assertTrue(result == 0)

    def testEvalGen(self):
        result = os.system(
            'tfkit-eval --model ' + self.MODEL_PATH + ' --valid ' + os.path.join(self.DATASET_DIR,
                                                                                 'generate.csv') + ' --metric emf1 --print')
        self.assertTrue(result == 0)
        result = os.system(
            'tfkit-eval  --model ' + self.MODEL_PATH + ' --valid ' + os.path.join(self.DATASET_DIR,
                                                                                  'generate.csv') + ' --metric emf1 --print')
        self.assertTrue(result == 0)
        result = os.system(
            'tfkit-eval --model ' + self.MODEL_PATH + ' --valid ' + os.path.join(self.DATASET_DIR,
                                                                                 'generate.csv') + ' --metric clas --print')
        self.assertTrue(result == 0)

    def testEvalMask(self):
        result = os.system(
            'tfkit-eval --model ' + self.MODEL_PATH + ' --valid ' + os.path.join(self.DATASET_DIR,
                                                                                 'mask.csv') + ' --metric clas --print')
        self.assertTrue(result == 0)

    def testEvalClassify(self):
        result = os.system(
            'tfkit-eval --model ' + self.MODEL_PATH + ' --valid ' + os.path.join(self.DATASET_DIR,
                                                                                 'classification.csv') + ' --metric clas --print')
        self.assertTrue(result == 0)

    def testEvalAddedTokenModel(self):
        result = os.system(
            'tfkit-eval --model ' + self.MODEL_PATH + ' --valid ' + os.path.join(self.DATASET_DIR,
                                                                                 'generate.csv') + ' --metric emf1 --print')
        result = os.system(
            'tfkit-eval --model ' + self.MODEL_PATH + ' --config ' + self.ADDED_TOK_MODEL + ' --valid ' + os.path.join(
                self.DATASET_DIR, 'generate.csv') + ' --metric emf1 --print')
