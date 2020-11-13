import unittest

import os

import tfkit


class TestEval(unittest.TestCase):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../../"))
    DATASET_DIR = os.path.join(ROOT_DIR, 'demo_data')
    ADDED_TOK_MODEL = os.path.join(ROOT_DIR, 'tfkit/test/cache/voidful/albert_chinese_tiny_added_tok')
    ONEBYONE_MODEL_PATH = os.path.join(ROOT_DIR, 'tfkit/test/cache/onebyone/2.pt')
    CLAS_MODEL_PATH = os.path.join(ROOT_DIR, 'tfkit/test/cache/clas/2.pt')
    MASK_MODEL_PATH = os.path.join(ROOT_DIR, 'tfkit/test/cache/mask/2.pt')
    MCQ_MODEL_PATH = os.path.join(ROOT_DIR, 'tfkit/test/cache/mcq/2.pt')
    TAG_MODEL_PATH = os.path.join(ROOT_DIR, 'tfkit/test/cache/tag/2.pt')
    QA_MODEL_PATH = os.path.join(ROOT_DIR, 'tfkit/test/cache/qa/2.pt')
    TAG_DATASET = os.path.join(DATASET_DIR, 'tag_row.csv')
    CLAS_DATASET = os.path.join(DATASET_DIR, 'classification.csv')
    GEN_DATASET = os.path.join(DATASET_DIR, 'generate.csv')
    MASK_DATASET = os.path.join(DATASET_DIR, 'mask.csv')
    MCQ_DATASET = os.path.join(DATASET_DIR, 'mcq.csv')
    QA_DATASET = os.path.join(DATASET_DIR, 'qa.csv')

    def testHelp(self):
        result = os.system('tfkit-eval -h')
        self.assertTrue(result == 0)

    def test_parser(self):
        parser = tfkit.eval.parse_eval_args(
            ['--model', 'onebyone', '--metric', 'emf1', '--valid', 'test.csv', '--print'])
        print(parser)
        self.assertTrue(parser.get('model') == 'onebyone')

    def testEvalGen(self):
        tfkit.eval.main(
            ['--model', self.ONEBYONE_MODEL_PATH, '--valid', self.GEN_DATASET, '--metric', 'emf1', '--print'])
        result = os.system(
            'tfkit-eval --model ' + self.ONEBYONE_MODEL_PATH + ' --valid ' + self.GEN_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)

    def testEvalMask(self):
        tfkit.eval.main(
            ['--model', self.MASK_MODEL_PATH, '--valid', self.MASK_DATASET, '--metric', 'clas', '--print'])
        result = os.system(
            'tfkit-eval --model ' + self.MASK_MODEL_PATH + ' --valid ' + self.MASK_DATASET + ' --metric clas --print')
        self.assertTrue(result == 0)

    def testEvalMCQ(self):
        tfkit.eval.main(
            ['--model', self.MCQ_MODEL_PATH, '--valid', self.MCQ_DATASET, '--metric', 'clas', '--print'])
        result = os.system(
            'tfkit-eval --model ' + self.MCQ_MODEL_PATH + ' --valid ' + self.MCQ_DATASET + ' --metric clas --print')
        self.assertTrue(result == 0)

    def testEvalQA(self):
        tfkit.eval.main(
            ['--model', self.QA_MODEL_PATH, '--valid', self.QA_DATASET, '--metric', 'emf1', '--print'])
        result = os.system(
            'tfkit-eval --model ' + self.QA_MODEL_PATH + ' --valid ' + self.QA_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)

    def testEvalClassify(self):
        tfkit.eval.main(
            ['--model', self.CLAS_MODEL_PATH, '--valid', self.CLAS_DATASET, '--metric', 'clas', '--print'])
        result = os.system(
            'tfkit-eval --model ' + self.CLAS_MODEL_PATH + ' --valid ' + self.CLAS_DATASET + ' --metric clas --print')
        self.assertTrue(result == 0)

    def testEvalTag(self):
        tfkit.eval.main(
            ['--model', self.TAG_MODEL_PATH, '--valid', self.TAG_DATASET, '--metric', 'clas', '--print'])
        result = os.system(
            'tfkit-eval --model ' + self.TAG_MODEL_PATH + ' --valid ' + self.TAG_DATASET + ' --metric clas --print')
        self.assertTrue(result == 0)

    def testEvalAddedTokenModel(self):
        result = os.system(
            'tfkit-eval --model ' + self.ONEBYONE_MODEL_PATH + ' --valid ' + self.GEN_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)
        result = os.system(
            'tfkit-eval --model ' + self.ONEBYONE_MODEL_PATH + ' --config ' + self.ADDED_TOK_MODEL + ' --valid ' + self.GEN_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)
