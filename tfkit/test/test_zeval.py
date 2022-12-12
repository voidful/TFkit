import unittest

import tfkit
from tfkit.test import *


class TestEval(unittest.TestCase):

    def testHelp(self):
        result = os.system('tfkit-eval -h')
        self.assertTrue(result == 0)

    def test_parser(self):
        parser, _ = tfkit.eval.parse_eval_args(
            ['--model', 'once', '--metric', 'emf1', '--valid', 'test.csv', '--print'])
        print(parser)
        self.assertTrue(parser.get('model') == ['once'])

        eval_parser, model_parser = tfkit.eval.parse_eval_args(
            ['--model', 'once', '--metric', 'emf1', '--valid', 'test.csv', '--print', '--decodenum', '2'])
        self.assertTrue(eval_parser.get('model') == ['once'])
        self.assertTrue(model_parser.get('decodenum') == '2')

    def testEvalGen(self):
        tfkit.eval.main(
            ['--model', ONCE_MODEL_PATH, '--valid', GEN_DATASET, '--metric', 'emf1', '--print'])
        result = os.system(
            'tfkit-eval --model ' + ONCE_MODEL_PATH + ' --valid ' + GEN_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)

    def testEvalGenOnce(self):
        tfkit.eval.main(
            ['--model', ONCE_MODEL_PATH, '--valid', GEN_DATASET, '--metric', 'emf1', '--print'])
        result = os.system(
            'tfkit-eval --model ' + ONCE_MODEL_PATH + ' --valid ' + GEN_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)

    def testEvalGenOnceCTC(self):
        tfkit.eval.main(
            ['--model', ONCECTC_MODEL_PATH, '--valid', GEN_DATASET, '--metric', 'emf1', '--print'])
        result = os.system(
            'tfkit-eval --model ' + ONCECTC_MODEL_PATH + ' --valid ' + GEN_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)

    def testEvalSeq2Seq(self):
        tfkit.eval.main(
            ['--model', SEQ2SEQ_MODEL_PATH, '--valid', GEN_DATASET, '--metric', 'emf1', '--print',
             '--decodenum', '2'])
        tfkit.eval.main(
            ['--model', SEQ2SEQ_MODEL_PATH, '--valid', GEN_DATASET, '--metric', 'emf1', '--print'])
        result = os.system(
            'tfkit-eval --model ' + SEQ2SEQ_MODEL_PATH + ' --valid ' + GEN_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)

    def testEvalCLM(self):
        tfkit.eval.main(
            ['--model', CLM_MODEL_PATH, '--valid', GEN_DATASET, '--metric', 'emf1', '--print'])
        result = os.system(
            'tfkit-eval --model ' + CLM_MODEL_PATH + ' --valid ' + GEN_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)

    def testEvalAddedTokenModel(self):
        result = os.system(
            'tfkit-eval --model ' + ADDTOKFILE_MODEL_PATH + ' --valid ' + ADDTOK_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)

    # def testEvalQA(self):
    #     tfkit.eval.main(
    #         ['--model', QA_MODEL_PATH, '--valid', QA_DATASET, '--metric', 'emf1', '--print'])
    #     result = os.system(
    #         'tfkit-eval --model ' + QA_MODEL_PATH + ' --valid ' + QA_DATASET + ' --metric emf1 --print')
    #     self.assertTrue(result == 0)
    #
    # def testEvalClassify(self):
    #     tfkit.eval.main(
    #         ['--model', CLAS_MODEL_PATH, '--valid', CLAS_DATASET, '--metric', 'clas', '--print'])
    #     result = os.system(
    #         'tfkit-eval --model ' + CLAS_MODEL_PATH + ' --valid ' + CLAS_DATASET + ' --metric clas --print')
    #     self.assertTrue(result == 0)
    #
    # def testEvalTag(self):
    #     tfkit.eval.main(
    #         ['--model', TAG_MODEL_PATH, '--valid', TAG_DATASET, '--metric', 'clas', '--print'])
    #     result = os.system(
    #         'tfkit-eval --model ' + TAG_MODEL_PATH + ' --valid ' + TAG_DATASET + ' --metric clas --print')
    #     self.assertTrue(result == 0)