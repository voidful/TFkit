import unittest

import tfkit
from tfkit.test import *


class TestEval(unittest.TestCase):

    def testHelp(self):
        result = os.system('tfkit-eval -h')
        self.assertTrue(result == 0)

    def test_parser(self):
        parser, _ = tfkit.eval.parse_eval_args(
            ['--task', 'onebyone', '--metric', 'emf1', '--valid', 'test.csv', '--print'])
        print(parser)
        self.assertTrue(parser.get('task') == ['onebyone'])

        eval_parser, model_parser = tfkit.eval.parse_eval_args(
            ['--task', 'onebyone', '--metric', 'emf1', '--valid', 'test.csv', '--print', '--decodenum', '2'])
        self.assertTrue(eval_parser.get('task') == ['onebyone'])
        self.assertTrue(model_parser.get('decodenum') == '2')

    def testEvalGen(self):
        tfkit.eval.main(
            ['--task', ONEBYONE_MODEL_PATH, '--valid', GEN_DATASET, '--metric', 'emf1', '--print'])
        result = os.system(
            'tfkit-eval --task ' + ONEBYONE_MODEL_PATH + ' --valid ' + GEN_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)

    def testEvalGenOnce(self):
        tfkit.eval.main(
            ['--task', ONCE_MODEL_PATH, '--valid', GEN_DATASET, '--metric', 'emf1', '--print'])
        result = os.system(
            'tfkit-eval --task ' + ONCE_MODEL_PATH + ' --valid ' + GEN_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)

    def testEvalGenOnceCTC(self):
        tfkit.eval.main(
            ['--task', ONCECTC_MODEL_PATH, '--valid', GEN_DATASET, '--metric', 'emf1', '--print'])
        result = os.system(
            'tfkit-eval --task ' + ONCECTC_MODEL_PATH + ' --valid ' + GEN_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)

    def testEvalSeq2Seq(self):
        tfkit.eval.main(
            ['--task', SEQ2SEQ_MODEL_PATH, '--valid', GEN_DATASET, '--metric', 'emf1', '--print',
             '--decodenum', '2'])
        tfkit.eval.main(
            ['--task', SEQ2SEQ_MODEL_PATH, '--valid', GEN_DATASET, '--metric', 'emf1', '--print'])
        result = os.system(
            'tfkit-eval --task ' + SEQ2SEQ_MODEL_PATH + ' --valid ' + GEN_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)

    def testEvalCLM(self):
        tfkit.eval.main(
            ['--task', CLM_MODEL_PATH, '--valid', GEN_DATASET, '--metric', 'emf1', '--print'])
        result = os.system(
            'tfkit-eval --task ' + CLM_MODEL_PATH + ' --valid ' + GEN_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)

    def testEvalQA(self):
        tfkit.eval.main(
            ['--task', QA_MODEL_PATH, '--valid', QA_DATASET, '--metric', 'emf1', '--print'])
        result = os.system(
            'tfkit-eval --task ' + QA_MODEL_PATH + ' --valid ' + QA_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)

    def testEvalClassify(self):
        tfkit.eval.main(
            ['--task', CLAS_MODEL_PATH, '--valid', CLAS_DATASET, '--metric', 'clas', '--print'])
        result = os.system(
            'tfkit-eval --task ' + CLAS_MODEL_PATH + ' --valid ' + CLAS_DATASET + ' --metric clas --print')
        self.assertTrue(result == 0)

    def testEvalTag(self):
        tfkit.eval.main(
            ['--task', TAG_MODEL_PATH, '--valid', TAG_DATASET, '--metric', 'clas', '--print'])
        result = os.system(
            'tfkit-eval --task ' + TAG_MODEL_PATH + ' --valid ' + TAG_DATASET + ' --metric clas --print')
        self.assertTrue(result == 0)

    def testEvalAddedTokenModel(self):
        result = os.system(
            'tfkit-eval --task ' + ADDTOKFILE_MODEL_PATH + ' --valid ' + ADDTOK_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)
        result = os.system(
            'tfkit-eval --task ' + ADDTOKFILE_MODEL_PATH + ' --valid ' + ADDTOK_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)
        result = os.system(
            'tfkit-eval --task ' + ADDTOKFREQ_MODEL_PATH + ' --valid ' + ADDTOK_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)
        result = os.system(
            'tfkit-eval --task ' + ADDTOKFREQ_MODEL_PATH + ' --valid ' + ADDTOK_DATASET + ' --metric emf1 --print')
        self.assertTrue(result == 0)
