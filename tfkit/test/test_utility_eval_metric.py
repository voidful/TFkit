import sys
import os

import pytest
import torch
from torch import nn
from torch.autograd import Variable

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import unittest
import tfkit
from transformers import *


class TestEval(unittest.TestCase):
    def testEMF1(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "abc", "abb[SEP]acc[SEP]abc", task='default')
        for s in eval.cal_score('emf1'):
            print(s)
            self.assertTrue(s[1]['EM'] == 1)
            self.assertTrue(s[1]['F1'] == 1)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "ab", "abb[SEP]acc[SEP]ab c", task='default')
        for s in eval.cal_score('emf1'):
            print(s)
            self.assertTrue(s[1]['EM'] == 0)
            self.assertTrue(s[1]['F1'] > 0)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "a b c", "a b b[SEP]a c c[SEP]", task='default')
        for s in eval.cal_score('emf1'):
            print(s)
            self.assertTrue(s[1]['EM'] == 0)
            self.assertTrue(s[1]['F1'] > 0)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "", "a b b[SEP]a c c", task='default')
        for s in eval.cal_score('emf1'):
            print(s)
            self.assertTrue(s[1]['EM'] == 0)
            self.assertTrue(s[1]['F1'] == 0)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "a", ["a"], task='default')
        for s in eval.cal_score('emf1'):
            print(s)
            self.assertTrue(s[1]['EM'] == 1)
            self.assertTrue(s[1]['F1'] == 1)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "a", ["b"], task='default')
        for s in eval.cal_score('emf1'):
            print(s)
            self.assertTrue(s[1]['EM'] == 0)
            self.assertTrue(s[1]['F1'] == 0)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "b", ["a"], task='default')
        for s in eval.cal_score('emf1'):
            print(s)
            self.assertTrue(s[1]['EM'] == 0)
            self.assertTrue(s[1]['F1'] == 0)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "b", ["b"], task='default')
        for s in eval.cal_score('emf1'):
            print(s)
            self.assertTrue(s[1]['EM'] == 1)
            self.assertTrue(s[1]['F1'] == 1)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "", [""], task='default')
        for s in eval.cal_score('emf1'):
            print(s)
            self.assertTrue(s[1]['EM'] == 1)
            self.assertTrue(s[1]['F1'] == 1)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "梵 語", ["梵 語"], task='default')
        eval.add_record("input", "梵 語", ["梵 語d"], task='default')
        eval.add_record("input", "梵 語", ["梵 語c"], task='default')
        for s in eval.cal_score('emf1'):
            self.assertAlmostEqual(s[1]['EM'], 0.3333333333)

    @pytest.mark.skip()
    def testNLG(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "a b c", "a b c[SEP]a c c[SEP]", task='default')
        for s in eval.cal_score('nlg'):
            self.assertAlmostEqual(s[1]['Bleu_1'], 1)
            print(s)

        eval1 = tfkit.utility.eval_metric.EvalMetric(tokenizer, max_candidate=1)
        eval1.add_record("input", "abc", " abc ", task='default')
        for s1 in eval1.cal_score('nlg'):
            self.assertAlmostEqual(s1[1]['Bleu_1'], 1)
            print(s1)

        eval1 = tfkit.utility.eval_metric.EvalMetric(tokenizer, max_candidate=1)
        eval1.add_record("input", "abb", " abc ", task='default')
        eval1.add_record("input", "abc", " abc ", task='default')
        eval1.add_record("input", "abd", " abc ", task='default')
        for s1 in eval1.cal_score('nlg'):
            self.assertAlmostEqual(s1[1]['Bleu_1'], 0.3333333)
            print(s1)

        eval3 = tfkit.utility.eval_metric.EvalMetric(tokenizer, max_candidate=3)
        eval3.add_record("input", "abc ", "abb [SEP]acc[SEP] abc ", task='default')
        for s3 in eval3.cal_score('nlg'):
            self.assertAlmostEqual(s3[1]['Bleu_1'], 1)
            print(s3)

        eval6 = tfkit.utility.eval_metric.EvalMetric(tokenizer, max_candidate=6)
        eval6.add_record("input", "abc", "abb [SEP] acc [SEP]abc", task='default')
        for s6 in eval6.cal_score('nlg'):
            self.assertAlmostEqual(s6[1]['Bleu_1'], 1)
            print(s6)
        self.assertTrue(s1[0] == s3[0] == s6[0])

        eval1 = tfkit.utility.eval_metric.EvalMetric(tokenizer, max_candidate=1)
        eval1.add_record("input", "opq", "abc", task='default')
        for s1 in eval1.cal_score('nlg'):
            self.assertAlmostEqual(s1[1]['Bleu_1'], 0)
            print(s1)

        eval3 = tfkit.utility.eval_metric.EvalMetric(tokenizer, max_candidate=3)
        eval3.add_record("input", "opq", "abb[SEP]acc[SEP]abc", task='default')
        for s3 in eval3.cal_score('nlg'):
            self.assertAlmostEqual(s3[1]['Bleu_1'], 0)
            print(s3)

        eval6 = tfkit.utility.eval_metric.EvalMetric(tokenizer, max_candidate=6)
        eval6.add_record("input", "opq", "abb [SEP] acc[SEP]abc", task='default')
        for s6 in eval6.cal_score('nlg'):
            self.assertAlmostEqual(s6[1]['Bleu_1'], 0)
            print(s6)
        self.assertTrue(s1[0] == s3[0] == s6[0])

    def testClassify(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "abc", "abb[SEP]acc[SEP]abc", task='default')
        for s in eval.cal_score('classification'):
            print(s[0])
            print(s[1])

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "你 好", "我 好[SEP]你 好 嗎[SEP]好 嗎", task='default')
        for s in eval.cal_score('classification'):
            print(s[0])
            print(s[1])

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "1 3 2", "1 2 3", task='default')
        eval.add_record("input", "1 3 2", "1 3 3", task='default')
        for s in eval.cal_score('classification'):
            print(s[0])
            print(s[1])

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", ["1", "3", "2"], ["1", "2", "3"], task='default')
        eval.add_record("input", ["1", "3", "2"], ["1", "3", "3"], task='default')
        for s in eval.cal_score('classification'):
            print(s[0])
            print(s[1])

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", ['O', 'B_Location', 'I_Location', 'I_Location', 'I_Location', 'I_Location', 'O'],
                        ['O', 'B_Location', 'I_Location', 'B_Location', 'I_Thing', 'I_Location', 'O'],
                        task='default')
        for s in eval.cal_score('classification'):
            print(s[0])
            print(s[1])

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", ['O', 'B_Location', 'I_Location', 'I_Location', 'I_Location', 'I_Location', 'O'],
                        ['O', 'B_Location', 'I_Location', 'B_Location', 'I_Thing', 'I_Location', 'O'],
                        task='default')
        eval.add_record("input", ['O', 'B_Location', 'I_Location', 'I_Location', 'I_Location', 'I_Location', 'O'],
                        ['O', 'B_Location', 'I_Location', 'B_Location', 'I_Thing', 'I_Location', 'O'],
                        task='default')
        for s in eval.cal_score('classification'):
            print(s[0])
            print(s[1])

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", ['O', 'B_Location', 'I_Location', 'I_Location', 'I_Location', 'I_Location', 'O'],
                        ['O', 'B_Location', 'I_Location', 'B_Location', 'I_Thing', 'I_Location', 'O'],
                        task='default')
        eval.add_record("input", ['O', 'B_Location', 'I_Location', 'I_Location', 'I_Location', 'I_Location'],
                        ['O', 'B_Location', 'I_Location', 'B_Location', 'I_Thing', 'I_Location', 'O'],
                        task='default')
        eval.add_record("input", ['O', 'B_Location', 'I_Location', 'I_Location', 'I_Location', 'I_Location', 'O', 'O'],
                        ['O', 'B_Location', 'I_Location', 'B_Location', 'I_Thing', 'I_Location', 'O'],
                        task='default')
        eval.add_record("input", [""] * 7,
                        ['O', 'B_Location', 'I_Location', 'B_Location', 'I_Thing', 'I_Location', 'O'],
                        task='default')
        eval.add_record("input", [],
                        ['O', 'B_Location', 'I_Location', 'B_Location', 'I_Thing', 'I_Location', 'O'],
                        task='default')
        for s in eval.cal_score('classification'):
            print(s[0])
            print(s[1])
            print(s)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", '攝影台', ['攝影台'], task='default')
        for s in eval.cal_score('classification'):
            print(s[0])
            print(s[1])
