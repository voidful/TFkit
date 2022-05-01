import sys
import os

import pytest

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import unittest
import tfkit
from transformers import BertTokenizer, AutoTokenizer


class TestEval(unittest.TestCase):
    @pytest.mark.skip()
    def testER(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "abc", "abb///acc///abc", task='default')
        for s in eval.cal_score('er'):
            print(s)
            self.assertTrue(s[1]['WER'] == 0)
            self.assertTrue(s[1]['CER'] == 0)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "ab", "abb///acc///ab c", task='default')
        for s in eval.cal_score('er'):
            print(s)
            self.assertTrue(s[1]['WER'] == 50)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "a b c", "a b b///a c c///", task='default')
        for s in eval.cal_score('er'):
            print(s)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "", "a b b///a c c", task='default')
        for s in eval.cal_score('er'):
            print(s)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "a", ["a"], task='default')
        for s in eval.cal_score('er'):
            print(s)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "a", ["b"], task='default')
        for s in eval.cal_score('er'):
            print(s)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "b", ["a"], task='default')
        for s in eval.cal_score('er'):
            print(s)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "b", ["b"], task='default')
        for s in eval.cal_score('er'):
            print(s)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "", [""], task='default')
        for s in eval.cal_score('er'):
            print(s)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "梵 語", ["梵 語"], task='default')
        eval.add_record("input", "梵 語", ["梵 語d"], task='default')
        eval.add_record("input", "梵 語", ["梵 語c"], task='default')
        for s in eval.cal_score('er'):
            print(s)

    def testEMF1(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "abc", "abb///acc///abc", task='default')
        for s in eval.cal_score('emf1'):
            print(s)
            self.assertTrue(s[1]['EM'] == 1)
            self.assertTrue(s[1]['F1'] == 1)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "ab", "abb///acc///ab c", task='default')
        for s in eval.cal_score('emf1'):
            print(s)
            self.assertTrue(s[1]['EM'] == 0)
            self.assertTrue(s[1]['F1'] > 0)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "a b c", "a b b///a c c///", task='default')
        for s in eval.cal_score('emf1'):
            print(s)
            self.assertTrue(s[1]['EM'] == 0)
            self.assertTrue(s[1]['F1'] > 0)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "", "a b b///a c c", task='default')
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

    def test_tokenize_text(self):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer, normalize_text=False)
        self.assertEqual(eval.tokenize_text("How's this work"), "How's this work")
        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer, normalize_text=True)
        self.assertEqual(eval.tokenize_text("How's this work"), "how ' s this work")

    @pytest.mark.skip()
    def testNLGWithPAD(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "a b", "a b ///a c c", task='default')
        eval.add_record("input", "d c", "a b c///a c c///d c", task='default')
        for s in eval.cal_score('nlg'):
            s1 = s[1]
            print(s1)

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "a b", "a b ///a c c", task='nlg')
        eval.add_record("input", "d c", "a b c///a c c///d c", task='nlg')
        for s in eval.cal_score('nlg'):
            s2 = s[1]
            print(s2)

        self.assertEqual(s1, s2)
        self.assertAlmostEqual(s2['Bleu_1'], 1)

    @pytest.mark.skip()
    def testNLG(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "a b c", "a b c///a c c///", task='default')
        for s in eval.cal_score('nlg'):
            self.assertAlmostEqual(s[1]['Bleu_1'], 1)
            print(s)

        eval1 = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval1.add_record("input", "abc", " abc ", task='default')
        for s1 in eval1.cal_score('nlg'):
            self.assertAlmostEqual(s1[1]['Bleu_1'], 1)
            print(s1)

        eval1 = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval1.add_record("input", "abb", " abc ", task='default')
        eval1.add_record("input", "abc", " abc ", task='default')
        eval1.add_record("input", "abd", " abc ", task='default')
        for s1 in eval1.cal_score('nlg'):
            self.assertAlmostEqual(s1[1]['Bleu_1'], 0.3333333)
            print(s1)

        eval3 = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval3.add_record("input", "abc ", "abb ///acc/// abc ", task='default')
        for s3 in eval3.cal_score('nlg'):
            self.assertAlmostEqual(s3[1]['Bleu_1'], 1)
            print(s3)

        eval6 = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval6.add_record("input", "abc", "abb /// acc ///abc", task='default')
        for s6 in eval6.cal_score('nlg'):
            self.assertAlmostEqual(s6[1]['Bleu_1'], 1)
            print(s6)
        self.assertTrue(s1[0] == s3[0] == s6[0])

        eval1 = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval1.add_record("input", "opq", "abc", task='default')
        for s1 in eval1.cal_score('nlg'):
            self.assertAlmostEqual(s1[1]['Bleu_1'], 0)
            print(s1)

        eval3 = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval3.add_record("input", "opq", "abb///acc///abc", task='default')
        for s3 in eval3.cal_score('nlg'):
            self.assertAlmostEqual(s3[1]['Bleu_1'], 0)
            print(s3)

        eval6 = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval6.add_record("input", "opq", "abb /// acc///abc", task='default')
        for s6 in eval6.cal_score('nlg'):
            self.assertAlmostEqual(s6[1]['Bleu_1'], 0)
            print(s6)
        self.assertTrue(s1[0] == s3[0] == s6[0])

    def testTag(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
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

    def testClassify(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "abc", "abb///acc///abc", task='default')
        for s in eval.cal_score('classification'):
            print(s[0])
            print(s[1])

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "你 好", "我 好///你 好 嗎///好 嗎", task='default')
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

    @pytest.mark.skip()
    def testNLGOnModel(self):

        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "An interesting research into chocolate",
                        "More Chocolate , Less Health</s>Chocolate and Blood Pressure</s>Advice on Eating Chocolate",
                        task='nlg')
        for s in eval.cal_score('nlg'):
            print(s)

        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "An interesting research into chocolate",
                        "More Chocolate , Less Health///Chocolate and Blood Pressure///Advice on Eating Chocolate",
                        task='nlg')
        for s in eval.cal_score('nlg'):
            print(s)
