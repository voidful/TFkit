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


class TestLoss(unittest.TestCase):
    outputs = Variable(torch.Tensor([[0.00000000000009, 5, 0.5], [0.00000000000000000001, 69, 9]]), requires_grad=False)
    targets = Variable(torch.Tensor([1, 1]).long(), requires_grad=False)
    alln_targets = Variable(torch.Tensor([-1, -1]).long(), requires_grad=False)
    onen_targets = Variable(torch.Tensor([1, -1]).long(), requires_grad=False)

    def testLabelSmoothingCrossEntropy(self):
        outputs = torch.Tensor([[0.00000000000009, 5, 0.5], [0.00000000000000000001, 69, 9]])
        targets = torch.Tensor([1, 1]).long()
        alln_targets = torch.Tensor([-1, -1]).long()
        onen_targets = torch.Tensor([1, -1]).long()

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        custom_criterion = tfkit.utility.loss.LabelSmoothingLoss(3, ignore_index=-1)

        self.assertTrue(criterion(outputs, targets).item() <
                        custom_criterion(outputs, targets).item())
        self.assertTrue((criterion(outputs, alln_targets).item() == custom_criterion(outputs, alln_targets).item()))
        self.assertTrue(criterion(outputs, onen_targets).item() <
                        custom_criterion(outputs, onen_targets).item())

    def testDiceLoss(self):
        custom_criterion = tfkit.utility.loss.DiceLoss(ignore_index=-1)
        self.assertTrue(0.8 < custom_criterion(self.outputs, self.targets).item() < 1)
        self.assertTrue(0.99 < custom_criterion(self.outputs, self.alln_targets).item() <= 1)
        self.assertTrue(0.8 < custom_criterion(self.outputs, self.onen_targets).item() < 1)

    def testLossDrop(self):
        outputs = torch.Tensor([[0.00000000000009, 5, 0.5], [0.00000000000000000001, 69, 9]])
        targets = torch.Tensor([1, 1]).long()
        norm_loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)  # -1 index = padding token
        masked_lm_loss = loss_fct(outputs, targets)
        masked_lm_loss = masked_lm_loss.view(-1, len(targets))  # view by batch size
        masked_lm_loss = masked_lm_loss.sum(dim=0)
        masked_lm_loss = masked_lm_loss.mean()
        print(masked_lm_loss.mean(), norm_loss_fct(outputs, targets).mean())

    def testNegativeCElLoss(self):
        outputs = torch.Tensor([[0.00000000000009, 5, 0.5], [0.00000000000000000001, 69, 9]])
        targets = torch.Tensor([1, 1]).long()
        alln_targets = torch.Tensor([-1, -1]).long()
        onen_targets = torch.Tensor([1, -1]).long()

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        custom_criterion = tfkit.utility.loss.NegativeCElLoss()
        self.assertTrue(
            criterion(outputs, targets).item() < custom_criterion(outputs, self.targets).item())
        self.assertTrue(criterion(outputs, alln_targets).item() == custom_criterion(outputs, alln_targets).item())
        self.assertTrue(criterion(outputs, onen_targets).item() < custom_criterion(outputs, onen_targets).item())

    def testFocalLoss(self):
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        custom_criterion = tfkit.utility.loss.FocalLoss(gamma=0)
        self.assertAlmostEqual(criterion(self.outputs, self.targets).item(),
                               custom_criterion(self.outputs, self.targets).item())
        self.assertAlmostEqual(criterion(self.outputs, self.alln_targets).item(),
                               custom_criterion(self.outputs, self.alln_targets).item())
        self.assertAlmostEqual(criterion(self.outputs, self.onen_targets).item(),
                               custom_criterion(self.outputs, self.onen_targets).item())

        custom_criterion = tfkit.utility.loss.FocalLoss(gamma=1)
        self.assertTrue(criterion(self.outputs, self.targets) > custom_criterion(self.outputs, self.targets))
        self.assertTrue(criterion(self.outputs, self.alln_targets).item() - custom_criterion(self.outputs,
                                                                                             self.alln_targets).item() < 1)
        self.assertTrue(criterion(self.outputs, self.onen_targets) > custom_criterion(self.outputs, self.onen_targets))


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
        eval.add_record("input", "", "a b b[SEP]a c c[SEP]", task='default')
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

    @pytest.mark.skip()
    def testNLG(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')

        eval = tfkit.utility.eval_metric.EvalMetric(tokenizer)
        eval.add_record("input", "a b c", "a b c[SEP]a c c[SEP]", task='default')
        for s in eval.cal_score('nlg'):
            print(s)

        eval1 = tfkit.utility.eval_metric.EvalMetric(tokenizer, max_candidate=1)
        eval1.add_record("input", "abc", " abc ", task='default')
        for s1 in eval1.cal_score('nlg'):
            print(s1)

        eval1 = tfkit.utility.eval_metric.EvalMetric(tokenizer, max_candidate=1)
        eval1.add_record("input", "abb", " abc ", task='default')
        eval1.add_record("input", "abc", " abc ", task='default')
        eval1.add_record("input", "abd", " abc ", task='default')
        for s1 in eval1.cal_score('nlg'):
            print(s1)

        eval3 = tfkit.utility.eval_metric.EvalMetric(tokenizer, max_candidate=3)
        eval3.add_record("input", "abc ", "abb [SEP]acc[SEP] abc ", task='default')
        for s3 in eval3.cal_score('nlg'):
            print(s3)

        eval6 = tfkit.utility.eval_metric.EvalMetric(tokenizer, max_candidate=6)
        eval6.add_record("input", "abc", "abb [SEP] acc [SEP]abc", task='default')
        for s6 in eval6.cal_score('nlg'):
            print(s6)
        self.assertTrue(s1[0] == s3[0] == s6[0])

        eval1 = tfkit.utility.eval_metric.EvalMetric(tokenizer, max_candidate=1)
        eval1.add_record("input", "opq", "abc", task='default')
        for s1 in eval1.cal_score('nlg'):
            print(s1)

        eval3 = tfkit.utility.eval_metric.EvalMetric(tokenizer, max_candidate=3)
        eval3.add_record("input", "opq", "abb[SEP]acc[SEP]abc", task='default')
        for s3 in eval3.cal_score('nlg'):
            print(s3)

        eval6 = tfkit.utility.eval_metric.EvalMetric(tokenizer, max_candidate=6)
        eval6.add_record("input", "opq", "abb [SEP] acc[SEP]abc", task='default')
        for s6 in eval6.cal_score('nlg'):
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