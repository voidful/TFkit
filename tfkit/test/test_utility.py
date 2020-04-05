import sys
import os

import torch
from torch import nn

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import unittest
import tfkit


class TestLoss(unittest.TestCase):
    outputs = torch.Tensor([[0.9, 0.5, 0.05], [0.01, 0.2, 0.7]])
    targets = torch.Tensor([0, 1]).long()
    alln_targets = torch.Tensor([-1, -1]).long()
    onen_targets = torch.Tensor([-1, 1]).long()

    def testLabelSmoothingCrossEntropy(self):
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        custom_criterion = tfkit.utility.loss.LabelSmoothingCrossEntropy(eps=0.0)
        self.assertTrue(criterion(self.outputs, self.targets) == custom_criterion(self.outputs, self.targets))
        self.assertTrue(criterion(self.outputs, self.alln_targets) == custom_criterion(self.outputs, self.alln_targets))
        self.assertTrue(criterion(self.outputs, self.onen_targets) == custom_criterion(self.outputs, self.onen_targets))

    def testNegativeCElLoss(self):
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        custom_criterion = tfkit.utility.loss.NegativeCElLoss()
        self.assertTrue(
            criterion(self.outputs, self.targets).item() > custom_criterion(self.outputs, self.targets).item())
        self.assertTrue(criterion(self.outputs, self.alln_targets).item() == custom_criterion(self.outputs,
                                                                                              self.alln_targets).item())
        self.assertTrue(criterion(self.outputs, self.onen_targets).item() > custom_criterion(self.outputs,
                                                                                             self.onen_targets).item())

    def testFocalLoss(self):
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        custom_criterion = tfkit.utility.loss.FocalLoss(gamma=0)
        self.assertTrue(criterion(self.outputs, self.targets) == custom_criterion(self.outputs, self.targets))
        self.assertTrue(criterion(self.outputs, self.alln_targets) == custom_criterion(self.outputs, self.alln_targets))
        self.assertTrue(criterion(self.outputs, self.onen_targets) == custom_criterion(self.outputs, self.onen_targets))

        custom_criterion = tfkit.utility.loss.FocalLoss(gamma=1)
        self.assertTrue(criterion(self.outputs, self.targets) > custom_criterion(self.outputs, self.targets))
        self.assertTrue(criterion(self.outputs, self.alln_targets) == custom_criterion(self.outputs, self.alln_targets))
        self.assertTrue(criterion(self.outputs, self.onen_targets) > custom_criterion(self.outputs, self.onen_targets))

    def testGWLoss(self):
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        custom_criterion = tfkit.utility.loss.GWLoss()
        self.assertTrue(criterion(self.outputs, self.targets) > custom_criterion(self.outputs, self.targets))
        self.assertTrue(criterion(self.outputs, self.alln_targets) == custom_criterion(self.outputs, self.alln_targets))
        self.assertTrue(criterion(self.outputs, self.onen_targets) > custom_criterion(self.outputs, self.onen_targets))


class TestEval(unittest.TestCase):
    def testEMF1(self):
        eval = tfkit.utility.eval_metric.EvalMetric()
        eval.add_record("input", "abc", "abb[SEP]acc[SEP]abc", task='default')
        for s in eval.cal_score('emf1'):
            print(s)
            self.assertTrue(s[1]['EM'] == 1)
            self.assertTrue(s[1]['F1'] == 1)

        eval = tfkit.utility.eval_metric.EvalMetric()
        eval.add_record("input", "a b c", "a b b[SEP]a c c[SEP]", task='default')
        for s in eval.cal_score('emf1'):
            print(s)
            self.assertTrue(s[1]['EM'] == 0)
            self.assertTrue(s[1]['F1'] > 0)

    def testNLG(self):
        eval1 = tfkit.utility.eval_metric.EvalMetric(max_candidate=1)
        eval1.add_record("input", "abc", "abc", task='default')
        for s1 in eval1.cal_score('nlg'):
            print(s1)

        eval3 = tfkit.utility.eval_metric.EvalMetric(max_candidate=3)
        eval3.add_record("input", "abc", "abb[SEP]acc[SEP]abc", task='default')
        for s3 in eval3.cal_score('nlg'):
            print(s3)

        eval6 = tfkit.utility.eval_metric.EvalMetric(max_candidate=6)
        eval6.add_record("input", "abc", "abb[SEP]acc[SEP]abc", task='default')
        for s6 in eval6.cal_score('nlg'):
            print(s6)
        self.assertTrue(s1 == s3 == s6)

    def testClassify(self):
        eval = tfkit.utility.eval_metric.EvalMetric()
        eval.add_record("input", "abc", "abb[SEP]acc[SEP]abc", task='default')
        for s in eval.cal_score('classification'):
            print(s[0])
            print(s[1])
