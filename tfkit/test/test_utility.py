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

    def testSoothingCElLoss(self):
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        custom_criterion = tfkit.utility.loss.SoothingCElLoss(smoothing=0)
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