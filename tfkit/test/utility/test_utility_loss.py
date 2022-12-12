import os
import sys

import torch
from torch import nn
from torch.autograd import Variable

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import unittest
import tfkit


class TestLoss(unittest.TestCase):
    outputs = Variable(torch.Tensor([[0.00000000000009, 5, 0.5], [0.00000000000000000001, 69, 9]]), requires_grad=False)
    targets = Variable(torch.Tensor([1, 1]).long(), requires_grad=False)
    alln_targets = Variable(torch.Tensor([-1, -1]).long(), requires_grad=False)
    onen_targets = Variable(torch.Tensor([1, -1]).long(), requires_grad=False)

    def testLabelSmoothingCrossEntropy(self):
        outputs = torch.Tensor([[0.00000000000009, 5, 0.5], [0.00000000000000000001, 69, 9]])
        targets = torch.Tensor([1, 1]).long()
        alln_targets = torch.Tensor([0, -1]).long()
        onen_targets = torch.Tensor([1, -1]).long()

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        custom_criterion = tfkit.utility.loss.LabelSmoothingLoss(3, ignore_index=-1)

        self.assertTrue(criterion(outputs, targets).item() <
                        custom_criterion(outputs, targets).item())
        self.assertTrue(criterion(outputs, onen_targets).item() <
                        custom_criterion(outputs, onen_targets).item())

        criterion = nn.CrossEntropyLoss()
        custom_criterion = tfkit.utility.loss.LabelSmoothingLoss(3)
        self.assertTrue(criterion(outputs, targets).item() <
                        custom_criterion(outputs, targets).item())

        custom_criterion = tfkit.utility.loss.LabelSmoothingLoss(3, reduction='none')
        print(custom_criterion(self.outputs, self.targets))
        self.assertTrue(list(custom_criterion(self.outputs, self.targets).shape) == [2])

    def testDiceLoss(self):
        custom_criterion = tfkit.utility.loss.DiceLoss(ignore_index=-1)
        self.assertTrue(0.8 < custom_criterion(self.outputs, self.targets).item() < 1)
        self.assertTrue(0.99 < custom_criterion(self.outputs, self.alln_targets).item() <= 1)
        self.assertTrue(0.8 < custom_criterion(self.outputs, self.onen_targets).item() < 1)

        custom_criterion = tfkit.utility.loss.DiceLoss(reduction='none')
        print(custom_criterion(self.outputs, self.targets))
        self.assertTrue(list(custom_criterion(self.outputs, self.targets).shape) == [2])

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

    def testBCEFocalLoss(self):
        outputs = torch.Tensor([[0, 1, 0], [0.2, 0, 0]])
        targets = torch.Tensor([[0, 1, 0], [1, 0, 0]])
        criterion = nn.BCELoss()
        custom_criterion = tfkit.utility.loss.BCEFocalLoss()
        self.assertTrue(criterion(outputs, targets).item() >
                        custom_criterion(outputs, targets).item())

    def testNegativeCElLoss(self):
        outputs = torch.Tensor([[0.00000000000009, 5, 0.5], [0.00000000000000000001, 69, 9]])
        targets = torch.Tensor([1, 1]).long()
        alln_targets = torch.Tensor([-1, -1]).long()
        onen_targets = torch.Tensor([1, -1]).long()

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        custom_criterion = tfkit.utility.loss.NegativeCElLoss()
        self.assertTrue(
            criterion(outputs, targets).item() < custom_criterion(outputs, self.targets).item())
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
