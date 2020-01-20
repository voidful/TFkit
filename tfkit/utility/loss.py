import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(BCEFocalLoss, self).__init__()
        self.gamma = 2

    def forward(self, input, target):
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


class BCEGWLoss(nn.Module):
    def __init__(self):
        super(BCEGWLoss, self).__init__()

    def gaussian(self, x, mean=0.5, variance=0.25):
        for i, v in enumerate(x.data):
            x.data[i] = math.exp(-(v - mean) ** 2 / (2.0 * variance ** 2))
        return x

    def forward(self, input, target):
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        BCE_loss = BCE_loss.view(-1)
        pt = sigmoid(BCE_loss)  # prevents nans when probability 0
        loss = (self.gaussian(pt, variance=0.1 * math.exp(1), mean=0.5) - 0.1 * pt) * BCE_loss
        return loss.mean()


class GWLoss(nn.Module):
    def __init__(self):
        super(GWLoss, self).__init__()
        self.softmax = nn.Softmax()
        self.nll = nn.NLLLoss(ignore_index=-1)

    def gaussian(self, x, mean=0.5, variance=0.25):
        for row in x.data:
            for i, v in enumerate(row):
                row[i] = math.exp(-(v - mean) ** 2 / (2.0 * variance ** 2))
        return x

    def forward(self, input, target):
        softmax = self.softmax(input)
        logpt = torch.log(softmax)
        pt = Variable(logpt.data.exp())
        return self.nll((self.gaussian(pt, variance=0.1 * math.exp(1), mean=0.5) - 0.1 * pt) * logpt, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.softmax = nn.Softmax()
        self.nll = nn.NLLLoss(ignore_index=-1)

    def forward(self, input, target):
        softmax = self.softmax(input)
        logpt = torch.log(softmax)
        pt = Variable(logpt.data.exp())
        return self.nll((1 - pt) ** self.gamma * logpt, target)


class NegativeCElLoss(nn.Module):
    def __init__(self, ratio=0.7):
        super(NegativeCElLoss, self).__init__()
        self.ratio = ratio
        self.softmax = nn.Softmax()
        self.nll = nn.NLLLoss(ignore_index=-1)

    def forward(self, input, target):
        nsoftmax = self.softmax(input)
        nsoftmax = torch.clamp((1.0 - nsoftmax), min=1e-5)
        return self.nll(torch.log(nsoftmax), target)


class FocalSmoothingLoss(nn.Module):
    def __init__(self, eps: float = 0.1, gamma=2):
        super(FocalSmoothingLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.softmax = nn.Softmax()
        self.nll = nn.NLLLoss(ignore_index=-1)

    def forward(self, input, target):
        c = input.size()[-1]
        softmax = self.softmax(input)
        logpt = torch.log(softmax)
        pt = Variable(logpt.data.exp())
        return -logpt.sum(dim=-1) * self.eps / c + (1 - self.eps) * self.nll((1 - pt) ** self.gamma * logpt, target)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps: float = 0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.softmax = nn.Softmax()
        self.eps, self.reduction = eps, reduction

    def forward(self, output, target):
        c = output.size()[-1]
        softmax = self.softmax(output)
        log_preds = torch.log(softmax)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':  loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                                 ignore_index=-1)


class NegativeLabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps: float = 0.1, reduction='mean'):
        super(NegativeLabelSmoothingCrossEntropy, self).__init__()
        self.eps, self.reduction = eps, reduction

    def forward(self, output, target):
        c = output.size()[-1]
        softmax = self.softmax(output, dim=-1)
        nsoftmax = torch.clamp((1.0 - softmax), min=1e-5)
        log_preds = torch.log(nsoftmax)

        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':  loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                                 ignore_index=-1)
