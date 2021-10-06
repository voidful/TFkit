import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.softmax = nn.Softmax(dim=1)
        self.nll = nn.NLLLoss(ignore_index=ignore_index)

    def forward(self, input, target):
        softmax = self.softmax(input)
        logpt = torch.log(softmax)
        pt = Variable(logpt.data.exp())
        return self.nll((1 - pt) ** self.gamma * logpt, target)


class SeqCTCLoss(nn.Module):
    def __init__(self, blank_index):
        super(SeqCTCLoss, self).__init__()
        self.blank_index = blank_index

    def forward(self, logits, input_lengths, targets, target_lengths):
        # lengths : (batch_size, )
        # log_logits : (T, batch_size, n_class), this kind of shape is required for ctc_loss
        # log_logits = logits + (logit_mask.unsqueeze(-1) + 1e-45).log()
        log_logits = logits.log_softmax(-1).transpose(0, 1)
        loss = F.ctc_loss(log_logits,
                          targets,
                          input_lengths,
                          target_lengths,
                          blank=self.blank_index,
                          reduction='mean',
                          zero_infinity=True)
        return loss


class SelfKDLoss(nn.Module):

    def __init__(self, alpha=0.1, temperature=2,ignore_index=-1):
        super(SelfKDLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(self, outputs, teacher_outputs, labels):
        loss = nn.KLDivLoss()(F.log_softmax(outputs / self.temperature, dim=-1),
                              F.softmax(teacher_outputs / self.temperature, dim=-1)) * (
                       self.alpha * self.temperature * self.temperature) + F.cross_entropy(outputs, labels,ignore_index=self.ignore_index,) * (
                       1. - self.alpha)
        return loss


class DiceLoss(nn.Module):
    """From 'Dice Loss for Data-imbalanced NLP Tasks'"""

    def __init__(self, ignore_index=None, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)
        if self.ignore_index is not None:
            mask = y_true == -1
            filtered_target = y_true
            filtered_target[mask] = 0
            torch.gather(y_pred, dim=1, index=filtered_target.unsqueeze(1))
            mask = mask.unsqueeze(1).expand(y_pred.data.size())
            y_pred[mask] = 0
        pred_prob = torch.gather(y_pred, dim=1, index=y_true.unsqueeze(1))
        dsc_i = 1 - ((1 - pred_prob) * pred_prob) / ((1 - pred_prob) * pred_prob + 1)
        if self.reduction == 'mean':
            return dsc_i.mean()
        else:
            return dsc_i.view(-1)


class NegativeCElLoss(nn.Module):
    def __init__(self, ignore_index=-1, reduction='mean'):
        super(NegativeCElLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.alpha = 1
        self.nll = nn.NLLLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, input, target):
        nsoftmax = self.softmax(input)
        nsoftmax = torch.clamp((1.0 - nsoftmax), min=1e-32)
        return self.nll(torch.log(nsoftmax) * self.alpha, target)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1, ignore_index=None, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            if self.ignore_index is not None:
                mask = target == -1
                filtered_target = target.clone()
                filtered_target[mask] = 0
                true_dist.scatter_(1, filtered_target.unsqueeze(1), self.confidence)
                mask = mask.unsqueeze(1).expand(pred.data.size())
                true_dist[mask] = 0
            else:
                true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        if self.reduction == 'mean':
            return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        else:
            return torch.sum(-true_dist * pred, dim=self.dim)
