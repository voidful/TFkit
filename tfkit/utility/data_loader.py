import numpy
import torch
from torch import nn
from torch.utils import data


def index_of(in_list, val):
    """
    get token index in list, return -1 when it is not in the list 
    :rtype: int
    :param in_list: query list
    :param val: query target
    :return: position index
    """
    try:
        return in_list.index(val)
    except ValueError:
        return -1


def pad_batch(batch):
    """
    reduce batch data shape by reduce their padding to common max
    it needs to Handel some exception since some key is no need to be padded
    :param batch: list of dict, with key input and target as model input and target
    :return: list of dict
    """
    keys = list(batch[0].keys())
    for k in keys:
        batch_key_length = [len(i[k]) if not isinstance(i[k], int) else 1 for i in batch]
        if len(set(batch_key_length)) > 1:  # is all value same? if no, it need to pad with max length
            pad_length = max(batch_key_length)
            for idx, _ in enumerate(batch):
                if f"{k}_pad" in batch[idx]:
                    padded = nn.ConstantPad1d((0, pad_length - len(batch[idx][k])), batch[idx][f"{k}_pad"][0])
                else:
                    padded = nn.ConstantPad1d((0, pad_length - len(batch[idx][k])), 0)
                # batch[idx][k] = torch.unsqueeze(padded(batch[idx][k]), 0)
                batch[idx][k] = padded(batch[idx][k])
    for ind, dat in enumerate(batch):
        for k, v in dat.items():
            batch[ind][k] = numpy.asarray(batch[ind][k])
    return batch


def dataloader_collate(batch):
    """
    dataloader_collate function to apply batch reduce padding
    :param batch: list of dict
    :return: batch: list of dict
    """
    # batch = copy.deepcopy(batch)
    return torch.utils.data._utils.collate.default_collate(pad_batch(batch))
