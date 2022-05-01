import copy

import numpy
import torch
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


def batch_reduce_pad(batch):
    """
    reduce batch data shape by reduce their padding to common max
    it needs to Handel some exception since some key is no need to be padded
    :param batch: list of dict, with key input and target as model input and target
    :return: list of dict
    """
    has_pad = all([dat['input'][-1] == batch[0]['input'][-1] for dat in batch]) and \
              batch[0]['input'][-1] == batch[0]['input'][-2]
    if has_pad:
        pad_token_input = batch[0]['input'][-1]
        pad_start = max([list(dat['input']).index(pad_token_input) for dat in batch])
        pad_token_target = batch[0]['target'][-1] if 'target' in batch[0] else None
        if not isinstance(pad_token_target, numpy.ndarray) and pad_token_target:
            # multi-label classification target will have an array target, should not pad this
            pad_start = max(pad_start, max([index_of(list(dat['target']), pad_token_target) for dat in batch]))
        if 'start' in batch[0]:
            pad_start = max(pad_start, max([data['start'] for data in batch if 'start' in data]) + 1)
        for ind, dat in enumerate(batch):
            for k, v in dat.items():
                if isinstance(v, list) and len(v) > 1 and k != 'task':  # not padding task name
                    batch[ind][k] = v[:pad_start]
                if k == 'input_length':
                    batch[ind][k] = pad_start - 1
                batch[ind][k] = numpy.asarray(batch[ind][k])
    else:
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
    batch = copy.deepcopy(batch)
    print("batch",batch)
    return torch.utils.data._utils.collate.default_collate(batch_reduce_pad(batch))
