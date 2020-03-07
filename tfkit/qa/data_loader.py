import csv
import os
import pickle
from collections import defaultdict

import numpy as np
from torch.utils import data
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer
from utility.tok import *


class loadQADataset(data.Dataset):
    def __init__(self, fpath, pretrained, maxlen=512, cache=False):
        samples = []
        if 'albert_chinese' in pretrained:
            tokenizer = BertTokenizer.from_pretrained(pretrained)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained)
        cache_path = fpath + ".cache"
        if os.path.isfile(cache_path) and cache:
            with open(cache_path, "rb") as cf:
                samples = pickle.load(cf)
        else:
            for i in tqdm(get_data_from_file(fpath)):
                tasks, task, input, target = i
                feature = get_feature_from_data(tokenizer, input, target, maxlen=maxlen)
                if len(feature['input']) <= maxlen:
                    samples.append(feature)
            if cache:
                with open(cache_path, 'wb') as cf:
                    pickle.dump(samples, cf)

        self.sample = samples

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]


def get_data_from_file(fpath):
    tasks = defaultdict(list)
    task = 'default'
    with open(fpath, 'r', encoding='utf8', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            context, start, end = row
            yield tasks, task, context, [start, end]


def get_feature_from_data(tokenizer, input, target=None, maxlen=512, separator=" "):
    row_dict = dict()
    row_dict['target'] = np.asarray([0, 0])
    tokenized_input = [tok_begin(tokenizer)] + tokenizer.tokenize(input) + [tok_sep(tokenizer)]
    input_id = tokenizer.convert_tokens_to_ids(tokenized_input)
    input = input.split()
    if target is not None:
        start, end = target
        start = int(start) + 1
        end = int(end)
        for pos, i in enumerate(input):
            length = len([x for x in tokenizer.tokenize(i) if
                          x not in tokenizer.all_special_tokens or x == tokenizer.unk_token])
            if length > 1:
                if pos < start:
                    start += length - 1
                if pos < end:
                    end += length - 1
        # print("ANS:", start, end, tokenized_input[start:end + 1])
        row_dict['target'] = np.asarray([start, end])

    mask_id = [1] * len(input_id)
    mask_id.extend([0] * (maxlen - len(mask_id)))
    row_dict['mask'] = np.asarray(mask_id)
    input_id.extend([0] * (maxlen - len(input_id)))
    row_dict['input'] = np.asarray(input_id)
    row_dict['raw_input'] = tokenized_input

    return row_dict
