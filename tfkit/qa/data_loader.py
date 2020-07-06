import csv
import os
import pickle
from collections import defaultdict

import numpy as np
from torch.utils import data
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer
from utility.tok import *
from random import choice


class loadQADataset(data.Dataset):
    def __init__(self, fpath, pretrained, maxlen=512, cache=False):
        samples = []
        if 'albert_chinese' in pretrained:
            tokenizer = BertTokenizer.from_pretrained(pretrained)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained)
        cache_path = fpath + "_maxlen" + str(maxlen) + "_" + pretrained.replace("/", "_") + ".cache"
        if os.path.isfile(cache_path) and cache:
            with open(cache_path, "rb") as cf:
                samples = pickle.load(cf)
        else:
            total_data = 0
            data_exceed_maxlen = 0
            for i in tqdm(get_data_from_file(fpath)):
                tasks, task, input, target = i
                feature = get_feature_from_data(tokenizer, input, target, maxlen=maxlen)
                if len(feature['input']) <= maxlen and 0 <= feature['target'][0] < maxlen and 0 <= feature['target'][
                    1] < maxlen:
                    samples.append(feature)
                else:
                    data_exceed_maxlen += 1
                total_data += 1

            print("Processed " + str(total_data) + " data, removed " + str(
                data_exceed_maxlen) + " data that exceed the maximum length.")

            if cache:
                with open(cache_path, 'wb') as cf:
                    pickle.dump(samples, cf)

        self.sample = samples

    def increase_with_sampling(self, total):
        inc_samp = [choice(self.sample) for i in range(total - len(self.sample))]
        self.sample.extend(inc_samp)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        self.sample[idx].update((k, np.asarray(v)) for k, v in self.sample[idx].items() if k != 'raw_input')
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
    row_dict['target'] = [0, 0]
    tokenized_input_ori = tokenizer.tokenize(input)
    tokenized_input = [tok_begin(tokenizer)] + tokenizer.tokenize(input) + [tok_sep(tokenizer)]
    input_id = tokenizer.convert_tokens_to_ids(tokenized_input)
    ext_tok = []
    input = input.split(" ")
    if target is not None:
        start, end = target
        ori_start = start = int(start)
        ori_end = end = int(end)
        ori_ans = input[ori_start:ori_end]

        for pos, i in enumerate(input):
            ext_tok.extend(tokenizer.tokenize(i))
            length = len(tokenizer.tokenize(i))
            if pos < ori_start:
                start += length - 1
            if pos < ori_end:
                end += length - 1

        # print("ORI ANS:", ori_ans, "TOK ANS:", tokenized_input[start + 1:end + 1])
        row_dict['target'] = [start + 1, end + 1]  # cls +1

    mask_id = [1] * len(input_id)
    mask_id.extend([0] * (maxlen - len(mask_id)))
    row_dict['mask'] = mask_id
    input_id.extend([0] * (maxlen - len(input_id)))
    row_dict['input'] = input_id
    row_dict['raw_input'] = tokenized_input

    return row_dict
