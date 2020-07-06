import sys
import os
from random import choice

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import csv
import json
import pickle
from collections import defaultdict

import numpy as np
from torch.utils import data
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer
from utility.tok import *


class loadColTaggerDataset(data.Dataset):
    def __init__(self, fpath, pretrained, maxlen=368, cache=False):
        samples = []
        if 'albert_chinese' in pretrained:
            tokenizer = BertTokenizer.from_pretrained(pretrained)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained)
        cache_path = fpath + pretrained + ".cache"
        if os.path.isfile(cache_path) and cache:
            with open(cache_path, "rb") as cf:
                savedict = pickle.load(cf)
                samples = savedict["samples"]
                labels = savedict["labels"]
        else:
            for i in get_data_from_file_col(fpath):
                tasks, task, input, target = i
                labels = tasks[task]
                feature = get_feature_from_data(tokenizer, labels, input, target, maxlen=maxlen)
                if len(feature['input']) == len(feature['target']) <= maxlen:
                    samples.append(feature)
            if cache:
                with open(cache_path, 'wb') as cf:
                    pickle.dump({'samples': samples, 'labels': labels}, cf)
        self.sample = samples
        self.label = labels

    def increase_with_sampling(self, total):
        inc_samp = [choice(self.sample) for i in range(total - len(self.sample))]
        self.sample.extend(inc_samp)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        self.sample[idx].update((k, np.asarray(v)) for k, v in self.sample[idx].items() if k != 'mapping')
        return self.sample[idx]


class loadRowTaggerDataset(data.Dataset):
    def __init__(self, fpath, pretrained, maxlen=512, separator=" ", cache=False):
        samples = []
        labels = []
        if 'albert_chinese' in pretrained:
            tokenizer = BertTokenizer.from_pretrained(pretrained)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained)
        cache_path = fpath + "_maxlen" + str(maxlen) + "_" + pretrained.replace("/", "_") + ".cache"
        if os.path.isfile(cache_path) and cache:
            with open(cache_path, "rb") as cf:
                savedict = pickle.load(cf)
                samples = savedict["samples"]
                labels = savedict["labels"]
        else:
            total_data = 0
            data_exceed_maxlen = 0

            for i in get_data_from_file_row(fpath):
                tasks, task, input, target = i
                labels = tasks[task]
                feature = get_feature_from_data(tokenizer, labels, input, target, maxlen=maxlen)
                if len(feature['input']) == len(feature['target']) <= maxlen:
                    samples.append(feature)
                else:
                    data_exceed_maxlen += 1
                total_data += 1

            print("Processed " + str(total_data) + " data, removed " + str(
                data_exceed_maxlen) + " data that exceed the maximum length.")

            if cache:
                with open(cache_path, 'wb') as cf:
                    pickle.dump({'samples': samples, 'labels': labels}, cf)
        self.sample = samples
        self.label = labels

    def increase_with_sampling(self, total):
        inc_samp = [choice(self.sample) for i in range(total - len(self.sample))]
        self.sample.extend(inc_samp)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        self.sample[idx].update((k, np.asarray(v)) for k, v in self.sample[idx].items() if k != 'mapping')
        return self.sample[idx]


def get_data_from_file_row(fpath, text_index: int = 0, label_index: int = 1, separator=" "):
    tasks = defaultdict(list)
    task = 'default'
    labels = []
    with open(fpath, 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            for i in row[1].split(separator):
                if i not in labels and len(i.strip()) > 0:
                    labels.append(i)
                    labels.sort()
    tasks[task] = labels
    with open(fpath, 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for row in tqdm(f_csv):
            yield tasks, task, row[text_index].strip(), row[label_index].strip()


def get_data_from_file_col(fpath, text_index: int = 0, label_index: int = 1, separator=" "):
    tasks = defaultdict(list)
    task = 'default'
    labels = []
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in tqdm(lines):
            rows = line.split(' ')
            if len(rows) > 1:
                if rows[label_index] not in labels and len(rows[label_index]) > 0:
                    labels.append(rows[label_index])
                    labels.sort()
    tasks[task] = labels
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        x, y = "", ""
        for line in tqdm(lines):
            rows = line.split(' ')
            if len(rows) == 1:
                yield tasks, task, x.strip(), y.strip()
                x, y = "", ""
            else:
                if len(rows[text_index]) > 0:
                    x += rows[text_index].replace(" ", "_") + separator
                    y += rows[label_index].replace(" ", "_") + separator


def get_feature_from_data(tokenizer, labels, input, target=None, maxlen=512, separator=" "):
    # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
    row_dict = dict()
    tokenized_input = [tok_begin(tokenizer)] + tokenizer.tokenize(input) + [tok_sep(tokenizer)]
    input_id = tokenizer.convert_tokens_to_ids(tokenized_input)
    input = input.split()
    mapping_index = []

    pos = 1  # cls as start 0
    for i in input:
        for _ in range(len(tokenizer.tokenize(i))):
            if _ < 1:
                mapping_index.append({'char': i, 'pos': pos})
            pos += 1

    if target is not None:
        target = target.split(separator)
        target_token = []

        for i, t in zip(input, target):
            for _ in range(len(tokenizer.tokenize(i))):
                target_token += [labels.index(t)]

        target_id = [labels.index("O")] + target_token + [labels.index("O")]
        if len(input_id) != len(target_id):
            print("input target len not equal", len(input_id), len(target_id), len(input), len(target),
                  len(tokenized_input), len(target_token))
        target_id.extend([0] * (maxlen - len(target_id)))
        row_dict['target'] = target_id

    row_dict['mapping'] = json.dumps(mapping_index, ensure_ascii=False)
    mask_id = [1] * len(input_id)
    mask_id.extend([0] * (maxlen - len(mask_id)))
    row_dict['mask'] = mask_id
    row_dict['end'] = len(input_id)
    input_id.extend([0] * (maxlen - len(input_id)))
    row_dict['input'] = input_id

    # if debug:
    #     print("*** Example ***")
    #     print(f"input: {len(input_id)}, {list(zip(enumerate(input_id)))} ")
    #     print(f"mask: {len(mask_id)}, {list(zip(enumerate(mask_id)))} ")
    #     if target is not None:
    #         print(f"target: {len(target_id)}, {list(zip(enumerate(mask_id)))} ")
    #     print(f"mapping: {row_dict['mapping']} ")

    return row_dict
