import os
import pickle
import csv
from collections import defaultdict
from random import choice

import numpy as np
from torch.utils import data
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from utility.tok import *


class loadClassifierDataset(data.Dataset):
    def __init__(self, fpath, pretrained, maxlen=512, cache=False):
        sample = []
        if 'albert_chinese' in pretrained:
            tokenizer = BertTokenizer.from_pretrained(pretrained)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained)

        cache_path = fpath + "_maxlen" + str(maxlen) + "_" + pretrained.replace("/", "_") + ".cache"
        if os.path.isfile(cache_path) and cache:
            with open(cache_path, "rb") as cf:
                outdata = pickle.load(cf)
                sample = outdata['sample']
                task_dict = outdata['task']
        else:
            task_dict = {}
            total_data = 0
            data_exceed_maxlen = 0
            for i in tqdm(get_data_from_file(fpath)):
                all_task, task, input, target = i
                task_dict.update(all_task)
                feature = get_feature_from_data(tokenizer, maxlen, all_task, task, input, target)
                if len(feature['input']) <= maxlen:
                    sample.append(feature)
                else:
                    data_exceed_maxlen += 1
                total_data += 1

            print("Processed " + str(total_data) + " data, removed " + str(
                data_exceed_maxlen) + " data that exceed the maximum length.")

            if cache:
                with open(cache_path, 'wb') as cf:
                    outdata = {'sample': sample, 'task': task_dict}
                    pickle.dump(outdata, cf)

        self.sample = sample
        self.task = task_dict

    def increase_with_sampling(self, total):
        inc_samp = [choice(self.sample) for i in range(total - len(self.sample))]
        self.sample.extend(inc_samp)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        self.sample[idx].update((k, np.asarray(v)) for k, v in self.sample[idx].items() if k != 'task')
        return self.sample[idx]


def get_data_from_file(fpath):
    tasks = defaultdict(list)
    with open(fpath, 'r', encoding='utf8', newline='') as csvfile:
        reader = csv.reader(csvfile)
        reader = list(reader)
        headers = ['input'] + ['target_' + str(i) for i in range(len(reader[0]) - 1)]
        for row in reader:
            start_pos = 1
            for pos, item in enumerate(row[start_pos:]):
                pos += start_pos
                task = headers[0] + "_" + headers[pos]
                item = item.strip()
                if '/' in item:
                    for i in item.split("/"):
                        tasks[task].append(i) if i not in tasks[task] else tasks[task]
                else:
                    tasks[task].append(item) if item not in tasks[task] else tasks[task]
                tasks[task].sort()

        for row in reader:
            start_pos = 1
            for pos, item in enumerate(row[start_pos:]):
                pos += start_pos
                task = headers[0] + "_" + headers[pos]
                item = item.strip()
                target = item.split('/') if '/' in item else [item]
                input = row[0]
                yield tasks, task, input, target


def get_feature_from_data(tokenizer, maxlen, task_lables, task, input, target=None):
    row_dict = dict()
    row_dict['task'] = task
    # bert embedding
    # inputs[id] += "[SEP]".join(task_lables)

    input_token = [tok_begin(tokenizer)] + tokenizer.tokenize(input) + [tok_sep(tokenizer)]
    tokenized_input_id = tokenizer.convert_tokens_to_ids(input_token)
    mask_id = [1] * len(tokenized_input_id)
    tokenized_input_id.extend([tokenizer.pad_token_id] * (maxlen - len(tokenized_input_id)))
    mask_id.extend([-1] * (maxlen - len(mask_id)))
    # tokenized_input = []
    # for i in list_in_windows(token_input_id, maxlen):
    #     tokenized_input.append(i)
    row_dict['input'] = tokenized_input_id
    row_dict['mask'] = mask_id
    row_dict['target'] = [-1]
    if target is not None:
        if 'multi_target' in task:
            mlb = MultiLabelBinarizer(classes=task_lables)
            tar = mlb.fit_transform([target.split("/")])
            tokenize_label = tar
        else:
            tokenize_label = [task_lables[task].index(target[0])]
        row_dict['target'] = tokenize_label

    return row_dict
