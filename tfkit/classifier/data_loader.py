import csv
from collections import defaultdict

import numpy as np
from torch.utils import data
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from utility.tok import *


class loadClassifierDataset(data.Dataset):
    def __init__(self, fpath, pretrained, maxlen=512, cache=False):
        samples = []
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        for i in tqdm(get_data_from_file(fpath)):
            tasks, task, input, target = i
            feature = get_feature_from_data(tokenizer, maxlen, tasks, task, input, target)
            if len(feature['input']) <= tokenizer.max_model_input_sizes[pretrained]:
                samples.append(feature)

        self.sample = samples
        self.task = tasks

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]


def get_data_from_file(fpath):
    tasks = defaultdict(list)
    with open(fpath, 'r', encoding='utf8', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)
        reader = list(reader)
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
    row_dict['input'] = np.asarray(tokenized_input_id)
    row_dict['mask'] = np.asarray(mask_id)
    row_dict['target'] = np.asarray([-1])
    if target is not None:
        if 'multi_target' in task:
            mlb = MultiLabelBinarizer(classes=task_lables)
            tar = mlb.fit_transform([target.split("/")])
            tokenize_label = tar
        else:
            tokenize_label = [task_lables[task].index(target[0])]
        row_dict['target'] = np.asarray(tokenize_label)

    return row_dict
