import csv
import os
import pickle
from collections import defaultdict

import numpy as np
from torch.utils import data
from transformers import AutoTokenizer
from tqdm import tqdm
from utility.tok import *
import gen_once


class loadOneByOneDataset(data.Dataset):
    def __init__(self, fpath, pretrained, maxlen=510, cache=False):
        sample = []
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        cache_path = fpath + ".cache"
        if os.path.isfile(cache_path) and cache:
            with open(cache_path, "rb") as cf:
                sample = pickle.load(cf)
        else:
            for i in get_data_from_file(fpath):
                tasks, task, input, target, negative_text = i
                for j in range(1, len(target) + 1):
                    feature = get_feature_from_data(tokenizer, maxlen, input, " ".join(target[:j - 1]),
                                                    " ".join(target[:j]),
                                                    ntarget=negative_text)
                    if len(feature['input']) == len(feature['target']) == len(feature['ntarget']) <= maxlen:
                        sample.append(feature)

                feature = get_feature_from_data(tokenizer, maxlen, input, " ".join(target), " ".join(target),
                                                ntarget=negative_text)
                if len(feature['input']) == len(feature['target']) == len(feature['ntarget']) <= maxlen:
                    sample.append(feature)

                # # sentence level negative loss
                # feature = gen_once.data_loader.get_feature_from_data(tokenizer, maxlen, input, " ".join(target),
                #                                                      ntarget=negative_text)
                # if len(feature['input']) == len(feature['target']) == len(feature['ntarget']) == tokenizer.max_model_input_sizes[pretrained]:
                #     sample.append(feature)

                # if "[SEP]" in target:
                #     feature = data_loader.get_feature_from_data(tokenizer, maxlen, input, target, ntarget=negative_text,
                #                                                 debug=True)

            if cache:
                with open(cache_path, 'wb') as cf:
                    pickle.dump(sample, cf)
        self.sample = sample

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]


def get_data_from_file(fpath):
    tasks = defaultdict(list)
    task = 'default'
    tasks[task] = []
    with open(fpath, encoding='utf') as csvfile:
        for i in tqdm(list(csv.reader(csvfile))):
            source_text = i[0]
            target_text = i[1].split(" ")
            negative_text = i[2] if len(i) > 2 else None
            input = source_text
            target = target_text
            yield tasks, task, input, target, negative_text


def get_feature_from_data(tokenizer, maxlen, input, previous, target=None, ntarget=None):
    row_dict = dict()

    tokenized_input = [tok_begin(tokenizer)] + tokenizer.tokenize(input) + [tok_sep(tokenizer)]
    tokenized_previous = tokenizer.tokenize(previous)
    tokenized_input.extend(tokenized_previous)
    tokenized_input.append('[MASK]')
    tokenized_input_id = tokenizer.convert_tokens_to_ids(tokenized_input)
    mask_id = [1] * len(tokenized_input)
    target_start = len(tokenized_input_id) - 1
    tokenized_input_id.extend([0] * (maxlen - len(tokenized_input_id)))

    row_dict['target'] = np.asarray([-1] * maxlen)
    row_dict['ntarget'] = np.asarray([-1] * maxlen)

    if target is not None:
        tokenized_target = tokenizer.tokenize(target)
        if previous == target:
            tokenized_target = [tok_sep(tokenizer)]
        tokenized_target_id = [-1] * target_start
        tokenized_target_id.append(tokenizer.convert_tokens_to_ids(tokenized_target)[-1])
        tokenized_target_id.extend([-1] * (maxlen - len(tokenized_target_id)))
        row_dict['target'] = np.asarray(tokenized_target_id)
    if ntarget is not None:
        tokenized_ntarget = tokenizer.tokenize(ntarget)
        tokenized_ntarget_id = [-1] * target_start
        ntarget_token = tokenized_ntarget if len(previous) < len(tokenized_ntarget) else ["[SEP]"]
        ntarget_token_id = tokenizer.convert_tokens_to_ids(ntarget_token)[-1]
        tokenized_ntarget_id.append(ntarget_token_id)
        tokenized_ntarget_id.extend([-1] * (maxlen - len(tokenized_ntarget_id)))
        if tokenized_ntarget_id != tokenized_target_id:
            row_dict['ntarget'] = np.asarray(tokenized_ntarget_id)

    mask_id.extend([0] * (maxlen - len(mask_id)))
    type_id = [0] * len(tokenized_input)
    type_id.extend([1] * (maxlen - len(type_id)))
    row_dict['input'] = np.asarray(tokenized_input_id)
    row_dict['type'] = np.asarray(type_id)
    row_dict['mask'] = np.asarray(mask_id)
    row_dict['start'] = target_start

    # if True:
    #     print("*** Example ***")
    #     print(f"input: {len(row_dict['input'])}, {row_dict['input']} ")
    #     print(f"type: {len(row_dict['type'])}, {row_dict['type']} ")
    #     print(f"mask: {len(row_dict['mask'])}, {row_dict['mask']} ")
    #     if target is not None:
    #         print(f"target: {len(row_dict['target'])}, {row_dict['target']} ")
    #     if ntarget is not None:
    #         print("POS", target_start, len(tokenized_ntarget))
    #         print("STR", tokenized_target, tokenized_ntarget)
    #         print("ANS", tokenized_target[target_start], tokenized_ntarget_id)
    #         print(f"ntarget: {len(tokenized_ntarget_id)}, {row_dict['ntarget']} ")

    return row_dict
