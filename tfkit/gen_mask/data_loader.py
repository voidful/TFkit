import csv
import os
import pickle
from collections import defaultdict
from random import choice

import numpy as np
from torch.utils import data
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer
import tfkit.utility.tok as tok


class loadMaskDataset(data.Dataset):
    def __init__(self, fpath, pretrained_config, maxlen=510, cache=False, handle_exceed='slide'):
        sample = []
        if 'albert_chinese' in pretrained_config:
            tokenizer = BertTokenizer.from_pretrained(pretrained_config)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_config)

        cache_path = fpath + "_maxlen" + str(maxlen) + "_" + pretrained_config.replace("/", "_") + ".cache"
        if os.path.isfile(cache_path) and cache:
            with open(cache_path, "rb") as cf:
                sample = pickle.load(cf)
        else:
            total_data = 0
            for i in get_data_from_file(fpath):
                tasks, task, input, target = i
                for feature in get_feature_from_data(tokenizer, maxlen, input, target, handle_exceed=handle_exceed):
                    print("feature", feature)
                    sample.append(feature)
                    total_data += 1
            print("Processed " + str(total_data) + " data.")

            if cache:
                with open(cache_path, 'wb') as cf:
                    pickle.dump(sample, cf)

        self.sample = sample

    def increase_with_sampling(self, total):
        inc_samp = [choice(self.sample) for i in range(total - len(self.sample))]
        self.sample.extend(inc_samp)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        self.sample[idx].update((k, np.asarray(v)) for k, v in self.sample[idx].items())
        return self.sample[idx]


def get_data_from_file(fpath):
    tasks = defaultdict(list)
    task = 'default'
    tasks[task] = []
    with open(fpath, encoding='utf') as csvfile:
        for i in tqdm(list(csv.reader(csvfile))):
            source_text = i[0]
            target_text = i[1]
            input = source_text
            target = target_text
            yield tasks, task, input, target


def get_feature_from_data(tokenizer, maxlen, input, target=None, handle_exceed='slide'):
    feature_dict_list = []
    t_input_list, _ = tok.handle_exceed(tokenizer, input, maxlen - 2, handle_exceed)
    for t_input in t_input_list:  # -2 for cls and sep
        row_dict = dict()
        tokenized_input = [tok.tok_begin(tokenizer)] + t_input + [tok.tok_sep(tokenizer)]
        row_dict['target'] = [-1] * maxlen
        tokenized_input_id = tokenizer.convert_tokens_to_ids(tokenized_input)
        if target is not None:
            targets_token = target.split(" ")
            tokenized_target = []
            targets_pointer = 0
            for tok_pos, text in enumerate(tokenized_input):
                if text == tok.tok_mask(tokenizer):
                    tok_target = tokenizer.tokenize(targets_token[targets_pointer])
                    tokenized_target.extend(tokenizer.convert_tokens_to_ids(tok_target))
                    targets_pointer += 1
                else:
                    tokenized_target.append(-1)

            mask_id = [1] * len(tokenized_target)
            type_id = [0] * len(tokenized_target)
            tokenized_target.extend([-1] * (maxlen - len(tokenized_target)))
            row_dict['target'] = tokenized_target
        else:
            mask_id = [1] * len(tokenized_input)
            type_id = [0] * len(tokenized_input)
        tokenized_input_id.extend(
            [tokenizer.convert_tokens_to_ids([tok.tok_pad(tokenizer)])[0]] * (maxlen - len(tokenized_input_id)))
        mask_id.extend([0] * (maxlen - len(mask_id)))
        type_id.extend([1] * (maxlen - len(type_id)))
        row_dict['input'] = tokenized_input_id
        row_dict['type'] = type_id
        row_dict['mask'] = mask_id
        feature_dict_list.append(row_dict)
    return feature_dict_list
