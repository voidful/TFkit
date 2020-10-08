import csv
import os
import pickle
from collections import defaultdict

import numpy as np
from torch.utils import data
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer
import tfkit.utility.tok as tok
from random import choice


class loadQADataset(data.Dataset):
    def __init__(self, fpath, pretrained, maxlen=512, cache=False, handle_exceed='slide'):
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
            for i in tqdm(get_data_from_file(fpath)):
                tasks, task, input, target = i
                for feature in get_feature_from_data(tokenizer, input, target, maxlen=maxlen,
                                                     handle_exceed=handle_exceed):
                    samples.append(feature)
                    total_data += 1

            print("Processed " + str(total_data) + " data.")

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
        self.sample[idx].pop('raw_input', None)
        self.sample[idx].update((k, np.asarray(v)) for k, v in self.sample[idx].items())
        return self.sample[idx]


def get_data_from_file(fpath):
    tasks = defaultdict(list)
    task = 'default'
    with open(fpath, 'r', encoding='utf8', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            context, start, end = row
            yield tasks, task, context, [start, end]


def get_feature_from_data(tokenizer, input_text, target=None, maxlen=512, separator=" ", handle_exceed='slide'):
    feature_dict_list = []

    mapping_index = []
    pos = 1  # cls as start 0
    input_text_list = input_text.split(" ")
    for i in input_text_list:
        for _ in range(len(tokenizer.tokenize(i))):
            if _ < 1:
                mapping_index.append({'char': i, 'pos': pos})
            pos += 1

    t_input_list, t_pos_list = tok.handle_exceed(tokenizer, input_text, maxlen - 2, mode=handle_exceed)
    for t_input, t_pos in zip(t_input_list, t_pos_list):  # -2 for cls and sep:
        row_dict = dict()
        row_dict['target'] = [0, 0]
        tokenized_input = [tok.tok_begin(tokenizer)] + t_input + [tok.tok_sep(tokenizer)]
        input_id = tokenizer.convert_tokens_to_ids(tokenized_input)
        ext_tok = []
        if target is not None:
            start, end = target
            ori_start = start = int(start)
            ori_end = end = int(end)
            ori_ans = input_text_list[ori_start:ori_end]

            if mapping_index[t_pos[0]]['pos'] > ori_end:
                start = 0
                end = 0
            else:
                for map_pos, map_tok in enumerate(mapping_index[t_pos[0]:]):
                    if t_pos[0] < map_tok['pos'] <= t_pos[1]:
                        length = len(tokenizer.tokenize(map_tok['char']))
                        if map_pos < ori_start:
                            start += length - 1
                        if map_pos < ori_end:
                            end += length - 1

            if ori_ans != tokenized_input[start + 1:end + 1]:
                if tokenizer.tokenize(" ".join(ori_ans)) != tokenized_input[start + 1:end + 1] and start != end != 0:
                    print("processed result change", "ORI ANS:", ori_ans, "TOK ANS:",
                          tokenized_input[start + 1:end + 1], ori_start, ori_end, start, end)
            row_dict['target'] = [start + 1, end + 1]  # cls +1

        mask_id = [1] * len(input_id)
        mask_id.extend([0] * (maxlen - len(mask_id)))
        row_dict['mask'] = mask_id
        input_id.extend([0] * (maxlen - len(input_id)))
        row_dict['input'] = input_id
        row_dict['raw_input'] = tokenized_input
        feature_dict_list.append(row_dict)

    return feature_dict_list
