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
import tfkit.gen_once as gen_once


class loadOneByOneDataset(data.Dataset):

    def __init__(self, fpath, pretrained_config, maxlen=510, cache=False, likelihood='', pos_ratio=1, neg_ratio=1,
                 handle_exceed='start_slice', **kwargs):
        self.maxlen = maxlen
        sample = []
        if 'albert_chinese' in pretrained_config:
            tokenizer = BertTokenizer.from_pretrained(pretrained_config)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_config)

        neg_info = "_" + likelihood + "_pos_" + str(pos_ratio) + "_neg_" + str(pos_ratio)
        cache_path = fpath + "_maxlen" + str(maxlen) + "_" + pretrained_config.replace("/", "_") + neg_info + ".cache"
        if os.path.isfile(cache_path) and cache:
            with open(cache_path, "rb") as cf:
                sample = pickle.load(cf)
        else:
            total_data = 0
            data_invalid = 0
            for i in get_data_from_file(fpath):
                tasks, task, input, target, negative_text = i
                input = input.strip()
                tokenized_target = tokenizer.tokenize(" ".join(target))
                # each word in sentence
                for j in range(1, len(tokenized_target) + 1):
                    feature = get_feature_from_data(tokenizer, maxlen, input, tokenized_target[:j - 1],
                                                    tokenized_target[:j], handle_exceed=handle_exceed)[-1]
                    if "neg" in likelihood or 'both' in likelihood:
                        # formatting neg data in csv
                        if negative_text is None:
                            ntext_arr = [tokenizer.convert_tokens_to_string(tokenized_target[:j - 1])]
                        elif "[SEP]" in negative_text:
                            ntext_arr = [ntext.strip() for ntext in negative_text.split("[SEP]")]
                        else:
                            ntext_arr = [negative_text.strip()]
                        # adding neg data
                        for neg_text in ntext_arr:
                            feature_neg = gen_once.data_loader.get_feature_from_data(tokenizer, maxlen, input,
                                                                                     ntarget=neg_text,
                                                                                     handle_exceed=handle_exceed)[-1]
                            feature['ntarget'] = feature_neg['ntarget']
                            if self.check_feature_valid(feature):
                                sample.append(feature)
                            else:
                                data_invalid += 1
                            total_data += 1
                    else:
                        if self.check_feature_valid(feature):
                            sample.append(feature)
                        else:
                            data_invalid += 1
                        total_data += 1

                # end of the last word
                feature = get_feature_from_data(tokenizer, maxlen, input, tokenized_target,
                                                [tok.tok_sep(tokenizer)], handle_exceed=handle_exceed)[-1]
                if "neg" in likelihood or 'both' in likelihood:
                    # formatting neg data in csv
                    if negative_text is None:
                        ntext_arr = [tokenizer.convert_tokens_to_string(tokenized_target[:j - 1])]
                    elif "[SEP]" in negative_text:
                        ntext_arr = [ntext.strip() for ntext in negative_text.split("[SEP]")]
                    else:
                        ntext_arr = [negative_text.strip()]
                    # adding neg data
                    for neg_text in ntext_arr:
                        feature_neg = gen_once.data_loader.get_feature_from_data(tokenizer, maxlen, input,
                                                                                 ntarget=neg_text)[-1]
                        feature['ntarget'] = feature_neg['ntarget']
                        if self.check_feature_valid(feature):
                            sample.append(feature)
                        else:
                            data_invalid += 1
                        total_data += 1
                else:
                    if self.check_feature_valid(feature):
                        sample.append(feature)
                    else:
                        data_invalid += 1
                    total_data += 1

                # whole sentence masking
                if 'pos' in likelihood:
                    feature = gen_once.data_loader.get_feature_from_data(tokenizer, maxlen, input,
                                                                         " ".join(target),
                                                                         handle_exceed=handle_exceed)[-1]
                    if self.check_feature_valid(feature):
                        for _ in range(int(pos_ratio)):
                            sample.append(feature)
                    else:
                        data_invalid += 1
                    total_data += 1
                elif 'both' in likelihood:
                    # formatting neg data in csv
                    if negative_text is None:
                        ntext_arr = [tokenizer.convert_tokens_to_string(tokenized_target[:j - 1])]
                    elif "[SEP]" in negative_text:
                        ntext_arr = [ntext.strip() for ntext in negative_text.split("[SEP]")]
                    else:
                        ntext_arr = [negative_text.strip()]

                    for neg_text in ntext_arr:
                        if 'neg' in likelihood:
                            feature = gen_once.data_loader.get_feature_from_data(tokenizer, maxlen, input,
                                                                                 ntarget=neg_text,
                                                                                 handle_exceed=handle_exceed)[-1]
                        elif 'both' in likelihood:
                            feature = gen_once.data_loader.get_feature_from_data(tokenizer, maxlen, input,
                                                                                 " ".join(target),
                                                                                 ntarget=neg_text,
                                                                                 handle_exceed=handle_exceed)[-1]
                        if self.check_feature_valid(feature):
                            for _ in range(int(neg_ratio)):
                                sample.append(feature)
                        else:
                            data_invalid += 1
                        total_data += 1

            print("Processed " + str(total_data) + " data, removed " + str(
                data_invalid) + " invalid data.")

            if cache:
                with open(cache_path, 'wb') as cf:
                    pickle.dump(sample, cf)
        self.sample = sample

    def increase_with_sampling(self, total):
        inc_samp = [choice(self.sample) for _ in range(total - len(self.sample))]
        self.sample.extend(inc_samp)

    def check_feature_valid(self, feature):
        if feature['target'][feature['start']] == feature['ntarget'][feature['start']]:
            feature['ntarget'][feature['start']] = -1
        return True

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
            target_text = i[1].strip().split(" ")
            negative_text = i[2].strip() if len(i) > 2 else None
            input = source_text
            target = target_text
            yield tasks, task, input, target, negative_text


def get_feature_from_data(tokenizer, maxlen, input, tokenized_previous, tokenized_target=None, ntarget=None,
                          reserved_len=0, handle_exceed='start_slice'):
    feature_dict_list = []
    t_input_list, _ = tok.handle_exceed(tokenizer, input, maxlen - 2, handle_exceed)
    for t_input in t_input_list:  # -2 for cls and sep
        row_dict = dict()
        tokenized_input = t_input
        tokenized_input = [tok.tok_begin(tokenizer)] + tokenized_input[:maxlen - reserved_len - 2] \
                          + [tok.tok_sep(tokenizer)]
        tokenized_input.extend(tokenized_previous)
        tokenized_input.append(tok.tok_mask(tokenizer))
        tokenized_input_id = tokenizer.convert_tokens_to_ids(tokenized_input)
        mask_id = [1] * len(tokenized_input)
        target_start = len(tokenized_input_id) - 1
        tokenized_input_id.extend([0] * (maxlen - len(tokenized_input_id)))

        row_dict['target'] = [-1] * maxlen
        row_dict['ntarget'] = [-1] * maxlen

        tokenized_target_id = None
        if tokenized_target is not None:
            tokenized_target_id = [-1] * target_start
            tokenized_target_id.append(tokenizer.convert_tokens_to_ids(tokenized_target)[-1])
            tokenized_target_id.extend([-1] * (maxlen - len(tokenized_target_id)))
            row_dict['target'] = tokenized_target_id
        if ntarget is not None:
            tokenized_ntarget = tokenizer.tokenize(ntarget)
            tokenized_ntarget_id = [-1] * target_start
            ntarget_token_id = tokenizer.convert_tokens_to_ids(tokenized_ntarget)[-1]
            tokenized_ntarget_id.append(ntarget_token_id)
            tokenized_ntarget_id.extend([-1] * (maxlen - len(tokenized_ntarget_id)))
            if tokenized_target_id is None or tokenized_ntarget_id != tokenized_target_id:
                row_dict['ntarget'] = tokenized_ntarget_id

        mask_id.extend([0] * (maxlen - len(mask_id)))
        type_id = [0] * len(tokenized_input)
        type_id.extend([1] * (maxlen - len(type_id)))
        row_dict['input'] = tokenized_input_id
        row_dict['type'] = type_id
        row_dict['mask'] = mask_id
        row_dict['start'] = target_start
        feature_dict_list.append(row_dict)

    return feature_dict_list
