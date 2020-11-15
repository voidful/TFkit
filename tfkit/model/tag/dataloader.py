import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import csv
from collections import defaultdict
from tqdm import tqdm
import tfkit.utility.tok as tok


def get_data_from_file(fpath, text_index: int = 0, label_index: int = 1, separator=" ", **kwargs):
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
            yield tasks, task, row[text_index].strip(), [row[label_index].strip()]


def get_data_from_file_col(fpath, text_index: int = 0, label_index: int = 1, separator=" ", **kwargs):
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
                yield tasks, task, x.strip(), [y.strip()]
                x, y = "", ""
            else:
                if len(rows[text_index]) > 0:
                    x += rows[text_index].replace(" ", "_") + separator
                    y += rows[label_index].replace(" ", "_") + separator


def preprocessing_data(item, tokenizer, maxlen=512, handle_exceed='slide', separator=" ", **kwargs):
    tasks, task, input, target = item
    param_dict = {'input': input, 'tokenizer': tokenizer, 'target': target[0], 'maxlen': maxlen,
                  'separator': separator, 'handle_exceed': handle_exceed, 'labels': tasks[task]}
    yield get_feature_from_data, param_dict


def get_feature_from_data(tokenizer, labels, input, target=None, maxlen=512, separator=" ", handle_exceed='slide'):
    feature_dict_list = []

    mapping_index = []
    pos = 0
    for i in input.split(" "):
        for _ in range(len(tokenizer.tokenize(i))):
            if _ < 1:
                mapping_index.append({'char': i, 'pos': pos})
            pos += 1
    if target is not None:
        target = target.split(separator)

    t_input_list, t_pos_list = tok.handle_exceed(tokenizer, input, maxlen - 2, mode=handle_exceed, keep_after_sep=False)
    for t_input, t_pos in zip(t_input_list, t_pos_list):  # -2 for cls and sep
        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        row_dict = dict()
        tokenized_input = [tok.tok_begin(tokenizer)] + t_input + [tok.tok_sep(tokenizer)]
        input_id = tokenizer.convert_tokens_to_ids(tokenized_input)

        if target is not None:
            target_token = []
            pev = 0
            for tok_map, target_label in zip(mapping_index, target):
                if t_pos[0] < tok_map['pos'] <= t_pos[1]:
                    for _ in range(tok_map['pos'] - pev):
                        target_token += [labels.index(target_label)]
                pev = tok_map['pos']

            if "O" in labels:
                target_id = [labels.index("O")] + target_token + [labels.index("O")]
            else:
                target_id = [target_token[0]] + target_token + [target_token[-1]]

            if len(input_id) != len(target_id):
                print(input, target)
                print("input target len not equal ", len(input_id), len(target_id))
                continue
            target_id.extend([0] * (maxlen - len(target_id)))
            row_dict['target'] = target_id

        map_start = 0
        map_end = len(mapping_index)
        for pos, tok_map in enumerate(mapping_index):
            if t_pos[0] == tok_map['pos']:
                map_start = pos
            elif t_pos[1] == tok_map['pos']:
                map_end = pos

        row_dict['mapping'] = mapping_index[map_start:map_end]
        mask_id = [1] * len(input_id)
        mask_id.extend([0] * (maxlen - len(mask_id)))
        row_dict['mask'] = mask_id
        row_dict['end'] = len(input_id)
        input_id.extend([0] * (maxlen - len(input_id)))
        row_dict['input'] = input_id
        row_dict['pos'] = [map_start, map_end]
        feature_dict_list.append(row_dict)

    return feature_dict_list
