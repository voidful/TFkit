import csv
from collections import defaultdict
import tfkit.utility.tok as tok


def get_data_from_file(fpath):
    tasks = defaultdict(list)
    task = 'default'
    with open(fpath, 'r', encoding='utf8', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            context, start, end = row
            yield tasks, task, context, [start, end]


def preprocessing_data(item, tokenizer, maxlen=512, handle_exceed='slide', **kwargs):
    tasks, task, input, target = item
    param_dict = {'input': input, 'tokenizer': tokenizer, 'target': target, 'maxlen': maxlen,
                  'handle_exceed': handle_exceed}
    yield get_feature_from_data, param_dict


def get_feature_from_data(tokenizer, input, target=None, maxlen=512, handle_exceed='slide', **kwargs):
    feature_dict_list = []

    mapping_index = []
    pos = 1  # cls as start 0
    input_text_list = input.split(" ")
    for i in input_text_list:
        for _ in range(len(tokenizer.tokenize(i))):
            if _ < 1:
                mapping_index.append({'char': i, 'pos': pos})
            pos += 1

    t_input_list, t_pos_list = tok.handle_exceed(tokenizer, input, maxlen - 2, mode=handle_exceed)
    for t_input, t_pos in zip(t_input_list, t_pos_list):  # -2 for cls and sep:
        row_dict = dict()
        row_dict['target'] = [0, 0]
        tokenized_input = [tok.tok_begin(tokenizer)] + t_input + [tok.tok_sep(tokenizer)]
        input_id = tokenizer.convert_tokens_to_ids(tokenized_input)
        if target is not None:
            start, end = target
            ori_start = start = int(start)
            ori_end = end = int(end)
            ori_ans = input_text_list[ori_start:ori_end]
            start -= t_pos[0]
            end -= t_pos[0]
            if mapping_index[start]['pos'] > ori_end or start < 0 or start > maxlen or end >= maxlen - 2:
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
            if ori_ans != tokenized_input[start + 1:end + 1] and tokenizer.tokenize(
                    " ".join(ori_ans)) != tokenized_input[start + 1:end + 1] and start != end != 0:
                continue
            row_dict['target'] = [start + 1, end + 1]  # cls +1

        mask_id = [1] * len(input_id)
        mask_id.extend([0] * (maxlen - len(mask_id)))
        row_dict['mask'] = mask_id
        input_id.extend([0] * (maxlen - len(input_id)))
        row_dict['input'] = input_id
        row_dict['raw_input'] = tokenized_input
        feature_dict_list.append(row_dict)

    return feature_dict_list
