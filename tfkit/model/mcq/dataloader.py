import csv
from collections import defaultdict
from tqdm import tqdm
import tfkit.utility.tok as tok


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
            yield tasks, task, input, [target]


def preprocessing_data(item, tokenizer, maxlen=512, handle_exceed='start_slice', **kwargs):
    tasks, task, input, target = item
    param_dict = {'input': input, 'tokenizer': tokenizer, 'target': target[0], 'maxlen': maxlen,
                  'handle_exceed': handle_exceed}
    yield get_feature_from_data, param_dict


def get_feature_from_data(tokenizer, maxlen, input, target=None, handle_exceed='start_slice', **kwargs):
    feature_dict_list = []
    t_input_list, _ = tok.handle_exceed(tokenizer, input, maxlen - 2, handle_exceed)

    for t_input in t_input_list:  # -2 for cls and sep
        row_dict = dict()
        tokenized_input = [tok.tok_begin(tokenizer)] + t_input + [tok.tok_sep(tokenizer)]
        tokenized_input_id = tokenizer.convert_tokens_to_ids(tokenized_input)

        row_dict['target'] = [-1] * maxlen
        if target is not None:
            tokenized_target = []
            targets_pointer = 0
            for tok_pos, text in enumerate(tokenized_input):
                if text == tok.tok_mask(tokenizer):
                    if targets_pointer == int(target):
                        tok_target = 1
                    else:
                        tok_target = 0
                    tokenized_target.extend([tok_target])
                    targets_pointer += 1
                else:
                    tokenized_target.append(-1)
            tokenized_target.extend([-1] * (maxlen - len(tokenized_target)))
            row_dict['target'] = tokenized_target
        target_pos_list = []
        for tok_pos, text in enumerate(tokenized_input):
            if text == tok.tok_mask(tokenizer):
                target_pos_list.append(tok_pos)
        target_pos_list.extend([0] * (4 - len(target_pos_list)))
        if len(target_pos_list) != 4:
            continue
        row_dict['target_pos'] = target_pos_list

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
