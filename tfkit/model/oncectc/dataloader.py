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
            yield tasks, task, input, [target, None]


def preprocessing_data(item, tokenizer, maxlen=512, handle_exceed='start_slice', **kwargs):
    tasks, task, input, targets = item
    p_target, n_target = targets
    param_dict = {'input': input, 'tokenizer': tokenizer, 'target': p_target, 'maxlen': maxlen,
                  'handle_exceed': handle_exceed, "ntarget": n_target}
    yield get_feature_from_data, param_dict


def get_feature_from_data(tokenizer, maxlen, input, target=None, ntarget=None, reserved_len=0,
                          handle_exceed='start_slice', add_end_tok=True, **kwargs):
    feature_dict_list = []
    tokenized_target = tokenizer.tokenize(target) if target is not None else []
    t_input_list, _ = tok.handle_exceed(tokenizer, input, maxlen - 3 - len(tokenized_target), handle_exceed)
    for t_input in t_input_list:  # -2 for cls and sep and prediction end sep
        row_dict = dict()
        tokenized_input = [tok.tok_begin(tokenizer)] + t_input[:maxlen - reserved_len - 3] + [tok.tok_sep(tokenizer)]

        row_dict['target'] = [-1] * maxlen
        row_dict['target_once'] = [-1] * maxlen
        tokenized_input_id = tokenizer.convert_tokens_to_ids(tokenized_input)
        target_start = len(tokenized_input_id)
        target_end = maxlen
        target_length = target_end - target_start

        if target is not None:
            if add_end_tok:
                tokenized_target += [tok.tok_sep(tokenizer)]
            tokenized_target_id = []
            tokenized_target_once_id = [-1] * len(tokenized_input)
            target_ids = tokenizer.convert_tokens_to_ids(tokenized_target)
            target_length = len(target_ids)
            tokenized_target_id.extend(target_ids)
            tokenized_target_once_id.extend(target_ids)
            target_end = len(tokenized_target_id) - 1
            tokenized_target_id.extend([-1] * (maxlen - len(tokenized_target_id)))
            tokenized_target_once_id.extend([-1] * (maxlen - len(tokenized_target_once_id)))
            row_dict['target'] = tokenized_target_id
            row_dict['target_once'] = tokenized_target_once_id

        input_length = min(maxlen, target_start * 3)
        tokenized_input_id.extend([tokenizer.mask_token_id] * (maxlen - len(tokenized_input_id)))
        mask_id = [1] * input_length
        mask_id.extend([0] * (maxlen - len(mask_id)))
        row_dict['input'] = tokenized_input_id
        row_dict['mask'] = mask_id
        row_dict['start'] = target_start
        row_dict['end'] = target_end
        row_dict['input_length'] = input_length
        row_dict['target_length'] = target_length
        feature_dict_list.append(row_dict)

    return feature_dict_list
