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
        mask_id = [1] * len(tokenized_input)
        type_id = [0] * len(tokenized_input)

        row_dict['target'] = [-1] * maxlen
        row_dict['ntarget'] = [-1] * maxlen

        tokenized_input_id = tokenizer.convert_tokens_to_ids(tokenized_input)
        target_start = len(tokenized_input_id)
        target_end = maxlen

        if target is not None:
            if add_end_tok:
                tokenized_target += [tok.tok_sep(tokenizer)]
            tokenized_target_id = [-1] * len(tokenized_input)
            tokenized_target_id.extend(tokenizer.convert_tokens_to_ids(tokenized_target))
            target_end = len(tokenized_target_id) - 1
            tokenized_target_id.extend([-1] * (maxlen - len(tokenized_target_id)))
            row_dict['target'] = tokenized_target_id

        if ntarget is not None:
            tokenized_ntarget = tokenizer.tokenize(ntarget)
            tokenized_ntarget_id = [-1] * target_start
            tokenized_ntarget_id.extend(tokenizer.convert_tokens_to_ids(tokenized_ntarget))
            tokenized_ntarget_id.extend([-1] * (maxlen - len(tokenized_ntarget_id)))
            if len(tokenized_ntarget_id) <= maxlen:
                row_dict['ntarget'] = tokenized_ntarget_id

        tokenized_input_id.extend([tokenizer.mask_token_id] * (maxlen - len(tokenized_input_id)))
        mask_id.extend([0] * (maxlen - len(mask_id)))
        type_id.extend([1] * (maxlen - len(type_id)))

        row_dict['input'] = tokenized_input_id
        row_dict['type'] = type_id
        row_dict['mask'] = mask_id
        row_dict['start'] = target_start
        row_dict['end'] = target_end
        feature_dict_list.append(row_dict)
    return feature_dict_list
