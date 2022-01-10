import numpy as np

from tfkit.utility.datafile import get_gen_data_from_file

get_data_from_file = get_gen_data_from_file


def preprocessing_data(item, tokenizer,  **kwargs):
    tasks, task, input, targets = item
    p_target, n_target = targets
    param_dict = {'input': input, 'tokenizer': tokenizer, 'target': p_target, "ntarget": n_target}
    param_dict.update(kwargs)
    yield get_feature_from_data, param_dict


def get_feature_from_data(tokenizer, maxlen, input, target=None, ntarget=None, **kwargs):
    row_dict = dict()
    mask_id = [1] * len(input)
    type_id = [0] * len(input)

    row_dict['target'] = [-1] * maxlen
    row_dict['ntarget'] = [-1] * maxlen

    tokenized_input_id = tokenizer.convert_tokens_to_ids(input)
    target_start = len(tokenized_input_id)
    target_end = maxlen

    if target is not None:
        tokenized_target_id = [-1] * len(input)
        tokenized_target_id.extend(tokenizer.convert_tokens_to_ids(target))
        target_end = len(tokenized_target_id) - 1
        tokenized_target_id.extend([-1] * (maxlen - len(tokenized_target_id)))
        row_dict['target'] = np.array(tokenized_target_id)

    if ntarget is not None:
        tokenized_ntarget_id = [-1] * target_start
        tokenized_ntarget_id.extend(tokenizer.convert_tokens_to_ids(ntarget))
        tokenized_ntarget_id.extend([-1] * (maxlen - len(tokenized_ntarget_id)))
        if len(tokenized_ntarget_id) <= maxlen:
            row_dict['ntarget'] = np.array(tokenized_ntarget_id)

    tokenized_input_id.extend([tokenizer.mask_token_id] * (maxlen - len(tokenized_input_id)))
    mask_id.extend([1] * (maxlen - len(mask_id)))
    type_id.extend([1] * (maxlen - len(type_id)))

    row_dict['input'] = np.array(tokenized_input_id)
    row_dict['type'] = np.array(type_id)
    row_dict['mask'] = np.array(mask_id)
    row_dict['start'] = np.array(target_start)
    row_dict['end'] = np.array(target_end)
    return row_dict
