import csv
from collections import defaultdict
from tqdm import tqdm
import tfkit.utility.tok as tok
import tfkit.model.once as once


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
            yield tasks, task, input, [target, negative_text]


def preprocessing_data(item, tokenizer, maxlen=512, handle_exceed='start_slice',
                       likelihood=['none', 'pos', 'neg', 'both'], reserved_len=0, **kwargs):
    likelihood = likelihood[0] if isinstance(likelihood, list) else likelihood
    tasks, task, input, targets = item
    p_target, n_target = targets
    input = input.strip()

    tokenized_target = tokenizer.tokenize(" ".join(p_target))
    param_dict = {'tokenizer': tokenizer, 'maxlen': maxlen, 'handle_exceed': handle_exceed,
                  'reserved_len': reserved_len}

    yield get_feature_from_data, {**{'input': input, 'previous': [],
                                     'target': tokenized_target, 'ntarget': None}, **param_dict}


def get_feature_from_data(tokenizer, maxlen, input, previous, target=None, ntarget=None, reserved_len=0,
                          handle_exceed='noop', **kwargs):
    feature_dict_list = []
    t_input_list, _ = tok.handle_exceed(tokenizer, input, maxlen - 2 - len(previous) - 1, handle_exceed)
    for t_input in t_input_list:  # -2 for cls and sep
        row_dict = dict()
        t_input = [tok.tok_begin(tokenizer)] + \
                  t_input[:maxlen - reserved_len - 2] + \
                  [tok.tok_sep(tokenizer)]
        t_input.extend(previous)
        t_input_id = tokenizer.convert_tokens_to_ids(t_input)
        mask_id = [1] * len(t_input)
        target_start = len(t_input_id) - 1
        target_end = maxlen
        t_input_id.extend([0] * (maxlen - len(t_input_id)))
        row_dict['target'] = [-100] * maxlen
        row_dict['ntarget'] = [-1] * maxlen
        tokenized_target_id = None
        if target is not None:
            tokenized_target_id = [-100] * (target_start + 1)
            tokenized_target_id.extend(tokenizer.convert_tokens_to_ids(target+[tok.tok_sep(tokenizer)]))
            target_end = len(tokenized_target_id) - 1
            tokenized_target_id.extend([-100] * (maxlen - len(tokenized_target_id)))
            row_dict['target'] = tokenized_target_id
        if ntarget is not None and len(tokenizer.tokenize(ntarget)) > 0:
            tokenized_ntarget = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ntarget))
            tokenized_ntarget_id = [-1] * (target_start + 1)
            tokenized_ntarget_id.extend(tokenized_ntarget)
            tokenized_ntarget_id.extend([-1] * (maxlen - len(tokenized_ntarget_id)))
            if len(tokenized_ntarget_id) <= maxlen:
                row_dict['ntarget'] = tokenized_ntarget_id

        mask_id.extend([0] * (maxlen - len(mask_id)))
        type_id = [0] * len(t_input)
        type_id.extend([1] * (maxlen - len(type_id)))
        row_dict['input'] = t_input_id
        row_dict['type'] = type_id
        row_dict['mask'] = mask_id
        row_dict['start'] = target_start
        row_dict['end'] = target_end
        feature_dict_list.append(row_dict)

    return feature_dict_list
