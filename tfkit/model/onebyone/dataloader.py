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

    # each word in sentence
    for j in range(1, len(tokenized_target) + 1):
        if "neg" in likelihood or 'both' in likelihood:
            # formatting neg data in csv
            if n_target is None:
                ntext_arr = [tokenizer.convert_tokens_to_string(tokenized_target[:j - 1])]
            elif "[SEP]" in n_target:
                ntext_arr = [ntext.strip() for ntext in n_target.split("[SEP]")]
            else:
                ntext_arr = [n_target.strip()]
            # adding neg data
            for neg_text in ntext_arr:
                yield once.get_feature_from_data, {
                    **{'input': input + " " + " ".join(tokenized_target[:j - 1]),
                       'target': tokenized_target[:j][-1], 'ntarget': neg_text, "add_end_tok": False},
                    **param_dict}
        else:
            yield get_feature_from_data, {**{'input': input, 'previous': tokenized_target[:j - 1],
                                             'target': tokenized_target[:j], 'ntarget': None}, **param_dict}

    # end of the last word
    if "neg" in likelihood or 'both' in likelihood:
        # formatting neg data in csv
        if n_target is None:
            ntext_arr = [tokenizer.convert_tokens_to_string(tokenized_target[:j - 1])]
        elif "[SEP]" in n_target:
            ntext_arr = [ntext.strip() for ntext in n_target.split("[SEP]")]
        else:
            ntext_arr = [n_target.strip()]
        # adding neg data
        for neg_text in ntext_arr:
            yield get_feature_from_data, {**{'input': input, 'previous': tokenized_target,
                                             'target': [tok.tok_sep(tokenizer)], 'ntarget': neg_text}, **param_dict}
    else:
        yield get_feature_from_data, {**{'input': input, 'previous': tokenized_target,
                                         'target': [tok.tok_sep(tokenizer)], 'ntarget': None}, **param_dict}

    # whole sentence masking
    if 'pos' in likelihood:
        yield once.get_feature_from_data, {**{'input': input, 'target': " ".join(p_target)}, **param_dict}
    elif 'both' in likelihood or "neg" in likelihood:
        # formatting neg data in csv
        if n_target is None:
            ntext_arr = [tokenizer.convert_tokens_to_string(tokenized_target[:j - 1])]
        elif "[SEP]" in n_target:
            ntext_arr = [ntext.strip() for ntext in n_target.split("[SEP]")]
        else:
            ntext_arr = [n_target.strip()]
        for neg_text in ntext_arr:
            yield once.get_feature_from_data, {**{'input': input, 'target': " ".join(p_target), 'ntarget': neg_text},
                                               **param_dict}

    return get_feature_from_data, param_dict


def get_feature_from_data(tokenizer, maxlen, input, previous, target=None, ntarget=None, reserved_len=0,
                          handle_exceed='start_slice', **kwargs):
    feature_dict_list = []
    t_input_list, _ = tok.handle_exceed(tokenizer, input, maxlen - 2 - len(previous) -1, handle_exceed)
    for t_input in t_input_list:  # -2 for cls and sep
        row_dict = dict()
        t_input = [tok.tok_begin(tokenizer)] + \
                  t_input[:maxlen - reserved_len - 2] + \
                  [tok.tok_sep(tokenizer)]
        t_input.extend(previous)
        t_input.append(tok.tok_mask(tokenizer))
        t_input_id = tokenizer.convert_tokens_to_ids(t_input)
        mask_id = [1] * len(t_input)
        target_start = len(t_input_id) - 1
        target_end = maxlen
        t_input_id.extend([0] * (maxlen - len(t_input_id)))

        row_dict['target'] = [-1] * maxlen
        row_dict['ntarget'] = [-1] * maxlen

        tokenized_target_id = None
        if target is not None:
            tokenized_target_id = [-1] * target_start
            tokenized_target_id.append(tokenizer.convert_tokens_to_ids(target)[-1])
            target_end = len(tokenized_target_id) - 1
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
        type_id = [0] * len(t_input)
        type_id.extend([1] * (maxlen - len(type_id)))
        row_dict['input'] = t_input_id
        row_dict['type'] = type_id
        row_dict['mask'] = mask_id
        row_dict['start'] = target_start
        row_dict['end'] = target_end
        feature_dict_list.append(row_dict)

    return feature_dict_list
