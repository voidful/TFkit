import tfkit.utility.tok as tok
import tfkit.model.once as once


from tfkit.utility.dataloader import get_gen_data_from_file

get_data_from_file = get_gen_data_from_file


def preprocessing_data(item, tokenizer, maxlen=512, handle_exceed='start_slice',
                       likelihood=['none', 'pos', 'neg', 'both'], reserved_len=0, **kwargs):
    likelihood = likelihood[0] if isinstance(likelihood, list) else likelihood
    tasks, task, input, targets = item
    p_target, n_target = targets
    tokenized_target = tokenizer.tokenize(p_target)
    param_dict = {'tokenizer': tokenizer, 'maxlen': maxlen, 'handle_exceed': handle_exceed,
                  'reserved_len': reserved_len}

    if "neg" in likelihood or 'both' in likelihood:
        # formatting neg data in csv
        if n_target is None:
            ntext_arr = [tokenizer.convert_tokens_to_string([tok.tok_begin(tokenizer)] + tokenized_target)]
        elif "[SEP]" in n_target:
            ntext_arr = [ntext.strip() for ntext in n_target.split("[SEP]")]
        else:
            ntext_arr = [n_target.strip()]
        for neg_text in ntext_arr:
            yield get_feature_from_data, {**{'input': input, 'previous': [],
                                             'target': tokenized_target, 'ntarget': neg_text}, **param_dict}
    else:
        yield get_feature_from_data, {**{'input': input, 'previous': [],
                                         'target': tokenized_target, 'ntarget': None}, **param_dict}

    # whole sentence masking
    if 'pos' in likelihood:
        yield once.get_feature_from_data, {**{'input': input, 'target': p_target}, **param_dict}
    elif 'both' in likelihood:
        for neg_text in ntext_arr:
            yield once.get_feature_from_data, {**{'input': input, 'target': p_target, 'ntarget': neg_text},
                                               **param_dict}

    return get_feature_from_data, param_dict


def get_feature_from_data(tokenizer, maxlen, input, previous, target=None, ntarget=None, reserved_len=0,
                          handle_exceed='noop', **kwargs):
    feature_dict_list = []
    pred_len = len(tokenizer.convert_tokens_to_ids(target)) if target is not None else len(previous)
    t_input_list, _ = tok.handle_exceed(tokenizer, input, maxlen - 3 - pred_len - reserved_len, handle_exceed)
    for t_input in t_input_list:  # -2 for cls and sep
        row_dict = dict()
        t_input = [tok.tok_begin(tokenizer)] + t_input + [tok.tok_begin(tokenizer)]
        t_input.extend(previous)
        t_input_id = tokenizer.convert_tokens_to_ids(t_input)
        target_start = len(t_input_id) - 1

        row_dict['target'] = [-1] * maxlen
        row_dict['ntarget'] = [-1] * maxlen

        if target is not None:
            t_input_id.extend(tokenizer.convert_tokens_to_ids(target[:len(target)]))
            tokenized_target_id = [-1] * target_start
            # tokenized_target_id = tokenizer.convert_tokens_to_ids(t_input[1:])
            tokenized_target_id.extend(tokenizer.convert_tokens_to_ids(target + [tok.tok_sep(tokenizer)]))
            tokenized_target_id.extend([-1] * (maxlen - len(tokenized_target_id)))
            row_dict['target'] = tokenized_target_id
        if ntarget is not None and len(tokenizer.tokenize(ntarget)) > 0:
            tokenized_ntarget = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ntarget))
            tokenized_ntarget_id = [-1] * target_start
            # tokenized_ntarget_id = tokenizer.convert_tokens_to_ids(t_input[1:])
            tokenized_ntarget_id.extend(tokenized_ntarget)
            tokenized_ntarget_id.extend([-1] * (maxlen - len(tokenized_ntarget_id)))
            if len(tokenized_ntarget_id) <= maxlen:
                row_dict['ntarget'] = tokenized_ntarget_id

        mask_id = [1] * len(t_input_id)
        t_input_id.extend(tokenizer.convert_tokens_to_ids([tok.tok_pad(tokenizer)]) * (maxlen - len(t_input_id)))
        mask_id.extend([0] * (maxlen - len(mask_id)))
        row_dict['input'] = t_input_id
        row_dict['mask'] = mask_id
        row_dict['start'] = target_start
        feature_dict_list.append(row_dict)

    return feature_dict_list
