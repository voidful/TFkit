import tfkit.utility.tok as tok
from tfkit.utility.dataloader import get_gen_data_from_file

get_data_from_file = get_gen_data_from_file


def preprocessing_data(item, tokenizer, maxlen=512, handle_exceed='start_slice', reserved_len=0, **kwargs):
    tasks, task, input, targets = item
    p_target, b_target = targets
    previous = []
    if tok.UNIVERSAL_SEP in input:
        part = input.split(tok.UNIVERSAL_SEP)
        previous = tokenizer.tokenize(part[-1])
        input = "".join(part[:-1])

    tokenized_target = tokenizer.tokenize(p_target)
    param_dict = {'tokenizer': tokenizer, 'maxlen': maxlen, 'handle_exceed': handle_exceed,
                  'reserved_len': reserved_len}

    if b_target and len(b_target) > 0:
        yield get_feature_from_data, {**{'input': input, 'previous': previous,
                                         'target': tokenized_target, 'btarget': b_target.strip()}, **param_dict}
    else:
        yield get_feature_from_data, {**{'input': input, 'previous': previous,
                                         'target': tokenized_target, 'btarget': None}, **param_dict}

    return get_feature_from_data, param_dict


def get_feature_from_data(tokenizer, maxlen, input, previous, target=None, btarget=None, reserved_len=0,
                          handle_exceed='noop', **kwargs):
    feature_dict_list = []
    tok_pad = tok.tok_pad(tokenizer)
    tok_bos = tok.tok_begin(tokenizer)
    tok_sep = tok.tok_sep(tokenizer)
    tok_mask = tok.tok_mask(tokenizer)
    pred_len = len(tokenizer.convert_tokens_to_ids(target)) + 1 if target is not None else len(previous) - 1
    t_input_list, _ = tok.handle_exceed(tokenizer, input, maxlen - 2 - pred_len, handle_exceed)
    for t_input in t_input_list:  # -2 for cls and sep
        row_dict = dict()
        t_input = [tok_bos] + \
                  t_input[:maxlen - reserved_len - 2] + \
                  [tok_sep]
        t_input_id = tokenizer.convert_tokens_to_ids(t_input)
        encoder_mask_id = [1] * (len(t_input))
        encoder_mask_id.extend([0] * (maxlen - len(encoder_mask_id)))
        t_input_id.extend(tokenizer.convert_tokens_to_ids([tok_pad]) * (maxlen - len(t_input_id)))

        if target is not None:
            tokenized_target_id = []
            if previous is None:  # pm
                tokenized_prev_id = [tokenizer.convert_tokens_to_ids(tok_mask)] * maxlen
            else:
                tokenized_prev_id = tokenizer.convert_tokens_to_ids([tok_sep] + target)
            tokenized_target_id.extend(tokenizer.convert_tokens_to_ids(target + [tok_sep]))
            decoder_mask_id = [1] * (len(tokenized_prev_id))
            decoder_mask_id.extend([0] * (maxlen - len(decoder_mask_id)))
            tokenized_prev_id.extend(
                tokenizer.convert_tokens_to_ids([tok_pad]) * (maxlen - len(tokenized_prev_id)))
            tokenized_target_id.extend([-1] * (maxlen - len(tokenized_target_id)))
            row_dict['target'] = tokenized_target_id
            row_dict['prev'] = tokenized_prev_id
            row_dict['btarget'] = tokenizer.convert_tokens_to_ids([tok_pad]) * maxlen
            if btarget is not None and len(tokenizer.tokenize(btarget)) > 0:
                tokenized_ntarget = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(btarget))
                tokenized_ntarget_id = tokenized_ntarget
                tokenized_ntarget_id.extend(
                    tokenizer.convert_tokens_to_ids([tok_pad]) * (maxlen - len(tokenized_ntarget_id)))
                if len(tokenized_ntarget_id) <= maxlen:
                    row_dict['btarget'] = tokenized_ntarget_id
        else:
            tokenized_prev_id = [tokenizer.convert_tokens_to_ids(tok_sep)]
            tokenized_prev_id.extend(tokenizer.convert_tokens_to_ids(previous))
            target_start = len(tokenized_prev_id) - 1
            row_dict['start'] = target_start
            decoder_mask_id = [1] * (len(tokenized_prev_id))
            row_dict['prev'] = tokenized_prev_id

        row_dict['input'] = t_input_id
        row_dict['encoder_mask'] = encoder_mask_id
        row_dict['decoder_mask'] = decoder_mask_id
        feature_dict_list.append(row_dict)

    return feature_dict_list
