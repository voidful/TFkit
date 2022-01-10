import tfkit.utility.tok as tok
from tfkit.utility.datafile import get_clas_data_from_file

get_data_from_file = get_clas_data_from_file


def get_feature_from_data(tokenizer, maxlen, input, target=None, **kwargs):
    row_dict = dict()
    tokenized_input = [tok.tok_begin(tokenizer)] + input + [tok.tok_sep(tokenizer)]
    row_dict['target'] = [-1] * maxlen
    tokenized_input_id = tokenizer.convert_tokens_to_ids(tokenized_input)
    if target is not None:
        targets_token = target.split(" ")
        tokenized_target = []
        targets_pointer = 0
        for tok_pos, text in enumerate(tokenized_input):
            if text == tok.tok_mask(tokenizer):
                tok_target = tokenizer.tokenize(targets_token[targets_pointer])
                tokenized_target.extend(tokenizer.convert_tokens_to_ids(tok_target))
                targets_pointer += 1
            else:
                tokenized_target.append(-1)

        mask_id = [1] * len(tokenized_target)
        type_id = [0] * len(tokenized_target)
        tokenized_target.extend([-1] * (maxlen - len(tokenized_target)))
        row_dict['target'] = tokenized_target
    else:
        mask_id = [1] * len(tokenized_input)
        type_id = [0] * len(tokenized_input)
    tokenized_input_id.extend(
        [tokenizer.convert_tokens_to_ids([tok.tok_pad(tokenizer)])[0]] * (maxlen - len(tokenized_input_id)))
    mask_id.extend([0] * (maxlen - len(mask_id)))
    type_id.extend([1] * (maxlen - len(type_id)))
    row_dict['input'] = tokenized_input_id
    row_dict['type'] = type_id
    row_dict['mask'] = mask_id
    return row_dict
