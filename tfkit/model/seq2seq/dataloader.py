import tfkit.utility.tok as tok
from tfkit.utility.datafile import get_gen_data_from_file
from tfkit.utility.datapreprocess import GeneralNLPPreprocessor

get_data_from_file = get_gen_data_from_file

preprocessor = GeneralNLPPreprocessor


class preprocessor(GeneralNLPPreprocessor):
    def custom_preprocess_fn(self, item, likelihood=['none', 'pos', 'neg', 'both'], **param_dict):
        likelihood = likelihood[0] if isinstance(likelihood, list) else likelihood
        tokenizer = param_dict['tokenizer']
        input, p_target, n_target = item['input'], item.get('target', None), item.get('ntarget', None)
        previous = []
        if tok.UNIVERSAL_SEP in input:
            part = input.split(tok.UNIVERSAL_SEP)
            previous = tokenizer.tokenize(part[-1])
            input = "".join(part[:-1])
        if p_target is None:
            yield {**{'input': input, 'previous': previous}, **param_dict}
        else:
            tokenized_target = tokenizer.tokenize(p_target)
            if "neg" in likelihood or 'both' in likelihood:
                # formatting neg data in csv
                if n_target is None:
                    ntext_arr = [tok.tok_sep(tokenizer) + tokenizer.convert_tokens_to_string(tokenized_target)]
                elif tok.tok_sep(tokenizer) in n_target:
                    ntext_arr = [ntext.strip() for ntext in n_target.split(tok.tok_sep(tokenizer))]
                else:
                    ntext_arr = [n_target.strip()]
                for neg_text in ntext_arr:
                    yield {**{'input': input, 'previous': previous,
                              'target': tokenized_target, 'ntarget': neg_text}, **param_dict}
            else:
                yield {**{'input': input, 'previous': previous,
                          'target': tokenized_target}, **param_dict}

            # whole sentence masking
            if 'pos' in likelihood:
                yield {**{'input': input, 'target': tokenized_target,
                          'previous': [tok.tok_mask(tokenizer)] * len(tokenized_target)},
                       **param_dict}
            elif 'both' in likelihood:
                for neg_text in ntext_arr:
                    yield {**{'input': input, 'target': tokenized_target,
                              'previous': [tok.tok_mask(tokenizer)] * len(tokenized_target), 'ntarget': neg_text},
                           **param_dict}


def get_feature_from_data(item, tokenizer, maxlen, task_dict={}, **kwargs):
    tok_pad = tok.tok_pad(tokenizer)
    tok_bos = tok.tok_begin(tokenizer)
    tok_sep = tok.tok_sep(tokenizer)
    tok_mask = tok.tok_mask(tokenizer)

    t_input, previous = item['input'], item['previous'],
    t_input_id = tokenizer.convert_tokens_to_ids(t_input)
    encoder_mask_id = [1] * (len(t_input))
    encoder_mask_id.extend([0] * (maxlen - len(encoder_mask_id)))
    t_input_id.extend(tokenizer.convert_tokens_to_ids([tok_pad]) * (maxlen - len(t_input_id)))
    row_dict = {}
    if 'target' in item:
        target = item['target']
        tokenized_target_id = []
        if len(previous) == len(target):
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
        row_dict['ntarget'] = [-1] * maxlen
        if 'ntarget' in item and len(item['ntarget'].strip()) > 0:
            ntarget = item['ntarget']
            tokenized_ntarget = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ntarget))
            tokenized_ntarget_id = tokenized_ntarget
            tokenized_ntarget_id.extend([-1] * (maxlen - len(tokenized_ntarget_id)))
            if len(tokenized_ntarget_id) <= maxlen:
                row_dict['ntarget'] = tokenized_ntarget_id
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
    return row_dict
