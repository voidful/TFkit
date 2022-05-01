import tfkit.utility.tok as tok
from tfkit.utility.datafile import get_gen_data_from_file
from tfkit.utility.preprocess import GeneralNLPPreprocessor


class Preprocessor(GeneralNLPPreprocessor):
    def read_file_to_data(self, path):
        return get_gen_data_from_file(path)

    def preprocess_component_convert_to_id(self, item, likelihood=['none', 'pos', 'neg', 'both'], **param_dict):
        likelihood = likelihood[0] if isinstance(likelihood, list) else likelihood
        tokenized_input, tokenized_target, n_target = item['input'], item.get('target', None), item.get('ntarget', None)
        if tokenized_target is None:
            yield {**{'input': self.tokenizer.convert_tokens_to_ids(tokenized_input)}, **param_dict}
        else:
            # each word in sentence
            for j in range(1, len(tokenized_target) + 1):
                if "neg" in likelihood or 'both' in likelihood:
                    # formatting neg data in csv
                    if n_target is None:
                        ntext_arr = [self.tokenizer.convert_tokens_to_string(tokenized_target[:j - 1])]
                    elif tok.tok_sep(self.tokenizer) in n_target:
                        ntext_arr = [ntext.strip() for ntext in n_target.split("[SEP]")]
                    else:
                        ntext_arr = [n_target.strip()]
                    # adding neg data
                    for neg_text in ntext_arr:
                        yield {
                            **{'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                               'previous': self.tokenizer.convert_tokens_to_ids(tokenized_target[:j - 1]),
                               'target': self.tokenizer.convert_tokens_to_ids(tokenized_target[:j]),
                               'ntarget': self.tokenizer.encode(neg_text)},
                            **param_dict}
                else:
                    yield {**{'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                              'previous': self.tokenizer.convert_tokens_to_ids(tokenized_target[:j - 1]),
                              'target': self.tokenizer.convert_tokens_to_ids(tokenized_target[:j])},
                           **param_dict}

            # end of the last word
            if "neg" in likelihood or 'both' in likelihood:
                # formatting neg data in csv
                if n_target is None:
                    ntext_arr = [self.tokenizer.convert_tokens_to_string(tokenized_target[:j])]
                elif "[SEP]" in n_target:
                    ntext_arr = [ntext.strip() for ntext in n_target.split("[SEP]")]
                else:
                    ntext_arr = [n_target.strip()]
                # adding neg data
                for neg_text in ntext_arr:
                    yield {
                        **{'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                           'previous': self.tokenizer.convert_tokens_to_ids(tokenized_target),
                           'target': self.tokenizer.convert_tokens_to_ids([tok.tok_sep(self.tokenizer)]),
                           'ntarget': self.tokenizer.encode(neg_text)},
                        **param_dict}
            else:
                yield {**{'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                          'previous': self.tokenizer.convert_tokens_to_ids(tokenized_target),
                          'target': self.tokenizer.convert_tokens_to_ids([tok.tok_sep(self.tokenizer)])},
                       **param_dict}

            # whole sentence masking
            if 'pos' in likelihood:
                yield {**{'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                          'previous': self.tokenizer.convert_tokens_to_ids(
                              [tok.tok_mask(self.tokenizer)] * len(tokenized_target)),
                          'target': self.tokenizer.convert_tokens_to_ids(tokenized_target)},
                       **param_dict}
            elif 'both' in likelihood:
                # formatting neg data in csv
                if n_target is None:
                    ntext_arr = []
                elif "[SEP]" in n_target:
                    ntext_arr = [ntext.strip() for ntext in n_target.split("[SEP]")]
                else:
                    ntext_arr = [n_target.strip()]
                for neg_text in ntext_arr:
                    yield {
                        **{'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                           'previous': self.tokenizer.convert_tokens_to_ids(
                               [tok.tok_mask(self.tokenizer)] * len(tokenized_target)),
                           'target': tokenized_target,
                           'ntarget': self.tokenizer.encode(neg_text)
                           },
                        **param_dict}

    def postprocess(self, item, tokenizer, maxlen, **kwargs):
        previous = item.get('previous', [])
        t_input_id = item['input']
        row_dict = {}
        t_input_id.extend(previous)
        t_input_id.append(tok.tok_mask_id(tokenizer))
        mask_id = [1] * len(t_input_id)
        target_start = len(t_input_id) - 1
        target_end = maxlen
        t_input_id.extend([0] * (maxlen - len(t_input_id)))
        if 'target' in item and item['target']:
            target = item['target']
            tokenized_target_id = [-1] * target_start
            tokenized_target_id.append(target[-1])
            target_end = len(tokenized_target_id) - 1
            tokenized_target_id.extend([-1] * (maxlen - len(tokenized_target_id)))
            row_dict['target'] = tokenized_target_id
        if 'ntarget' in item and len(item['ntarget']) > 0:
            tokenized_ntarget = item['ntarget']
            tokenized_ntarget_id = [-1] * target_start
            tokenized_ntarget_id.extend(tokenized_ntarget)
            tokenized_ntarget_id.extend([-1] * (maxlen - len(tokenized_ntarget_id)))
            if len(tokenized_ntarget_id) <= maxlen:
                row_dict['ntarget'] = tokenized_ntarget_id

        mask_id.extend([0] * (maxlen - len(mask_id)))
        type_id = [0] * len(t_input_id)
        type_id.extend([1] * (maxlen - len(type_id)))
        row_dict['input'] = t_input_id
        row_dict['type'] = type_id
        row_dict['mask'] = mask_id
        row_dict['start'] = target_start
        row_dict['end'] = target_end
        return row_dict
