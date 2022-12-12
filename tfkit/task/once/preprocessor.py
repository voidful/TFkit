import tfkit.utility.tok as tok
from tfkit.utility.datafile import get_gen_data_from_file
from tfkit.utility.preprocess import GeneralNLPPreprocessor


class Preprocessor(GeneralNLPPreprocessor):
    def read_file_to_data(self, path):
        return get_gen_data_from_file(path)

    def set_global_parameters(self):
        self.tokenize_target = True

    def preprocess_component_convert_to_id(self, item, likelihood=['none', 'pos', 'neg', 'both'], **param_dict):
        likelihood = likelihood[0] if isinstance(likelihood, list) else likelihood
        tokenized_input, tokenized_target, n_target = item['input'], item.get('target', None), item.get('ntarget', None)
        yield {'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
               'target': self.tokenizer.convert_tokens_to_ids(tokenized_target)}
        if "neg" in likelihood:
            # formatting neg data in csv
            if n_target is None:
                ntext_arr = [
                    tok.tok_sep(self.tokenizer) + self.tokenizer.convert_tokens_to_string(tokenized_target)]
            elif tok.tok_sep(self.tokenizer) in n_target:
                ntext_arr = [ntext.strip() for ntext in n_target.split(tok.tok_sep(self.tokenizer))]
            else:
                ntext_arr = [n_target.strip()]
            for neg_text in ntext_arr:
                yield {'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                       'target': self.tokenizer.convert_tokens_to_ids(tokenized_target),
                       'ntarget': self.tokenizer.convert_tokens_to_ids(neg_text)}

    def postprocess(self, item, tokenizer, maxlen, **kwargs):
        tok_pad = tok.tok_pad_id(tokenizer)
        tok_bos = tok.tok_begin_id(tokenizer)
        tok_sep = tok.tok_sep_id(tokenizer)
        tok_mask = tok.tok_mask_id(tokenizer)

        row_dict = {}
        t_input_id = item['input']
        encoder_mask_id = [1] * (len(t_input_id))
        encoder_mask_id.extend([0] * (maxlen - len(encoder_mask_id)))
        target_start = len(t_input_id)
        target_end = maxlen
        target_length = target_end - target_start
        t_input_id.extend([tok_pad] * (maxlen - len(t_input_id)))
        if 'target' in item and item['target'] is not None:
            target = item['target'] + [tok_sep]
            target.extend([-1] * (maxlen - len(target)))
            row_dict['target'] = target
            row_dict['ntarget'] = [-1] * maxlen
            if 'ntarget' in item and len(item['ntarget'].strip()) > 0:
                tokenized_ntarget_id = item['ntarget']
                tokenized_ntarget_id.extend([-1] * (maxlen - len(tokenized_ntarget_id)))
                if len(tokenized_ntarget_id) <= maxlen:
                    row_dict['ntarget'] = tokenized_ntarget_id

        input_length = min(maxlen, target_start * 3)
        row_dict['input'] = t_input_id
        row_dict['mask'] = encoder_mask_id
        row_dict['start'] = target_start
        row_dict['end'] = maxlen
        row_dict['input_length'] = input_length
        row_dict['target_length'] = target_length
        return row_dict
