import torch

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
        tokenized_input, tokenized_target, n_target, b_target = item['input'], \
                                                      item.get('target', None), \
                                                      item.get('ntarget', None), \
                                                      item.get('btarget', None)
        previous = item.get("previous", [])
        if tokenized_target is None:
            yield {'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                   'previous': self.tokenizer.convert_tokens_to_ids(previous)}
        elif b_target and len(b_target) > 0:
            yield {'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                   'previous': self.tokenizer.convert_tokens_to_ids(previous),
                   'target': self.tokenizer.convert_tokens_to_ids(tokenized_target),
                   'btarget': self.tokenizer.encode(b_target)}
        else:
            if "neg" in likelihood or 'both' in likelihood:
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
                           'previous': self.tokenizer.convert_tokens_to_ids(previous),
                           'target': self.tokenizer.convert_tokens_to_ids(tokenized_target),
                           'ntarget': self.tokenizer.encode(neg_text)}
            else:
                yield {'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                       'previous': self.tokenizer.convert_tokens_to_ids(previous),
                       'target': self.tokenizer.convert_tokens_to_ids(tokenized_target)}

            # whole sentence masking
            if 'pos' in likelihood:
                yield {'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                       'target': self.tokenizer.convert_tokens_to_ids(tokenized_target),
                       'previous': self.tokenizer.convert_tokens_to_ids(
                           [tok.tok_mask(self.tokenizer)] * len(tokenized_target))}
            elif 'both' in likelihood:
                for neg_text in ntext_arr:
                    yield {'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                           'target': self.tokenizer.convert_tokens_to_ids(tokenized_target),
                           'previous': self.tokenizer.convert_tokens_to_ids(
                               [tok.tok_mask(self.tokenizer)] * len(tokenized_target)),
                           'ntarget': self.tokenizer.encode(neg_text)}

    def postprocess(self, item, tokenizer, maxlen, **kwargs):
        t_input_id, previous = item['input'], item['previous']
        row_dict = {}
        if 'target' in item:
            target = item['target']
            tokenized_target_id = []
            if len(previous) == len(target):
                tokenized_prev_id = [self.tok_mask_id] * maxlen
            else:
                tokenized_prev_id = [self.tok_sep_id] + target
            tokenized_target_id.extend(target + [self.tok_sep_id])
            row_dict['target'] = tokenized_target_id
            row_dict['target_pad'] = [-1]
            row_dict['prev'] = tokenized_prev_id
            row_dict['ntarget'] = [-1] * maxlen
            if 'ntarget' in item and len(item['ntarget']) > 0:
                tokenized_ntarget_id = item['ntarget']
                if len(tokenized_ntarget_id) <= maxlen:
                    row_dict['ntarget'] = tokenized_ntarget_id
            if 'btarget' in item and len(item['btarget']) > 0:
                row_dict['btarget'] = tokenizer.encode(item['btarget'])
        else:
            tokenized_prev_id = [self.tok_sep_id]
            tokenized_prev_id.extend(previous)
            row_dict['prev'] = tokenized_prev_id

        row_dict['input'] = t_input_id
        row_dict['encoder_mask'] = [1] * len(t_input_id)
        row_dict['decoder_mask'] = [1] * len(tokenized_prev_id)
        return {key: torch.tensor(value) for key, value in row_dict.items()}
