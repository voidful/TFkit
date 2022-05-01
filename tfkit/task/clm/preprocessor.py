import tfkit.utility.tok as tok

from tfkit.utility.datafile import get_gen_data_from_file
from tfkit.utility.preprocess import GeneralNLPPreprocessor


class Preprocessor(GeneralNLPPreprocessor):
    def read_file_to_data(self, path):
        return get_gen_data_from_file(path)

    def preprocess_component_convert_to_id(self, item, likelihood=['none', 'pos', 'neg', 'both'], **param_dict):
        likelihood = likelihood[0] if isinstance(likelihood, list) else likelihood
        tokenized_input, tokenized_target, n_target = item['input'], item.get('target', None), item.get('ntarget', None)
        previous = item.get("previous", [])

        if tokenized_target is None:
            yield {'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                   'previous': self.tokenizer.convert_tokens_to_ids(previous)}
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
                           'ntarget': self.tokenizer.convert_tokens_to_ids(neg_text)}
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
                           'ntarget': self.tokenizer.convert_tokens_to_ids(neg_text)}
        return item

    def postprocess(self, item, tokenizer, maxlen, **kwargs):
        tok_pad = tok.tok_pad_id(tokenizer)
        tok_bos = tok.tok_begin_id(tokenizer)
        tok_sep = tok.tok_sep_id(tokenizer)
        tok_mask = tok.tok_mask_id(tokenizer)

        t_input_id, previous = item['input'], item['previous']

        encoder_mask_id = [1] * (len(t_input_id))
        encoder_mask_id.extend([0] * (maxlen - len(encoder_mask_id)))
        t_input_id.extend([tok_pad] * (maxlen - len(t_input_id)))
        row_dict = {}
        if 'target' in item:
            target = item['target']
            tokenized_target_id = []
            if len(previous) == len(target):
                tokenized_prev_id = [tok_mask] * maxlen
            else:
                tokenized_prev_id = [tok_sep] + target
            tokenized_target_id.extend(target + [tok_sep])
            decoder_mask_id = [1] * (len(tokenized_prev_id))
            decoder_mask_id.extend([0] * (maxlen - len(decoder_mask_id)))
            tokenized_prev_id.extend([tok_pad] * (maxlen - len(tokenized_prev_id)))
            tokenized_target_id.extend([-1] * (maxlen - len(tokenized_target_id)))

            row_dict['target'] = tokenized_target_id
            row_dict['prev'] = tokenized_prev_id
            row_dict['ntarget'] = [-1] * maxlen
            if 'ntarget' in item and len(item['ntarget']) > 0:
                tokenized_ntarget_id = item['ntarget']
                tokenized_ntarget_id.extend([-1] * (maxlen - len(tokenized_ntarget_id)))
                if len(tokenized_ntarget_id) <= maxlen:
                    row_dict['ntarget'] = tokenized_ntarget_id
        else:
            tokenized_prev_id = [tok_sep]
            tokenized_prev_id.extend(previous)
            target_start = len(tokenized_prev_id) - 1
            row_dict['start'] = target_start
            decoder_mask_id = [1] * (len(tokenized_prev_id))
            row_dict['prev'] = tokenized_prev_id

        row_dict['input'] = t_input_id
        row_dict['encoder_mask'] = encoder_mask_id
        row_dict['decoder_mask'] = decoder_mask_id
        return row_dict
