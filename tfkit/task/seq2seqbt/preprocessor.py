from tfkit.utility import tok
from tfkit.utility.datafile import get_gen_data_from_file
from tfkit.utility.preprocess import GeneralNLPPreprocessor


class Preprocessor(GeneralNLPPreprocessor):
    def read_file_to_data(self, path):
        return get_gen_data_from_file(path)

    def preprocess_component_convert_to_id(self, item, **param_dict):
        tokenized_input, tokenized_target, b_target = item['input'], item.get('target', None), item.get('btarget', None)
        previous = item.get("previous", [])
        if tokenized_target is None:
            yield {'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                   'previous': self.tokenizer.convert_tokens_to_ids(previous)}
        else:
            if b_target and len(b_target) > 0:
                yield {'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                       'previous': self.tokenizer.convert_tokens_to_ids(previous),
                       'target': self.tokenizer.convert_tokens_to_ids(tokenized_target),
                       'btarget': self.tokenizer.encode(b_target)}
            else:
                yield {'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                       'previous': self.tokenizer.convert_tokens_to_ids(previous),
                       'target': self.tokenizer.convert_tokens_to_ids(tokenized_target),
                       'btarget': None}

    def postprocess(self, item, tokenizer, maxlen, **kwargs):
        tok_pad = tok.tok_pad_id(tokenizer)
        tok_bos = tok.tok_begin_id(tokenizer)
        tok_sep = tok.tok_sep_id(tokenizer)
        tok_mask = tok.tok_mask_id(tokenizer)

        t_input_id, previous = item['input'], item['previous'],
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
            row_dict['btarget'] = [-1] * maxlen
            if 'btarget' in item and len(item['btarget']) > 0:
                tokenized_ntarget_id = item['btarget']
                tokenized_ntarget_id.extend([-1] * (maxlen - len(tokenized_ntarget_id)))
                if len(tokenized_ntarget_id) <= maxlen:
                    row_dict['btarget'] = tokenized_ntarget_id
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
