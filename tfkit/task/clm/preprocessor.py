import torch

from tfkit.utility.datafile import get_gen_data_from_file
from tfkit.utility.preprocess import GeneralNLPPreprocessor


class Preprocessor(GeneralNLPPreprocessor):
    def read_file_to_data(self, path):
        return get_gen_data_from_file(path)

    def preprocess_component_convert_to_id(self, item, **param_dict):
        tokenized_input, target = item['input'], item.get('target', None)
        tokenized_target = self.tokenizer.tokenize(target) if target else None
        previous = item.get("previous", [])
        if tokenized_target is None:
            yield {'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                   'previous': self.tokenizer.convert_tokens_to_ids(previous)}
        else:
            yield {'input': self.tokenizer.convert_tokens_to_ids(tokenized_input),
                   'previous': self.tokenizer.convert_tokens_to_ids(previous),
                   'target': self.tokenizer.convert_tokens_to_ids(tokenized_target)}

    def postprocess(self, item, tokenizer, maxlen, **kwargs):
        t_input_id, previous = item['input'], item['previous']
        row_dict = {}
        if 'target' in item:
            target = item['target']
            t_target_id = [-1] * len(t_input_id)
            mask_id = [0] * (len(t_target_id))
            t_target_id += target + [self.tok_sep_id]
            mask_id += [1] * (len(target + [self.tok_sep_id]))

            row_dict['start'] = [len(t_input_id)]
            t_input_id += [self.tok_bos_id] + target
            mask_id = [1] * (len(t_input_id))
            row_dict['target'] = t_target_id
        else:
            t_prev_id = [self.tok_sep_id] + previous
            t_input_id.extend(t_prev_id)
            mask_id = [1] * (len(t_input_id))
            row_dict['start'] = [len(t_input_id) - 1]
        row_dict['input'] = t_input_id
        row_dict['mask'] = mask_id
        row_dict['target_pad'] = [-1]
        return {key: torch.tensor(value) for key, value in row_dict.items()}
