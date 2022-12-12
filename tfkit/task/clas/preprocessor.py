import torch
from sklearn.preprocessing import MultiLabelBinarizer
from tfkit.utility import tok
from tfkit.utility.datafile import get_multiclas_data_from_file
from tfkit.utility.preprocess import GeneralNLPPreprocessor


class Preprocessor(GeneralNLPPreprocessor):

    def read_file_to_data(self, path):
        return get_multiclas_data_from_file(path)

    def preprocess_component_convert_to_id(self, item, **param_dict):
        item['input'] = self.tokenizer.convert_tokens_to_ids(item['input'])
        yield item

    def postprocess(self, item, tokenizer, maxlen, **kwargs):
        tinput, task = item['input'], item['task']
        row_dict = {'task': list(task.encode("utf-8"))}
        tokenized_input_id = [tok.tok_begin_id(tokenizer)] + tinput + [tok.tok_sep_id(tokenizer)]
        mask_id = [1] * len(tokenized_input_id)
        tokenized_input_id.extend([tokenizer.pad_token_id] * (maxlen - len(tokenized_input_id)))
        mask_id.extend([-1] * (maxlen - len(mask_id)))
        row_dict['input'] = tokenized_input_id
        row_dict['mask'] = mask_id
        row_dict['target'] = [-1]
        if 'target' in item:
            target = item['target']
            if 'multi_label' in task:
                mlb = MultiLabelBinarizer(classes=item['task_dict'][task])
                tar = mlb.fit_transform(target)
                tokenize_label = tar
            else:
                tokenize_label = [item['task_dict'][task].index(target[0])]
            row_dict['target'] = tokenize_label
        return {key: torch.tensor(value) for key, value in row_dict.items()}
