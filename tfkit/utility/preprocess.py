import numpy as np
import torch
from numpy import uint16

from tfkit.utility import tok


class GeneralNLPPreprocessor:
    """
    The design of NLPPreprocessor is to handle a pure text input,
    perform preprocessing on it base on model constrain,
    return ids as output

    This class will be applied before model training, splitting and prepare the data for model input
    it will call get feature from data when it's converting to model input
    """

    def __init__(self, tokenizer, maxlen=512, handle_exceed='slide', reserved_len=0, uint16_save=False,
                 kwargs={}):
        self.tokenizer = tokenizer
        self.uint16_save = uint16_save
        self.parameters = {**{'tokenizer': tokenizer, 'maxlen': maxlen, 'handle_exceed': handle_exceed,
                              'reserved_len': reserved_len}, **kwargs}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # item = {key: value.tolist() for key, value in item.items()}
        self.tok_pad_id = tok.tok_pad_id(tokenizer)
        self.tok_bos_id = tok.tok_begin_id(tokenizer)
        self.tok_sep_id = tok.tok_sep_id(tokenizer)
        self.tok_mask_id = tok.tok_mask_id(tokenizer)

    def read_file_to_data(self, filepath):
        assert 'plz override this funciton'

    def set_global_parameters(self):
        self.tokenize_target = False

    def preprocess(self, item):
        self.set_global_parameters()
        preprocessed_data = []
        item = self.preprocess_component_prepare_input(item)
        # target may be none in eval
        t_input_list, t_target_list, t_input_index, t_target_index = self.preprocess_component_split_into_list(
            item['input'],
            item.get('target'))
        for t_input, t_target, t_input_index, t_target_index in zip(t_input_list,
                                                                    t_target_list,
                                                                    t_input_index,
                                                                    t_target_index):
            slice_length = self.parameters['maxlen'] - self.parameters.get('reserved_len') - 3
            item['input'] = [tok.tok_begin(self.tokenizer)] + t_input[:slice_length]
            item['input_index'] = t_input_index
            item['target_index'] = t_target_index
            if len(t_target) > 0:
                item['target'] = t_target
            for convert_feature_input_dict in self.preprocess_component_convert_to_id(item):
                if self.uint16_save:
                    data_item = {k: np.array(v, dtype=uint16) if isinstance(v, list) else v for k, v in
                                 convert_feature_input_dict.items()}
                else:
                    data_item = convert_feature_input_dict
                preprocessed_data.append(data_item)
        return preprocessed_data

    def preprocess_component_prepare_input(self, item):
        if tok.UNIVERSAL_SEP in item['input']:
            part = item['input'].split(tok.UNIVERSAL_SEP)
            item['previous'] = self.tokenizer.tokenize(part[-1])
            item['input'] = "".join(part[:-1])
        return item

    def preprocess_component_split_into_list(self, input_text, target_text=None):
        t_input_list, t_input_index = tok.handle_exceed(self.tokenizer, input_text,
                                                        maxlen=self.parameters['maxlen'] - 3,
                                                        mode=self.parameters.get('handle_exceed'))
        if self.tokenize_target and target_text:
            t_target_list, t_target_index = tok.handle_exceed(self.tokenizer, target_text,
                                                              maxlen=self.parameters['maxlen'] - 3,
                                                              mode=self.parameters.get('handle_exceed'))
        elif target_text:
            t_target_list, t_target_index = [target_text * len(t_input_list)], [[0] * len(t_input_list)]
        else:
            t_target_list, t_target_index = ['' * len(t_input_list)], [[0] * len(t_input_list)]
        return t_input_list, t_target_list, t_input_index, t_target_index

    def preprocess_component_convert_to_id(self, item):
        yield {k: self.tokenizer.convert_tokens_to_ids(v) if isinstance(v, list) else v for k, v in item.items()}

    def postprocess(self, item, tokenizer, maxlen, **kwargs):
        return {key: torch.tensor(value).to(self.device) for key, value in item.items() if isinstance(value, list)}

    def postprocess_batch(self, feature_dict, **kwargs):
        return {key: torch.unsqueeze(torch.tensor(value), 0).to(self.device) for key, value in feature_dict.items()}


class GeneralCVPreprocessor:
    def __init__(self, feature_extractor, kwargs={}):
        self.feature_extractor = feature_extractor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.parameters = {**{'feature_extractor': feature_extractor}, **kwargs}

    def read_file_to_data(self, filepath):
        assert 'plz override this funciton'

    def preprocess(self, item):
        preprocessed_data = []
        preprocessed_data.append(item)
        return preprocessed_data

    def postprocess(self, item, **kwargs):
        item['input'] = self.feature_extractor(item['input'])
        return {key: torch.tensor(value).to(self.device) for key, value in item.items()}


class GeneralSpeechPreprocessor:
    def __init__(self, feature_extractor, kwargs={}):
        self.feature_extractor = feature_extractor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.parameters = {**{'feature_extractor': feature_extractor}, **kwargs}

    def read_file_to_data(self, filepath):
        assert 'plz override this funciton'

    def preprocess(self, item):
        preprocessed_data = []
        preprocessed_data.append(item)
        return preprocessed_data

    def postprocess(self, item, **kwargs):
        item['input'] = self.feature_extractor(item['input'])
        return {key: torch.tensor(value).to(self.device) for key, value in item.items()}
