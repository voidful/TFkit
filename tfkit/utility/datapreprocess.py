import numpy as np
import pandas as pd
from numpy import uint16

from tfkit.utility import tok


class GeneralNLPPreprocessor:
    def __init__(self, tokenizer, maxlen=512, handle_exceed='start_slice', reserved_len=0, kwargs={}):
        self.tokenizer = tokenizer
        self.parameters = {**{'tokenizer': tokenizer, 'maxlen': maxlen, 'handle_exceed': handle_exceed,
                              'reserved_len': reserved_len}, **kwargs}

    def custom_preprocess_fn(self, item, **param_dict):
        yield {k: self.tokenizer.convert_tokens_to_ids(v) for k, v in item.items()}

    def prepare(self, item):
        preprocessed_data = []
        maxlen = self.parameters.get('maxlen')
        t_input_list, _ = tok.handle_exceed(self.tokenizer, item['input'],
                                            maxlen=maxlen - 3,
                                            mode=self.parameters.get('handle_exceed'))
        for t_input in t_input_list:
            slice_length = maxlen - self.parameters.get('reserved_len') - 3
            item['input'] = [tok.tok_begin(self.tokenizer)] + t_input[:slice_length]
            for convert_feature_input_dict in self.custom_preprocess_fn(item, **self.parameters):
                preprocessed_data.append({k: np.array(v, dtype=uint16) for k, v in convert_feature_input_dict.items()})
        return preprocessed_data
