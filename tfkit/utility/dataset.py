import os
import pickle
from random import choice

import nlp2
import numpy
import torch
from torch.utils import data
import copy


def check_type_for_dataloader(data_item):
    if (isinstance(data_item, list) and not isinstance(data_item[-1], str) and check_type_for_dataloader(
            data_item[-1])) or \
            isinstance(data_item, numpy.ndarray) or \
            isinstance(data_item, int):
        return True
    else:
        return False


def batch_reduce_pad(batch):
    has_pad = all([dat['input'][-1] == batch[0]['input'][-1] for dat in batch]) and batch[0]['input'][-1] == \
              batch[0]['input'][-2]
    if has_pad:
        pad_token_input = batch[0]['input'][-1]
        pad_start = max([list(dat['input']).index(pad_token_input) for dat in batch])
        pad_token_target = batch[0]['target'][-1] if 'target' in batch[0] else None
        if isinstance(pad_token_target, int):
            pad_start = max(pad_start, max([list(dat['target']).index(pad_token_target) for dat in batch]))
        for ind, dat in enumerate(batch):
            for k, v in dat.items():
                if (isinstance(v, list) and len(v) > 1):
                    batch[ind][k] = v[:pad_start]
                if k == 'input_length':
                    batch[ind][k] = pad_start - 1
                batch[ind][k] = numpy.asarray(batch[ind][k])
    return batch


def dataloader_collate(batch):
    batch = copy.deepcopy(batch)
    return torch.utils.data._utils.collate.default_collate(batch_reduce_pad(batch))


def get_dataset(file_path, model_class, tokenizer, parameter):
    panel = nlp2.Panel()
    all_arg = nlp2.function_get_all_arg_with_value(model_class.preprocessor)
    if parameter.get('panel'):
        print("Operation panel for data preprocessing.")
        for missarg in nlp2.function_check_missing_arg(model_class.preprocessor,
                                                       parameter):
            panel.add_element(k=missarg, v=all_arg[missarg], msg=missarg, default=all_arg[missarg])
        filled_arg = panel.get_result_dict()
        parameter.update(filled_arg)
    ds = LoadDataset(fpath=file_path, tokenizer=tokenizer,
                     get_data_from_file=model_class.get_data_from_file,
                     preprocessor=model_class.preprocessor,
                     get_feature_from_data=model_class.get_feature_from_data,
                     **parameter)
    return ds


class LoadDataset(data.Dataset):
    def __init__(self, fpath, tokenizer, get_data_from_file, preprocessor, get_feature_from_data, preprocessing_arg={},
                 cache=False, **kwargs):
        cache_path = fpath + "_" + tokenizer.name_or_path.replace("/", "_") + ".cache"
        self.task_dict = {}
        self.preprocessor = preprocessor(tokenizer, kwargs=preprocessing_arg)
        self.get_feature_from_data = get_feature_from_data
        self.tokenizer = tokenizer
        if os.path.isfile(cache_path) and cache:
            with open(cache_path, "rb") as cf:
                outdata = pickle.load(cf)
                sample = outdata['sample']
                self.task_dict = outdata['task']
        else:
            print(f"Start preprocessing...")
            tasks, items = get_data_from_file(fpath)
            sample = self.preprocessor.prepare(items)
            self.task_dict = tasks
            print(f"There are {len(sample)} datas after preprocessing.")
            if cache:
                with open(cache_path, 'wb') as cf:
                    outdata = {'sample': sample, 'task': self.task_dict}
                    pickle.dump(outdata, cf)
        self.sample = sample
        self.task = self.task_dict

    def increase_with_sampling(self, total):
        inc_samp = [choice(self.sample) for _ in range(total - len(self.sample))]
        if len(inc_samp) > 0:
            self.sample.extend(inc_samp)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.get_feature_from_data(self.sample[idx], self.tokenizer, self.preprocessor.parameters['maxlen'],
                                          self.task_dict)
