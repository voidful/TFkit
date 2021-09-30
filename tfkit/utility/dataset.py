import os
import pickle
from random import choice

import nlp2
import numpy
import numpy as np
import torch
from torch.utils import data
import copy
from tqdm.contrib.concurrent import process_map


def check_type_for_dataloader(data_item):
    if (isinstance(data_item, list) and not isinstance(data_item[-1], str) and check_type_for_dataloader(
            data_item[-1])) or \
            isinstance(data_item, str) or \
            isinstance(data_item, numpy.ndarray) or \
            isinstance(data_item, int):
        return True
    else:
        return False


def dataloader_collate(batch):
    batch = copy.deepcopy(batch)
    has_pad = all([dat['input'][-1] == batch[0]['input'][-1] for dat in batch])
    if has_pad:
        pad_token = batch[0]['input'][-1]
        pad_start = max([list(dat['input']).index(pad_token) for dat in batch])
        for ind, dat in enumerate(batch):
            for k, v in dat.items():
                if isinstance(v, numpy.ndarray) and v.size > 1:
                    batch[ind][k] = v[:pad_start]
                if k == 'input_length':
                    batch[ind][k] = pad_start - 1
    return torch.utils.data._utils.collate.default_collate(batch)


def get_dataset(file_path, model_class, tokenizer, parameter):
    panel = nlp2.Panel()
    all_arg = nlp2.function_get_all_arg_with_value(model_class.preprocessing_data)
    if parameter.get('panel'):
        for missarg in nlp2.function_check_missing_arg(model_class.preprocessing_data,
                                                       parameter):
            panel.add_element(k=missarg, v=all_arg[missarg], msg=missarg, default=all_arg[missarg])
        filled_arg = panel.get_result_dict()
        parameter.update(filled_arg)
    ds = LoadDataset(fpath=file_path, tokenizer=tokenizer,
                     get_data_from_file=model_class.get_data_from_file,
                     preprocessing_data=model_class.preprocessing_data,
                     cache=parameter.get('cache'),
                     input_arg=parameter)
    return ds


class LoadDataset(data.Dataset):
    def preprocess(self, i):
        tasks, task, input_text, target, *other = i
        self.task_dict.update(tasks)
        sample = []
        for get_feature_from_data, feature_param in self.preprocessing_data(i, self.tokenizer, **self.input_arg):
            for feature in get_feature_from_data(**feature_param):
                feature = {k: v for k, v in feature.items() if check_type_for_dataloader(v)}
                feature.update((k, np.asarray(v)) for k, v in feature.items() if k != 'task')
                sample.append(feature)
        return sample

    def __init__(self, fpath, tokenizer, get_data_from_file, preprocessing_data, cache=False, input_arg={}):
        cache_path = fpath + "_" + tokenizer.name_or_path.replace("/", "_") + ".cache"
        self.task_dict = {}
        self.preprocessing_data = preprocessing_data
        self.tokenizer = tokenizer
        self.input_arg = input_arg

        if os.path.isfile(cache_path) and cache:
            with open(cache_path, "rb") as cf:
                outdata = pickle.load(cf)
                sample = outdata['sample']
                self.task_dict = outdata['task']
        else:
            sample = []
            [sample.extend(i) for i in process_map(self.preprocess, list(get_data_from_file(fpath)),
                                                   max_workers=input_arg['worker'],
                                                   chunksize=1000)]
            print(f"There are {len(sample)} datas after preprocessing.")
            if cache:
                with open(cache_path, 'wb') as cf:
                    outdata = {'sample': sample, 'task': self.task_dict}
                    pickle.dump(outdata, cf)
        self.sample = np.array(sample)
        self.task = self.task_dict

    def increase_with_sampling(self, total):
        inc_samp = [choice(self.sample) for _ in range(total - len(self.sample))]
        self.sample = np.concatenate([self.sample, np.array(inc_samp)])

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]
