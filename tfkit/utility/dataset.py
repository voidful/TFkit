import os
from collections import defaultdict
from random import choice

import joblib
import nlp2
from torch.utils import data
from tqdm.contrib.concurrent import process_map

from tfkit.utility.constants import CACHE_EXTENSION

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None


def get_dataset(file_path, task_class, tokenizer, parameter):
    panel = nlp2.Panel()
    # all_arg = nlp2.function_get_all_arg_with_value(task_class.preprocessor.prepare_convert_to_id)
    # if parameter.get('panel'):
    #     print("Operation panel for data preprocessing.")
    #     for missarg in nlp2.function_check_missing_arg(task_class.preprocessor,
    #                                                    parameter):
    #         panel.add_element(k=missarg, v=all_arg[missarg], msg=missarg, default=all_arg[missarg])
    #     filled_arg = panel.get_result_dict()
    #     parameter.update(filled_arg)
    if load_dataset is not None and not os.path.isfile(file_path):
        try:
            hf_ds = load_dataset(file_path, split=parameter.get('split', 'train'))
            return HFDataset(hf_ds, tokenizer=tokenizer,
                             preprocessor=task_class.Preprocessor,
                             preprocessing_arg=parameter)
        except Exception:
            pass
    ds = TFKitDataset(fpath=file_path, tokenizer=tokenizer,
                      preprocessor=task_class.Preprocessor,
                      preprocessing_arg=parameter)
    return ds


class TFKitDataset(data.Dataset):
    def __init__(self, fpath, tokenizer, preprocessor, preprocessing_arg={}):
        cache_path = fpath + "_" + tokenizer.name_or_path.replace("/", "_") + CACHE_EXTENSION
        self.task_dict = {}
        self.preprocessor = preprocessor(tokenizer, kwargs=preprocessing_arg)
        self.tokenizer = tokenizer
        if os.path.isfile(cache_path) and preprocessing_arg.get('cache', False):
            with open(cache_path, "rb") as fo:
                outdata = joblib.load(fo)
                sample = outdata['sample']
                length = outdata['length']
                self.task_dict = outdata['task']
        else:
            print(f"Start preprocessing...")
            sample = defaultdict(list)
            length = 0
            get_data_item = self.preprocessor.read_file_to_data(fpath)
            while True:
                try:
                    for items in process_map(self.preprocessor.preprocess, next(get_data_item),
                                             chunksize=1000):
                        for i in items:
                            length += 1
                            for k, v in i.items():
                                sample[k].append(v)
                    print(f"loaded {length} data.")
                except StopIteration as e:
                    tasks = e.value
                    break
            self.task_dict = tasks
            print(f"There are {length} datas after preprocessing.")
            if preprocessing_arg.get('cache', False):
                with open(cache_path, 'wb') as fo:
                    outdata = {'sample': sample, 'task': self.task_dict, 'length': length}
                    joblib.dump(outdata, fo)
        self.length = length
        self.sample = sample
        self.task = self.task_dict

    def increase_with_sampling(self, total):
        for _ in range(total - self.length):
            for key in self.sample.keys():
                self.sample[key].append(choice(self.sample[key]))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.preprocessor.postprocess(
            {**{'task_dict': self.task_dict}, **{key: self.sample[key][idx] for key in self.sample.keys()}},
            self.tokenizer,
            maxlen=self.preprocessor.parameters['maxlen'])


class HFDataset(data.Dataset):
    """Dataset wrapper for the HuggingFace datasets library."""

    def __init__(self, hf_dataset, tokenizer, preprocessor, preprocessing_arg=None):
        preprocessing_arg = preprocessing_arg or {}
        self.task_dict = {}
        self.sample = defaultdict(list)
        self.preprocessor = preprocessor(tokenizer, kwargs=preprocessing_arg)
        self.tokenizer = tokenizer

        print("Start preprocessing with HuggingFace dataset...")
        length = 0
        for raw_item in hf_dataset:
            for items in self.preprocessor.preprocess(raw_item):
                length += 1
                for k, v in items.items():
                    self.sample[k].append(v)
        self.length = length
        self.task = self.task_dict

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.preprocessor.postprocess(
            {**{'task_dict': self.task_dict}, **{key: self.sample[key][idx] for key in self.sample.keys()}},
            self.tokenizer,
            maxlen=self.preprocessor.parameters['maxlen'])
