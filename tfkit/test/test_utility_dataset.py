import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import unittest
import tfkit


class TestDataset(unittest.TestCase):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../../"))
    DATASET_DIR = os.path.join(ROOT_DIR, 'demo_data')
    CLAS_DATASET = os.path.join(DATASET_DIR, 'classification.csv')
    GEN_DATASET = os.path.join(DATASET_DIR, 'generate.csv')

    def test_check_type_for_dataloader(self):
        self.assertTrue(tfkit.utility.check_type_for_dataloader('a'))
        self.assertTrue(tfkit.utility.check_type_for_dataloader(123))
        self.assertFalse(tfkit.utility.check_type_for_dataloader(['a']))
        self.assertFalse(tfkit.utility.check_type_for_dataloader([{'a'}]))

    def test_get_dataset(self):
        model_class = tfkit.utility.load_model_class('clas')
        file_path = self.CLAS_DATASET
        dataset_arg = {
            'maxlen': 123,
            'handle_exceed': 'slide',
            'config': 'voidful/albert_chinese_tiny',
            'cache': False
        }
        ds = tfkit.utility.get_dataset(file_path=file_path, model_class=model_class, parameter=dataset_arg)
        print(ds, ds[0])

        model_class = tfkit.utility.load_model_class('onebyone')
        file_path = self.GEN_DATASET
        dataset_arg = {
            'maxlen': 123,
            'handle_exceed': 'slide',
            'config': 'voidful/albert_chinese_tiny',
            'cache': False
        }
        ds = tfkit.utility.get_dataset(file_path=file_path, model_class=model_class, parameter=dataset_arg)
        print(len(ds), ds[0])

        model_class = tfkit.utility.load_model_class('onebyone')
        file_path = self.GEN_DATASET
        dataset_arg = {
            'maxlen': 123,
            'handle_exceed': 'slide',
            'config': 'voidful/albert_chinese_tiny',
            'cache': False,
            'likelihood': 'both'
        }
        ds = tfkit.utility.get_dataset(file_path=file_path, model_class=model_class, parameter=dataset_arg)
        print(len(ds), ds[0])
