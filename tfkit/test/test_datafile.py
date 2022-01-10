import unittest

from tfkit.test import *
from tfkit.utility.datafile import *


class TestDataFile(unittest.TestCase):

    def test_get_multiclas_data_from_file(self):
        task_label_dict, datas = get_multiclas_data_from_file(CLAS_DATASET)
        for k, v in task_label_dict.items():
            print(k, v)
            self.assertTrue(isinstance(v, list))
        print(datas)

    def test_get_clas_data_from_file(self):
        task_label_dict, datas = get_clas_data_from_file(CLAS_DATASET)
        for k, v in task_label_dict.items():
            print(k, v)
            self.assertTrue(isinstance(v, list))
        print(datas)

    def test_get_gen_data_from_file(self):
        task_label_dict, datas = get_clas_data_from_file(GEN_DATASET)
        for k, v in task_label_dict.items():
            print(k, v)
            self.assertTrue(isinstance(v, list))
        print(datas)

    def test_get_qa_data_from_file(self):
        task_label_dict, datas = get_qa_data_from_file(QA_DATASET)
        for k, v in task_label_dict.items():
            print(k, v)
            self.assertTrue(isinstance(v, list))
        print(datas)

    def test_get_tag_data_from_file(self):
        task_label_dict, datas = get_tag_data_from_file(TAG_DATASET)
        for k, v in task_label_dict.items():
            print(k, v)
            self.assertTrue(isinstance(v, list))
        print(datas)
