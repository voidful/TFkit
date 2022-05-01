import unittest

from tfkit.test import *
from tfkit.utility.datafile import *


class TestDataPreprocess(unittest.TestCase):

    def test_get_x_data_from_file(self):
        for get_x_iter in [get_gen_data_from_file(GEN_DATASET),
                           get_qa_data_from_file(QA_DATASET),
                           get_tag_data_from_file(TAG_DATASET),
                           get_clas_data_from_file(CLAS_DATASET),
                           get_multiclas_data_from_file(CLAS_DATASET)]:
            while True:
                try:
                    print(next(get_x_iter))
                except StopIteration as e:
                    task_label_dict = e.value
                    break
            print(task_label_dict)
            for k, v in task_label_dict.items():
                print(k, v)
                self.assertTrue(isinstance(v, list))
