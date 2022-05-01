import unittest

from transformers import AutoTokenizer

import tfkit
from tfkit.test import *
from tfkit.utility.dataloader import batch_reduce_pad


class TestDataLoader(unittest.TestCase):

    def test_batch_reduce_pad(self):
        k = [{'input': [1, 2, 3, 0, 0]}, {'input': [3, 4, 0, 0, 0]}]
        reduced_batch = batch_reduce_pad(k)
        self.assertEqual(len(reduced_batch[0]['input']), len(reduced_batch[1]['input']))
        self.assertCountEqual(reduced_batch[0]['input'], [1, 2, 3])
        self.assertCountEqual(reduced_batch[1]['input'], [3, 4, 0])

