import unittest

import torch

from tfkit.utility.dataloader import pad_batch


class TestUtilityDataLoader(unittest.TestCase):

    def test_batch_reduce_pad(self):
        k = [{'input': torch.tensor([1, 2, 3])},
             {'input': torch.tensor([3, 4])},
             {'input': torch.tensor([5])}]
        reduced_batch = pad_batch(k)
        self.assertEqual(len(reduced_batch[0]['input']), len(reduced_batch[1]['input']))
        print(reduced_batch)
        self.assertCountEqual(reduced_batch[0]['input'], [1, 2, 3])
        self.assertCountEqual(reduced_batch[1]['input'], [3, 4, 0])
