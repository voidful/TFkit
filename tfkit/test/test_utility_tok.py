import sys
import os

import pytest
import torch
from torch import nn
from torch.autograd import Variable

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import unittest
import tfkit
from transformers import *


class TestTok(unittest.TestCase):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../../"))
    DATASET_DIR = os.path.join(ROOT_DIR, 'demo_data')

    def testTok(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        begin = tfkit.utility.tok.tok_begin(tokenizer)
        self.assertEqual(begin, "[CLS]")
        sep = tfkit.utility.tok.tok_sep(tokenizer)
        self.assertEqual(sep, "[SEP]")
        mask = tfkit.utility.tok.tok_mask(tokenizer)
        self.assertEqual(mask, "[MASK]")
        pad = tfkit.utility.tok.tok_pad(tokenizer)
        self.assertEqual(pad, "[PAD]")

    def testTok(self):
        tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        begin = tfkit.utility.tok.tok_begin(tokenizer)
        self.assertEqual(begin, "<s>")
        sep = tfkit.utility.tok.tok_sep(tokenizer)
        self.assertEqual(sep, "</s>")
        mask = tfkit.utility.tok.tok_mask(tokenizer)
        self.assertEqual(mask, "<mask>")
        pad = tfkit.utility.tok.tok_pad(tokenizer)
        self.assertEqual(pad, "<pad>")

    def testGetXUnkToken(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        result = tfkit.utility.tok.get_topP_unk_token(tokenizer, file_paths=[], topP=0.5)
        self.assertFalse(result)
        result = tfkit.utility.tok.get_freqK_unk_token(tokenizer, file_paths=[], freqK=10)
        self.assertFalse(result)
        result = tfkit.utility.tok.get_freqK_unk_token(tokenizer, file_paths=[self.DATASET_DIR + '/unk_tok.csv'],
                                                       freqK=1)
        self.assertTrue(len(result) > 0)
        result = tfkit.utility.tok.get_topP_unk_token(tokenizer, file_paths=[self.DATASET_DIR + '/unk_tok.csv'],
                                                      topP=0.9)
        self.assertTrue(len(result) > 0)

    def testHandleExceed(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        seq = " ".join([str(_) for _ in range(100)])
        maxlen = 50
        for mode in ['remove', 'slide', 'start_slice', 'end_slice']:
            rlt, _ = tfkit.utility.tok.handle_exceed(tokenizer, seq, maxlen, mode=mode)
            if mode == 'remove':
                self.assertTrue(len(rlt) == 0)
            if mode == 'slide':
                self.assertTrue(len(rlt) > 1)
            for i in rlt:
                print(i)
                self.assertTrue(len(i) == 50)
