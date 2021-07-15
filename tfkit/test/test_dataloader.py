import os
import unittest

import tfkit
from transformers import BertTokenizer, AutoTokenizer
from tfkit.utility.dataset import LoadDataset
from tfkit.test import *


class TestDataLoader(unittest.TestCase):

    def testTag(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        for i in tfkit.tag.get_data_from_file(TAG_DATASET):
            print(i)
        maxlen = 128
        for i in LoadDataset(TAG_DATASET,
                             tokenizer=tokenizer,
                             get_data_from_file=tfkit.tag.get_data_from_file,
                             preprocessing_data=tfkit.tag.preprocessing_data,
                             input_arg={'maxlen': maxlen}):
            print(i)
            self.assertTrue(len(i['input']) == maxlen)
            self.assertTrue(len(i['target']) == maxlen)

        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        feature = tfkit.tag.get_feature_from_data(tokenizer, ["B_Thing", "I_Thing", "O"], "狼 煙 逝 去 ， 幽 夢 醒 來 。",
                                                  target="O O O O O O O O O O", maxlen=5, separator=" ",
                                                  handle_exceed='slide')
        self.assertEqual(len(feature[0]['target']), 5)

        print("start_slice")
        feature = tfkit.tag.get_feature_from_data(tokenizer, ["B_Thing", "I_Thing", "O"], "狼 煙 逝 去 ， 幽 夢 醒 來 。",
                                                  target="O O O O O O O O O O", maxlen=5, separator=" ",
                                                  handle_exceed='start_slice')
        self.assertEqual(feature[0]['pos'], [0, 4])

        print("end_slice")
        feature = tfkit.tag.get_feature_from_data(tokenizer, ["B_Thing", "I_Thing", "O"], "狼 煙 逝 去 ， 幽 夢 醒 來 。",
                                                  target="O O O O O O O O O O", maxlen=5, separator=" ",
                                                  handle_exceed='end_slice')
        self.assertEqual(feature[0]['pos'], [6, 10])

    def testMask(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        for i in tfkit.mask.get_data_from_file(MASK_DATASET):
            print("get_data_from_file", i)
            tfkit.mask.preprocessing_data(i, BertTokenizer.from_pretrained('voidful/albert_chinese_tiny'))

        maxlen = 512
        for i in LoadDataset(MASK_DATASET,
                             tokenizer=tokenizer,
                             get_data_from_file=tfkit.mask.get_data_from_file,
                             preprocessing_data=tfkit.mask.preprocessing_data,
                             input_arg={'maxlen': maxlen}):
            print(i)
            self.assertTrue(len(i['input']) == maxlen)
            self.assertTrue(len(i['target']) == maxlen)

    def testMCQ(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        for i in tfkit.mcq.get_data_from_file(MCQ_DATASET):
            print("get_data_from_file", i)

        maxlen = 512
        for i in LoadDataset(MCQ_DATASET,
                             tokenizer=tokenizer,
                             get_data_from_file=tfkit.model.mcq.get_data_from_file,
                             preprocessing_data=tfkit.model.mcq.preprocessing_data,
                             input_arg={'maxlen': maxlen}):
            print(i)
            self.assertTrue(len(i['input']) == maxlen)

    def testOnce(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        for i in tfkit.once.get_data_from_file(GEN_DATASET):
            print(i)
        maxlen = 512
        for i in LoadDataset(GEN_DATASET,
                             tokenizer=tokenizer,
                             get_data_from_file=tfkit.model.once.get_data_from_file,
                             preprocessing_data=tfkit.model.once.preprocessing_data,
                             input_arg={'maxlen': maxlen}):
            print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['input'])))
            print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['target'])))
            print(len(i['target']))
            self.assertEqual(maxlen, len(i['input']))
            self.assertEqual(maxlen, len(i['target']))

    def testOnceCTC(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        for i in tfkit.oncectc.get_data_from_file(GEN_DATASET):
            print(i)
        maxlen = 100
        for i in LoadDataset(GEN_DATASET,
                             tokenizer=tokenizer,
                             get_data_from_file=tfkit.model.oncectc.get_data_from_file,
                             preprocessing_data=tfkit.model.oncectc.preprocessing_data,
                             input_arg={'maxlen': maxlen}):
            print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['input'])))
            print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['target_once'])))
            print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['target'])))
            print(i)
            print(len(i['target']))
            self.assertEqual(maxlen, len(i['input']))
            self.assertEqual(maxlen, len(i['target']))

    def testCLM(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')

        maxlen = 10
        feature = tfkit.clm.get_feature_from_data(tokenizer, maxlen, "go go go go go go go", '',
                                                  target=["hi"],
                                                  reserved_len=3)[-1]
        print(feature)

        for i in tfkit.clm.get_data_from_file(GEN_DATASET):
            print(i)

        maxlen = 100
        for likelihood in ['none', 'neg', 'pos', 'both']:
            print(likelihood)
            for i in LoadDataset(GEN_DATASET,
                                 tokenizer=tokenizer,
                                 get_data_from_file=tfkit.model.clm.get_data_from_file,
                                 preprocessing_data=tfkit.model.clm.preprocessing_data,
                                 input_arg={'maxlen': maxlen, 'likelihood': likelihood}):
                start_pos = i['start']
                print(likelihood, i)
                # self.assertTrue(tokenizer.mask_token_id == i['input'][start_pos])
                if 'neg' in likelihood and i['target'][start_pos] == -1:
                    self.assertTrue(i['ntarget'][start_pos] != -1)
                else:
                    self.assertTrue(i['target'][start_pos] != -1)
                self.assertEqual(maxlen, len(i['input']))
                self.assertEqual(maxlen, len(i['target']))

    def testOnebyone(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')

        maxlen = 10
        feature = tfkit.onebyone.get_feature_from_data(tokenizer, maxlen, "go go go go go go go", '',
                                                       target=["hi"],
                                                       reserved_len=3)[-1]
        print(feature)
        self.assertTrue(feature['start'] == maxlen - 3)  ## -reserved_len

        for i in tfkit.onebyone.get_data_from_file(GEN_DATASET):
            print(i)

        maxlen = 100
        for likelihood in ['none', 'neg', 'pos', 'both']:
            print(likelihood)
            for i in LoadDataset(GEN_DATASET,
                                 tokenizer=tokenizer,
                                 get_data_from_file=tfkit.model.onebyone.get_data_from_file,
                                 preprocessing_data=tfkit.model.onebyone.preprocessing_data,
                                 input_arg={'maxlen': maxlen, 'likelihood': likelihood}):
                start_pos = i['start']
                print(likelihood, i)
                self.assertTrue(tokenizer.mask_token_id == i['input'][start_pos])
                if 'neg' in likelihood and i['target'][start_pos] == -1:
                    self.assertTrue(i['ntarget'][start_pos] != -1)
                else:
                    self.assertTrue(i['target'][start_pos] != -1)

                self.assertEqual(maxlen, len(i['input']))
                self.assertEqual(maxlen, len(i['target']))

    def testGPT(self):
        tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
        maxlen = 512
        for likelihood in ['none', 'neg', 'pos', 'both']:
            output = []
            for i in LoadDataset(GEN_DATASET,
                                 tokenizer=tokenizer,
                                 get_data_from_file=tfkit.model.onebyone.get_data_from_file,
                                 preprocessing_data=tfkit.model.onebyone.preprocessing_data,
                                 input_arg={'maxlen': maxlen, 'likelihood': likelihood}):
                start_pos = i['start']
                output.extend(tokenizer.convert_ids_to_tokens([i['target'][start_pos]]))
                print(output)
            print(tokenizer.convert_tokens_to_string(output) + "\n")

    def testClassifier(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        for i in tfkit.model.clas.get_data_from_file(CLAS_DATASET):
            print(i)
        for i in LoadDataset(CLAS_DATASET,
                             tokenizer=tokenizer,
                             get_data_from_file=tfkit.model.clas.get_data_from_file,
                             preprocessing_data=tfkit.model.clas.preprocessing_data):
            print(i)
            self.assertTrue(len(i['input']) <= 512)
            self.assertTrue(len(i['target']) < 512)

    def testQA(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_small')
        for i in tfkit.qa.get_data_from_file(QA_DATASET):
            print(i)
        for i in LoadDataset(QA_DATASET,
                             tokenizer=tokenizer,
                             get_data_from_file=tfkit.model.qa.get_data_from_file,
                             preprocessing_data=tfkit.model.qa.preprocessing_data):
            print(i)
            self.assertTrue(len(i['input']) <= 512)
            self.assertTrue(len(i['target']) == 2)
        for i in LoadDataset(QA_DATASET,
                             tokenizer=tokenizer,
                             get_data_from_file=tfkit.model.qa.get_data_from_file,
                             preprocessing_data=tfkit.model.qa.preprocessing_data):
            self.assertTrue(len(i['input']) <= 512)
            self.assertTrue(len(i['target']) == 2)

    def testLen(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_small')
        ds = LoadDataset(QA_DATASET,
                         tokenizer=tokenizer,
                         get_data_from_file=tfkit.model.qa.get_data_from_file,
                         preprocessing_data=tfkit.model.qa.preprocessing_data)
        old_len = ds.__len__()
        ds.increase_with_sampling(2)
        self.assertEqual(old_len, ds.__len__())

        ds = LoadDataset(QA_DATASET,
                         tokenizer=tokenizer,
                         get_data_from_file=tfkit.model.qa.get_data_from_file,
                         preprocessing_data=tfkit.model.qa.preprocessing_data)
        print("before increase_with_sampling", ds.__len__())
        ds.increase_with_sampling(50)
        print("after increase_with_sampling", ds.__len__())
        self.assertTrue(ds.__len__() == 50)

    def testSeq2seq(self):
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

        maxlen = 10
        feature = tfkit.seq2seq.get_feature_from_data(tokenizer, maxlen, "go go go go go go go", [],
                                                      reserved_len=3)[-1]
        print(feature)
        self.assertTrue(len(feature['prev']) > 0)

        feature = tfkit.seq2seq.get_feature_from_data(tokenizer, maxlen, "go go go go go go go", '',
                                                      target=["hi", "bye"],
                                                      reserved_len=3)[-1]
        print("feature", feature)

        for i in tfkit.seq2seq.get_data_from_file(GEN_DATASET):
            print("data", i)

        maxlen = 512
        for likelihood in ['none', 'neg', 'pos', 'both']:
            print(likelihood)
            for i in LoadDataset(GEN_DATASET,
                                 tokenizer=tokenizer,
                                 get_data_from_file=tfkit.seq2seq.get_data_from_file,
                                 preprocessing_data=tfkit.seq2seq.preprocessing_data,
                                 input_arg={'maxlen': maxlen, 'likelihood': likelihood}):
                print(likelihood, i)
                print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['input'])))
                self.assertTrue(len(i['input']) == maxlen)
                self.assertTrue(len(i['target']) == maxlen)

    def testSeq2seqWithPrev(self):
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
        maxlen = 10
        for i in tfkit.seq2seq.get_data_from_file(GEN_DATASET):
            print("data", i)
            for j in tfkit.seq2seq.preprocessing_data(i, tokenizer, maxlen=maxlen):
                print(j)
            break