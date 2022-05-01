import os
import sys

import numpy
from transformers import AutoTokenizer

from tfkit.utility.dataset import get_dataset, TFKitDataset
from tfkit.utility.model import load_pretrained_tokenizer, load_model_class

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import unittest
import tfkit
from tfkit.test import *


class TestDataset(unittest.TestCase):

    def check_type_for_dataloader(self, data_item):
        if (isinstance(data_item, list) and not isinstance(data_item[-1], str) and self.check_type_for_dataloader(
                data_item[-1])) or \
                isinstance(data_item, numpy.ndarray) or \
                isinstance(data_item, int):
            return True
        else:
            return False

    def test_check_type_for_dataloader(self):
        self.assertTrue(self.check_type_for_dataloader(123))
        self.assertFalse(self.check_type_for_dataloader('a'))
        self.assertFalse(self.check_type_for_dataloader(['a']))
        self.assertFalse(self.check_type_for_dataloader([{'a'}]))

    def test_get_dataset(self):
        tokenizer = load_pretrained_tokenizer('voidful/albert_chinese_tiny')
        model_class = load_model_class('clas')
        file_path = CLAS_DATASET
        dataset_arg = {
            'maxlen': 123,
            'handle_exceed': 'slide',
            'config': 'voidful/albert_chinese_tiny',
            'cache': False
        }
        ds = get_dataset(file_path=file_path, task_class=model_class, tokenizer=tokenizer,
                                       parameter=dataset_arg)
        print(ds, ds[0])

    # def testClassifier(self):
    #     for i in tfkit.task.clas.get_data_from_file(CLAS_DATASET):
    #         print(i)
    #     tokenizer = AutoTokenizer.from_pretrained('voidful/albert_chinese_tiny')
    #
    #     for i in TFKitDataset(CLAS_DATASET,
    #                          tokenizer=tokenizer,
    #                          get_data_from_file=tfkit.task.clas.get_data_from_file,
    #                          preprocessor=tfkit.task.clas.preprocessor,
    #                          get_feature_from_data=tfkit.task.clas.get_feature_from_data):
    #         print(i)
    #         self.assertTrue(len(i['input']) <= 512)
    #         self.assertTrue(len(i['target']) <= 512)
    # def testProcessingArg_maxlen(self):
    #     maxlen = 128
    #     tokenizer = AutoTokenizer.from_pretrained('voidful/albert_chinese_small')
    #     for m in [tfkit.task.seq2seq, tfkit.task.clas]:
    #         for i in TFKitDataset(GEN_DATASET,
    #                              tokenizer=tokenizer,
    #                              get_data_from_file=m.get_data_from_file,
    #                              preprocessor=m.preprocessor,
    #                              get_feature_from_data=m.get_feature_from_data,
    #                              preprocessing_arg={'maxlen': maxlen, }):
    #             print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['input'])))
    #             print(len(i['input']), i['input'])
    #             self.assertTrue(len(i['input']) <= maxlen)
    #             print(len(i['target']), i['target'])
    #             self.assertTrue(len(i['target']) <= maxlen)
    #
    # def testSeq2seq(self):
    #     tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    #
    #     for i in tfkit.task.seq2seq.get_data_from_file(GEN_DATASET):
    #         print(i)
    #     for i in TFKitDataset(GEN_DATASET,
    #                          tokenizer=tokenizer,
    #                          get_data_from_file=tfkit.task.seq2seq.get_data_from_file,
    #                          preprocessor=tfkit.task.seq2seq.preprocessor,
    #                          get_feature_from_data=tfkit.task.seq2seq.get_feature_from_data):
    #         print(i.keys())
    #         self.assertTrue(len(i['prev']) == 512)
    #         self.assertTrue(len(i['input']) == 512)
    #         self.assertTrue(len(i['target']) == 512)
    #
    #     maxlen = 100
    #     for likelihood in ['none', 'neg', 'pos', 'both']:
    #         for i in TFKitDataset(GEN_DATASET,
    #                              tokenizer=tokenizer,
    #                              get_data_from_file=tfkit.task.seq2seq.get_data_from_file,
    #                              preprocessor=tfkit.task.seq2seq.preprocessor,
    #                              get_feature_from_data=tfkit.task.seq2seq.get_feature_from_data,
    #                              preprocessing_arg={'maxlen': maxlen, 'likelihood': likelihood}):
    #             print(likelihood, i)
    #             print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['input'])))
    #             print(len(i['input']), i['input'])
    #             self.assertTrue(len(i['input']) == maxlen)
    #             print(len(i['target']), i['target'])
    #             self.assertTrue(len(i['target']) == maxlen)
    #
    # def testTag(self):
    #     tokenizer = AutoTokenizer.from_pretrained('voidful/albert_chinese_tiny')
    #     for i in tfkit.tag.get_data_from_file(TAG_DATASET):
    #         print(i)
    #     maxlen = 128
    #     for i in TFKitDataset(TAG_DATASET,
    #                          tokenizer=tokenizer,
    #                          get_data_from_file=tfkit.tag.get_data_from_file,
    #                          preprocessing_data=tfkit.tag.preprocessing_data,
    #                          input_arg={'maxlen': maxlen}):
    #         print(i)
    #         self.assertTrue(len(i['input']) == maxlen)
    #         self.assertTrue(len(i['target']) == maxlen)
    #
    #     tokenizer = AutoTokenizer.from_pretrained('voidful/albert_chinese_tiny')
    #     feature = tfkit.tag.preprocessing_data(tokenizer, ["B_Thing", "I_Thing", "O"], "狼 煙 逝 去 ， 幽 夢 醒 來 。",
    #                                            target="O O O O O O O O O O", maxlen=5, separator=" ",
    #                                            handle_exceed='slide')
    #     self.assertEqual(len(feature[0]['target']), 5)
    #
    #     print("start_slice")
    #     feature = tfkit.tag.preprocessing_data(tokenizer, ["B_Thing", "I_Thing", "O"], "狼 煙 逝 去 ， 幽 夢 醒 來 。",
    #                                            target="O O O O O O O O O O", maxlen=5, separator=" ",
    #                                            handle_exceed='start_slice')
    #     self.assertEqual(feature[0]['pos'], [0, 4])
    #
    #     print("end_slice")
    #     feature = tfkit.tag.preprocessing_data(tokenizer, ["B_Thing", "I_Thing", "O"], "狼 煙 逝 去 ， 幽 夢 醒 來 。",
    #                                            target="O O O O O O O O O O", maxlen=5, separator=" ",
    #                                            handle_exceed='end_slice')
    #     self.assertEqual(feature[0]['pos'], [6, 10])
    #
    # def testOnce(self):
    #     tokenizer = AutoTokenizer.from_pretrained('voidful/albert_chinese_tiny')
    #     for i in tfkit.once.get_data_from_file(GEN_DATASET):
    #         print(i)
    #     maxlen = 512
    #     for i in TFKitDataset(GEN_DATASET,
    #                          tokenizer=tokenizer,
    #                          get_data_from_file=tfkit.task.once.get_data_from_file,
    #                          preprocessing_data=tfkit.task.once.preprocessing_data,
    #                          input_arg={'maxlen': maxlen}):
    #         print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['input'])))
    #         print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['target'])))
    #         print(i)
    #         self.assertEqual(maxlen, len(i['input']))
    #         self.assertEqual(maxlen, len(i['target']))
    #
    # def testOnceCTC(self):
    #     tokenizer = AutoTokenizer.from_pretrained('voidful/albert_chinese_tiny')
    #     for i in tfkit.oncectc.get_data_from_file(GEN_DATASET):
    #         print(i)
    #     maxlen = 100
    #     for i in TFKitDataset(GEN_DATASET,
    #                          tokenizer=tokenizer,
    #                          get_data_from_file=tfkit.task.oncectc.get_data_from_file,
    #                          preprocessing_data=tfkit.task.oncectc.preprocessing_data,
    #                          input_arg={'maxlen': maxlen}):
    #         print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['input'])))
    #         print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['target_once'])))
    #         print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['target'])))
    #         print(i)
    #         print(len(i['target']))
    #         self.assertEqual(maxlen, len(i['input']))
    #         self.assertEqual(maxlen, len(i['target']))
    #
    # def testCLM(self):
    #     tokenizer = AutoTokenizer.from_pretrained('voidful/albert_chinese_tiny')
    #
    #     for i in tfkit.clm.get_data_from_file(GEN_DATASET):
    #         print(i)
    #
    #     maxlen = 100
    #     for likelihood in ['none', 'neg', 'pos', 'both']:
    #         print(likelihood)
    #         for i in TFKitDataset(GEN_DATASET,
    #                              tokenizer=tokenizer,
    #                              get_data_from_file=tfkit.task.clm.get_data_from_file,
    #                              preprocessing_data=tfkit.task.clm.preprocessing_data,
    #                              input_arg={'maxlen': maxlen, 'likelihood': likelihood}):
    #             start_pos = i['start']
    #             print(likelihood, i)
    #             # self.assertTrue(tokenizer.mask_token_id == i['input'][start_pos])
    #             if 'neg' in likelihood and i['target'][start_pos] == -1:
    #                 self.assertTrue(i['ntarget'][start_pos] != -1)
    #             else:
    #                 self.assertTrue(i['target'][start_pos] != -1)
    #             self.assertEqual(maxlen, len(i['input']))
    #             self.assertEqual(maxlen, len(i['target']))
    #
    # def testOnebyone(self):
    #     tokenizer = AutoTokenizer.from_pretrained('voidful/albert_chinese_tiny')
    #
    #     maxlen = 10
    #
    #     for i in tfkit.onebyone.get_data_from_file(GEN_DATASET):
    #         print(i)
    #
    #     maxlen = 100
    #     for likelihood in ['none', 'neg', 'pos', 'both']:
    #         print(likelihood)
    #         for i in TFKitDataset(GEN_DATASET,
    #                              tokenizer=tokenizer,
    #                              get_data_from_file=tfkit.task.onebyone.get_data_from_file,
    #                              preprocessing_data=tfkit.task.onebyone.preprocessing_data,
    #                              input_arg={'maxlen': maxlen, 'likelihood': likelihood}):
    #             start_pos = i['start']
    #             print(likelihood, i)
    #             self.assertTrue(tokenizer.mask_token_id == i['input'][start_pos])
    #             if 'neg' in likelihood and i['target'][start_pos] == -1:
    #                 self.assertTrue(i['ntarget'][start_pos] != -1)
    #             else:
    #                 self.assertTrue(i['target'][start_pos] != -1)
    #
    #             self.assertEqual(maxlen, len(i['input']))
    #             self.assertEqual(maxlen, len(i['target']))
    #
    # def testGPT(self):
    #     tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
    #     maxlen = 512
    #     for likelihood in ['none', 'neg', 'pos', 'both']:
    #         output = []
    #         for i in TFKitDataset(GEN_DATASET,
    #                              tokenizer=tokenizer,
    #                              get_data_from_file=tfkit.task.onebyone.get_data_from_file,
    #                              preprocessing_data=tfkit.task.onebyone.preprocessing_data,
    #                              input_arg={'maxlen': maxlen, 'likelihood': likelihood}):
    #             start_pos = i['start']
    #             output.extend(tokenizer.convert_ids_to_tokens([i['target'][start_pos]]))
    #             print(output)
    #         print(tokenizer.convert_tokens_to_string(output) + "\n")
    #
    # def testQA(self):
    #     tokenizer = AutoTokenizer.from_pretrained('voidful/albert_chinese_small')
    #     for i in tfkit.qa.get_data_from_file(QA_DATASET):
    #         print(i)
    #     for i in TFKitDataset(QA_DATASET,
    #                          tokenizer=tokenizer,
    #                          get_data_from_file=tfkit.task.qa.get_data_from_file,
    #                          preprocessing_data=tfkit.task.qa.preprocessing_data):
    #         print(i)
    #         self.assertTrue(len(i['input']) <= 512)
    #         self.assertTrue(len(i['target']) == 2)
    #     for i in TFKitDataset(QA_DATASET,
    #                          tokenizer=tokenizer,
    #                          get_data_from_file=tfkit.task.qa.get_data_from_file,
    #                          preprocessing_data=tfkit.task.qa.preprocessing_data):
    #         self.assertTrue(len(i['input']) <= 512)
    #         self.assertTrue(len(i['target']) == 2)
    #
    # def testLen(self):
    #     tokenizer = AutoTokenizer.from_pretrained('voidful/albert_chinese_small')
    #     ds = TFKitDataset(QA_DATASET,
    #                      tokenizer=tokenizer,
    #                      get_data_from_file=tfkit.task.qa.get_data_from_file,
    #                      preprocessing_data=tfkit.task.qa.preprocessing_data)
    #     old_len = ds.__len__()
    #     ds.increase_with_sampling(2)
    #     self.assertEqual(old_len, ds.__len__())
    #
    #     ds = TFKitDataset(QA_DATASET,
    #                      tokenizer=tokenizer,
    #                      get_data_from_file=tfkit.task.qa.get_data_from_file,
    #                      preprocessing_data=tfkit.task.qa.preprocessing_data)
    #     print("before increase_with_sampling", ds.__len__())
    #     ds.increase_with_sampling(50)
    #     print("after increase_with_sampling", ds.__len__())
    #     self.assertTrue(ds.__len__() == 50)
    #
    # def testSeq2seqWithPrev(self):
    #     tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    #     maxlen = 10
    #     for i in tfkit.seq2seq.get_data_from_file(GEN_DATASET):
    #         print("data", i)
    #
    # def testSeq2seqBT(self):
    #     tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    #
    #     for i in tfkit.seq2seqbt.get_data_from_file(GEN_DATASET):
    #         print("data", i)
    #
    #     maxlen = 10
