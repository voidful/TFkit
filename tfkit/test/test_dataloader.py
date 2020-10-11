import os
import unittest

from transformers import BertTokenizer, AutoTokenizer

import tfkit


class TestDataLoader(unittest.TestCase):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../"))
    DATASET_DIR = os.path.join(ROOT_DIR, 'demo_data')

    def testTagRow(self):
        for i in tfkit.tag.get_data_from_file_row(os.path.join(TestDataLoader.DATASET_DIR, 'tag_row.csv')):
            print(i)
        for i in tfkit.tag.loadRowTaggerDataset(os.path.join(TestDataLoader.DATASET_DIR, 'tag_row.csv'),
                                                pretrained='voidful/albert_chinese_small', maxlen=128):
            print(i)
            self.assertTrue(len(i['input']) < 512)
            self.assertTrue(len(i['target']) < 512)

    def testTagCol(self):

        for i in tfkit.tag.get_data_from_file_col(os.path.join(TestDataLoader.DATASET_DIR, 'tag_col.csv')):
            print(i)
        for i in tfkit.tag.loadColTaggerDataset(os.path.join(TestDataLoader.DATASET_DIR, 'tag_col.csv'),
                                                pretrained='voidful/albert_chinese_small', maxlen=128):
            self.assertTrue(len(i['input']) < 512)
            self.assertTrue(len(i['target']) < 512)

    def testMask(self):
        for i in tfkit.gen_mask.get_data_from_file(os.path.join(TestDataLoader.DATASET_DIR, 'mask.csv')):
            print(i)

        maxlen = 100
        for i in tfkit.gen_mask.loadMaskDataset(os.path.join(TestDataLoader.DATASET_DIR, 'mask.csv'),
                                                pretrained_config='voidful/albert_chinese_tiny',
                                                maxlen=maxlen):
            print(i)
            self.assertTrue(len(i['input']) == maxlen)
            self.assertTrue(len(i['target']) == maxlen)

    def testOnce(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for i in tfkit.gen_once.get_data_from_file(os.path.join(TestDataLoader.DATASET_DIR, 'gen_long.csv')):
            print(i)
        for i in tfkit.gen_once.loadOnceDataset(os.path.join(TestDataLoader.DATASET_DIR, 'gen_long.csv'),
                                                pretrained='bert-base-uncased',
                                                maxlen=128):
            print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['input'])))
            print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['target'])))
            print(i)
            self.assertTrue(len(i['input']) < 512)
            self.assertTrue(len(i['target']) < 512)

    def testOnebyone(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')

        maxlen = 10
        feature = tfkit.gen_onebyone.get_feature_from_data(tokenizer, maxlen, "go go go go go go go", '',
                                                           tokenized_target=["hi"],
                                                           reserved_len=3)[-1]
        print(feature)
        self.assertTrue(feature['start'] == maxlen - 3)  ## -reserved_len

        maxlen = 100
        for i in tfkit.gen_onebyone.get_data_from_file(os.path.join(TestDataLoader.DATASET_DIR, 'generate.csv')):
            print(i)

        for likelihood in ['onebyone', 'onebyone-neg', 'onebyone-pos', 'onebyone-both']:
            # for likelihood in ['onebyone-both']:
            for i in tfkit.gen_onebyone.loadOneByOneDataset(os.path.join(TestDataLoader.DATASET_DIR, 'generate.csv'),
                                                            pretrained_config='voidful/albert_chinese_tiny',
                                                            maxlen=maxlen, likelihood=likelihood):
                start_pos = i['start']
                print(i)
                self.assertTrue(tokenizer.mask_token_id == i['input'][start_pos])
                if 'neg' in likelihood and i['target'][start_pos] == -1:
                    self.assertTrue(i['ntarget'][start_pos] != -1)
                else:
                    self.assertTrue(i['target'][start_pos] != -1)
                self.assertTrue(len(i['input']) == maxlen)
                self.assertTrue(len(i['target']) == maxlen)

    def testGPT(self):
        tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
        maxlen = 512
        for likelihood in ['onebyone', 'onebyone-neg', 'onebyone-pos', 'onebyone-both']:
            output = []
            for i in tfkit.gen_onebyone.loadOneByOneDataset(os.path.join(TestDataLoader.DATASET_DIR, 'generate.csv'),
                                                            pretrained_config='openai-gpt',
                                                            maxlen=maxlen, likelihood=likelihood):
                start_pos = i['start']
                output.append(tokenizer.convert_ids_to_tokens(i['target'])[start_pos])
                print(output)
            print(tokenizer.convert_tokens_to_string(output) + "\n")

    def testClassifier(self):

        for i in tfkit.classifier.get_data_from_file(os.path.join(TestDataLoader.DATASET_DIR, 'classification.csv')):
            print(i)
        for i in tfkit.classifier.loadClassifierDataset(os.path.join(TestDataLoader.DATASET_DIR, 'classification.csv'),
                                                        pretrained='voidful/albert_chinese_small',
                                                        maxlen=512):
            self.assertTrue(len(i['input']) <= 512)
            self.assertTrue(len(i['target']) < 512)

        for i in tfkit.classifier.get_data_from_file(
                os.path.join(TestDataLoader.DATASET_DIR, 'multi_label_classification.csv')):
            print(i)
        for i in tfkit.classifier.loadClassifierDataset(
                os.path.join(TestDataLoader.DATASET_DIR, 'multi_label_classification.csv'),
                pretrained='voidful/albert_chinese_small',
                maxlen=512):
            self.assertTrue(len(i['input']) <= 512)
            self.assertTrue(len(i['target']) < 512)

    def testQA(self):
        for i in tfkit.qa.get_data_from_file(os.path.join(TestDataLoader.DATASET_DIR, 'qa.csv')):
            print(i)
        for i in tfkit.qa.loadQADataset(os.path.join(TestDataLoader.DATASET_DIR, 'qa.csv'),
                                        pretrained='voidful/albert_chinese_small',
                                        maxlen=512):
            self.assertTrue(len(i['input']) <= 512)
            self.assertTrue(len(i['target']) == 2)
        for i in tfkit.qa.loadQADataset(os.path.join(TestDataLoader.DATASET_DIR, 'qa.csv'),
                                        pretrained='voidful/albert_chinese_tiny',
                                        maxlen=512):
            self.assertTrue(len(i['input']) <= 512)
            self.assertTrue(len(i['target']) == 2)

    def testLen(self):
        ds = tfkit.qa.loadQADataset(os.path.join(TestDataLoader.DATASET_DIR, 'qa.csv'),
                                    pretrained='voidful/albert_chinese_small',
                                    maxlen=512)
        print(ds.__len__())
        ds.increase_with_sampling(20)
        self.assertTrue(ds.__len__() == 20)

        ds = tfkit.qa.loadQADataset(os.path.join(TestDataLoader.DATASET_DIR, 'qa.csv'),
                                    pretrained='voidful/albert_chinese_small',
                                    maxlen=512)
        print("before increase_with_sampling", ds.__len__())
        ds.increase_with_sampling(30)
        print("after increase_with_sampling", ds.__len__())
        self.assertTrue(ds.__len__() == 30)
