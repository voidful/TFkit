import unittest

from transformers import BertTokenizer

import tfkit


class TestDataLoader(unittest.TestCase):

    def testTagRow(self):
        for i in tfkit.tag.get_data_from_file_row('../demo_data/tag_row.csv'):
            print(i)
        for i in tfkit.tag.loadRowTaggerDataset('../demo_data/tag_row.csv', pretrained='bert-base-chinese', maxlen=128):
            self.assertTrue(len(i['input']) < 512)
            self.assertTrue(len(i['target']) < 512)

    def testTagCol(self):
        for i in tfkit.tag.get_data_from_file_col('../demo_data/tag_col.csv'):
            print(i)
        for i in tfkit.tag.loadColTaggerDataset('../demo_data/tag_col.csv', pretrained='bert-base-chinese', maxlen=128):
            self.assertTrue(len(i['input']) < 512)
            self.assertTrue(len(i['target']) < 512)

    def testOnce(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        for i in tfkit.gen_once.get_data_from_file('../demo_data/generate.csv'):
            print(i)
        for i in tfkit.gen_once.loadOnceDataset('../demo_data/generate.csv', pretrained='bert-base-chinese',
                                                maxlen=128):
            print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['input'])))
            print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['target'])))
            self.assertTrue(len(i['input']) < 512)
            self.assertTrue(len(i['target']) < 512)

    def testOnebyone(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        for i in tfkit.gen_onebyone.get_data_from_file('../demo_data/generate.csv'):
            print(i)
        maxlen = 512
        for likelihood in ['onebyone-neg', 'onebyone-pos', 'onebyone-both']:
            for i in tfkit.gen_onebyone.loadOneByOneDataset('../demo_data/generate.csv', pretrained='bert-base-cased',
                                                            maxlen=maxlen, likelihood=likelihood):
                # print(likelihood, i)
                print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['input'])))
                print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['target'])))
                start_pos = i['start']
                self.assertTrue(tokenizer.mask_token_id == i['input'][start_pos])
                if 'neg' in likelihood and i['target'][start_pos] == -1:
                    self.assertTrue(i['ntarget'][start_pos] != -1)
                else:
                    self.assertTrue(i['target'][start_pos] != -1)
                self.assertTrue(len(i['input']) == maxlen)
                self.assertTrue(len(i['target']) == maxlen)

    def testClassifier(self):
        for i in tfkit.classifier.get_data_from_file('../demo_data/classification.csv'):
            print(i)
        for i in tfkit.classifier.loadClassifierDataset('../demo_data/classification.csv',
                                                        pretrained='bert-base-chinese',
                                                        maxlen=512):
            self.assertTrue(len(i['input']) <= 512)
            self.assertTrue(len(i['target']) < 512)

    def testQA(self):
        for i in tfkit.qa.get_data_from_file('../demo_data/qa.csv'):
            print(i)
        for i in tfkit.qa.loadQADataset('../demo_data/qa.csv',
                                        pretrained='bert-base-chinese',
                                        maxlen=512):
            print(i['raw_input'][int(i['target'][0]):int(i['target'][1])])
            self.assertTrue(len(i['input']) <= 512)
            self.assertTrue(len(i['target']) == 2)
        for i in tfkit.qa.loadQADataset('../demo_data/qa.csv',
                                        pretrained='voidful/albert_chinese_tiny',
                                        maxlen=512):
            print(i['raw_input'][int(i['target'][0]):int(i['target'][1])])
            self.assertTrue(len(i['input']) <= 512)
            self.assertTrue(len(i['target']) == 2)

    def testLen(self):
        ds = tfkit.qa.loadQADataset('../demo_data/qa.csv',
                                    pretrained='bert-base-chinese',
                                    maxlen=512)
        print(ds.__len__())
        ds.increase_with_sampling(20)
        self.assertTrue(ds.__len__() == 20)

        ds = tfkit.qa.loadQADataset('../demo_data/qa.csv',
                                    pretrained='bert-base-chinese',
                                    maxlen=512)
        print(ds.__len__())
        ds.increase_with_sampling(12)
        self.assertTrue(ds.__len__() == 14)
