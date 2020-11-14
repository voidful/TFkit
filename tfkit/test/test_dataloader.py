import os
import unittest

import tfkit
from transformers import BertTokenizer, AutoTokenizer
from tfkit.utility.dataset import LoadDataset


class TestDataLoader(unittest.TestCase):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../../"))
    DATASET_DIR = os.path.join(ROOT_DIR, 'demo_data')

    def testTagRow(self):
        for i in tfkit.tag.get_data_from_file(os.path.join(TestDataLoader.DATASET_DIR, 'tag_row.csv')):
            print(i)
        maxlen = 128
        for i in LoadDataset(os.path.join(TestDataLoader.DATASET_DIR, 'tag_row.csv'),
                             pretrained_config='voidful/albert_chinese_small',
                             get_data_from_file=tfkit.tag.get_data_from_file,
                             preprocessing_data=tfkit.tag.preprocessing_data,
                             input_arg={'maxlen': maxlen}):
            print(i)
            self.assertTrue(len(i['input']) == maxlen)
            self.assertTrue(len(i['target']) == maxlen)

    def testTagCol(self):

        for i in tfkit.tag.get_data_from_file_col(os.path.join(TestDataLoader.DATASET_DIR, 'tag_col.csv')):
            print(i)
        maxlen = 512
        for i in LoadDataset(os.path.join(TestDataLoader.DATASET_DIR, 'tag_col.csv'),
                             pretrained_config='voidful/albert_chinese_small',
                             get_data_from_file=tfkit.tag.get_data_from_file_col,
                             preprocessing_data=tfkit.tag.preprocessing_data,
                             input_arg={'maxlen': maxlen}):
            self.assertTrue(len(i['input']) == 512)
            self.assertTrue(len(i['target']) == 512)
        #
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        feature = tfkit.tag.get_feature_from_data(tokenizer, ["B_Thing", "I_Thing", "O"], "狼 煙 逝 去 ， 幽 夢 醒 來 。",
                                        target="O O O O O O O O O O", maxlen=5, separator=" ",
                                        handle_exceed='slide')
        for _ in feature:
            print(_)

        print("start_slice")
        feature = tfkit.tag.get_feature_from_data(tokenizer, ["B_Thing", "I_Thing", "O"], "狼 煙 逝 去 ， 幽 夢 醒 來 。",
                                                  target="O O O O O O O O O O", maxlen=5, separator=" ",
                                                  handle_exceed='start_slice')
        for _ in feature:
            print(_)

        print("end_slice")
        feature = tfkit.tag.get_feature_from_data(tokenizer, ["B_Thing", "I_Thing", "O"], "狼 煙 逝 去 ， 幽 夢 醒 來 。",
                                                  target="O O O O O O O O O O", maxlen=5, separator=" ",
                                                  handle_exceed='end_slice')
        for _ in feature:
            print(_)
    def testMask(self):
        for i in tfkit.mask.get_data_from_file(os.path.join(TestDataLoader.DATASET_DIR, 'mask.csv')):
            print("get_data_from_file", i)
            tfkit.mask.preprocessing_data(i, BertTokenizer.from_pretrained('voidful/albert_chinese_tiny'))

        maxlen = 512
        for i in LoadDataset(os.path.join(TestDataLoader.DATASET_DIR, 'mask.csv'),
                             pretrained_config='voidful/albert_chinese_tiny',
                             get_data_from_file=tfkit.mask.get_data_from_file,
                             preprocessing_data=tfkit.mask.preprocessing_data,
                             input_arg={'maxlen': maxlen}):
            print(i)
            self.assertTrue(len(i['input']) == maxlen)
            self.assertTrue(len(i['target']) == maxlen)

    def testMCQ(self):
        for i in tfkit.mcq.get_data_from_file(os.path.join(TestDataLoader.DATASET_DIR, 'mcq.csv')):
            print("get_data_from_file", i)

        maxlen = 512
        for i in LoadDataset(os.path.join(TestDataLoader.DATASET_DIR, 'mcq.csv'),
                             pretrained_config='voidful/albert_chinese_tiny',
                             get_data_from_file=tfkit.mcq.get_data_from_file,
                             preprocessing_data=tfkit.mcq.preprocessing_data,
                             input_arg={'maxlen': maxlen}):
            print(i)
            self.assertTrue(len(i['input']) == maxlen)

    def testOnce(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        for i in tfkit.once.get_data_from_file(os.path.join(TestDataLoader.DATASET_DIR, 'gen_long.csv')):
            print(i)
        maxlen = 512
        for i in LoadDataset(os.path.join(TestDataLoader.DATASET_DIR, 'gen_long.csv'),
                             pretrained_config='voidful/albert_chinese_tiny',
                             get_data_from_file=tfkit.once.get_data_from_file,
                             preprocessing_data=tfkit.once.preprocessing_data,
                             input_arg={'maxlen': maxlen}):
            # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['input'])))
            # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i['target'])))
            print(len(i['target']))
            self.assertTrue(len(i['input']) == 512)
            self.assertTrue(len(i['target']) == 512)

    def testOnebyone(self):
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')

        maxlen = 10
        feature = tfkit.onebyone.get_feature_from_data(tokenizer, maxlen, "go go go go go go go", '',
                                                       target=["hi"],
                                                       reserved_len=3)[-1]
        print(feature)
        self.assertTrue(feature['start'] == maxlen - 3)  ## -reserved_len

        for i in tfkit.onebyone.get_data_from_file(os.path.join(TestDataLoader.DATASET_DIR, 'generate.csv')):
            print(i)

        maxlen = 10
        for likelihood in ['none', 'neg', 'pos', 'both']:
            print(likelihood)
            for i in LoadDataset(os.path.join(TestDataLoader.DATASET_DIR, 'generate.csv'),
                                 pretrained_config='voidful/albert_chinese_tiny',
                                 get_data_from_file=tfkit.onebyone.get_data_from_file,
                                 preprocessing_data=tfkit.onebyone.preprocessing_data,
                                 input_arg={'maxlen': maxlen, 'likelihood': likelihood}):
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
        for likelihood in ['none', 'neg', 'pos', 'both']:
            output = []
            for i in LoadDataset(os.path.join(TestDataLoader.DATASET_DIR, 'generate.csv'),
                                 pretrained_config='openai-gpt',
                                 get_data_from_file=tfkit.onebyone.get_data_from_file,
                                 preprocessing_data=tfkit.onebyone.preprocessing_data,
                                 input_arg={'maxlen': maxlen, 'likelihood': likelihood}):
                start_pos = i['start']
                output.append(tokenizer.convert_ids_to_tokens(i['target'])[start_pos])
                print(output)
            print(tokenizer.convert_tokens_to_string(output) + "\n")

    def testClassifier(self):
        for i in tfkit.clas.get_data_from_file(os.path.join(TestDataLoader.DATASET_DIR, 'classification.csv')):
            print(i)
        for i in LoadDataset(os.path.join(TestDataLoader.DATASET_DIR, 'classification.csv'),
                             pretrained_config='voidful/albert_chinese_small',
                             get_data_from_file=tfkit.clas.get_data_from_file,
                             preprocessing_data=tfkit.clas.preprocessing_data):
            self.assertTrue(len(i['input']) <= 512)
            self.assertTrue(len(i['target']) < 512)

        # multi-label data
        for i in tfkit.clas.get_data_from_file(
                os.path.join(TestDataLoader.DATASET_DIR, 'multi_label_classification.csv')):
            print(i)

        for i in LoadDataset(
                os.path.join(TestDataLoader.DATASET_DIR, 'multi_label_classification.csv'),
                pretrained_config='voidful/albert_chinese_small',
                get_data_from_file=tfkit.clas.get_data_from_file,
                preprocessing_data=tfkit.clas.preprocessing_data):
            print(i)
            self.assertTrue(len(i['input']) <= 512)
            self.assertTrue(len(i['target']) < 512)

    def testQA(self):
        for i in tfkit.qa.get_data_from_file(os.path.join(TestDataLoader.DATASET_DIR, 'qa.csv')):
            print(i)
        for i in LoadDataset(os.path.join(TestDataLoader.DATASET_DIR, 'qa.csv'),
                             pretrained_config='voidful/albert_chinese_small',
                             get_data_from_file=tfkit.qa.get_data_from_file,
                             preprocessing_data=tfkit.qa.preprocessing_data):
            print(i)
            self.assertTrue(len(i['input']) <= 512)
            self.assertTrue(len(i['target']) == 2)
        for i in LoadDataset(os.path.join(TestDataLoader.DATASET_DIR, 'qa.csv'),
                             pretrained_config='voidful/albert_chinese_tiny',
                             get_data_from_file=tfkit.qa.get_data_from_file,
                             preprocessing_data=tfkit.qa.preprocessing_data):
            self.assertTrue(len(i['input']) <= 512)
            self.assertTrue(len(i['target']) == 2)

    def testLen(self):
        ds = LoadDataset(os.path.join(TestDataLoader.DATASET_DIR, 'qa.csv'),
                         pretrained_config='voidful/albert_chinese_small',
                         get_data_from_file=tfkit.qa.get_data_from_file,
                         preprocessing_data=tfkit.qa.preprocessing_data)
        old_len = ds.__len__()
        ds.increase_with_sampling(10)
        self.assertTrue(ds.__len__() == old_len)

        ds = LoadDataset(os.path.join(TestDataLoader.DATASET_DIR, 'qa.csv'),
                         pretrained_config='voidful/albert_chinese_small',
                         get_data_from_file=tfkit.qa.get_data_from_file,
                         preprocessing_data=tfkit.qa.preprocessing_data)
        print("before increase_with_sampling", ds.__len__())
        ds.increase_with_sampling(50)
        print("after increase_with_sampling", ds.__len__())
        self.assertTrue(ds.__len__() == 50)
