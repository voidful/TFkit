import unittest
import tfkit


class TestDataLoader(unittest.TestCase):

    def testTagRow(self):
        for i in tfkit.tag.get_data_from_file_row('../demo_data/tag_row.csv'):
            print(i)
        for i in tfkit.tag.loadRowTaggerDataset('../demo_data/tag_row.csv', tokenizer='bert-base-chinese', maxlen=128):
            print(i)

    def testTagCol(self):
        for i in tfkit.tag.get_data_from_file_col('../demo_data/tag_col.csv'):
            print(i)
        for i in tfkit.tag.loadColTaggerDataset('../demo_data/tag_col.csv', tokenizer='bert-base-chinese', maxlen=128):
            print(i)

    def testOnce(self):
        for i in tfkit.gen_once.get_data_from_file('../demo_data/generate.csv'):
            print(i)
        for i in tfkit.gen_once.loadOnceDataset('../demo_data/generate.csv', tokenizer='bert-base-chinese', maxlen=128):
            print(i)

    def testOnebyone(self):
        for i in tfkit.gen_onebyone.get_data_from_file('../demo_data/generate.csv'):
            print(i)
        for i in tfkit.gen_onebyone.loadOneByOneDataset('../demo_data/generate.csv', tokenizer='bert-base-chinese',
                                                        maxlen=512):
            print(i)

    def testClassifier(self):
        for i in tfkit.classifier.get_data_from_file('../demo_data/classification.csv'):
            print(i)
        for i in tfkit.classifier.loadClassifierDataset('../demo_data/classification.csv',
                                                        tokenizer='bert-base-chinese',
                                                        maxlen=512):
            print(i)
