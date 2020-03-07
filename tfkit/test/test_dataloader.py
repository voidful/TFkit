import unittest
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
        for i in tfkit.gen_once.get_data_from_file('../demo_data/generate.csv'):
            print(i)
        for i in tfkit.gen_once.loadOnceDataset('../demo_data/generate.csv', pretrained='bert-base-chinese',
                                                maxlen=128):
            self.assertTrue(len(i['input']) < 512)
            self.assertTrue(len(i['target']) < 512)

    def testOnebyone(self):
        for i in tfkit.gen_onebyone.get_data_from_file('../demo_data/generate.csv'):
            print(i)
        for i in tfkit.gen_onebyone.loadOneByOneDataset('../demo_data/generate.csv', pretrained='bert-base-chinese',
                                                        maxlen=24):
            # print(len(i['input']))
            # print(len(i['target']))
            # print(i)
            self.assertTrue(len(i['input']) <= 24)
            self.assertTrue(len(i['target']) <= 24)

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
