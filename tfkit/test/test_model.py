import unittest

from transformers import AutoTokenizer

import tfkit
from torch.utils import data


class TestDataLoader(unittest.TestCase):

    def testClassifier(self):
        model = tfkit.classifier.BertMtClassifier(tasks_detail={"taskA": ["a", "b"]}, model_config="albert-base-v2")

    def testQA(self):
        input = "梵 語 在 社 交 中 口 頭 使 用 , 並 且 在 早 期 古 典 梵 語 文 獻 的 發 展 中 維 持 口 頭 傳 統 。 在 印 度 , 書 寫 形 式 是 當 梵 語 發 展 成 俗 語 之 後 才 出 現 的 ; 在 書 寫 梵 語 的 時 候 , 書 寫 系 統 的 選 擇 受 抄 寫 者 所 處 地 域 的 影 響 。 同 樣 的 , 所 有 南 亞 的 主 要 書 寫 系 統 事 實 上 都 用 於 梵 語 文 稿 的 抄 寫 。 自 1 9 世 紀 晚 期 , 天 城 文 被 定 為 梵 語 的 標 準 書 寫 系 統 , 十 分 可 能 的 原 因 是 歐 洲 人 有 用 這 種 文 字 印 刷 梵 語 文 本 的 習 慣 。 最 早 的 已 知 梵 語 碑 刻 可 確 定 為 公 元 前 一 世 紀 。 它 們 採 用 了 最 初 用 於 俗 語 而 非 梵 語 的 婆 羅 米 文 。 第 一 個 書 寫 梵 語 的 證 據 , 出 現 在 晚 於 它 的 俗 語 的 書 寫 證 據 之 後 的 幾 個 世 紀 , 這 被 描 述 為 一 種 悖 論 。 在 梵 語 被 書 寫 下 來 的 時 候 , 它 首 先 用 於 行 政 、 文 學 或 科 學 類 的 文 本 。 宗 教 文 本 口 頭 傳 承 , 在 相 當 晚 的 時 候 才 「 不 情 願 」 地 被 書 寫 下 來 。 [Question] 最 初 梵 語 以 什 麼 書 寫 系 統 被 記 錄 下 來 ?"
        target = [201, 205]
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        feature = tfkit.qa.get_feature_from_data(tokenizer, input, target, maxlen=512)
        for k, v in feature.items():
            feature[k] = [v, v]
        model = tfkit.qa.BertQA(model_config="distilbert-base-multilingual-cased")
        # print(model(feature))
        # print(model(feature, eval=True))
        print(model.predict(input=input))

    # def testTagRow(self):
    #     for i in tfkit.tag.get_data_from_file_row('../demo_data/tag_row.csv'):
    #         print(i)
    #     for i in tfkit.tag.loadRowTaggerDataset('../demo_data/tag_row.csv', pretrained='bert-base-chinese', maxlen=128):
    #         self.assertTrue(len(i['input']) < 512)
    #         self.assertTrue(len(i['target']) < 512)
    #
    # def testTagCol(self):
    #     for i in tfkit.tag.get_data_from_file_col('../demo_data/tag_col.csv'):
    #         print(i)
    #     for i in tfkit.tag.loadColTaggerDataset('../demo_data/tag_col.csv', pretrained='bert-base-chinese', maxlen=128):
    #         self.assertTrue(len(i['input']) < 512)
    #         self.assertTrue(len(i['target']) < 512)
    #
    # def testOnce(self):
    #     for i in tfkit.gen_once.get_data_from_file('../demo_data/generate.csv'):
    #         print(i)
    #     for i in tfkit.gen_once.loadOnceDataset('../demo_data/generate.csv', pretrained='bert-base-chinese', maxlen=128):
    #         self.assertTrue(len(i['input']) < 512)
    #         self.assertTrue(len(i['target']) < 512)
    #
    # def testOnebyone(self):
    #     for i in tfkit.gen_onebyone.get_data_from_file('../demo_data/generate.csv'):
    #         print(i)
    #     for i in tfkit.gen_onebyone.loadOneByOneDataset('../demo_data/generate.csv', pretrained='bert-base-chinese',
    #                                                     maxlen=24):
    #         # print(len(i['input']))
    #         # print(len(i['target']))
    #         # print(i)
    #         self.assertTrue(len(i['input']) <= 24)
    #         self.assertTrue(len(i['target']) <= 24)
    #
    # def testClassifier(self):
    #     for i in tfkit.classifier.get_data_from_file('../demo_data/classification.csv'):
    #         print(i)
    #     for i in tfkit.classifier.loadClassifierDataset('../demo_data/classification.csv',
    #                                                     pretrained='bert-base-chinese',
    #                                                     maxlen=512):
    #         self.assertTrue(len(i['input']) <= 512)
    #         self.assertTrue(len(i['target']) < 512)
