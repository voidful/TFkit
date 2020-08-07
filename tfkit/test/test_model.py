import os
import unittest

import pytest
from torch import Tensor
from transformers import BertTokenizer, AutoModel

import tfkit


class TestModel(unittest.TestCase):

    def testClassifier(self):
        input = "One hundred thirty-four patients suspected of having pancreas cancer successfully underwent gray scale ultrasound examination of the pancreas ."
        target = "a"
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        pretrained = AutoModel.from_pretrained('voidful/albert_chinese_tiny')

        feature = tfkit.classifier.get_feature_from_data(tokenizer, task_lables={"taskA": ["a", "b"]}, task="taskA",
                                                         input=input, target=target, maxlen=512)
        for k, v in feature.items():
            feature[k] = [v, v]
        model = tfkit.classifier.MtClassifier({"taskA": ["a", "b"]}, tokenizer, pretrained)
        print(model(feature))
        self.assertTrue(isinstance(model(feature), Tensor))
        print(model(feature, eval=True))
        model_dict = model(feature, eval=True)
        self.assertTrue('label_prob_all' in model_dict)
        self.assertTrue('label_map' in model_dict)
        print(model.predict(task="taskA", input=input))
        print(model.predict(task="taskA", input=input, topk=2))
        # test exceed 512
        result, model_dict = model.predict(task="taskA", input="T " * 512)
        self.assertTrue(isinstance(result, list))
        self.assertTrue(len(result) == 0)

    def testQA(self):
        input = "梵 語 在 社 交 中 口 頭 使 用 , 並 且 在 早 期 古 典 梵 語 文 獻 的 發 展 中 維 持 口 頭 傳 統 。 在 印 度 , 書 寫 形 式 是 當 梵 語 發 展 成 俗 語 之 後 才 出 現 的 ; 在 書 寫 梵 語 的 時 候 , 書 寫 系 統 的 選 擇 受 抄 寫 者 所 處 地 域 的 影 響 。 同 樣 的 , 所 有 南 亞 的 主 要 書 寫 系 統 事 實 上 都 用 於 梵 語 文 稿 的 抄 寫 。 自 1 9 世 紀 晚 期 , 天 城 文 被 定 為 梵 語 的 標 準 書 寫 系 統 , 十 分 可 能 的 原 因 是 歐 洲 人 有 用 這 種 文 字 印 刷 梵 語 文 本 的 習 慣 。 最 早 的 已 知 梵 語 碑 刻 可 確 定 為 公 元 前 一 世 紀 。 它 們 採 用 了 最 初 用 於 俗 語 而 非 梵 語 的 婆 羅 米 文 。 第 一 個 書 寫 梵 語 的 證 據 , 出 現 在 晚 於 它 的 俗 語 的 書 寫 證 據 之 後 的 幾 個 世 紀 , 這 被 描 述 為 一 種 悖 論 。 在 梵 語 被 書 寫 下 來 的 時 候 , 它 首 先 用 於 行 政 、 文 學 或 科 學 類 的 文 本 。 宗 教 文 本 口 頭 傳 承 , 在 相 當 晚 的 時 候 才 「 不 情 願 」 地 被 書 寫 下 來 。 [Question] 最 初 梵 語 以 什 麼 書 寫 系 統 被 記 錄 下 來 ?"
        target = [201, 205]
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        pretrained = AutoModel.from_pretrained('voidful/albert_chinese_tiny')

        feature = tfkit.qa.get_feature_from_data(tokenizer, input, target, maxlen=512)
        for k, v in feature.items():
            feature[k] = [v, v]
        model = tfkit.qa.QA(tokenizer, pretrained, maxlen=512)
        print(model(feature))
        self.assertTrue(isinstance(model(feature), Tensor))
        print(model(feature, eval=True))
        model_dict = model(feature, eval=True)
        self.assertTrue('label_prob_all' in model_dict)
        self.assertTrue('label_map' in model_dict)
        result, model_dict = model.predict(input=input)
        print("model_dict", model_dict, input, result)
        self.assertTrue('label_prob_all' in model_dict)
        self.assertTrue('label_map' in model_dict)
        self.assertTrue(len(result) == 1)
        result, model_dict = model.predict(input=input, topk=2)
        self.assertTrue('label_prob_all' in model_dict)
        self.assertTrue('label_map' in model_dict)
        self.assertTrue(len(result) == 2)
        # test exceed 512
        result, model_dict = model.predict(input="T " * 512)
        self.assertTrue(isinstance(result, list))
        self.assertTrue(len(result) == 0)

    def testTag(self):
        input = "在 歐 洲 , 梵 語 的 學 術 研 究 , 由 德 國 學 者 陸 特 和 漢 斯 雷 頓 開 創 。 後 來 威 廉 · 瓊 斯 發 現 印 歐 語 系 , 也 要 歸 功 於 對 梵 語 的 研 究 。 此 外 , 梵 語 研 究 , 也 對 西 方 文 字 學 及 歷 史 語 言 學 的 發 展 , 貢 獻 不 少 。 1 7 8 6 年 2 月 2 日 , 亞 洲 協 會 在 加 爾 各 答 舉 行 。 [SEP] 陸 特 和 漢 斯 雷 頓 開 創 了 哪 一 地 區 對 梵 語 的 學 術 研 究 ?"
        target = "O A A O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O"
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        pretrained = AutoModel.from_pretrained('voidful/albert_chinese_tiny')
        feature = tfkit.tag.get_feature_from_data(tokenizer, labels=["O", "A"], input=input, target=target, maxlen=512)
        for k, v in feature.items():
            feature[k] = [v, v]
        model = tfkit.tag.Tagger(["O", "A"],
                                 tokenizer, pretrained)

        self.assertTrue(isinstance(model(feature), Tensor))
        # print(model(feature, eval=True))
        model_dict = model(feature, eval=True)
        self.assertTrue('label_prob_all' in model_dict)
        self.assertTrue('label_map' in model_dict)
        result, model_dict = model.predict(input=input)
        self.assertTrue('label_prob_all' in model_dict)
        self.assertTrue('label_map' in model_dict)
        # print(result, len(result))
        self.assertTrue(isinstance(result, list))
        self.assertTrue(isinstance(result[0][0], str))
        # test exceed 512
        result, model_dict = model.predict(input="T " * 512)
        self.assertTrue(isinstance(result, list))
        self.assertTrue(len(result) == 0)

    def testOnce(self):
        input = "See you next time"
        target = "下 次 見"
        ntarget = "不 見 不 散"

        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        pretrained = AutoModel.from_pretrained('voidful/albert_chinese_tiny')

        feature = tfkit.gen_once.get_feature_from_data(tokenizer, input=input, target=target, maxlen=512)
        for k, v in feature.items():
            feature[k] = [v, v]
        model = tfkit.gen_once.Once(tokenizer, pretrained)

        print(model(feature))
        self.assertTrue(isinstance(model(feature), Tensor))
        model_dict = model(feature, eval=True)
        self.assertTrue('label_prob_all' in model_dict)
        self.assertTrue('label_map' in model_dict)
        result, model_dict = model.predict(input=input)
        self.assertTrue('label_prob_all' in model_dict)
        self.assertTrue('label_map' in model_dict)
        print(result, len(result))
        self.assertTrue(isinstance(result, list))
        self.assertTrue(isinstance(result[0][0], str))
        # test exceed 512
        result, model_dict = model.predict(input="T " * 512)
        self.assertTrue(isinstance(result, list))
        self.assertTrue(len(result) == 0)

    def testOnebyone(self):
        input = "See you next time"
        previous = "下 次"
        target = "下 次 見"
        ntarget = "不 見 不 散"

        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        pretrained = AutoModel.from_pretrained('voidful/albert_chinese_tiny')

        feature = tfkit.gen_onebyone.get_feature_from_data(tokenizer, input=input,
                                                           tokenized_previous=tokenizer.tokenize(" ".join(previous)),
                                                           tokenized_target=tokenizer.tokenize(" ".join(target)),
                                                           maxlen=512)
        for k, v in feature.items():
            feature[k] = [v, v]
        model = tfkit.gen_onebyone.OneByOne(tokenizer, pretrained)

        print(model(feature))
        self.assertTrue(isinstance(model(feature), Tensor))
        model_dict = model(feature, eval=True)
        self.assertTrue('label_prob_all' in model_dict)
        self.assertTrue('label_map' in model_dict)
        result, model_dict = model.predict(input=input)
        self.assertTrue('label_prob_all' in model_dict)
        self.assertTrue('label_map' in model_dict)
        print(result, len(result))
        self.assertTrue(isinstance(result, list))
        self.assertTrue(isinstance(result[0][0], str))
        result, model_dict = model.predict(input=input, beamsearch=True, beamsize=3)
        print("beamsaerch", result, len(result), model_dict)
        # test exceed 512
        result, model_dict = model.predict(input="T " * 512)
        self.assertTrue(isinstance(result, list))
        print(result)
        self.assertTrue(len(result) == 0)

    def testOnebyoneWithOutSpace(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../"))
        DATASET_DIR = os.path.join(ROOT_DIR, 'demo_data')

        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')

        for i in tfkit.gen_onebyone.get_data_from_file(os.path.join(DATASET_DIR, 'generate.csv')):
            tasks, task, input, target, negative_text = i
            input = input.strip()
            tokenized_target = tokenizer.tokenize(" ".join(target))
            for j in range(1, len(tokenized_target) + 1):
                feature = tfkit.gen_onebyone.get_feature_from_data(tokenizer, input=input,
                                                                   tokenized_previous=tokenized_target[:j - 1],
                                                                   tokenized_target=tokenized_target[:j],
                                                                   maxlen=20, outspacelen=0)
                target_start = feature['start']
                print(f"input: {len(feature['input'])}, {tokenizer.decode(feature['input'][:target_start])} ")
                print(f"type: {len(feature['type'])}, {feature['type'][:target_start]} ")
                print(f"mask: {len(feature['mask'])}, {feature['mask'][:target_start]} ")
                if tokenized_target is not None:
                    print(f"target: {len(feature['target'])}, {tokenizer.convert_ids_to_tokens(feature['target'][target_start])} ")
