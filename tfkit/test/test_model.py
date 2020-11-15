import os
import unittest

from torch import Tensor
from transformers import BertTokenizer, AutoModel

import tfkit


class TestModel(unittest.TestCase):

    def testClas(self):
        input = "One hundred thirty-four patients suspected of having pancreas cancer successfully underwent gray scale ultrasound examination of the pancreas ."
        target = "a"
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        pretrained = AutoModel.from_pretrained('voidful/albert_chinese_tiny')

        model = tfkit.model.clas.Model(tokenizer, pretrained, tasks_detail={"taskA": ["a", "b"]})
        for feature in tfkit.model.clas.get_feature_from_data(tokenizer, tasks={"taskA": ["a", "b"]},
                                                              task="taskA",
                                                              input=input, target=target, maxlen=512):
            for k, v in feature.items():
                feature[k] = [v, v]

            print(feature)
            # test train
            print(model(feature))
            self.assertTrue(isinstance(model(feature), Tensor))
            # test eval
            print(model(feature, eval=True))
            model_dict = model(feature, eval=True)
            self.assertTrue('label_prob_all' in model_dict)
            self.assertTrue('label_map' in model_dict)
            # test predict
            tok_label = model.predict(task="taskA", input=input)
            self.assertTrue(len(tok_label) == 2)
            # test predict with top k 2
            top_k_label, top_k_dict = model.predict(task="taskA", input=input, topk=2)
            print("test predict with top k 2, ", top_k_label, top_k_dict)
            self.assertTrue(len(top_k_label) == 2)

        # test exceed 512
        for merge_strategy in ['minentropy', 'maxcount', 'maxprob']:
            result, model_dict = model.predict(task="taskA", input=" ".join([str(i) for i in range(2000)]),
                                               merge_strategy=merge_strategy)
            print(result, len(model_dict), model_dict)
            self.assertTrue(isinstance(result, list))
            self.assertTrue(len(result) == 1)

    def testQA(self):
        input = "梵 語 在 社 交 中 口 頭 使 用 , 並 且 在 早 期 古 典 梵 語 文 獻 的 發 展 中 維 持 口 頭 傳 統 。 在 印 度 , 書 寫 形 式 是 當 梵 語 發 展 成 俗 語 之 後 才 出 現 的 ; 在 書 寫 梵 語 的 時 候 , 書 寫 系 統 的 選 擇 受 抄 寫 者 所 處 地 域 的 影 響 。 同 樣 的 , 所 有 南 亞 的 主 要 書 寫 系 統 事 實 上 都 用 於 梵 語 文 稿 的 抄 寫 。 自 1 9 世 紀 晚 期 , 天 城 文 被 定 為 梵 語 的 標 準 書 寫 系 統 , 十 分 可 能 的 原 因 是 歐 洲 人 有 用 這 種 文 字 印 刷 梵 語 文 本 的 習 慣 。 最 早 的 已 知 梵 語 碑 刻 可 確 定 為 公 元 前 一 世 紀 。 它 們 採 用 了 最 初 用 於 俗 語 而 非 梵 語 的 婆 羅 米 文 。 第 一 個 書 寫 梵 語 的 證 據 , 出 現 在 晚 於 它 的 俗 語 的 書 寫 證 據 之 後 的 幾 個 世 紀 , 這 被 描 述 為 一 種 悖 論 。 在 梵 語 被 書 寫 下 來 的 時 候 , 它 首 先 用 於 行 政 、 文 學 或 科 學 類 的 文 本 。 宗 教 文 本 口 頭 傳 承 , 在 相 當 晚 的 時 候 才 「 不 情 願 」 地 被 書 寫 下 來 。 [Question] 最 初 梵 語 以 什 麼 書 寫 系 統 被 記 錄 下 來 ?"
        target = [201, 205]
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        pretrained = AutoModel.from_pretrained('voidful/albert_chinese_tiny')
        model = tfkit.model.qa.Model(tokenizer, pretrained, maxlen=512)

        for feature in tfkit.model.qa.get_feature_from_data(tokenizer, input, target, maxlen=512):
            for k, v in feature.items():
                feature[k] = [v, v]

            # test train
            print(model(feature))
            self.assertTrue(isinstance(model(feature), Tensor))
            # test eval
            print(model(feature, eval=True))
            model_dict = model(feature, eval=True)
            self.assertTrue('label_prob_all' in model_dict)
            self.assertTrue('label_map' in model_dict)

        # test predict
        result, model_dict = model.predict(input=input)
        print("model_dict", model_dict, input, result)
        self.assertTrue('label_prob_all' in model_dict[0])
        self.assertTrue('label_map' in model_dict[0])
        self.assertTrue(len(result) == 1)

        # test eval top k = 2
        top_k_label, top_k_dict = model.predict(input=input, topk=10)
        print("top_k_label", top_k_label)
        self.assertTrue(len(top_k_label) == 10)

        # test exceed 512
        for merge_strategy in ['minentropy', 'maxcount', 'maxprob']:
            result, model_dict = model.predict(input=" ".join([str(i) for i in range(550)]),
                                               merge_strategy=merge_strategy)
            print(result, len(model_dict))
            self.assertTrue(isinstance(result, list))
            self.assertTrue(len(result) == 1)

    def testTag(self):
        input = "在 歐 洲 , 梵 語 的 學 術 研 究 , 由 德 國 學 者 陸 特 和 漢 斯 雷 頓 開 創 。 後 來 威 廉 · 瓊 斯 發 現 印 歐 語 系 , 也 要 歸 功 於 對 梵 語 的 研 究 。 此 外 , 梵 語 研 究 , 也 對 西 方 文 字 學 及 歷 史 語 言 學 的 發 展 , 貢 獻 不 少 。 1 7 8 6 年 2 月 2 日 , 亞 洲 協 會 在 加 爾 各 答 舉 行 。 [SEP] 陸 特 和 漢 斯 雷 頓 開 創 了 哪 一 地 區 對 梵 語 的 學 術 研 究 ?"
        target = "O A A O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O"
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_small')
        pretrained = AutoModel.from_pretrained('voidful/albert_chinese_small')

        model = tfkit.model.tag.Model(tokenizer=tokenizer, pretrained=pretrained, tasks_detail={"default": ["O", "A"]})

        # model, model_task, model_class = tfkit.utility.model_loader.load_trained_model('./cache/model.pt',
        #                                                                                'voidful/albert_chinese_small')
        # # test exceed 512
        # for merge_strategy in ['minentropy']:
        #     result, model_dict = model.predict(
        #         input="""
        #               明朝（1368年1月23日－1644年4月25日[註 1]）是中國歷史上最後一個由漢族建立的大一統王朝，歷經十二世、十六位皇帝，國祚二百七十六年[參 4]。\n\n元朝末年政治腐敗，種族紛爭，天災不斷，民不聊生，民變暴動屢禁不止，平民朱元璋加入紅巾軍並在其中乘勢崛起，跟隨佔據濠州的郭子興。郭子興死後，朱元璋被當時反抗軍擁立的小明王韓林兒封為左副元帥，並率部眾先後攻占滁州、和州等地，並最終攻佔集慶（今江蘇南京），採取朱升所建議的「高築牆，廣積糧，緩稱王」的政策，以鞏固根據地，讓士兵屯田積糧減少百姓負擔，以示自己為仁義之師而避免受敵。1364年，朱元璋稱吳王，建立西吳政權。1368年，在掃滅陳友諒、張士誠和方國珍等群雄勢力後，朱元璋於當年農曆正月初四日登基稱帝，立國號為大明[參 5]，定都應天府（今南京市），其轄區稱為京師，由因皇室姓朱，故又稱朱明，之後以「驅逐胡虜，恢復中華」[參 6]為號召北伐中原[參 7][參 8]，並收回了燕雲十六州[參 9]，結束蒙元在中國漢地的統治，統一天下。\n\n明初天下大定，經過朱元璋的休養生息，社會經濟得以恢復和發展，國力迅速恢復，史稱洪武之治。朱元璋去世後，其孫朱允炆即位，但其在靖難之役中敗於駐守燕京的朱元璋第四子朱棣，也自此失蹤。朱棣登基後遷都至順天府（今北京市），將北平布政司升為京師，原京師改稱南京[參 3]。成祖朱棣時期，開疆拓土，又派遣鄭和七下西洋，此後許多漢人遠赴海外，國勢達到頂峰，史稱永樂盛世。其後的仁宗和宣宗時期國家仍處於興盛時期，史稱仁宣之治[參 10]。英宗和代宗時期，遭遇土木之變，國力中衰，經于謙等人抗敵，最終解除國家危機。憲宗和孝宗相繼與民休息，孝宗則力行節儉，減免稅賦，百姓安居樂業，史稱弘治中興[參 11]。武宗時期爆發了南巡之爭和寧王之亂。世宗即位初，引發大禮議之爭，他清除宦官和權臣勢力後總攬朝綱，實現嘉靖中興，並於屯門海戰與西草灣之戰中擊退葡萄牙殖民侵略，任用胡宗憲和俞大猷等將領平定東南沿海的倭患。世宗駕崩後經過隆慶新政國力得到恢復，神宗前期任用張居正，推行萬曆新政，國家收入大增，商品經濟空前繁榮、科學巨匠迭出、社會風尚呈現出活潑開放的新鮮氣息，史稱萬曆中興[參 12]。後經過萬曆三大征平定內憂外患，粉碎豐臣秀吉攻占朝鮮進而入明的計劃，然而因為國本之爭，皇帝逐漸疏於朝政，史稱萬曆怠政，同時東林黨爭也帶來了明中期的政治混亂。\n\n萬曆一朝成為明朝由盛轉衰的轉折期[參 13]。光宗繼位不久因紅丸案暴斃，熹宗繼承大統改元天啟，天啟年間魏忠賢閹黨禍亂朝綱，至明思宗即位後剷除閹黨，但閹黨倒臺後，黨爭又起，政治腐敗以及連年天災[註 2][註 3]，導致國力衰退，最終爆發大規模民變。1644年4月25日（舊曆三月十九），李自成所建立的大順軍攻破北京，思宗自縊於煤山，是為甲申之變。隨後吳三桂倒戈相向，滿族建立的滿清入主中原。明朝宗室於江南地區相繼成立南明諸政權，而原本反明的流寇在李自成等領袖死後亦加入南明陣營，這些政權被清朝統治者先後以「為君父報仇」為名各個殲滅，1662年，明朝宗室最後政權被剷除，永曆帝被俘後被殺，滿清又陸續擊敗各地反抗軍，以及攻取台灣、澎湖，1683年，奉大明為正朔的明鄭向清朝投降，漢族抗爭勢力方為清朝所消滅。[參 16]。\n\n明代的核心領土囊括漢地[註 4]，東北到外興安嶺及黑龍江流域[參 19]，後縮為遼河流域；初年北達戈壁沙漠一帶，後改為今長城；西北至新疆哈密，後改為嘉峪關；西南臨孟加拉灣[註 5]，後折回約今雲南境；曾經在今中國東北、新疆東部及西藏等地設有羈縻機構[參 21]。不過，明朝是否實際統治了西藏國際上尚存在有一定的爭議[註 6]。明成祖時期曾短暫征服及統治安南[參 22]，永樂二十二年（1424年），明朝國土面積達到極盛，在東南亞設置舊港宣慰司[註 7]等行政機構，加強對東南洋一帶的管理[參 23][參 24]。\n\n明代商品經濟繁榮，出現商業集鎮，而手工業及文化藝術呈現世俗化趨勢[參 25]。根據《明實錄》所載的人口峰值於成化十五年（1479年）達七千餘萬人[參 26]，不過許多學者考慮到當時存在大量隱匿戶口，故認為明朝人口峰值實際上逾億[參 27]，還有學者認為晚明人口峰值接近2億[註 8]。這一時期，其GDP總量所占的世界比例在中國古代史上也是最高的，1600年明朝GDP總量為960億美元，占世界經濟總量的29.2%，晚明中國人均GDP在600美元[註 9]。\n\n明朝政治則是權力趨於集中，明太祖在誅殺胡惟庸後廢除傳統的丞相制，六部直接對皇帝負責，後來設置內閣；地方上由承宣布政使司、提刑按察使司、都指揮使司分掌權力，加強地方管理。仁宗、宣宗之後，文官治國的思想逐漸濃厚，行政權向內閣和六部轉移。同時還設有都察院等監察機構，為加強對全國臣民的監視，明太祖設立特務機構錦衣衛，明成祖設立東廠，明憲宗時再設西廠（後取消），明武宗又設內行廠（後取消），合稱「廠衛」。但明朝皇帝並非完全獨斷獨行，有許多事還必須經過經廷推、廷議、廷鞫程序，同時，能將原旨退還的給事中亦可對皇權形成制衡。[參 33]到了後期皇帝出現了怠政，宦官行使大權的陋習[參 3]，儘管決策權始終集中在皇帝手中，然而政務大部分已經由內閣處理，此外，到了明代中晚期文官集團的集體意見足以與皇帝抗衡，在遇到事情決斷兩相僵持不下時，也容易產生一種類似於「憲政危機（英語：Constitutional crisis）」的情況，因此「名義上他是天子，實際上他受制於廷臣。」[參 34]但明朝皇權受制於廷臣主要是基於道德上而非法理上，因為明朝當時風氣普遍注重名節，受儒家教育的皇帝通常不願被冠以「昏君」之名。但雖然皇權受制衡，皇帝仍可任意動用皇權，例如明世宗「大禮議」事件最後以廷杖朝臣多人的方式結束[參 35]，明神宗在國本之爭失利後也以長期拒絕參與政事向朝臣們示威[1][2][3]。\n\n有學者認為明代是繼漢唐之後的黃金時期，也被稱為最後一個可以和漢唐媲美的盛世[參 36]。清代張廷玉等修的官修《明史》評價明朝為「治隆唐宋」[註 10]、「遠邁漢唐」[參 37]。
        #               """,
        #         merge_strategy=merge_strategy, start_contain="B_",
        #         end_contain="I_")
        #     print(result)
        #     self.assertTrue(isinstance(result, list))

        for feature in tfkit.model.tag.get_feature_from_data(tokenizer, labels=["O", "A"], input=input, target=target,
                                                             maxlen=512):
            for k, v in feature.items():
                feature[k] = [v, v]
            print(feature)

            # test train
            print(model(feature))
            self.assertTrue(isinstance(model(feature), Tensor))
            # test eval
            model_dict = model(feature, eval=True)
            self.assertTrue('label_prob_all' in model_dict)
            self.assertTrue('label_map' in model_dict)

        # test predict
        result, model_dict = model.predict(input=input, start_contain="A", end_contain="A")
        print("model_dict", model_dict)
        self.assertTrue('label_prob_all' in model_dict[0])
        self.assertTrue('label_map' in model_dict[0])
        print("result", result, len(result))
        self.assertTrue(isinstance(result, list))

        # test exceed 512
        for merge_strategy in ['minentropy', 'maxcount', 'maxprob']:
            result, model_dict = model.predict(input=" ".join([str(i) for i in range(1000)]),
                                               merge_strategy=merge_strategy, start_contain="A", end_contain="A")
            print(result)
            self.assertTrue(isinstance(result, list))

    def testMask(self):
        input = "今 天 [MASK] 情 [MASK] 好"
        target = "心 很"

        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_small')
        pretrained = AutoModel.from_pretrained('voidful/albert_chinese_small')
        model = tfkit.model.mask.Model(tokenizer, pretrained)

        for feature in tfkit.model.mask.get_feature_from_data(tokenizer, input=input, target=target, maxlen=512):
            for k, v in feature.items():
                feature[k] = [v]

            print(feature)
            self.assertTrue(isinstance(model(feature), Tensor))

            model_dict = model(feature, eval=True)
            print(model_dict)
            self.assertTrue('label_map' in model_dict)
            self.assertTrue('label_prob' in model_dict)

        result, model_dict = model.predict(input=input)
        self.assertTrue('label_prob' in model_dict[0])
        self.assertTrue('label_map' in model_dict[0])
        print("predict", result, len(result))

        self.assertTrue(isinstance(result, list))
        self.assertTrue(isinstance(result[0][0], str))

        # test exceed 512
        result, model_dict = model.predict(input="T " * 512)
        self.assertTrue(isinstance(result, list))
        self.assertTrue(len(result[0][0]) == 0)

    def testMCQ(self):
        input = "你 是 誰 [SEP] [MASK] 我 [MASK] 你 [MASK] 他"
        target = 1

        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_small')
        pretrained = AutoModel.from_pretrained('voidful/albert_chinese_small')
        model = tfkit.model.mcq.Model(tokenizer, pretrained)

        for feature in tfkit.model.mcq.get_feature_from_data(tokenizer, input=input, target=target, maxlen=512):
            for k, v in feature.items():
                feature[k] = [v]

            print(feature)
            self.assertTrue(isinstance(model(feature), Tensor))

            model_dict = model(feature, eval=True)
            print(model_dict)
            self.assertTrue('label_map' in model_dict)
            self.assertTrue('label_max' in model_dict)

        result, model_dict = model.predict(input=input)
        self.assertTrue('label_max' in model_dict[0])
        self.assertTrue('label_map' in model_dict[0])
        print("predict", result, len(result))

        self.assertTrue(isinstance(result, list))
        print(result)
        self.assertTrue(isinstance(result[0], str))

        # test exceed 512
        result, model_dict = model.predict(input="T " * 300 + "[MASK]" + "T " * 300)
        self.assertTrue(isinstance(result, list))

    def testOnce(self):
        input = "See you next time"
        target = "下 次 見"
        ntarget = "不 見 不 散"

        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        pretrained = AutoModel.from_pretrained('voidful/albert_chinese_tiny')

        for feature in tfkit.model.once.get_feature_from_data(tokenizer, input=input, target=target, maxlen=512):
            for k, v in feature.items():
                feature[k] = [v, v]
            model = tfkit.model.once.Model(tokenizer, pretrained)
            self.assertTrue(isinstance(model(feature), Tensor))
            model_dict = model(feature, eval=True)
            self.assertTrue('label_prob_all' in model_dict)
            self.assertTrue('label_map' in model_dict)

        result, model_dict = model.predict(input=input)
        self.assertTrue('label_prob_all' in model_dict[0])
        self.assertTrue('label_map' in model_dict[0])
        print(result, len(result))
        self.assertTrue(isinstance(result, list))
        self.assertTrue(isinstance(result[0][0], str))
        # test exceed 512
        result, model_dict = model.predict(input="T " * 512)
        self.assertTrue(isinstance(result, list))

    def testOnebyone(self):
        input = "See you next time"
        previous = "下 次"
        target = "下 次 見"
        ntarget = "不 見 不 散"

        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_small')
        pretrained = AutoModel.from_pretrained('voidful/albert_chinese_small')

        model = tfkit.model.onebyone.Model(tokenizer, pretrained)
        # package = torch.load('./cache/model.pt', map_location='cpu')
        # for model_tag, state_dict in zip(package['tags'], package['models']):
        #     model.load_state_dict(state_dict)

        # test filter sim
        dummy_result = [[['表', '示', '事', '業'], 4.438073633101437],
                        [['表', '示', '事', '情'], 9.86092332722302],
                        [['表', '示', '事', '情'], 9.86092332722302]]
        model._filterSimilar(dummy_result, 3)
        self.assertTrue(len(dummy_result), 3)

        dummy_result = [[['表', '示', '事', '業'], 4.438073633101437],
                        [['表', '示', '事', '情'], 9.86092332722302]]
        model._filterSimilar(dummy_result, 2)
        self.assertTrue(len(dummy_result), 2)

        for feature in tfkit.model.onebyone.get_feature_from_data(tokenizer, input=input,
                                                                  previous=tokenizer.tokenize(
                                                                      " ".join(previous)),
                                                                  target=tokenizer.tokenize(
                                                                      " ".join(target)),
                                                                  maxlen=512):
            for k, v in feature.items():
                feature[k] = [v, v]

            print(model(feature))
            self.assertTrue(isinstance(model(feature), Tensor))
            model_dict = model(feature, eval=True)
            self.assertTrue('label_map' in model_dict)

        # greedy
        result, model_dict = model.predict(input=input)
        print(result, model_dict)
        self.assertTrue('label_map' in model_dict[0])
        self.assertTrue(len(result) == 1)
        self.assertTrue(isinstance(result, list))
        self.assertTrue(isinstance(result[0][0], str))

        # TopK
        result, model_dict = model.predict(input=input, decodenum=3, mode='topK', topK=3, filtersim=False)
        print("TopK no filter sim", result, len(result), model_dict)
        self.assertTrue('label_map' in model_dict[0])
        self.assertTrue(len(result) == 3)
        self.assertTrue(isinstance(result, list))
        self.assertTrue(isinstance(result[0][0], str))

        # beamsearch
        result, model_dict = model.predict(input=input, decodenum=3)
        print("beamsaerch", result, len(result), model_dict)
        self.assertTrue('label_map' in model_dict[0])
        self.assertTrue(len(result) == 3)
        self.assertTrue(isinstance(result, list))
        self.assertTrue(isinstance(result[0][0], str))

        # TopK
        result, model_dict = model.predict(input=input, decodenum=3, mode='topK', topK=20)
        print("TopK", result, len(result), model_dict)
        self.assertTrue('label_map' in model_dict[0])
        self.assertTrue(len(result) == 3)
        self.assertTrue(isinstance(result, list))
        self.assertTrue(isinstance(result[0][0], str))

        # TopP
        result, model_dict = model.predict(input=input, decodenum=3, mode='topP', topP=0.8)
        print("TopP", result, len(result), model_dict)
        self.assertTrue('label_map' in model_dict[0])
        self.assertTrue(len(result) == 3)
        self.assertTrue(isinstance(result, list))
        self.assertTrue(isinstance(result[0][0], str))

        # test exceed 512
        result, model_dict = model.predict(input="T " * 540)
        self.assertTrue(isinstance(result, list))
        print("exceed max len", result)
        result, model_dict = model.predict(input="T " * 550, reserved_len=10)
        self.assertTrue(isinstance(result, list))
        print("exceed max len with reserved len:", result)
        self.assertTrue(result)

    def testOnebyoneWithReservedLen(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../../"))
        DATASET_DIR = os.path.join(ROOT_DIR, 'demo_data')

        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')

        for i in tfkit.model.onebyone.get_data_from_file(os.path.join(DATASET_DIR, 'generate.csv')):
            tasks, task, input, [target, negative_text] = i
            input = input.strip()
            tokenized_target = tokenizer.tokenize(" ".join(target))
            for j in range(1, len(tokenized_target) + 1):
                feature = tfkit.model.onebyone.get_feature_from_data(tokenizer, input=input,
                                                                     previous=tokenized_target[:j - 1],
                                                                     target=tokenized_target[:j],
                                                                     maxlen=20, reserved_len=0)[-1]
                target_start = feature['start']
                print(f"input: {len(feature['input'])}, {tokenizer.decode(feature['input'][:target_start])} ")
                print(f"type: {len(feature['type'])}, {feature['type'][:target_start]} ")
                print(f"mask: {len(feature['mask'])}, {feature['mask'][:target_start]} ")
                if tokenized_target is not None:
                    print(
                        f"target: {len(feature['target'])}, {tokenizer.convert_ids_to_tokens(feature['target'][target_start])} ")
