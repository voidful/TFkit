<p  align="center">
    <br>
    <img src="https://raw.githubusercontent.com/voidful/TFkit/master/doc/img/tfkit.png" width="400"/>
    <br>
<p>
<br/>
<p align="center">
    <a href="https://github.com/voidful/tfkit/releases/">
        <img alt="Release" src="https://img.shields.io/github/v/release/voidful/tfkit">
    </a>
    <a href="https://pypi.org/project/tfkit/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/tfkit">
    </a>
    <a href="https://github.com/voidful/tfkit">
        <img alt="Download" src="https://img.shields.io/pypi/dm/tfkit">
    </a>
    <a href="https://github.com/voidful/tfkit">
        <img alt="Build" src="https://img.shields.io/github/workflow/status/voidful/tfkit/Python package">
    </a>
    <a href="https://github.com/voidful/tfkit">
        <img alt="Last Commit" src="https://img.shields.io/github/last-commit/voidful/tfkit">
    </a>
</p>
<br/>

## 功能
- 支持 Bert/GPT/GPT2/XLM/XLNet/RoBERTa/CTRL/ALBert 各種模型，隨心換 [全部支持的模型](https://huggingface.co/models)   
- 模組換資料載入部分，也針對此做了資料下載預處理套件 [NLPrep](https://github.com/voidful/NLPrep)   
- 易於修改，可以加入自己的fine-tune架構
- 加入更多Loss Function: FocalLoss/ FocalBCELoss/ NegativeCrossEntropyLoss/ SmoothCrossEntropyLoss  
- 支持不同的指標驗證 - EM / F1 / BLEU / METEOR / ROUGE / CIDEr / Classification Report / ...
- 支持不同的解碼方式 - beamsearch/greedy/sampling
- 可以在所有模型中搭配多任務 
- 多任務 多分類 多標籤 分類器
- 文本生成
- 序列標注模型
- 閱讀理解模型


## Package Overview

<table>
<tr>
    <td><b> tfkit </b></td>
    <td> NLP library for different downstream tasks, built on huggingface project </td>
</tr>
<tr>
    <td><b> tfkit.classifier </b></td>
    <td> 多類別 多任務 多標籤 分類器</td>
</tr>
<tr>
    <td><b> tfkit.gen_once </b></td>
    <td> 基於MASKLM的文本生成,一次輸出結果 </td>
</tr>
<tr>
    <td><b> tfkit.gen_onebyone </b></td>
    <td> 基於MASKLM的文本生成,語言模型的方式輸出結果 </td>
</tr>
<tr>
    <td><b> tfkit.tag </b></td>
    <td> 序列標注模型 </td>
</tr>
<tr>
    <td><b> tfkit.qa </b></td>
    <td> 閱讀理解模型 </td>
</tr>
<tr>
    <td><b> tfkit.train.py </b></td>
    <td> 模型訓練 </td>
</tr>
<tr>
    <td><b> tfkit.eval.py </b></td>
    <td> 模型驗證 </td>
</tr>
</table>

## 安裝

TFKit 需要 **Python 3.6** 以上版本.   

### pip安裝
```bash
pip install tfkit
```

## Running TFKit

安裝好以後，可以用 `tfkit-train` 或者 `tfkit-eval` 驗證

```
$ tfkit-train
Run training

arguments:
  --train       訓練資料位置       
  --valid       驗證資料位置       
  --maxlen      文本最大長度       
  --model       模型的類型         ['once', 'onebyone', 'classify', 'tagRow', 'tagCol','qa']
  --config      預訓練             bert-base-multilingual-cased... etc (you can find one on https://huggingface.co/models)

optional arguments:
  -h, --help    幫助資料
  --resume      回復之前的訓練，模型位置
  --savedir     模型儲存資料夾
  --worker      dataloader中worker數量
  --batch       Batch size
  --lr          learning rate
  --epoch       epoch數
  --tensorboard 是否開啟tensorboard
  --cache       是否cache訓練資料
```

```
$ tfkit-eval
Run evaluation on different benchmark
arguments:
  --model       驗證資料位置       
  --valid       驗證模型位置        
  --metric      驗證的指標           ['em', 'nlg', 'classification']

optional arguments:
  -h, --help    幫助資料
  --batch       Batch size
  --outprint    印出預測結果
  --beamsearch  是否開啟beamsearch(只有文本生成有效)
```

## 資料格式
### once
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/generate.csv)   
兩行的CSV檔案 - 輸入, 目標  
每個字之間用空格隔開  
不需要header   
例子:   
```
"i go to school by bus","我 坐 巴 士 上 學"
```
### onebyone
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/generate.csv)   
兩行的CSV檔案 - 輸入, 目標  
每個字之間用空格隔開  
不需要header   
例子:   
```
"i go to school by bus","我 坐 巴 士 上 學"
```
### qa
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/qa.csv)   
兩行的CSV檔案 - 輸入, 答案開始位置, 答案結束位置  
每個字之間用空格隔開  
不需要header   
例子: 
```
"在 歐 洲 , 梵 語 的 學 術 研 究 , 由 德 國 學 者 陸 特 和 漢 斯 雷 頓 開 創 。 後 來 威 廉 · 瓊 斯 發 現 印 歐 語 系 , 也 要 歸 功 於 對 梵 語 的 研 究 。 此 外 , 梵 語 研 究 , 也 對 西 方 文 字 學 及 歷 史 語 言 學 的 發 展 , 貢 獻 不 少 。 1 7 8 6 年 2 月 2 日 , 亞 洲 協 會 在 加 爾 各 答 舉 行 。 會 中 , 威 廉 · 瓊 斯 發 表 了 下 面 這 段 著 名 的 言 論 : 「 梵 語 儘 管 非 常 古 老 , 構 造 卻 精 妙 絕 倫 : 比 希 臘 語 還 完 美 , 比 拉 丁 語 還 豐 富 , 精 緻 之 處 同 時 勝 過 此 兩 者 , 但 在 動 詞 詞 根 和 語 法 形 式 上 , 又 跟 此 兩 者 無 比 相 似 , 不 可 能 是 巧 合 的 結 果 。 這 三 種 語 言 太 相 似 了 , 使 任 何 同 時 稽 考 三 者 的 語 文 學 家 都 不 得 不 相 信 三 者 同 出 一 源 , 出 自 一 種 可 能 已 經 消 逝 的 語 言 。 基 於 相 似 的 原 因 , 儘 管 缺 少 同 樣 有 力 的 證 據 , 我 們 可 以 推 想 哥 德 語 和 凱 爾 特 語 , 雖 然 混 入 了 迥 然 不 同 的 語 彙 , 也 與 梵 語 有 著 相 同 的 起 源 ; 而 古 波 斯 語 可 能 也 是 這 一 語 系 的 子 裔 。 」 [Question] 印 歐 語 系 因 為 哪 一 門 語 言 而 被 發 現 ?",47,49
```

### classify
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/classification.csv)   
有Header的CSV檔案   
Header - 輸入,任務1,任務2...任務N  
Row    - 輸入, 任務1目標,任務2目標...任務N目標  
如果是多標籤的, 用 `/` 隔開不同的標籤 - 標籤1/標籤2/標籤3  
Example:   
```
SENTENCE,LABEL,Task2
"The prospective ultrasound findings were correlated with the final diagnoses , laparotomy findings , and pathology findings .",outcome/other,1
```
### tagRow
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/tag_row.csv)   
兩行的CSV檔案 - 輸入, 目標  
每個字之間用空格隔開  
不需要header   
例子:  
```
"在 歐 洲 , 梵 語 的 學 術 研 究 , 由 德 國 學 者 陸 特 和 漢 斯 雷 頓 開 創 。 後 來 威 廉 · 瓊 斯 發 現 印 歐 語 系 , 也 要 歸 功 於 對 梵 語 的 研 究 。 此 外 , 梵 語 研 究 , 也 對 西 方 文 字 學 及 歷 史 語 言 學 的 發 展 , 貢 獻 不 少 。 1 7 8 6 年 2 月 2 日 , 亞 洲 協 會 在 加 爾 各 答 舉 行 。 [SEP] 陸 特 和 漢 斯 雷 頓 開 創 了 哪 一 地 區 對 梵 語 的 學 術 研 究 ?",O A A O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
```
### tagCol
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/tag_col.csv)   
兩行的CSV檔案 - 輸入, 目標  
每個字之間用空格隔開  
不需要header   
例子:  
```
別 O
只 O
能 R
想 O
自 O
己 O
， O
想 M
你 M
周 O
圍 O
的 O
人 O
。 O
```
