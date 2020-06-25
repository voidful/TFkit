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

Read this in other languages: [中文](https://github.com/voidful/TFkit/blob/master/README.zh.md).

## Feature
- [Model list](https://huggingface.co/models): support Bert/GPT/GPT2/XLM/XLNet/RoBERTa/CTRL/ALBert 
- [NLPrep](https://github.com/voidful/NLPrep): create a data preprocessing library on many task   
- [nlp2go](https://github.com/voidful/nlp2go): create model hosting library for demo  
- multi-class multi-task multi-label classifier  
- Multi-Task on ALL model
- word/sentence level text generation  
- support greedy, beam-search & nucleus decoding 
- token tagging
- special loss function for handling different cases: FocalLoss/ FocalBCELoss/ NegativeCrossEntropyLoss/ SmoothCrossEntropyLoss  
- eval on different benchmark - EM / F1 / BLEU / METEOR / ROUGE / CIDEr / Classification Report / ...
- modularize data loading
- easy to modify


## DEMO

### albert multi-dataset QA model
dataset：
```bash
nlprep --dataset multiqa --task qa --outdir ./multiqa/   
tfkit-train --maxlen 512 --savedir ./multiqa_qa_model/ --train ./multiqa/train --valid ./multiqa/valid --model qa --config voidful/albert_chinese_small  --cache
nlp2go --model ./multiqa_qa_model/3.pt --cli 
```

### Distilbert NER model
three line code train and host NER model [Colab](https://colab.research.google.com/drive/1x5DLBQ6ufRUfi1PPmHcXtYqTl_9krRWz)
```bash
nlprep --dataset clner --task tagRow --outdir ./clner_row --util s2t 
tfkit-train --batch 10 --epoch 3 --lr 5e-6 --train ./clner_row/train --valid ./clner_row/test --maxlen 512 --model tagRow --config distilbert-base-multilingual-cased 
nlp2go --model ./checkpoints/3.pt  --cli     
```

### albert QA model
three line code train and host QA model [Colab](https://colab.research.google.com/drive/1hqaTKxd3VtX2XkvjiO0FMtY-rTZX30MJ)
```bash
nlprep --dataset zhqa --task qa --outdir ./zhqa/   
tfkit-train --maxlen 512 --savedir ./drcd_qa_model/ --train ./zhqa/drcd-train --valid ./zhqa/drcd-test --model qa --config voidful/albert_chinese_small  --cache
nlp2go --model ./drcd_qa_model/3.pt --cli 
```
### multi-task 
```bash
nlprep --dataset clner --task tagRow --outdir ./clner_row --util s2t 
nlprep --dataset zhqa --task qa --outdir ./zhqa/ 
tfkit-train --maxlen 300 --savedir ./mt-qaner --train ./clner_row/train ./zhqa/drcd-train --valid ./clner_row/test ./zhqa/drcd-test --model tagRow qa --config voidful/albert_chinese_small
nlp2go --model ./mt-qaner/3.pt --cli 
```

## Benchmark
#### DRCD Test
| model | EM | F1 | 
| :----:|:----: |:----: |
| <a href="https://huggingface.co/voidful/albert_chinese_small">albert-small</a>	| 74.45% | 86.08% |
| <a href="https://huggingface.co/hfl/chinese-electra-small-discriminator">electra-small</a>	| 76.64% | 87.49% |
| <a href="https://huggingface.co/voidful/albert_chinese_base">albert-base</a>	| 80.17% | 89.87% |

#### DRCD Dev
| model | EM | F1 | 
| :----:|:----: |:----: |
| <a href="https://huggingface.co/voidful/albert_chinese_small">albert-small</a>	| 73.70% | 85.33% |
| <a href="https://huggingface.co/hfl/chinese-electra-small-discriminator">electra-small</a>	| 77.61% | 87.33% |
| <a href="https://huggingface.co/voidful/albert_chinese_base">albert-base</a>	| 80.52% | 89.92% |


## Flow Overview
![nlp kit flow](https://raw.githubusercontent.com/voidful/TFkit/master/img/flow.png)

## Package Overview

<table>
<tr>
    <td><b> tfkit </b></td>
    <td> NLP library for different downstream tasks, built on huggingface project </td>
</tr>
<tr>
    <td><b> tfkit.classifier </b></td>
    <td> multi-class multi-task multi-label classifier</td>
</tr>
<tr>
    <td><b> tfkit.gen_once </b></td>
    <td> text generation in one time built on masklm model</td>
</tr>
<tr>
    <td><b> tfkit.gen_onebyone </b></td>
    <td> text generation in one word by one word built on masklm model</td>
</tr>
<tr>
    <td><b> tfkit.tag </b></td>
    <td> token tagging model </td>
</tr>
<tr>
    <td><b> tfkit.qa </b></td>
    <td> qa model predicting start and end position </td>
</tr>
<tr>
    <td><b> tfkit.train.py </b></td>
    <td> Run training </td>
</tr>
<tr>
    <td><b> tfkit.eval.py </b></td>
    <td> Run evaluation </td>
</tr>
</table>

## Installation

TFKit requires **Python 3.6** or later.   

### Installing via pip
```bash
pip install tfkit
```

## Running TFKit

Once you've installed TFKit, you can run train.py for training or eval.py for evaluation.  

```
$ tfkit-train
Run training

arguments:
  --train       training data path       
  --valid       validation data path       
  --maxlen      maximum text length       
  --model       type of model         ['once', 'onebyone', 'classify', 'tagRow', 'tagCol','qa']
  --config      pre-train model       bert-base-multilingual-cased... etc (you can find one on https://huggingface.co/models)

optional arguments:
  -h, --help    show this help message and exit
  --resume      resume from previous training
  --savedir     dir for model saving
  --worker      number of worker
  --batch       batch size
  --lr          learning rate
  --epoch       epoch rate
  --tensorboard enable tensorboard
  --cache       enable data caching
```

```
$ tfkit-eval
Run evaluation on different benchmark
arguments:
  --model       model for evaluate       
  --valid       validation data path        
  --metric      metric for evaluate         ['emf1', 'nlg', 'classification']Ω

optional arguments:
  -h, --help    show this help message and exit
  --batch       batch size
  --outprint    enable printing result in console
  --outfile     enable writing prediction result to file
  --beamsearch  enable beamsearch for text generation task
```

## Dataset format
### once
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/generate.csv)   
csv file with 2 row - input, target  
each token separate by space  
no header needed   
Example:   
```
"i go to school by bus","我 坐 巴 士 上 學"
```
### onebyone
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/generate.csv)   
csv file with 2 row - input, target  
each token separate by space  
no header needed   
Example:   
```
"i go to school by bus","我 坐 巴 士 上 學"
```
### qa
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/qa.csv)   
csv file with 3 row - input, start_pos, end_pos  
each token separate by space  
no header needed   
Example:   
```
"在 歐 洲 , 梵 語 的 學 術 研 究 , 由 德 國 學 者 陸 特 和 漢 斯 雷 頓 開 創 。 後 來 威 廉 · 瓊 斯 發 現 印 歐 語 系 , 也 要 歸 功 於 對 梵 語 的 研 究 。 此 外 , 梵 語 研 究 , 也 對 西 方 文 字 學 及 歷 史 語 言 學 的 發 展 , 貢 獻 不 少 。 1 7 8 6 年 2 月 2 日 , 亞 洲 協 會 在 加 爾 各 答 舉 行 。 會 中 , 威 廉 · 瓊 斯 發 表 了 下 面 這 段 著 名 的 言 論 : 「 梵 語 儘 管 非 常 古 老 , 構 造 卻 精 妙 絕 倫 : 比 希 臘 語 還 完 美 , 比 拉 丁 語 還 豐 富 , 精 緻 之 處 同 時 勝 過 此 兩 者 , 但 在 動 詞 詞 根 和 語 法 形 式 上 , 又 跟 此 兩 者 無 比 相 似 , 不 可 能 是 巧 合 的 結 果 。 這 三 種 語 言 太 相 似 了 , 使 任 何 同 時 稽 考 三 者 的 語 文 學 家 都 不 得 不 相 信 三 者 同 出 一 源 , 出 自 一 種 可 能 已 經 消 逝 的 語 言 。 基 於 相 似 的 原 因 , 儘 管 缺 少 同 樣 有 力 的 證 據 , 我 們 可 以 推 想 哥 德 語 和 凱 爾 特 語 , 雖 然 混 入 了 迥 然 不 同 的 語 彙 , 也 與 梵 語 有 著 相 同 的 起 源 ; 而 古 波 斯 語 可 能 也 是 這 一 語 系 的 子 裔 。 」 [Question] 印 歐 語 系 因 為 哪 一 門 語 言 而 被 發 現 ?",47,49
```

### classify
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/classification.csv)   
csv file with header  
header - input,task1,task2...taskN  
if some task have multiple label, use / to separate each label - label1/label2/label3  
Example:   
```
SENTENCE,LABEL,Task2
"The prospective ultrasound findings were correlated with the final diagnoses , laparotomy findings , and pathology findings .",outcome/other,1
```
### tagRow
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/tag_row.csv)   
csv file with 2 row - input, target  
each token separate by space  
no header needed   
Example:   
```
"在 歐 洲 , 梵 語 的 學 術 研 究 , 由 德 國 學 者 陸 特 和 漢 斯 雷 頓 開 創 。 後 來 威 廉 · 瓊 斯 發 現 印 歐 語 系 , 也 要 歸 功 於 對 梵 語 的 研 究 。 此 外 , 梵 語 研 究 , 也 對 西 方 文 字 學 及 歷 史 語 言 學 的 發 展 , 貢 獻 不 少 。 1 7 8 6 年 2 月 2 日 , 亞 洲 協 會 在 加 爾 各 答 舉 行 。 [SEP] 陸 特 和 漢 斯 雷 頓 開 創 了 哪 一 地 區 對 梵 語 的 學 術 研 究 ?",O A A O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
```
### tagCol
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/tag_col.csv)   
csv file with 2 row - input, target  
each token separate by space  
no header needed  
Example:      
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
