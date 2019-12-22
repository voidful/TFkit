# TFKit - Transformer Kit  
NLP library for different downstream tasks, built on huggingface project,   
for developing wide variety of nlp tasks.

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
    <td><b> tfkit.train.py </b></td>
    <td> Run training </td>
</tr>
<tr>
    <td><b> tfkit.eval.py </b></td>
    <td> Run evaluation </td>
</tr>
</table>

## Installation

TFKit requires Python 3.6.1 or later.   

### Installing via pip
```bash
pip install tfkit
```

## Running TFKit

Once you've installed TFKit, you can run train.py for training or eval.py for evaluation.  

```
$ train.py
Run training

arguments:
  --train       training data path       
  --valid       validation data path       
  --maxlen      maximum text length       
  --model       type of model         ['once', 'onebyone', 'classify', 'tagRow', 'tagCol']
  --config      pre-train model       bert-base-multilingual-cased

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
$ eval.py
Run evaluation on different benchmark
arguments:
  --model       model for evaluate       
  --valid       validation data path        
  --metric      metric for evaluate         ['em', 'nlg', 'classification']
  --config      pre-train model             bert-base-multilingual-cased

optional arguments:
  -h, --help    show this help message and exit
  --batch       batch size
  --topk        select top k result in classification task 
  --outprint    enable printing result in console
  --beamsearch  enable beamsearch for text generation task
```

## Dataset format
### once
csv file with 2 row - input, target  
each token separate by space  
no header needed   
Example:   
```
"i go to school by bus","我 坐 巴 士 上 學"
```
### onebyone
csv file with 2 row - input, target  
each token separate by space  
no header needed   
Example:   
```
"i go to school by bus","我 坐 巴 士 上 學"
```
### classify
csv file with header  
header - input,task1,task2...taskN  
if some task have multiple label, use / to separate each label - label1/label2/label3  
Example:   
```
SENTENCE,LABEL,Task2
"The prospective ultrasound findings were correlated with the final diagnoses , laparotomy findings , and pathology findings .",outcome/other,1
```
### tagRow
csv file with 2 row - input, target  
each token separate by space  
no header needed   
Example:   
```
"在 歐 洲 , 梵 語 的 學 術 研 究 , 由 德 國 學 者 陸 特 和 漢 斯 雷 頓 開 創 。 後 來 威 廉 · 瓊 斯 發 現 印 歐 語 系 , 也 要 歸 功 於 對 梵 語 的 研 究 。 此 外 , 梵 語 研 究 , 也 對 西 方 文 字 學 及 歷 史 語 言 學 的 發 展 , 貢 獻 不 少 。 1 7 8 6 年 2 月 2 日 , 亞 洲 協 會 在 加 爾 各 答 舉 行 。 [SEP] 陸 特 和 漢 斯 雷 頓 開 創 了 哪 一 地 區 對 梵 語 的 學 術 研 究 ?",O A A O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
```
### tagCol
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
