<p  align="center">
    <br>
    <img src="https://raw.githubusercontent.com/voidful/TFkit/master/docs/img/tfkit.png" width="300"/>
    <br>
</p>
<br/>
<p align="center">
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


TFKit lets everyone make use of  transformer architecture on many tasks and models in small change of config.   
At the same time, it can do multi-task multi-model learning, and can introduce its own data sets and tasks through simple modifications.    

## Feature
- One-click replacement of different pre-trained models
- Support multi-model and multi-task
- Classifier with multiple labels and multiple classifications
- Unify input formats for different tasks
- Separation of data reading and model architecture
- Support various loss function and indicators


## Supplement
- [Model list](https://huggingface.co/models): Support Bert/GPT/GPT2/XLM/XLNet/RoBERTa/CTRL/ALBert/...   
- [NLPrep](https://github.com/voidful/NLPrep): download and preprocessing data in one line     
- [nlp2go](https://github.com/voidful/nlp2go): create demo api as quickly as possible.


## Quick Start

### Installing via pip
```bash
pip install tfkit
```
### Running TFKit to train a ner model
install nlprep and nlp2go      
```bash
pip install nlprep  nlp2go -U
```
download dataset using nlprep
```bash
nlprep --dataset tag_clner  --outdir ./clner_row --util s2t
```
train model with albert
```bash
tfkit-train --batch 20 \
--epoch 5 \
--lr 5e-5 \
--train ./clner_row/train.csv \
--test ./clner_row/test.csv \
--maxlen 512 \
--model tagRow \
--savedir ./albert_ner \
--config voidful/albert_chinese_small
```
eval model
```bash
tfkit-eval --model ./albert_ner/3.pt --valid ./clner_row/validation.csv --metric clas
```     
result
```text
Task : default report 
TASK:  default 0
                precision    recall  f1-score   support

    B_Abstract       0.00      0.00      0.00         1
    B_Location       1.00      1.00      1.00         1
      B_Metric       1.00      1.00      1.00         1
B_Organization       0.00      0.00      0.00         1
      B_Person       1.00      1.00      1.00         1
    B_Physical       0.00      0.00      0.00         1
       B_Thing       1.00      1.00      1.00         1
        B_Time       1.00      1.00      1.00         1
    I_Abstract       1.00      1.00      1.00         1
    I_Location       1.00      1.00      1.00         1
      I_Metric       1.00      1.00      1.00         1
I_Organization       0.00      0.00      0.00         1
      I_Person       1.00      1.00      1.00         1
    I_Physical       0.00      0.00      0.00         1
       I_Thing       1.00      1.00      1.00         1
        I_Time       1.00      1.00      1.00         1
             O       1.00      1.00      1.00         1

     micro avg       1.00      0.71      0.83        17
     macro avg       0.71      0.71      0.71        17
  weighted avg       0.71      0.71      0.71        17
   samples avg       1.00      0.71      0.83        17
```    
host prediction service
```bash
nlp2go --model ./albert_ner/3.pt --api_path ner
```

**You can also try tfkit in Google Colab: [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg "tfkit")](https://colab.research.google.com/drive/1hqaTKxd3VtX2XkvjiO0FMtY-rTZX30MJ?usp=sharing)**


## Overview
### Train
```
$ tfkit-train
Run training

arguments:
  --train TRAIN [TRAIN ...]     train dataset path
  --test TEST [TEST ...]        test dataset path
  --config CONFIG               distilbert-base-multilingual-cased/bert-base-multilingual-cased/voidful/albert_chinese_small
  --model {once,twice,onebyone,clas,tagRow,tagCol,qa,onebyone-neg,onebyone-pos,onebyone-both} [{once,twice,onebyone,clas,tagRow,tagCol,qa,onebyone-neg,onebyone-pos,onebyone-both} ...]
                                model task
  --savedir SAVEDIR     model saving dir, default /checkpoints
optional arguments:
  -h, --help            show this help message and exit
  --batch BATCH         batch size, default 20
  --lr LR [LR ...]      learning rate, default 5e-5
  --epoch EPOCH         epoch, default 10
  --maxlen MAXLEN       max tokenized sequence length, default 368
  --lossdrop            loss dropping for text generation
  --tag TAG [TAG ...]   tag to identity task in multi-task
  --seed SEED           random seed, default 609
  --worker WORKER       number of worker on pre-processing, default 8
  --grad_accum          gradient accumulation, default 1
  --tensorboard         Turn on tensorboard graphing
  --resume RESUME       resume training
  --cache               cache training data

```
### Eval  
```
$ tfkit-eval
Run evaluation on different benchmark
arguments:
  --model MODEL             model path
  --metric {emf1,nlg,clas}  evaluate metric
  --valid VALID             evaluate data path

optional arguments:
  -h, --help            show this help message and exit
  --print               print each pair of evaluate data
  --enable_arg_panel    enable panel to input argument

```

## Contributing
Thanks for your interest.There are many ways to contribute to this project. Get started [here](https://github.com/voidful/tfkit/blob/master/CONTRIBUTING.md).

## License ![PyPI - License](https://img.shields.io/github/license/voidful/tfkit)

* [License](https://github.com/voidful/tfkit/blob/master/LICENSE)

## Icons reference
Icons modify from <a href="http://www.freepik.com/" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a>      
Icons modify from <a href="https://www.flaticon.com/authors/nikita-golubev" title="Nikita Golubev">Nikita Golubev</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a>      
