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
    <a href="https://www.codefactor.io/repository/github/voidful/tfkit/overview/master">
        <img src="https://www.codefactor.io/repository/github/voidful/tfkit/badge/master" alt="CodeFactor" />
    </a>
    <a href="https://github.com/voidful/tfkit">
        <img src="https://visitor-badge.glitch.me/badge?page_id=voidful.tfkit" alt="Visitor" />
    </a>
    <a href="https://codecov.io/gh/voidful/TFkit">
      <img src="https://codecov.io/gh/voidful/TFkit/branch/master/graph/badge.svg" />
    </a>
</p>

## What is it
TFKit is a tool kit mainly for language generation.  
It leverages the use of transformers on many tasks with different models in this all-in-one framework.   
All you need is a little change of config.  

## Task Supported
With transformer models - BERT/ALBERT/T5/BART......  
|  |  |
|-|-|
| Text Generation | :memo: seq2seq language model |
| Text Generation | :pen: causal language model |
| Text Generation | :printer: once generation model / once generation model with ctc loss |
| Text Generation | :pencil: onebyone generation model |

# Getting Started
Learn more from the [document](https://voidful.github.io/TFkit/).  

## How To Use

### Step 0: Install
Simple installation from PyPI
```bash
pip install git+https://github.com/voidful/TFkit.git@refactor-dataset
```

### Step 1: Prepare dataset in csv format
[Task format](https://voidful.tech/TFkit/tasks/)
``` 
input, target
```

### Step 2: Train model
```bash
tfkit-train \
--task clas \
--config xlm-roberta-base \
--train training_data.csv \
--test testing_data.csv \
--lr 4e-5 \
--maxlen 384 \
--epoch 10 \
--savedir roberta_sentiment_classificer
```

### Step 3: Evaluate
```bash
tfkit-eval \
--task roberta_sentiment_classificer/1.pt \
--metric clas \
--valid testing_data.csv
```

## Advanced features
<details>
  <summary>Multi-task training </summary>

  ```bash
  tfkit-train \
    --task clas clas \
    --config xlm-roberta-base \
    --train training_data_taskA.csv training_data_taskB.csv \
    --test testing_data_taskA.csv testing_data_taskB.csv \
    --lr 4e-5 \
    --maxlen 384 \
    --epoch 10 \
    --savedir roberta_sentiment_classificer_multi_task
  ```
</details>

## Not maintained task
Due to time constraints, the following tasks are temporarily not supported
|  |  |
|-|-|
| Classification  | :label: multi-class and multi-label classification |
| Question Answering  | :page_with_curl: extractive qa |
| Question Answering  | :radio_button: multiple-choice qa |
| Tagging  | :eye_speech_bubble: sequence level tagging / sequence level with crf  |
| Self-supervise Learning | :diving_mask: mask language model |

## Supplement
- [transformers models list](https://huggingface.co/models): you can find any pretrained models here   
- [nlprep](https://github.com/voidful/NLPrep): download and preprocessing data in one line     
- [nlp2go](https://github.com/voidful/nlp2go): create demo api as quickly as possible.


## Contributing
Thanks for your interest.There are many ways to contribute to this project. Get started [here](https://github.com/voidful/tfkit/blob/master/CONTRIBUTING.md).

## License ![PyPI - License](https://img.shields.io/github/license/voidful/tfkit)

* [License](https://github.com/voidful/tfkit/blob/master/LICENSE)

## Icons reference
Icons modify from <a href="http://www.freepik.com/" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a>      
Icons modify from <a href="https://www.flaticon.com/authors/nikita-golubev" title="Nikita Golubev">Nikita Golubev</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a>      
