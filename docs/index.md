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

## Getting started

### Installing via pip
```bash
pip install tfkit
```

* You can use tfkit for model training and evaluation with `tfkit-train` and `tfkit-eval`.

### Running TFKit on the task you wanted

### First step - prepare your dataset
The key to combine different task together is to make different task with same data format.

**notice**  

* All data will be in csv format - tfkit will use **csv** for all task, normally it will have two columns, first columns is the input of models, the second column is the output of models.
* Plane text with no tokenization - there is no need to tokenize text before training, or do re-calculating for tokenization, tfkit will handle it for you.
* No header is needed.

For example, a sentiment classification dataset will be like:
```csv
how dare you,negative
```

!!! hint 
    For the detail and example format on different, you can check [here](tasks/) 

!!! hint 
    nlprep is a tool for data split/preprocessing/argumentation, it can help you to create ready to train data for tfkit, check [here](https://github.com/voidful/NLPrep)

### Second step - model training

Using `tfkit-train` for model training, you can use 

Before training a model, there is something you need to clarify:

- `--model` what is your model to handle this task? check [here](models/) to the detail of models.
- `--config` what pretrained model you want to use？ you can go [https://huggingface.co/models](https://huggingface.co/models) to search for available pretrained models.
- `--train` and `--test` training and testing dataset path, which is in csv format.
- `--savedir` model saving directory, default will be in '/checkpoints' folder
  
you can leave the rest to the default config, or use `tfkit-train -h` to more configuration.

An example about training a sentiment classifier:
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

#### Third step - model eval

Using `tfkit-eval` for model evaluation.   
- `--model` saved model's path.  
- `--metric` the evaluation metric eg: emf1, nlg(BLEU/ROUGE), clas(confusion matrix).  
- `--valid` validation data, also in csv format.  
- `--panel` a input panel for model specific parameter.  

for more configuration detail, you may use `tfkit-eval -h`.

After evaluate, It will print evaluate result in your console, and also generate three report for debugging.  
- `*_score.csv` overall score, it is the copy of the console result.  
- `*each_data_score.csv` score on each data, 3 column `predicted,targets,score`, ranked from the lowest to the highest.  
- `*predicted.csv` csv file include 3 column `input,predicted,targets`.  

!!! hint 
    nlp2go is a tool for demonstration, with CLI and Restful interface. check [here](https://github.com/voidful/nlp2go) 

### Example
#### Use distilbert to train NER Model
```bash
nlprep --dataset tag_clner  --outdir ./clner_row --util s2t
tfkit-train --batch 10 --epoch 3 --lr 5e-6 --train ./clner_row/train --test ./clner_row/test --maxlen 512 --task tag --config distilbert-base-multilingual-cased 
nlp2go --task ./checkpoints/3.pt  --cli     
```

#### Use Albert to train DRCD Model Model
```bash
nlprep --dataset qa_zh --outdir ./zhqa/   
tfkit-train --maxlen 512 --savedir ./drcd_qa_model/ --train ./zhqa/drcd-train --test ./zhqa/drcd-test --task qa --config voidful/albert_chinese_small  --cache
nlp2go --task ./drcd_qa_model/3.pt --cli 
```

#### Use Albert to train both DRCD Model and NER Model
```bash
nlprep --dataset tag_clner  --outdir ./clner_row --util s2t
nlprep --dataset qa_zh --outdir ./zhqa/ 
tfkit-train --maxlen 300 --savedir ./mt-qaner --train ./clner_row/train ./zhqa/drcd-train --test ./clner_row/test ./zhqa/drcd-test --task tag qa --config voidful/albert_chinese_small
nlp2go --task ./mt-qaner/3.pt --cli 
```

**You can also try tfkit in Google Colab: [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg "tfkit")](https://colab.research.google.com/drive/1hqaTKxd3VtX2XkvjiO0FMtY-rTZX30MJ?usp=sharing)**

## Contributing
Thanks for your interest.There are many ways to contribute to this project. Get started [here](https://github.com/voidful/tfkit/blob/master/CONTRIBUTING.md).

## License 
![PyPI - License](https://img.shields.io/github/license/voidful/tfkit)

* [License](https://github.com/voidful/tfkit/blob/master/LICENSE)

## Icons reference
Icons modify from <a href="http://www.freepik.com/" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a>      
Icons modify from <a href="https://www.flaticon.com/authors/nikita-golubev" title="Nikita Golubev">Nikita Golubev</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a>      
