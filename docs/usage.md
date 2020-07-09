#Usage
## Overview
Flow  
![Flow](https://raw.githubusercontent.com/voidful/TFkit/master/docs/img/flow.png)

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
  --batch               batch size, default 20
  --lr LR [LR ...]      learning rate, default 5e-5
  --epoch               epoch, default 10
  --maxlen              max tokenized sequence length, default 368
  --lossdrop            loss dropping for text generation
  --add_tokens          auto add top x percent UNK token to word table, default 0, range 0-100
  --tag     [TAG ...]   tag to identity task in multi-task
  --seed                random seed, default 609
  --worker              number of worker on pre-processing, default 8
  --grad_accum          gradient accumulation, default 1
  --tensorboard         Turn on tensorboard graphing
  --resume              resume training
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
  --config              pre-trained model path after add token
  --print               print each pair of evaluate data
  --enable_arg_panel    enable panel to input argument

```



## Example

### Use distilbert to train NER Model
```bash
nlprep --dataset tag_clner  --outdir ./clner_row --util s2t
tfkit-train --batch 10 --epoch 3 --lr 5e-6 --train ./clner_row/train --test ./clner_row/test --maxlen 512 --model tagRow --config distilbert-base-multilingual-cased 
nlp2go --model ./checkpoints/3.pt  --cli     
```

### Use Albert to train DRCD QA Model
```bash
nlprep --dataset qa_zh --outdir ./zhqa/   
tfkit-train --maxlen 512 --savedir ./drcd_qa_model/ --train ./zhqa/drcd-train --test ./zhqa/drcd-test --model qa --config voidful/albert_chinese_small  --cache
nlp2go --model ./drcd_qa_model/3.pt --cli 
```

### Use Albert to train both DRCD QA and NER Model
```bash
nlprep --dataset tag_clner  --outdir ./clner_row --util s2t
nlprep --dataset qa_zh --outdir ./zhqa/ 
tfkit-train --maxlen 300 --savedir ./mt-qaner --train ./clner_row/train ./zhqa/drcd-train --test ./clner_row/test ./zhqa/drcd-test --model tagRow qa --config voidful/albert_chinese_small
nlp2go --model ./mt-qaner/3.pt --cli 
```
