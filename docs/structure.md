## Overview
Flow  
![Flow](https://raw.githubusercontent.com/voidful/TFkit/master/docs/img/flow.png)

Project directory:
```
.
├─ demo_data/                          # Example data for training and evaluation
├─ docs/                               # Documents
├─ tfkit/
│  ├─ model/                           # all of the models, subdir name will be model name 
│  │  ├─ model_name                    # - name will be dynamic import to tfkit-train
│  │  │  ├─ __init__.py                
│  │  │  ├─ dataloader.py              # - for data loading and preprocessing
│  │  │  └─ model.py                   # - model forward and prediction
│  │  └─ __init__.py                   
│  ├─ test/                            # project unit test
│  │  ├─ __init__.py                   
│  │  ├─ test_atrain.py                # - test tfkit-train
│  │  ├─ test_dataloader.py            # - test all model/*/dataloader.py
│  │  ├─ test_model.py                 # - test all model/*/model.py
│  │  ├─ test_package.py               # - test package import
│  │  ├─ test_utility_dataset.py       # - test utility/dataset.py
│  │  ├─ test_utility_eval_metric.py   # - test utility/eval_metric.py
│  │  ├─ test_utility_logger.py        # - test utility/logger.py
│  │  ├─ test_utility_loss.py          # - test utility/loss.py
│  │  ├─ test_utility_model_loader.py  # - test utility/model_loader.py
│  │  ├─ test_utility_tok.py           # - test utility/predictor.py
│  │  ├─ test_zeval.py                 # - test tfkit-eval
│  │  └─ test_zzdump.py                # - test tfkit-dump
│  ├─ utility/                         # project utility
│  │  ├─ __init__.py                   
│  │  ├─ dataset.py                    # - handle dataset loading
│  │  ├─ eval_metric.py                # - handle evaluation metric calculation
│  │  ├─ logger.py                     # - handle logging and printing
│  │  ├─ loss.py                       # - custom loss function
│  │  ├─ model_loader.py               # - handle model loading
│  │  ├─ predictor.py                  # - handle model prediction
│  │  └─ tok.py                        # - handle tokenization
│  ├─ __init__.py                      # package init
│  ├─ dump.py                          # tfkit-dump handler
│  ├─ eval.py                          # tfkit-eval handler
│  └─ train.py                         # tfkit-train handler
├─ Dockerfile                          # recommend docker file
├─ mkdocs.yml                          # document config
├─ README.md                           # project readme
├─ requirements.txt                    # package requirement
└─ setup.py                            # package setup
```