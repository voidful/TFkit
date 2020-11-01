import importlib
import os

import inquirer
import nlp2
import torch
from transformers import BertTokenizer, AutoTokenizer, AutoModel


def list_all_model(ignore_list=[]):
    dataset_dir = os.path.abspath(__file__ + "/../../") + '/model'
    return list(filter(
        lambda x: os.path.isdir(os.path.join(dataset_dir, x)) and '__pycache__' not in x and x not in ignore_list,
        os.listdir(dataset_dir)))


def load_predict_parameter(model, enable_arg_panel=False):
    """use inquirer panel to let user input model parameter or just use default value"""
    return nlp2.function_argument_panel(model.predict, disable_input_panel=(not enable_arg_panel), func_parent=model,
                                        ignore_empty=True)


def load_model_class(model_name):
    return importlib.import_module('.' + model_name, 'tfkit.model')


def load_trained_model(model_path, pretrained_config=None, model_type=None):
    """loading saved model"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torchpack = torch.load(model_path, map_location=device)

    print("===model info===")
    [print(key, ':', torchpack[key]) for key in torchpack.keys() if 'state_dict' not in key and 'models' not in key]
    print('==========')

    if 'tags' in torchpack and len(torchpack['tags']) > 1:
        if model_type is None:
            print("Pick which models to use in multi-task models")
            inquirer_res = inquirer.prompt(
                [inquirer.List('model_type', message="Select model", choices=torchpack['tags'])])
            model_type = inquirer_res['model_type']
        type_ind = torchpack['tags'].index(model_type)
    else:
        type_ind = 0

    print("loading saved model")

    # get all loading parameter
    maxlen = torchpack['maxlen']
    if pretrained_config is not None:
        config = pretrained_config
    else:
        config = torchpack['model_config'] if 'model_config' in torchpack else torchpack['bert']
    model_types = [torchpack['type']] if not isinstance(torchpack['type'], list) else torchpack['type']
    models_state = torchpack['models'] if 'models' in torchpack else [torchpack['model_state_dict']]
    type = model_types[type_ind]

    # load model
    if 'albert_chinese' in config:
        tokenizer = BertTokenizer.from_pretrained(config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config)
    pretrained = AutoModel.from_pretrained(config)

    if 'tag' in type:  # for old version model
        type = 'tag'
    elif 'onebyone' in type:
        type = 'onebyone'

    model_class = load_model_class(type)
    task_detail = {}
    if 'task-label' in torchpack:
        task_detail = torchpack['task-label']
    elif 'label' in torchpack:
        task_detail = {'label': torchpack['label']}

    model = model_class.Model(tokenizer=tokenizer, pretrained=pretrained, tasks_detail=task_detail,
                              maxlen=maxlen)
    model = model.to(device)
    model.load_state_dict(models_state[type_ind], strict=False)

    print("finish loading")
    return model, type, model_class
