import copy
import importlib
import os

import inquirer
import nlp2
import torch
from transformers import BertTokenizer, AutoTokenizer, AutoModel
import copy


def list_all_model(ignore_list=[]):
    dataset_dir = os.path.abspath(__file__ + "/../../") + '/model'
    return list(filter(
        lambda x: os.path.isdir(os.path.join(dataset_dir, x)) and '__pycache__' not in x and x not in ignore_list,
        os.listdir(dataset_dir)))


def load_predict_parameter(model, model_arg={}, enable_arg_panel=False):
    """use inquirer panel to let user input model parameter or just use default value"""
    return nlp2.function_argument_panel(model.predict, model_arg, disable_input_panel=(not enable_arg_panel),
                                        func_parent=model,
                                        ignore_empty=True)


def load_model_class(model_name):
    return importlib.import_module('.' + model_name, 'tfkit.model')


def load_pretrained_model(pretrained_config, model_type):
    pretrained = AutoModel.from_pretrained(pretrained_config)
    if 'clm' in model_type:
        pretrained.config.is_decoder = True
    return pretrained


def load_pretrained_tokenizer(pretrained_config):
    if 'albert_chinese' in pretrained_config:
        tokenizer = BertTokenizer.from_pretrained(pretrained_config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_config)
    return tokenizer


def add_tokens_to_pretrain(pretrained, tokenizer, add_tokens):
    origin_vocab_size = tokenizer.vocab_size
    print("===ADD TOKEN===")
    num_added_toks = tokenizer.add_tokens(add_tokens)
    print('We have added', num_added_toks, 'tokens')
    pretrained.resize_token_embeddings(len(tokenizer))
    input_embedding = pretrained.get_input_embeddings()
    state_dict_weight = input_embedding.state_dict()['weight']
    state_dict_weight[origin_vocab_size:len(tokenizer)] = copy.copy(
        state_dict_weight[100:100 + num_added_toks])
    pretrained.set_input_embeddings(input_embedding)
    print("===============")
    return pretrained, tokenizer


def load_trained_model(model_path, pretrained_config=None, tag=None):
    """loading saved model"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torchpack = torch.load(model_path, map_location=device)

    model_info = {key: torchpack[key] for key in torchpack.keys() if 'state_dict' not in key and 'models' not in key}
    print("===model info===")
    [print(k, v[:10], "...") if isinstance(v, list) and len(v) > 10 else print(k, v) for k, v in model_info.items()]
    print('===============')

    if 'tags' in torchpack and len(torchpack['tags']) > 1:
        if tag is None:
            print("Pick which models to use in multi-task models")
            inquirer_res = inquirer.prompt(
                [inquirer.List('tag', message="Select model", choices=torchpack['tags'])])
            tag = inquirer_res['tag']
        type_ind = torchpack['tags'].index(tag)
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
    add_tokens = torchpack['add_tokens'] if 'add_tokens' in torchpack else None
    # load model
    if 'albert_chinese' in config:
        tokenizer = BertTokenizer.from_pretrained(config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config)
    pretrained = AutoModel.from_pretrained(config)

    pretrained, tokenizer = add_tokens_to_pretrain(pretrained, tokenizer, add_tokens)

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
    return model, type, model_class, model_info


def save_model(models, input_arg, models_tag, epoch, fname, logger, add_tokens=None):
    save_model = {
        'models': [m.state_dict() for m in models],
        'model_config': input_arg.get('config'),
        'add_tokens': add_tokens,
        'tags': models_tag,
        'type': input_arg.get('model'),
        'maxlen': input_arg.get('maxlen'),
        'epoch': epoch
    }

    for ind, m in enumerate(input_arg.get('model')):
        if 'tag' in m:
            save_model['label'] = models[ind].labels
        if "clas" in m:
            save_model['task-label'] = models[ind].tasks_detail

    torch.save(save_model, f"{fname}.pt")
    logger.write_log(f"weights were saved to {fname}.pt")
