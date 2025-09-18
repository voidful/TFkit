import copy
import importlib
import os
from typing import Iterable, List, Sequence

import inquirer
import nlp2
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from tfkit.utility.constants import ENV_TRUST_REMOTE_CODE


def list_all_model(ignore_list=[]):
    dataset_dir = os.path.abspath(__file__ + "/../../") + '/task'
    return list(filter(
        lambda x: os.path.isdir(os.path.join(dataset_dir, x)) and '__pycache__' not in x and x not in ignore_list,
        os.listdir(dataset_dir)))


def load_predict_parameter(model, model_arg={}, enable_arg_panel=False):
    """use inquirer panel to let user input task parameter or just use default value"""
    return nlp2.function_argument_panel(model.predictor.wrap_input, model_arg,
                                        disable_input_panel=(not enable_arg_panel),
                                        func_parent=model,
                                        ignore_empty=True)


def load_model_class(model_name):
    return importlib.import_module('.' + model_name, 'tfkit.task')


def _should_trust_remote_code(explicit: bool | None = None) -> bool:
    """Determine whether remote code should be trusted when loading models."""

    if explicit is not None:
        return explicit

    env_value = os.getenv(ENV_TRUST_REMOTE_CODE, "1").strip().lower()
    return env_value in {"1", "true", "yes", "on"}


def _normalize_model_types(model_type: Iterable[str] | str | None) -> Sequence[str]:
    """Normalize model type input into a sequence for easier processing."""

    if model_type is None:
        return ()
    if isinstance(model_type, str):
        return (model_type,)
    return tuple(model_type)


def _requires_seq2seq_model(tasks: Sequence[str], config) -> bool:
    """Determine if a sequence-to-sequence model should be used."""

    if getattr(config, "is_encoder_decoder", False):
        return True
    return any(task in {"seq2seq"} for task in tasks)


def _requires_causal_model(tasks: Sequence[str], config) -> bool:
    """Determine if a causal language model should be used."""

    if any(task in {"clm"} for task in tasks):
        return True
    architectures = getattr(config, "architectures", []) or []
    return any("ForCausalLM" in arch for arch in architectures)


def load_pretrained_model(pretrained_config, model_type, trust_remote_code: bool | None = None):
    """Load a pretrained model compatible with the requested tasks."""

    trust_remote = _should_trust_remote_code(trust_remote_code)
    tasks = _normalize_model_types(model_type)
    config = AutoConfig.from_pretrained(pretrained_config, trust_remote_code=trust_remote)

    if _requires_seq2seq_model(tasks, config):
        try:
            return AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_config,
                config=config,
                trust_remote_code=trust_remote,
            )
        except (ValueError, OSError):
            pass

    if _requires_causal_model(tasks, config):
        try:
            return AutoModelForCausalLM.from_pretrained(
                pretrained_config,
                config=config,
                trust_remote_code=trust_remote,
            )
        except (ValueError, OSError):
            # Fall back to encoder model when causal head is unavailable
            config.is_decoder = True

    return AutoModel.from_pretrained(
        pretrained_config,
        config=config,
        trust_remote_code=trust_remote,
    )


def load_pretrained_tokenizer(pretrained_config, trust_remote_code: bool | None = None):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_config,
        trust_remote_code=_should_trust_remote_code(trust_remote_code),
    )
    return tokenizer


def resize_pretrain_tok(pretrained, tokenizer):
    if pretrained.config.vocab_size != len(tokenizer):
        pretrained.resize_token_embeddings(len(tokenizer))
    return pretrained, tokenizer


def add_tokens_to_pretrain(pretrained, tokenizer, add_tokens, sample_init=False):
    if not add_tokens:
        return pretrained, tokenizer

    origin_vocab_size = tokenizer.vocab_size
    print("===ADD TOKEN===")
    num_added_toks = tokenizer.add_tokens(add_tokens)
    print('We have added', num_added_toks, 'tokens')
    pretrained.resize_token_embeddings(len(tokenizer))
    if sample_init:
        input_embedding = pretrained.get_input_embeddings()
        state_dict_weight = input_embedding.state_dict()['weight']
        state_dict_weight[origin_vocab_size:len(tokenizer)] = copy.copy(
            state_dict_weight[100:100 + num_added_toks])
        pretrained.set_input_embeddings(input_embedding)
    print("===============")
    return pretrained, tokenizer


def load_trained_model(model_path, pretrained_config=None, tag=None, trust_remote_code: bool | None = None):
    """loading saved task"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torchpack = torch.load(model_path, map_location=device)

    model_info = {key: torchpack[key] for key in torchpack.keys() if 'state_dict' not in key and 'models' not in key}
    print("===task info===")
    [print(k, v[:10], "...") if isinstance(v, list) and len(v) > 10 else print(k, v) for k, v in model_info.items()]
    print('===============')

    if 'tags' in torchpack and len(torchpack['tags']) > 1:
        if tag is None:
            print("Pick which models to use in multi-task models")
            inquirer_res = inquirer.prompt(
                [inquirer.List('tag', message="Select task", choices=torchpack['tags'])])
            tag = inquirer_res['tag']
        type_ind = torchpack['tags'].index(tag)
    else:
        type_ind = 0
    print("loading saved task")

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
    # load task
    trust_remote = _should_trust_remote_code(trust_remote_code)
    tokenizer = load_pretrained_tokenizer(config, trust_remote_code=trust_remote)
    pretrained = load_pretrained_model(config, type, trust_remote_code=trust_remote)

    pretrained, tokenizer = add_tokens_to_pretrain(pretrained, tokenizer, add_tokens)

    model_class = load_model_class(type)
    task_detail = {}
    if 'task-label' in torchpack:
        task_detail = torchpack['task-label']
    elif 'label' in torchpack:
        task_detail = {'label': torchpack['label']}

    model = model_class.Model(tokenizer=tokenizer, pretrained=pretrained, tasks_detail=task_detail,
                              maxlen=maxlen)
    model.load_state_dict(models_state[type_ind], strict=False)
    model = model.to(device)

    preprocessor = model_class.Preprocessor(tokenizer)

    print("finish loading")
    return model, type, model_class, model_info, preprocessor


def save_model(models, input_arg, models_tag, epoch, fname, logger, accelerator, add_tokens=None):
    accelerator.wait_for_everyone()
    save_model = {
        'models': [accelerator.get_state_dict(m) for m in models],
        'model_config': input_arg.get('config'),
        'add_tokens': add_tokens,
        'tags': models_tag,
        'type': input_arg.get('task'),
        'maxlen': input_arg.get('maxlen'),
        'epoch': epoch
    }

    for ind, m in enumerate(input_arg.get('task')):
        if 'tag' in m:
            save_model['label'] = models[ind].labels
        if "clas" in m:
            save_model['task-label'] = models[ind].tasks_detail

    torch.save(save_model, f"{fname}.pt")
    logger.write_log(f"weights were saved to {fname}.pt")


def tie_encoder_decoder_weights(encoder, decoder, base_model_prefix):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        print(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Module,
            encoder_pointer: nn.Module,
            module_name: str,
            uninitialized_encoder_weights: List[str],
            depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight"):
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                    len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                            encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your task."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights)
    if len(uninitialized_encoder_weights) > 0:
        print(
            f"The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}"
        )
    else:
        print("All encoder weights tied to the decoder")
