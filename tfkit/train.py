import argparse
import logging
import os
import sys
import time
from datetime import timedelta
from itertools import zip_longest

import nlp2
import torch
from accelerate import Accelerator
from torch.utils import data

import tfkit
import tfkit.utility.tok as tok
from tfkit.utility.constants import (
    DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_MAXLEN,
    DEFAULT_SEED, DEFAULT_WORKER_COUNT, DEFAULT_GRADIENT_ACCUMULATION,
    DEFAULT_CHECKPOINT_DIR, DEFAULT_PRETRAINED_MODEL,
    ENV_TOKENIZERS_PARALLELISM, ENV_OMP_NUM_THREADS
)
from tfkit.utility.data_loader import dataloader_collate
from tfkit.utility.dataset import get_dataset
from tfkit.utility.logger import Logger
from tfkit.utility.config import ConfigManager
from tfkit.utility.model import (
    load_model_class, save_model, load_pretrained_tokenizer, 
    load_pretrained_model, resize_pretrain_tok
)
from tfkit.utility.training_utils import TrainingManager

transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.CRITICAL)

os.environ[ENV_TOKENIZERS_PARALLELISM] = "false"
os.environ[ENV_OMP_NUM_THREADS] = "1"


def parse_train_args(args):
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="Train TFKit models")
    exceed_mode = nlp2.function_get_all_arg_with_value(tok.handle_exceed)['mode']
    
    # Training parameters
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, 
                       help=f"batch size, default {DEFAULT_BATCH_SIZE}")
    parser.add_argument("--lr", type=float, nargs='+', default=[DEFAULT_LEARNING_RATE], 
                       help=f"learning rate, default {DEFAULT_LEARNING_RATE}")
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCHS, 
                       help=f"epoch, default {DEFAULT_EPOCHS}")
    parser.add_argument("--maxlen", type=int, default=0, 
                       help="max tokenized sequence length, default task max len")
    parser.add_argument("--handle_exceed", choices=exceed_mode,
                       help='select ways to handle input exceed max length')
    parser.add_argument("--grad_accum", type=int, default=DEFAULT_GRADIENT_ACCUMULATION, 
                       help=f"gradient accumulation, default {DEFAULT_GRADIENT_ACCUMULATION}")
    
    # Model and data parameters
    parser.add_argument("--config", type=str, default=DEFAULT_PRETRAINED_MODEL, required=True,
                       help='pretrained model config (e.g., bert-base-multilingual-cased)')
    parser.add_argument("--tok_config", type=str, help='tokenizer config')
    parser.add_argument("--task", type=str, required=True, nargs='+',
                       choices=tfkit.utility.model.list_all_model(), help="task type")
    parser.add_argument("--tag", type=str, nargs='+', help="tag to identify task in multi-task")
    
    # Data paths
    parser.add_argument("--train", type=str, nargs='+', required=True, help="train dataset path")
    parser.add_argument("--test", type=str, nargs='+', required=True, help="test dataset path")
    parser.add_argument("--savedir", type=str, default=DEFAULT_CHECKPOINT_DIR, 
                       help=f"task saving dir, default {DEFAULT_CHECKPOINT_DIR}")
    
    # Token handling
    parser.add_argument("--add_tokens_freq", type=int, default=0,
                       help="auto add freq >= x UNK token to word table")
    parser.add_argument("--add_tokens_file", type=str, help="add token from a list file")
    parser.add_argument("--add_tokens_config", type=str, help="add token from tokenizer config")
    
    # System parameters
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, 
                       help=f"random seed, default {DEFAULT_SEED}")
    parser.add_argument("--worker", type=int, default=DEFAULT_WORKER_COUNT, 
                       help=f"number of worker on pre-processing, default {DEFAULT_WORKER_COUNT}")
    
    # Options
    parser.add_argument("--no_eval", action='store_true', help="not running evaluation")
    parser.add_argument('--tensorboard', action='store_true', help='Turn on tensorboard graphing')
    parser.add_argument('--wandb', action='store_true', help='Turn on wandb with project name')
    parser.add_argument("--resume", help='resume training')
    parser.add_argument("--cache", action='store_true', help='cache training data')
    parser.add_argument("--panel", action='store_true', help="enable panel to input argument")
    
    # Configuration file support
    parser.add_argument("--config_file", type=str, help="path to configuration file (YAML or JSON)")
    parser.add_argument("--save_config", type=str, help="save current configuration to file")

    input_arg, model_arg = parser.parse_known_args(args)
    input_arg = {k: v for k, v in vars(input_arg).items() if v is not None}
    model_arg = {k.replace("--", ""): v for k, v in zip(model_arg[:-1:2], model_arg[1::2])}

    return input_arg, model_arg


# Training functions now handled by TrainingManager class
# These functions have been moved to tfkit.utility.training_utils


def load_model_and_datas(tokenizer, pretrained, accelerator, model_arg, input_arg):
    models = []
    train_dataset = []
    test_dataset = []
    train_ds_maxlen = 0
    test_ds_maxlen = 0
    for model_class_name, train_file, test_file in zip_longest(input_arg.get('task'), input_arg.get('train'),
                                                               input_arg.get('test'),
                                                               fillvalue=""):
        # get task class
        model_class = load_model_class(model_class_name)

        # load dataset
        ds_parameter = {**model_arg, **input_arg}
        train_ds = get_dataset(train_file, model_class, tokenizer, ds_parameter)
        test_ds = get_dataset(test_file, model_class, tokenizer, ds_parameter)

        # load task
        model = model_class.Model(tokenizer=tokenizer, pretrained=pretrained, tasks_detail=train_ds.task,
                                  maxlen=input_arg.get('maxlen'), **model_arg)

        # append to max len
        train_ds_maxlen = train_ds.__len__() if train_ds.__len__() > train_ds_maxlen else train_ds_maxlen
        test_ds_maxlen = test_ds.__len__() if test_ds.__len__() > test_ds_maxlen else test_ds_maxlen

        train_dataset.append(train_ds)
        test_dataset.append(test_ds)
        models.append(model)

    return models, train_dataset, test_dataset, train_ds_maxlen, test_ds_maxlen


def main(arg=None):
    input_arg, model_arg = parse_train_args(sys.argv[1:]) if arg is None else parse_train_args(arg)
    
    # Handle configuration file
    config_manager = None
    if input_arg.get('config_file'):
        config_manager = ConfigManager(input_arg['config_file'])
        # Update configuration with command line arguments (CLI args override config file)
        config_manager.update_from_args(input_arg, section='training')
        # Get the final arguments from the configuration
        input_arg = config_manager.get_training_args()
        # Remove None values and convert to expected format
        input_arg = {k: v for k, v in input_arg.items() if v is not None}
    
    # Save configuration if requested
    if input_arg.get('save_config'):
        if config_manager is None:
            config_manager = ConfigManager()
            config_manager.update_from_args(input_arg, section='training')
        config_manager.save_config(input_arg['save_config'])
        print(f"Configuration saved to {input_arg['save_config']}")
    
    # Validate configuration
    if config_manager:
        validation_errors = config_manager.validate_config()
        if validation_errors:
            print("Configuration validation errors:")
            for error in validation_errors:
                print(f"  - {error}")
            print("Please fix the configuration and try again.")
            return
    
    accelerator = Accelerator()
    nlp2.get_dir_with_notexist_create(input_arg.get('savedir'))
    logger = Logger(savedir=input_arg.get('savedir'), tensorboard=input_arg.get('tensorboard', False),
                    wandb=input_arg.get('wandb', False), print_fn=accelerator.print)
    logger.write_log("Accelerator")
    logger.write_log("=======================")
    logger.write_log(accelerator.state)
    logger.write_log("=======================")
    nlp2.set_seed(input_arg.get('seed'))

    tokenizer = load_pretrained_tokenizer(input_arg.get('tok_config', input_arg['config']))
    pretrained = load_pretrained_model(input_arg.get('config'), input_arg.get('task'))
    pretrained, tokenizer = resize_pretrain_tok(pretrained, tokenizer)
    if input_arg.get('maxlen') == 0:
        if hasattr(pretrained.config, 'max_position_embeddings'):
            input_arg.update({'maxlen': pretrained.config.max_position_embeddings})
        else:
            input_arg.update({'maxlen': 1024})

    # handling add tokens
    add_tokens = None
    if input_arg.get('add_tokens_freq', None):
        logger.write_log("Calculating Unknown Token")
        add_tokens = tok.get_freqK_unk_token(tokenizer, input_arg.get('train') + input_arg.get('test'),
                                             input_arg.get('add_tokens_freq'))
    if input_arg.get('add_tokens_file', None):
        logger.write_log("Add token from file")
        add_tokens = nlp2.read_files_into_lines(input_arg.get('add_tokens_file'))

    if input_arg.get('add_tokens_config', None):
        logger.write_log("Add token from config")
        add_tokens = tok.get_all_tok_from_config(input_arg.get('add_tokens_config'))

    if add_tokens:
        pretrained, tokenizer = tfkit.utility.model.add_tokens_to_pretrain(pretrained, tokenizer, add_tokens,
                                                                           sample_init=True)

    # load task and data
    models_tag = input_arg.get('tag') if input_arg.get('tag', None) is not None else [m.lower() + "_" + str(ind) for
                                                                                      ind, m in
                                                                                      enumerate(input_arg.get('task'))]

    models, train_dataset, test_dataset, train_ds_maxlen, test_ds_maxlen = load_model_and_datas(tokenizer, pretrained,
                                                                                                accelerator, model_arg,
                                                                                                input_arg)
    # balance sample for multi-task
    for ds in train_dataset:
        ds.increase_with_sampling(train_ds_maxlen)
    for ds in test_dataset:
        ds.increase_with_sampling(test_ds_maxlen)
    logger.write_config(input_arg)
    logger.write_log("TRAIN PARAMETER")
    logger.write_log("=======================")
    [logger.write_log(str(key) + " : " + str(value)) for key, value in input_arg.items()]
    logger.write_log("=======================")

    train_dataloaders = [data.DataLoader(dataset=ds,
                                         batch_size=input_arg.get('batch'),
                                         shuffle=True,
                                         pin_memory=False,
                                         collate_fn=dataloader_collate,
                                         num_workers=input_arg.get('worker')) for ds in
                         train_dataset]
    test_dataloaders = [data.DataLoader(dataset=ds,
                                        batch_size=input_arg.get('batch'),
                                        shuffle=False,
                                        pin_memory=False,
                                        collate_fn=dataloader_collate,
                                        num_workers=input_arg.get('worker')) for ds in
                        test_dataset]

    # loading task back
    start_epoch = 1
    if input_arg.get('resume'):
        logger.write_log("Loading back:", input_arg.get('resume'))
        package = torch.load(input_arg.get('resume'))
        if 'model_state_dict' in package:
            models[0].load_state_dict(package['model_state_dict'])
        else:
            if len(models) != len(package['models']) and not input_arg.get('tag'):
                raise Exception(
                    f"resuming from multi-task task, you should specific which task to use with --tag, from {package['tags']}")
            elif len(models) != len(package['models']):
                tags = input_arg.get('tag')
            else:
                tags = package['tags']
            for ind, model_tag in enumerate(tags):
                tag_ind = package['tags'].index(model_tag)
                models[ind].load_state_dict(package['models'][tag_ind])
        start_epoch = int(package.get('epoch', 1)) + 1

    # Initialize training manager
    trainer = TrainingManager(accelerator, logger)
    
    # train/eval loop
    logger.write_log("training batch : " + str(input_arg.get('batch') * input_arg.get('grad_accum')))
    for epoch in range(start_epoch, start_epoch + input_arg.get('epoch')):
        start_time = time.time()
        fname = os.path.join(input_arg.get('savedir'), str(epoch))

        logger.write_log(f"=========train at epoch={epoch}=========")
        try:
            # Prepare models and optimizers
            prepared_models, optims_schs, data_iters, total_iter_length = trainer.prepare_models_and_optimizers(
                models, train_dataloaders, input_arg
            )
            
            # Train for one epoch
            train_avg_loss = trainer.train_epoch(
                prepared_models, optims_schs, data_iters, models_tag, 
                input_arg, epoch, fname, add_tokens, total_iter_length
            )
            logger.write_metric("train_loss/epoch", train_avg_loss, epoch)
        except KeyboardInterrupt:
            save_model(models, input_arg, models_tag, epoch, fname + "_interrupt", logger, 
                      add_tokens=add_tokens, accelerator=accelerator)
            pass

        logger.write_log(f"=========save at epoch={epoch}=========")
        save_model(models, input_arg, models_tag, epoch, fname, logger, 
                  add_tokens=add_tokens, accelerator=accelerator)

        if input_arg.get('no_eval') is False:
            logger.write_log(f"=========eval at epoch={epoch}=========")
            eval_avg_loss = trainer.evaluate_models(models, test_dataloaders, fname, input_arg, epoch)
            logger.write_metric("eval_loss/epoch", eval_avg_loss, epoch)
        logger.write_log(f"=== Epoch execution time: {timedelta(seconds=(time.time() - start_time))} ===")


if __name__ == "__main__":
    main()
