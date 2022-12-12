import argparse
import sys
import os
from datetime import timedelta
import time
from itertools import zip_longest

import torch
from torch.utils import data
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
import nlp2
import tfkit
import tfkit.utility.tok as tok
from tfkit.utility.dataloader import dataloader_collate
from tfkit.utility.dataset import get_dataset

from tfkit.utility.logger import Logger
from tfkit.utility.model import load_model_class, save_model, load_pretrained_tokenizer, load_pretrained_model, \
    resize_pretrain_tok
import logging
from accelerate import Accelerator

transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.CRITICAL)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"


def parse_train_args(args):
    parser = argparse.ArgumentParser()
    exceed_mode = nlp2.function_get_all_arg_with_value(tok.handle_exceed)['mode']
    parser.add_argument("--batch", type=int, default=20, help="batch size, default 20")
    parser.add_argument("--lr", type=float, nargs='+', default=[5e-5], help="learning rate, default 5e-5")
    parser.add_argument("--epoch", type=int, default=10, help="epoch, default 10")
    parser.add_argument("--maxlen", type=int, default=0, help="max tokenized sequence length, default task max len")
    parser.add_argument("--handle_exceed", choices=exceed_mode,
                        help='select ways to handle input exceed max length')
    parser.add_argument("--savedir", type=str, default="checkpoints/", help="task saving dir, default /checkpoints")
    parser.add_argument("--add_tokens_freq", type=int, default=0,
                        help="auto add freq >= x UNK token to word table")
    parser.add_argument("--add_tokens_file", type=str,
                        help="add token from a list file")
    parser.add_argument("--add_tokens_config", type=str,
                        help="add token from tokenizer config")
    parser.add_argument("--train", type=str, nargs='+', required=True, help="train dataset path")
    parser.add_argument("--test", type=str, nargs='+', required=True, help="test dataset path")
    parser.add_argument("--no_eval", action='store_true', help="not running evaluation")
    parser.add_argument("--task", type=str, required=True, nargs='+',
                        choices=tfkit.utility.model.list_all_model(), help="task task")
    parser.add_argument("--tag", type=str, nargs='+', help="tag to identity task in multi-task")
    parser.add_argument("--config", type=str, default='bert-base-multilingual-cased', required=True,
                        help='distilbert-base-multilingual-cased|voidful/albert_chinese_small')
    parser.add_argument("--tok_config", type=str,
                        help='tokenizer config')
    parser.add_argument("--seed", type=int, default=609, help="random seed, default 609")
    parser.add_argument("--worker", type=int, default=8, help="number of worker on pre-processing, default 8")
    parser.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation, default 1")
    parser.add_argument('--tensorboard', action='store_true', help='Turn on tensorboard graphing')
    parser.add_argument('--wandb', action='store_true', help='Turn on wandb with project name')
    parser.add_argument("--resume", help='resume training')
    parser.add_argument("--cache", action='store_true', help='cache training data')
    parser.add_argument("--panel", action='store_true', help="enable panel to input argument")

    input_arg, model_arg = parser.parse_known_args(args)
    input_arg = {k: v for k, v in vars(input_arg).items() if v is not None}
    model_arg = {k.replace("--", ""): v for k, v in zip(model_arg[:-1:2], model_arg[1::2])}

    return input_arg, model_arg


def optimizer(model, lr, total_step):
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(total_step * 0.05),
                                                num_training_steps=total_step)
    return [optim, scheduler]


def model_train(models_list, dataloaders, models_tag, input_arg, epoch, logger, accelerator, fname, add_tokens):
    optims_schs = []
    models = []
    total_iter = 0
    t_loss = 0
    end = False
    total_iter_length = len(dataloaders[0])
    pbar = tqdm(total=total_iter_length)

    data_iters = []
    for i, (model, dataloader) in enumerate(zip(models_list, dataloaders)):
        if not accelerator.state.backend:
            model = torch.nn.DataParallel(model)
        model.train()
        optim = optimizer(model, input_arg.get('lr')[i] if i < len(input_arg.get('lr')) else input_arg.get('lr')[0],
                          total_iter_length)
        model, optim, dataloader = accelerator.prepare(model, optim, dataloader)
        optims_schs.append(optim)
        models.append(model)
        data_iters.append(iter(dataloader))

    while not end:
        for (model, optim_sch, mtag, train_batch) in zip(models, optims_schs, models_tag, data_iters):
            optim = optim_sch[0]
            scheduler = optim_sch[1]
            train_batch = next(train_batch, None)
            if train_batch is not None:
                loss = model(train_batch)
                loss = loss / input_arg.get('grad_accum')
                accelerator.backward(loss.mean())
                if (total_iter + 1) % input_arg.get('grad_accum') == 0:
                    optim.step()
                    model.zero_grad()
                    scheduler.step()
                t_loss += loss.mean().detach()
                logger.write_metric("loss/step", loss.mean().detach(), epoch)
                if total_iter % 100 == 0 and total_iter != 0:  # monitoring
                    logger.write_log(
                        f"epoch: {epoch}, tag: {mtag}, task: {model.__class__.__name__}, step: {total_iter}, loss: {t_loss / total_iter if total_iter > 0 else 0}, total:{total_iter_length}")
                if total_iter % 50000 == 0 and total_iter != 0:  # cache
                    save_model(models, input_arg, models_tag, epoch,
                               f"{fname}_epoch_{epoch}_iter_{total_iter}", logger,
                               add_tokens=add_tokens,
                               accelerator=accelerator)
            else:
                end = True
        pbar.update(1)
        total_iter += 1
    pbar.close()
    logger.write_log(
        f"epoch: {epoch}, step: {total_iter}, loss: {t_loss / total_iter if total_iter > 0 else 0}, total: {total_iter}")
    return t_loss / total_iter


def model_eval(models, dataloaders, fname, input_arg, epoch, logger, accelerator):
    t_loss = 0
    t_length = 0
    for m in models:
        m.eval()
    with torch.no_grad():
        total_iter_length = len(dataloaders[0])
        iters = [iter(accelerator.prepare(ds)) for ds in dataloaders]
        end = False
        pbar = tqdm(total=total_iter_length)
        while not end:
            for model, batch in zip(models, iters):
                test_batch = next(batch, None)
                if test_batch is not None:
                    loss = model(test_batch)
                    loss = loss / input_arg.get('grad_accum')
                    t_loss += loss.mean().detach()
                    t_length += 1
                    pbar.update(1)
                else:
                    end = True
        pbar.close()

    avg_t_loss = t_loss / t_length if t_length > 0 else 0
    logger.write_log(f"task: {fname}, Total Loss: {avg_t_loss}")
    logger.write_metric("eval_loss/step", avg_t_loss, epoch)
    return avg_t_loss


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

    # train/eval loop
    logger.write_log("training batch : " + str(input_arg.get('batch') * input_arg.get('grad_accum')))
    for epoch in range(start_epoch, start_epoch + input_arg.get('epoch')):
        start_time = time.time()
        fname = os.path.join(input_arg.get('savedir'), str(epoch))

        logger.write_log(f"=========train at epoch={epoch}=========")
        try:
            train_avg_loss = model_train(models, train_dataloaders, models_tag, input_arg, epoch, logger, accelerator,
                                         fname, add_tokens)
            logger.write_metric("train_loss/epoch", train_avg_loss, epoch)
        except KeyboardInterrupt:
            save_model(models, input_arg, models_tag, epoch, fname + "_interrupt", logger, add_tokens=add_tokens,
                       accelerator=accelerator)
            pass

        logger.write_log(f"=========save at epoch={epoch}=========")
        save_model(models, input_arg, models_tag, epoch, fname, logger, add_tokens=add_tokens, accelerator=accelerator)

        if input_arg.get('no_eval') is False:
            logger.write_log(f"=========eval at epoch={epoch}=========")
            eval_avg_loss = model_eval(models, test_dataloaders, fname, input_arg, epoch, logger, accelerator)
            logger.write_metric("eval_loss/epoch", eval_avg_loss, epoch)
        logger.write_log(f"=== Epoch execution time: {timedelta(seconds=(time.time() - start_time))} ===")


if __name__ == "__main__":
    main()
