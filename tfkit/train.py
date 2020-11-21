import argparse
import sys

import nlp2
import tfkit
import torch
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer, AutoModel
from torch.utils import data
from itertools import zip_longest
import os
import tfkit.utility.tok as tok
from tfkit.utility.dataset import get_dataset
from tfkit.utility.logger import Logger
from tfkit.utility.model_loader import load_model_class


def parse_train_args(args):
    parser = argparse.ArgumentParser()
    exceed_mode = nlp2.function_get_all_arg_with_value(tok.handle_exceed)['mode']
    parser.add_argument("--batch", type=int, default=20, help="batch size, default 20")
    parser.add_argument("--lr", type=float, nargs='+', default=[5e-5], help="learning rate, default 5e-5")
    parser.add_argument("--epoch", type=int, default=10, help="epoch, default 10")
    parser.add_argument("--maxlen", type=int, default=512, help="max tokenized sequence length, default 512")
    parser.add_argument("--handle_exceed", choices=exceed_mode,
                        help='select ways to handle input exceed max length')
    parser.add_argument("--savedir", type=str, default="checkpoints/", help="model saving dir, default /checkpoints")
    parser.add_argument("--add_tokens", type=int, default=0,
                        help="auto add freq > x UNK token to word table")
    parser.add_argument("--train", type=str, nargs='+', required=True, help="train dataset path")
    parser.add_argument("--test", type=str, nargs='+', required=True, help="test dataset path")
    parser.add_argument("--model", type=str, required=True, nargs='+',
                        choices=tfkit.model_loader.list_all_model(), help="model task")
    parser.add_argument("--tag", type=str, nargs='+', help="tag to identity task in multi-task")
    parser.add_argument("--config", type=str, default='bert-base-multilingual-cased', required=True,
                        help='distilbert-base-multilingual-cased|voidful/albert_chinese_small')
    parser.add_argument("--seed", type=int, default=609, help="random seed, default 609")
    parser.add_argument("--worker", type=int, default=8, help="number of worker on pre-processing, default 8")
    parser.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation, default 1")
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
    parser.add_argument("--resume", help='resume training')
    parser.add_argument("--cache", action='store_true', help='cache training data')
    parser.add_argument("--panel", action='store_true', help="enable panel to input argument")

    input_arg, model_arg = parser.parse_known_args(args)
    input_arg = {k: v for k, v in vars(input_arg).items() if v is not None}
    model_arg = {k.replace("--", ""): v for k, v in zip(model_arg[:-1:2], model_arg[1::2])}

    return input_arg, model_arg


def optimizer(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr)


def model_train(models_list, train_dataset, models_tag, input_arg, epoch, logger):
    optims = []
    models = []
    for i, m in enumerate(models_list):
        model = torch.nn.DataParallel(m)
        model.train()
        models.append(model)
        optims.append(optimizer(m, input_arg.get('lr')[i] if i < len(input_arg.get('lr')) else input_arg.get('lr')[0]))

    total_iter = 0
    t_loss = 0

    iters = [iter(ds) for ds in train_dataset]
    total_iter_length = len(iters[0])
    end = False
    pbar = tqdm(total=total_iter_length)
    while not end:
        for (model, optim, mtag, batch) in zip(models, optims, models_tag, iters):
            train_batch = next(batch, None)
            if train_batch is not None:
                loss = model(train_batch)
                loss = loss / input_arg.get('grad_accum')
                loss.mean().backward()
                if (total_iter + 1) % input_arg.get('grad_accum') == 0:
                    optim.step()
                    optim.zero_grad()
                    model.zero_grad()
                t_loss += loss.mean().item()
                logger.write_metric("loss/step", loss.mean().item(), epoch)
                if total_iter % 100 == 0 and total_iter != 0:  # monitoring
                    logger.write_log(
                        f"epoch: {epoch}, tag: {mtag}, model: {model.module.__class__.__name__}, step: {total_iter}, loss: {t_loss / total_iter if total_iter > 0 else 0}, total:{total_iter_length}")
            else:
                end = True
        pbar.update(1)
        total_iter += 1
    pbar.close()
    logger.write_log(
        f"epoch: {epoch}, step: {total_iter}, loss: {t_loss / total_iter if total_iter > 0 else 0}, total: {total_iter}")
    return t_loss / total_iter


def model_eval(models, test_dataset, fname, input_arg, epoch, logger):
    t_loss = 0
    t_length = 0
    for m in models:
        m.eval()

    with torch.no_grad():
        iters = [iter(ds) for ds in test_dataset]
        end = False
        total_iter_length = len(iters[0])
        pbar = tqdm(total=total_iter_length)
        while not end:
            for model, batch in zip(models, iters):
                test_batch = next(batch, None)
                if test_batch is not None:
                    loss = model(test_batch)
                    loss = loss / input_arg.get('grad_accum')
                    t_loss += loss.mean().item()
                    t_length += 1
                    pbar.update(1)
                else:
                    end = True
        pbar.close()

    avg_t_loss = t_loss / t_length if t_length > 0 else 0
    logger.write_log(f"model: {fname}, Total Loss: {avg_t_loss}")
    logger.write_metric("eval_loss/step", avg_t_loss, epoch)
    return avg_t_loss


def load_model_and_datas(tokenizer, pretrained, device, model_arg, input_arg):
    models = []
    train_dataset = []
    test_dataset = []
    train_ds_maxlen = 0
    test_ds_maxlen = 0
    for model_class_name, train_file, test_file in zip_longest(input_arg.get('model'), input_arg.get('train'),
                                                               input_arg.get('test'),
                                                               fillvalue=""):
        # get model class
        model_class = load_model_class(model_class_name)

        # load dataset
        ds_parameter = {**model_arg, **input_arg}
        train_ds = get_dataset(train_file, model_class, ds_parameter)
        test_ds = get_dataset(test_file, model_class, ds_parameter)

        # load model
        model = model_class.Model(tokenizer=tokenizer, pretrained=pretrained, tasks_detail=train_ds.task,
                                  maxlen=input_arg.get('maxlen'))
        model = model.to(device)

        # append to max len
        train_ds_maxlen = train_ds.__len__() if train_ds.__len__() > train_ds_maxlen else train_ds_maxlen
        test_ds_maxlen = test_ds.__len__() if test_ds.__len__() > test_ds_maxlen else test_ds_maxlen

        train_dataset.append(train_ds)
        test_dataset.append(test_ds)
        models.append(model)

    return models, train_dataset, test_dataset, train_ds_maxlen, test_ds_maxlen


def main(arg=None):
    input_arg, model_arg = parse_train_args(sys.argv[1:]) if arg is None else parse_train_args(arg)
    nlp2.get_dir_with_notexist_create(input_arg.get('savedir'))
    logger = Logger(savedir=input_arg.get('savedir'), tensorboard=input_arg.get('tensorboard'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nlp2.set_seed(input_arg.get('seed'))

    logger.write_log("TRAIN PARAMETER")
    logger.write_log("=======================")
    [logger.write_log(str(key) + " : " + str(value)) for key, value in input_arg.items()]
    logger.write_log("=======================")

    # load pre-train model
    pretrained_config = input_arg.get('config')
    if 'albert_chinese' in pretrained_config:
        tokenizer = BertTokenizer.from_pretrained(pretrained_config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_config)
    pretrained = AutoModel.from_pretrained(pretrained_config)

    # handling add tokens
    if input_arg.get('add_tokens'):
        logger.write_log("Calculating Unknown Token")
        add_tokens = tok.get_freqK_unk_token(tokenizer, input_arg.get('train') + input_arg.get('test'),
                                             input_arg.get('add_tokens'))
        num_added_toks = tokenizer.add_tokens(add_tokens)
        logger.write_log('We have added', num_added_toks, 'tokens')
        pretrained.resize_token_embeddings(len(tokenizer))
        save_path = os.path.join(input_arg.get('savedir'), pretrained_config + "_added_tok")
        pretrained.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.write_log('New pre-train model saved at ', save_path)
        logger.write_log("=======================")

    # load model and data
    models_tag = input_arg.get('tag') if input_arg.get('tag', None) is not None else [m.lower() + "_" + str(ind) for
                                                                                      ind, m in
                                                                                      enumerate(input_arg.get('model'))]

    models, train_dataset, test_dataset, train_ds_maxlen, test_ds_maxlen = load_model_and_datas(tokenizer, pretrained,
                                                                                                device, model_arg,
                                                                                                input_arg)
    # balance sample for multi-task
    for ds in train_dataset:
        ds.increase_with_sampling(train_ds_maxlen)
    for ds in test_dataset:
        ds.increase_with_sampling(test_ds_maxlen)

    train_dataset = [data.DataLoader(dataset=ds,
                                     batch_size=input_arg.get('batch'),
                                     shuffle=True,
                                     num_workers=input_arg.get('worker')) for ds in train_dataset]
    test_dataset = [data.DataLoader(dataset=ds,
                                    batch_size=input_arg.get('batch'),
                                    shuffle=True,
                                    num_workers=input_arg.get('worker')) for ds in test_dataset]

    # loading model back
    start_epoch = 1
    if input_arg.get('resume'):
        logger.write_log("Loading back:", input_arg.get('resume'))
        package = torch.load(input_arg.get('resume'), map_location=device)
        if 'model_state_dict' in package:
            models[0].load_state_dict(package['model_state_dict'])
        else:
            for model_tag, state_dict in zip(package['tags'], package['models']):
                tag_ind = package['tags'].index(model_tag)
                models[tag_ind].load_state_dict(state_dict)
        start_epoch = int(package.get('epoch', 1)) + 1

    # train/eval loop
    logger.write_log("training batch : " + str(input_arg.get('batch') * input_arg.get('grad_accum')))
    for epoch in range(start_epoch, start_epoch + input_arg.get('epoch')):
        fname = os.path.join(input_arg.get('savedir'), str(epoch))

        logger.write_log(f"=========train at epoch={epoch}=========")
        train_avg_loss = model_train(models, train_dataset, models_tag, input_arg, epoch, logger)

        logger.write_log(f"=========save at epoch={epoch}=========")
        save_model = {
            'models': [m.state_dict() for m in models],
            'model_config': input_arg.get('config'),
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

        logger.write_log(f"=========eval at epoch={epoch}=========")
        eval_avg_loss = model_eval(models, test_dataset, fname, input_arg, epoch, logger)

        logger.write_metric("train_loss/epoch", train_avg_loss, epoch)
        logger.write_metric("eval_loss/epoch", eval_avg_loss, epoch)


if __name__ == "__main__":
    main()
