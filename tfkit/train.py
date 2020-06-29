import argparse
import random

import torch
from torch import nn
from tqdm import tqdm
from transformers import *
import numpy as np
import tensorboardX as tensorboard
from torch.utils import data
from itertools import zip_longest
import os
import gen_once
import gen_twice
import gen_onebyone
import classifier
import tag
import qa


def write_log(*args):
    line = ' '.join([str(a) for a in args])
    with open(os.path.join(arg.savedir, "message.log"), "a", encoding='utf8') as log_file:
        log_file.write(line + '\n')
    print(line)


def optimizer(model, lr):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_parameters, lr=lr)
    return optimizer


def train(models_list, train_dataset, models_tag, arg, epoch):
    optims = []
    models = []
    for i, m in enumerate(models_list):
        model = nn.DataParallel(m)
        model.train()
        models.append(model)
        optims.append(optimizer(m, arg.lr[i] if i < len(arg.lr) else arg.lr[0]))

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
                loss = loss / arg.grad_accum
                loss.mean().backward()
                if (total_iter + 1) % arg.grad_accum == 0:
                    optim.step()
                    model.zero_grad()
                t_loss += loss.mean().item()
                if arg.tensorboard:
                    writer.add_scalar("loss/step", loss.mean().item(), epoch)
                if total_iter % 100 == 0 and total_iter != 0:  # monitoring
                    write_log(
                        f"tag: {mtag}, model: {model.module.__class__.__name__}, step: {total_iter}, loss: {t_loss / total_iter if total_iter > 0 else 0}, total:{total_iter_length}")
            else:
                end = True
        pbar.update(1)
        total_iter += 1
    pbar.close()
    write_log(f"step: {total_iter}, loss: {t_loss / total_iter if total_iter > 0 else 0}, total: {total_iter}")
    return t_loss / total_iter


def eval(models, test_dataset, fname, epoch):
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
                    t_loss += loss.mean().item()
                    t_length += 1
                    pbar.update(1)
                else:
                    end = True
        pbar.close()

    avg_t_loss = t_loss / t_length if t_length > 0 else 0
    write_log(f"model: {fname}, Total Loss: {avg_t_loss}")
    if arg.tensorboard:
        writer.add_scalar("eval_loss/step", avg_t_loss, epoch)
    return avg_t_loss


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--lr", type=float, nargs='+', default=[5e-5])
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--maxlen", type=int, default=368)
    parser.add_argument("--savedir", type=str, default="checkpoints/")
    parser.add_argument("--add_tokens", action='store_true', help="add new token if not exist")
    parser.add_argument("--train", type=str, nargs='+', required=True)
    parser.add_argument("--valid", type=str, nargs='+', required=True)
    parser.add_argument("--model", type=str, required=True, nargs='+',
                        choices=['once', 'twice', 'onebyone', 'clas', 'tagRow', 'tagCol', 'qa',
                                 'onebyone-neg', 'onebyone-pos', 'onebyone-both'])
    parser.add_argument("--lossdrop", action='store_true', help="loss dropping for text generation")
    parser.add_argument("--tag", type=str, nargs='+', help="tag to identity task in multi-task")
    parser.add_argument("--config", type=str, default='bert-base-multilingual-cased', required=True,
                        help='distilbert-base-multilingual-cased/bert-base-multilingual-cased/voidful/albert_chinese_small')
    parser.add_argument("--seed", type=int, default=609)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
    parser.add_argument("--resume", help='resume training')
    parser.add_argument("--cache", action='store_true', help='Caching training data')
    global arg
    arg = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(arg.savedir): os.makedirs(arg.savedir)

    write_log("TRAIN PARAMETER")
    write_log("=======================")
    [write_log(var, ':', vars(arg)[var]) for var in vars(arg)]
    write_log("=======================")

    # load pre-train model
    if 'albert_chinese' in arg.config:
        tokenizer = BertTokenizer.from_pretrained(arg.config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(arg.config)
    pretrained = AutoModel.from_pretrained(arg.config)

    models = []
    models_tag = arg.tag if arg.tag is not None else [m.lower() + "_" + str(ind) for ind, m in enumerate(arg.model)]
    train_dataset = []
    test_dataset = []
    train_ds_maxlen = 0
    test_ds_maxlen = 0
    for model_type, train_file, valid_file in zip_longest(arg.model, arg.train, arg.valid, fillvalue=""):
        model_type = model_type.lower()
        if "once" in model_type:
            train_ds = gen_once.loadOnceDataset(train_file, pretrained=arg.config, maxlen=arg.maxlen,
                                                cache=arg.cache)
            test_ds = gen_once.loadOnceDataset(valid_file, pretrained=arg.config, maxlen=arg.maxlen,
                                               cache=arg.cache)
            model = gen_once.Once(tokenizer, pretrained, maxlen=arg.maxlen)
        elif "twice" in model_type:
            train_ds = gen_once.loadOnceDataset(train_file, pretrained=arg.config, maxlen=arg.maxlen,
                                                cache=arg.cache)
            test_ds = gen_once.loadOnceDataset(valid_file, pretrained=arg.config, maxlen=arg.maxlen,
                                               cache=arg.cache)
            model = gen_twice.Twice(tokenizer, pretrained, maxlen=arg.maxlen)
        elif "onebyone" in model_type:
            train_ds = gen_onebyone.loadOneByOneDataset(train_file, pretrained=arg.config, maxlen=arg.maxlen,
                                                        cache=arg.cache,
                                                        likelihood=model_type)
            test_ds = gen_onebyone.loadOneByOneDataset(valid_file, pretrained=arg.config, maxlen=arg.maxlen,
                                                       cache=arg.cache)
            model = gen_onebyone.OneByOne(tokenizer, pretrained, maxlen=arg.maxlen, lossdrop=arg.lossdrop)
        elif 'clas' in model_type:
            train_ds = classifier.loadClassifierDataset(train_file, pretrained=arg.config, cache=arg.cache)
            test_ds = classifier.loadClassifierDataset(valid_file, pretrained=arg.config, cache=arg.cache)
            model = classifier.MtClassifier(train_ds.task, tokenizer, pretrained)
        elif 'tag' in model_type:
            if "row" in model_type:
                train_ds = tag.loadRowTaggerDataset(train_file, pretrained=arg.config, maxlen=arg.maxlen,
                                                    cache=arg.cache)
                test_ds = tag.loadRowTaggerDataset(valid_file, pretrained=arg.config, maxlen=arg.maxlen,
                                                   cache=arg.cache)
            elif "col" in model_type:
                train_ds = tag.loadColTaggerDataset(train_file, pretrained=arg.config, maxlen=arg.maxlen,
                                                    cache=arg.cache)
                test_ds = tag.loadColTaggerDataset(valid_file, pretrained=arg.config, maxlen=arg.maxlen,
                                                   cache=arg.cache)
            model = tag.Tagger(train_ds.label, tokenizer, pretrained, maxlen=arg.maxlen)
        elif 'qa' in model_type:
            train_ds = qa.loadQADataset(train_file, pretrained=arg.config, cache=arg.cache)
            test_ds = qa.loadQADataset(valid_file, pretrained=arg.config, cache=arg.cache)
            model = qa.QA(tokenizer, pretrained, maxlen=arg.maxlen)

        model = model.to(device)
        train_ds_maxlen = train_ds.__len__() if train_ds.__len__() > train_ds_maxlen else train_ds_maxlen
        test_ds_maxlen = test_ds.__len__() if test_ds.__len__() > test_ds_maxlen else test_ds_maxlen
        train_dataset.append(train_ds)
        test_dataset.append(test_ds)
        models.append(model)

    # balance sample for multi-task
    for ds in train_dataset:
        ds.increase_with_sampling(train_ds_maxlen)
    for ds in test_dataset:
        ds.increase_with_sampling(test_ds_maxlen)

    train_dataset = [data.DataLoader(dataset=ds,
                                     batch_size=arg.batch,
                                     shuffle=True,
                                     num_workers=arg.worker) for ds in train_dataset]
    test_dataset = [data.DataLoader(dataset=ds,
                                    batch_size=arg.batch,
                                    shuffle=True,
                                    num_workers=arg.worker) for ds in test_dataset]

    if arg.tensorboard:
        global writer
    writer = tensorboard.SummaryWriter()

    start_epoch = 1

    if arg.resume:
        print("Loading back:", arg.resume)
        package = torch.load(arg.resume, map_location=device)
        if 'model_state_dict' in package:
            models[0].load_state_dict(package['model_state_dict'])
        else:
            for model_tag, state_dict in zip(package['tags'], package['models']):
                tag_ind = package['tags'].index(model_tag)
                models[tag_ind].load_state_dict(state_dict)
        start_epoch = int(package.get('epoch', 1)) + 1

    set_seed(arg.seed)

    write_log("training batch : " + str(arg.batch * arg.grad_accum))
    for epoch in range(start_epoch, start_epoch + arg.epoch):
        fname = os.path.join(arg.savedir, str(epoch))

        write_log(f"=========train at epoch={epoch}=========")
        train_avg_loss = train(models, train_dataset, models_tag, arg, epoch)

        write_log(f"=========save at epoch={epoch}=========")
        save_model = {
            'models': [m.state_dict() for m in models],
            'model_config': arg.config,
            'tags': models_tag,
            'type': arg.model,
            'maxlen': arg.maxlen,
            'epoch': epoch
        }

        for ind, m in enumerate(arg.model):
            if 'tag' in m:
                save_model['label'] = models[ind].labels
            if "clas" in m:
                save_model['task-label'] = models[ind].tasks_detail

        torch.save(save_model, f"{fname}.pt")
        write_log(f"weights were saved to {fname}.pt")

        write_log(f"=========eval at epoch={epoch}=========")
        eval_avg_loss = eval(models, test_dataset, fname, epoch)

        if arg.tensorboard:
            writer.add_scalar("train_loss/epoch", train_avg_loss, epoch)
        writer.add_scalar("eval_loss/epoch", eval_avg_loss, epoch)


if __name__ == "__main__":
    main()
