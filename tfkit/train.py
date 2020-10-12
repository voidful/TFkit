import argparse
import random
import nlp2
import torch
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer, AutoModel
import numpy as np
import tensorboardX as tensorboard
from torch.utils import data
from itertools import zip_longest
import os
import tfkit.gen_once as gen_once
import tfkit.gen_onebyone as gen_onebyone
import tfkit.classifier as classifier
import tfkit.tag as tag
import tfkit.qa as qa
import tfkit.gen_mask as gen_mask
from tfkit.utility import get_freqK_unk_token
import tfkit.utility.tok as tok

input_arg = {}


def write_log(*args):
    line = ' '.join([str(a) for a in args])
    with open(os.path.join(input_arg.savedir, "message.log"), "a", encoding='utf8') as log_file:
        log_file.write(line + '\n')
    print(line)


def optimizer(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr)


def model_train(models_list, train_dataset, models_tag, input_arg, epoch, writer):
    optims = []
    models = []
    for i, m in enumerate(models_list):
        model = torch.nn.DataParallel(m)
        model.train()
        models.append(model)
        optims.append(optimizer(m, input_arg.lr[i] if i < len(input_arg.lr) else input_arg.lr[0]))

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
                loss = loss / input_arg.grad_accum
                loss.mean().backward()
                if (total_iter + 1) % input_arg.grad_accum == 0:
                    optim.step()
                    optim.zero_grad()
                    model.zero_grad()
                t_loss += loss.mean().item()
                if input_arg.tensorboard:
                    writer.add_scalar("loss/step", loss.mean().item(), epoch)
                if total_iter % 100 == 0 and total_iter != 0:  # monitoring
                    write_log(
                        f"epoch: {epoch}, tag: {mtag}, model: {model.module.__class__.__name__}, step: {total_iter}, loss: {t_loss / total_iter if total_iter > 0 else 0}, total:{total_iter_length}")
            else:
                end = True
        pbar.update(1)
        total_iter += 1
    pbar.close()
    write_log(
        f"epoch: {epoch}, step: {total_iter}, loss: {t_loss / total_iter if total_iter > 0 else 0}, total: {total_iter}")
    return t_loss / total_iter


def model_eval(models, test_dataset, fname, epoch, writer):
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
                    loss = loss / input_arg.grad_accum
                    t_loss += loss.mean().item()
                    t_length += 1
                    pbar.update(1)
                else:
                    end = True
        pbar.close()

    avg_t_loss = t_loss / t_length if t_length > 0 else 0
    write_log(f"model: {fname}, Total Loss: {avg_t_loss}")
    if input_arg.tensorboard:
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


def _load_model_and_data(pretrained_config, tokenizer, pretrained, device):
    models = []
    train_dataset = []
    test_dataset = []
    train_ds_maxlen = 0
    test_ds_maxlen = 0
    for model_type, train_file, test_file in zip_longest(input_arg.model, input_arg.train, input_arg.test,
                                                         fillvalue=""):
        model_type = model_type.lower()
        if "once" in model_type:
            train_ds = gen_once.loadOnceDataset(train_file, pretrained=pretrained_config, maxlen=input_arg.maxlen,
                                                cache=input_arg.cache)
            test_ds = gen_once.loadOnceDataset(test_file, pretrained=pretrained_config, maxlen=input_arg.maxlen,
                                               cache=input_arg.cache)
            model = gen_once.Once(tokenizer, pretrained, maxlen=input_arg.maxlen)
        elif "mask" in model_type:
            train_ds = gen_mask.loadMaskDataset(train_file, pretrained_config=pretrained_config,
                                                maxlen=input_arg.maxlen, cache=input_arg.cache,
                                                handle_exceed=input_arg.handle_exceed)
            test_ds = gen_mask.loadMaskDataset(test_file, pretrained_config=pretrained_config, maxlen=input_arg.maxlen,
                                               cache=input_arg.cache, handle_exceed=input_arg.handle_exceed)
            model = gen_mask.Mask(tokenizer, pretrained, maxlen=input_arg.maxlen)
        elif "onebyone" in model_type:
            panel = nlp2.Panel()
            inputted_arg = {"pretrained_config": pretrained_config, "maxlen": input_arg.maxlen,
                            "cache": input_arg.cache, "likelihood": model_type}
            all_arg = nlp2.function_get_all_arg_with_value(gen_onebyone.loadOneByOneDataset)
            if input_arg.enable_arg_panel:
                for missarg in nlp2.function_check_missing_arg(gen_onebyone.loadOneByOneDataset,
                                                               inputted_arg):
                    panel.add_element(k=missarg, v=all_arg[missarg], msg=missarg, default=all_arg[missarg])
                filled_arg = panel.get_result_dict()
                inputted_arg.update(filled_arg)
            train_ds = gen_onebyone.loadOneByOneDataset(train_file, **inputted_arg)
            test_ds = gen_onebyone.loadOneByOneDataset(test_file, **inputted_arg)
            model = gen_onebyone.OneByOne(tokenizer, pretrained, **inputted_arg)
        elif 'clas' in model_type:
            train_ds = classifier.loadClassifierDataset(train_file, pretrained=pretrained_config, cache=input_arg.cache,
                                                        handle_exceed=input_arg.handle_exceed)
            test_ds = classifier.loadClassifierDataset(test_file, pretrained=pretrained_config, cache=input_arg.cache,
                                                       handle_exceed=input_arg.handle_exceed)
            model = classifier.MtClassifier(train_ds.task, tokenizer, pretrained)
        elif 'tag' in model_type:
            if "row" in model_type:
                train_ds = tag.loadRowTaggerDataset(train_file, pretrained=pretrained_config, maxlen=input_arg.maxlen,
                                                    cache=input_arg.cache)
                test_ds = tag.loadRowTaggerDataset(test_file, pretrained=pretrained_config, maxlen=input_arg.maxlen,
                                                   cache=input_arg.cache)
            elif "col" in model_type:
                train_ds = tag.loadColTaggerDataset(train_file, pretrained=pretrained_config, maxlen=input_arg.maxlen,
                                                    cache=input_arg.cache)
                test_ds = tag.loadColTaggerDataset(test_file, pretrained=pretrained_config, maxlen=input_arg.maxlen,
                                                   cache=input_arg.cache)
            model = tag.Tagger(train_ds.label, tokenizer, pretrained, maxlen=input_arg.maxlen)
        elif 'qa' in model_type:
            train_ds = qa.loadQADataset(train_file, pretrained=pretrained_config, cache=input_arg.cache)
            test_ds = qa.loadQADataset(test_file, pretrained=pretrained_config, cache=input_arg.cache)
            model = qa.QA(tokenizer, pretrained, maxlen=input_arg.maxlen)

        model = model.to(device)
        train_ds_maxlen = train_ds.__len__() if train_ds.__len__() > train_ds_maxlen else train_ds_maxlen
        test_ds_maxlen = test_ds.__len__() if test_ds.__len__() > test_ds_maxlen else test_ds_maxlen
        train_dataset.append(train_ds)
        test_dataset.append(test_ds)
        models.append(model)
    return models, train_dataset, test_dataset, train_ds_maxlen, test_ds_maxlen


def main():
    parser = argparse.ArgumentParser()
    exceed_mode = nlp2.function_get_all_arg_with_value(tok.handle_exceed)['mode']
    parser.add_argument("--batch", type=int, default=20, help="batch size, default 20")
    parser.add_argument("--lr", type=float, nargs='+', default=[5e-5], help="learning rate, default 5e-5")
    parser.add_argument("--epoch", type=int, default=10, help="epoch, default 10")
    parser.add_argument("--maxlen", type=int, default=512, help="max tokenized sequence length, default 512")
    parser.add_argument("--handle_exceed", choices=exceed_mode,
                        default=exceed_mode[0],
                        help='select ways to handle input exceed max length')
    parser.add_argument("--savedir", type=str, default="checkpoints/", help="model saving dir, default /checkpoints")
    parser.add_argument("--add_tokens", type=int, default=0,
                        help="auto add freq > x UNK token to word table")
    parser.add_argument("--train", type=str, nargs='+', required=True, help="train dataset path")
    parser.add_argument("--test", type=str, nargs='+', required=True, help="test dataset path")
    parser.add_argument("--model", type=str, required=True, nargs='+',
                        choices=['once', 'twice', 'onebyone', 'clas', 'tagRow', 'tagCol', 'qa',
                                 'onebyone-neg', 'onebyone-pos', 'onebyone-both', 'mask'], help="model task")
    parser.add_argument("--tag", type=str, nargs='+', help="tag to identity task in multi-task")
    parser.add_argument("--config", type=str, default='bert-base-multilingual-cased', required=True,
                        help='distilbert-base-multilingual-cased|voidful/albert_chinese_small')
    parser.add_argument("--seed", type=int, default=609, help="random seed, default 609")
    parser.add_argument("--worker", type=int, default=8, help="number of worker on pre-processing, default 8")
    parser.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation, default 1")
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
    parser.add_argument("--resume", help='resume training')
    parser.add_argument("--cache", action='store_true', help='cache training data')
    parser.add_argument("--enable_arg_panel", action='store_true', help="enable panel to input argument")
    global input_arg
    input_arg = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nlp2.get_dir_with_notexist_create(input_arg.savedir)

    write_log("TRAIN PARAMETER")
    write_log("=======================")
    [write_log(var, ':', vars(input_arg)[var]) for var in vars(input_arg)]
    write_log("=======================")

    # load pre-train model
    pretrained_config = input_arg.config
    if 'albert_chinese' in pretrained_config:
        tokenizer = BertTokenizer.from_pretrained(pretrained_config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_config)
    pretrained = AutoModel.from_pretrained(pretrained_config)

    # handling add tokens
    if input_arg.add_tokens:
        write_log("Calculating Unknown Token")
        add_tokens = get_freqK_unk_token(tokenizer, input_arg.train + input_arg.test, input_arg.add_tokens)
        num_added_toks = tokenizer.add_tokens(add_tokens)
        write_log('We have added', num_added_toks, 'tokens')
        pretrained.resize_token_embeddings(len(tokenizer))
        save_path = os.path.join(input_arg.savedir, pretrained_config + "_added_tok")
        pretrained_config = save_path
        pretrained.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        write_log('New pre-train model saved at ', save_path)
        write_log("=======================")

    # load model and data
    models_tag = input_arg.tag if input_arg.tag is not None else [m.lower() + "_" + str(ind) for ind, m in
                                                                  enumerate(input_arg.model)]
    models, train_dataset, test_dataset, train_ds_maxlen, test_ds_maxlen = _load_model_and_data(pretrained_config,
                                                                                                tokenizer, pretrained,
                                                                                                device)
    # balance sample for multi-task
    for ds in train_dataset:
        ds.increase_with_sampling(train_ds_maxlen)
    for ds in test_dataset:
        ds.increase_with_sampling(test_ds_maxlen)

    train_dataset = [data.DataLoader(dataset=ds,
                                     batch_size=input_arg.batch,
                                     shuffle=True,
                                     # collate_fn=data_collate,
                                     num_workers=input_arg.worker) for ds in train_dataset]
    test_dataset = [data.DataLoader(dataset=ds,
                                    batch_size=input_arg.batch,
                                    shuffle=True,
                                    # collate_fn=data_collate,
                                    num_workers=input_arg.worker) for ds in test_dataset]

    writer = tensorboard.SummaryWriter() if input_arg.tensorboard else None
    start_epoch = 1

    if input_arg.resume:
        write_log("Loading back:", input_arg.resume)
        package = torch.load(input_arg.resume, map_location=device)
        if 'model_state_dict' in package:
            models[0].load_state_dict(package['model_state_dict'])
        else:
            for model_tag, state_dict in zip(package['tags'], package['models']):
                tag_ind = package['tags'].index(model_tag)
                models[tag_ind].load_state_dict(state_dict)
        start_epoch = int(package.get('epoch', 1)) + 1

    set_seed(input_arg.seed)
    write_log("training batch : " + str(input_arg.batch * input_arg.grad_accum))
    for epoch in range(start_epoch, start_epoch + input_arg.epoch):
        fname = os.path.join(input_arg.savedir, str(epoch))

        write_log(f"=========train at epoch={epoch}=========")
        train_avg_loss = model_train(models, train_dataset, models_tag, input_arg, epoch, writer)

        write_log(f"=========save at epoch={epoch}=========")
        save_model = {
            'models': [m.state_dict() for m in models],
            'model_config': input_arg.config,
            'tags': models_tag,
            'type': input_arg.model,
            'maxlen': input_arg.maxlen,
            'epoch': epoch
        }

        for ind, m in enumerate(input_arg.model):
            if 'tag' in m:
                save_model['label'] = models[ind].labels
            if "clas" in m:
                save_model['task-label'] = models[ind].tasks_detail

        torch.save(save_model, f"{fname}.pt")
        write_log(f"weights were saved to {fname}.pt")

        write_log(f"=========eval at epoch={epoch}=========")
        eval_avg_loss = model_eval(models, test_dataset, fname, epoch, writer)

        if input_arg.tensorboard:
            writer.add_scalar("train_loss/epoch", train_avg_loss, epoch)
            writer.add_scalar("eval_loss/epoch", eval_avg_loss, epoch)


if __name__ == "__main__":
    main()
