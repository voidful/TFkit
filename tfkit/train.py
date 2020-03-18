import argparse
import tensorboardX as tensorboard
from gen_once import *
from gen_twice import *
from gen_onebyone import *
from classifier import *
from tag import *
from qa import *
from torch.utils import data
from tqdm import tqdm
from utility.optim import *


def write_log(*args):
    line = ' '.join([str(a) for a in args])
    with open(os.path.join(arg.savedir, "message.log"), "a", encoding='utf8') as log_file:
        log_file.write(line + '\n')
    print(line)


def optimizer(model, arg):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    # optimizer = torch.optim.Adamax(optimizer_parameters, lr=arg.lr)
    # optimizer = torch.optim.Adamax(model.parameters(), lr=arg.lr)
    optimizer = AdamW(optimizer_parameters, lr=arg.lr)
    return optimizer


def train(model, iterator, arg, fname, epoch):
    model = nn.DataParallel(model)
    optim = optimizer(model, arg)
    t_loss = 0
    model.train()
    for i, batch in tqdm(enumerate(iterator)):
        loss = model(batch)
        optim.zero_grad()
        loss.mean().backward()
        optim.step()
        t_loss += loss.mean().item()

        if arg.tensorboard:
            writer.add_scalar("loss/step", loss.mean().item(), epoch)
        if i % 100 == 0 and i != 0:  # monitoring
            write_log(f"step: {i}, loss: {t_loss / (i + 1)}, total: {len(iterator)}")

    write_log(f"step: {len(iterator)}, loss: {t_loss / len(iterator)}, total: {len(iterator)}")
    return t_loss / len(iterator)


def eval(model, iterator, fname, epoch):
    model.eval()
    t_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            loss = model(batch)
            t_loss += loss.mean().item()
    avg_t_loss = t_loss / len(iterator)
    write_log(f"model: {fname}, Total Loss: {avg_t_loss}")
    if arg.tensorboard:
        writer.add_scalar("eval_loss/step", avg_t_loss, epoch)
    return avg_t_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--maxlen", type=int, default=368)
    parser.add_argument("--savedir", type=str, default="checkpoints/")
    parser.add_argument("--train", type=str, nargs='+', default="train.csv", required=True)
    parser.add_argument("--valid", type=str, nargs='+', default="valid.csv", required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=['once', 'twice', 'onebyone', 'classify', 'tagRow', 'tagCol', 'qa'])
    parser.add_argument("--neg", type=str, choices=['token', 'sent', 'both'], help='gen onebyone unlikelihood loss')
    parser.add_argument("--config", type=str, default='bert-base-multilingual-cased', required=True,
                        help='distilbert-base-multilingual-cased/bert-base-multilingual-cased/bert-base-chinese')
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
    parser.add_argument("--resume", help='resume training')
    parser.add_argument("--cache", action='store_true', help='Caching training data')
    global arg
    arg = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(arg.savedir): os.makedirs(arg.savedir)

    arg.model = arg.model.lower()
    if "once" in arg.model:
        train_dataset = loadOnceDataset(arg.train[0], pretrained=arg.config, maxlen=arg.maxlen, cache=arg.cache)
        eval_dataset = loadOnceDataset(arg.valid[0], pretrained=arg.config, maxlen=arg.maxlen, cache=arg.cache)
        model = BertOnce(model_config=arg.config, maxlen=arg.maxlen)
    if "twice" in arg.model:
        train_dataset = loadOnceDataset(arg.train[0], pretrained=arg.config, maxlen=arg.maxlen, cache=arg.cache)
        eval_dataset = loadOnceDataset(arg.valid[0], pretrained=arg.config, maxlen=arg.maxlen, cache=arg.cache)
        model = BertTwice(model_config=arg.config, maxlen=arg.maxlen)
    elif "onebyone" in arg.model:
        neg_token = False
        neg_sent = False
        if arg.neg == 'token':
            neg_token = True
        elif arg.neg == 'sent':
            neg_sent = True
        elif arg.neg == 'both':
            neg_token = True
            neg_sent = True
        train_dataset = loadOneByOneDataset(arg.train[0], pretrained=arg.config, maxlen=arg.maxlen, cache=arg.cache,
                                            neg_token=neg_token, neg_sent=neg_sent)
        eval_dataset = loadOneByOneDataset(arg.valid[0], pretrained=arg.config, maxlen=arg.maxlen, cache=arg.cache)
        model = BertOneByOne(model_config=arg.config, maxlen=arg.maxlen)
    elif 'classify' in arg.model:
        train_dataset = loadClassifierDataset(arg.train[0], pretrained=arg.config, cache=arg.cache)
        eval_dataset = loadClassifierDataset(arg.valid[0], pretrained=arg.config, cache=arg.cache)
        model = BertMtClassifier(train_dataset.task, arg.config)
    elif 'tag' in arg.model:
        if "row" in arg.model:
            train_dataset = loadRowTaggerDataset(arg.train[0], pretrained=arg.config, maxlen=arg.maxlen,
                                                 cache=arg.cache)
            eval_dataset = loadRowTaggerDataset(arg.valid[0], pretrained=arg.config, maxlen=arg.maxlen, cache=arg.cache)
        elif "col" in arg.model:
            train_dataset = loadColTaggerDataset(arg.train[0], pretrained=arg.config, maxlen=arg.maxlen,
                                                 cache=arg.cache)
            eval_dataset = loadColTaggerDataset(arg.valid[0], pretrained=arg.config, maxlen=arg.maxlen, cache=arg.cache)
        model = BertTagger(train_dataset.label, arg.config, maxlen=arg.maxlen)
    elif 'qa' in arg.model:
        train_dataset = loadQADataset(arg.train[0], pretrained=arg.config, cache=arg.cache)
        eval_dataset = loadQADataset(arg.valid[0], pretrained=arg.config, cache=arg.cache)
        model = BertQA(model_config=arg.config, maxlen=arg.maxlen)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=arg.batch,
                                 shuffle=True,
                                 num_workers=arg.worker)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=arg.batch,
                                shuffle=False,
                                num_workers=arg.worker)
    if arg.tensorboard:
        global writer
        writer = tensorboard.SummaryWriter()

    model = model.to(device)
    start_epoch = 1
    if arg.resume:
        package = torch.load(arg.resume, map_location=device)
        model.load_state_dict(package['model_state_dict'])
        start_epoch = int(package.get('epoch', 1))

    write_log("batch : " + str(arg.batch))
    for epoch in range(start_epoch, start_epoch + arg.epoch):
        fname = os.path.join(arg.savedir, str(epoch))

        write_log(f"=========train at epoch={epoch}=========")
        train_avg_loss = train(model, train_iter, arg, fname, epoch)

        write_log(f"=========save at epoch={epoch}=========")
        save_model = {
            'model_state_dict': model.state_dict(),
            'model_config': arg.config,
            'type': arg.model,
            'maxlen': model.maxlen
        }
        if 'classify' in arg.model:
            save_model['task'] = train_dataset.task
        elif 'tag' in arg.model:
            save_model['label'] = model.labels
        torch.save(save_model, f"{fname}.pt")
        write_log(f"weights were saved to {fname}.pt")

        write_log(f"=========eval at epoch={epoch}=========")
        eval_avg_loss = eval(model, eval_iter, fname, epoch)

        if arg.tensorboard:
            writer.add_scalar("train_loss/epoch", train_avg_loss, epoch)
            writer.add_scalar("eval_loss/epoch", eval_avg_loss, epoch)


if __name__ == "__main__":
    main()
