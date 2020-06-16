import argparse
import torch
import gen_once
import gen_twice
import gen_onebyone
import qa
import classifier
import tag
from tqdm import tqdm
from utility.eval_metric import EvalMetric
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--dumpdir", required=True, type=str)
    arg = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    package = torch.load(arg.model, map_location=device)

    maxlen = package['maxlen']
    type = package['type']
    config = package['model_config'] if 'model_config' in package else package['bert']
    type = type.lower()

    print("===model info===")
    print("maxlen", maxlen)
    print("type", type)
    print("pretrain", config)
    print('==========')

    if "once" in type:
        model = gen_once.Once(model_config=config, maxlen=maxlen)
    elif "twice" in type:
        model = gen_twice.Twice(model_config=config, maxlen=maxlen)
    elif "onebyone" in type:
        model = gen_onebyone.OneByOne(model_config=config, maxlen=maxlen)
    elif 'clas' in type:
        model = classifier.MtClassifier(package['task'], model_config=config)
    elif 'tag' in type:
        model = tag.Tagger(package['label'], model_config=config, maxlen=maxlen)
    elif 'qa' in type:
        model = qa.QA(model_config=config, maxlen=maxlen)

    model = model.to(device)
    model.load_state_dict(package['model_state_dict'], strict=False)
    model.pretrained.save_pretrained(arg.dumpdir)
    print('==================')
    print("Finish model dump.")


if __name__ == "__main__":
    main()
