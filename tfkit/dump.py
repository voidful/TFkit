import argparse
import torch
from transformers import BertTokenizer, AutoTokenizer, AutoModel

import tfkit.gen_once as gen_once
import tfkit.gen_onebyone as gen_onebyone
import tfkit.qa as qa
import tfkit.classifier as classifier
import tfkit.tag as tag
import tfkit.gen_mask as mask
import nlp2go

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--dumpdir", required=True, type=str)
    arg = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    package = torch.load(arg.model, map_location=device)

    maxlen = package['maxlen']
    model_type = (package['type'] if isinstance(package['type'], str) else package['type'])[0]
    config = package['model_config'] if 'model_config' in package else package['bert']
    models_state = (package['models'] if 'models' in package else [package['model_state_dict']])[0]
    model_type = model_type.lower()

    print("===model info===")
    print("maxlen", maxlen)
    print("model_type", model_type)
    print("pretrain", config)
    print('==========')

    if 'albert_chinese' in config:
        tokenizer = BertTokenizer.from_pretrained(config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config)
    pretrained = AutoModel.from_pretrained(config)

    if "once" in model_type:
        model = gen_once.Once(tokenizer, pretrained, maxlen=maxlen)
    elif "onebyone" in model_type:
        model = gen_onebyone.OneByOne(tokenizer, pretrained, maxlen=maxlen)
    elif 'clas' in model_type:
        model = classifier.MtClassifier(package['task-label'], tokenizer, pretrained, maxlen=maxlen)
    elif 'tag' in model_type:
        model = tag.Tagger(package['label'], tokenizer, pretrained, maxlen=maxlen)
    elif 'qa' in model_type:
        model = qa.QA(tokenizer, pretrained, maxlen=maxlen)
    elif 'mask' in model_type:
        model = mask.Mask(tokenizer, pretrained, maxlen=maxlen)

    model = model.to(device)
    model.load_state_dict(models_state, strict=False)
    model.pretrained.save_pretrained(arg.dumpdir)
    print('==================')
    print("Finish model dump.")


if __name__ == "__main__":
    main()
