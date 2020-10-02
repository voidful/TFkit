import argparse
import torch
import tfkit.gen_once as gen_once
import tfkit.gen_onebyone as gen_onebyone
import tfkit.qa as qa
import tfkit.classifier as classifier
import tfkit.tag as tag
import tfkit.gen_mask as mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--dumpdir", required=True, type=str)
    arg = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    package = torch.load(arg.model, map_location=device)

    maxlen = package['maxlen']
    model_type = package['type']
    config = package['model_config'] if 'model_config' in package else package['bert']
    model_type = model_type.lower()

    print("===model info===")
    print("maxlen", maxlen)
    print("model_type", model_type)
    print("pretrain", config)
    print('==========')

    if "once" in model_type:
        model = gen_once.Once(model_config=config, maxlen=maxlen)
    elif "onebyone" in model_type:
        model = gen_onebyone.OneByOne(model_config=config, maxlen=maxlen)
    elif 'clas' in model_type:
        model = classifier.MtClassifier(package['task'], model_config=config)
    elif 'tag' in model_type:
        model = tag.Tagger(package['label'], model_config=config, maxlen=maxlen)
    elif 'qa' in model_type:
        model = qa.QA(model_config=config, maxlen=maxlen)
    elif 'mask' in model_type:
        model = mask.Mask(model_config=config, maxlen=maxlen)

    model = model.to(device)
    model.load_state_dict(package['model_state_dict'], strict=False)
    model.pretrained.save_pretrained(arg.dumpdir)
    print('==================')
    print("Finish model dump.")


if __name__ == "__main__":
    main()
