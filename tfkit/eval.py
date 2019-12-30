import argparse
import torch
import gen_once
import gen_onebyone
import classifier
import tag
from utility.eval_metric import EvalMetric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--valid", type=str)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--type", type=str, choices=['once', 'onebyone', 'classify', 'tagRow', 'tagCol'])
    parser.add_argument("--metric", type=str, choices=['em', 'nlg', 'classification'])
    parser.add_argument("--outprint", action='store_true')
    parser.add_argument("--beamsearch", action='store_true')
    parser.add_argument("--topk", type=int, default=1)
    arg = parser.parse_args()
    arg.type = arg.type.lower()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    package = torch.load(arg.model, map_location=device)

    maxlen = package['maxlen']
    type = arg.type if arg.type else package['type']
    config = package['model_config'] if 'model_config' in package else package['bert']

    if "once" in type:
        eval_dataset = gen_once.get_data_from_file(arg.valid)
        model = gen_once.BertOnce(model_config=config, maxlen=maxlen)
    elif "onebyone" in type:
        eval_dataset = gen_once.get_data_from_file(arg.valid)
        model = gen_onebyone.BertOneByOne(model_config=config, maxlen=maxlen)
    elif 'classify' in type:
        eval_dataset = classifier.get_data_from_file(arg.valid)
        model = classifier.BertMtClassifier(package['task'], model_config=config)
    elif 'tag' in type:
        if "row" in type:
            eval_dataset = tag.get_data_from_file_row(arg.valid)
        elif "col" in type:
            eval_dataset = tag.get_data_from_file_col(arg.valid)
        model = tag.BertTagger(package['label'], model_config=config, maxlen=maxlen)

    model = model.to(device)
    model.load_state_dict(package['model_state_dict'], strict=False)

    eval_metric = EvalMetric()
    for i in eval_dataset:
        tasks = i[0]
        task = i[1]
        input = i[2]
        target = i[3]
        result, outprob = model.predict(task=task, input=input)
        print(input, target, result)
        eval_metric.add_record(result, target)

    print(list(eval_metric.cal_score(arg.metric, arg)))
