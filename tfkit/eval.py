from transformers import *
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
    parser.add_argument("--metric", required=True, type=str, choices=['emf1', 'nlg', 'classification'])
    parser.add_argument("--valid", required=True, type=str, nargs='+')
    parser.add_argument("--tag", type=str)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--print", action='store_true')
    parser.add_argument("--outfile", action='store_true')
    parser.add_argument("--beamsearch", action='store_true')
    parser.add_argument("--beamsize", type=int, default=3)
    parser.add_argument("--beamfiltersim", action='store_true')
    parser.add_argument("--topP", type=int, default=1)
    parser.add_argument("--topK", type=float, default=0.6)
    arg = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    package = torch.load(arg.model, map_location=device)

    maxlen = package['maxlen']
    config = package['model_config'] if 'model_config' in package else package['bert']
    model_types = package['type']
    model_types = [model_types] if not isinstance(model_types, list) else model_types
    models_state = package['models'] if 'models' in package else [package['model_state_dict']]
    models_tag = package['tags'] if 'tags' in package else model_types

    # load pre-train model
    if 'albert_chinese' in config:
        tokenizer = BertTokenizer.from_pretrained(config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config)
    pretrained = AutoModel.from_pretrained(config)

    if arg.tag is not None and arg.tag not in models_tag:
        print("tag must select from models tag: ", models_tag)
        raise ValueError("tag must select from models tag")

    if arg.tag is None:
        tag_ind = 0
    else:
        tag_ind = models_tag.index(arg.tag)

    valid = arg.valid[0]
    model_state = models_state[tag_ind]
    model_type = model_types[tag_ind]

    print("===model info===")
    print("maxlen", maxlen)
    print("type", model_type)
    print("tag", arg.tag)
    print("valid", valid)
    print('==========')

    if "once" in model_type:
        eval_dataset = gen_once.get_data_from_file(valid)
        model = gen_once.Once(tokenizer, pretrained, maxlen=maxlen)
    elif "twice" in model_type:
        eval_dataset = gen_once.get_data_from_file(valid)
        model = gen_twice.Twice(tokenizer, pretrained, maxlen=maxlen)
    elif "onebyone" in model_type:
        eval_dataset = gen_once.get_data_from_file(valid)
        model = gen_onebyone.OneByOne(tokenizer, pretrained, maxlen=maxlen)
    elif 'classify' in model_type:
        eval_dataset = classifier.get_data_from_file(valid)
        model = classifier.MtClassifier(package['task'], tokenizer, pretrained)
    elif 'tag' in model_type:
        if "row" in model_type:
            eval_dataset = tag.get_data_from_file_row(valid)
        elif "col" in model_type:
            eval_dataset = tag.get_data_from_file_col(valid)
        model = tag.Tagger(package['label'], tokenizer, pretrained, maxlen=maxlen)
    elif 'qa' in model_type:
        eval_dataset = qa.get_data_from_file(valid)
        model = qa.QA(tokenizer, pretrained, maxlen=maxlen)

    model.load_state_dict(model_state, strict=False)
    model = model.to(device)

    if not arg.beamsearch:
        eval_metrics = [EvalMetric(tokenizer)]
    else:
        eval_metrics = [EvalMetric(tokenizer) for _ in range(arg.beamsize)]

    for i in tqdm(eval_dataset):
        tasks = i[0]
        task = i[1]
        input = i[2]
        target = i[3]

        predict_param = {'input': input, 'task': task}
        if arg.beamsearch and 'onebyone' in model_type:
            predict_param['beamsearch'] = True
            predict_param['beamsize'] = arg.beamsize
            predict_param['filtersim'] = arg.beamfiltersim
        elif 'onebyone' in model_type:
            predict_param['topP'] = arg.topP
            predict_param['topK'] = arg.topK

        result, result_dict = model.predict(**predict_param)
        for eval_pos, eval_metric in enumerate(eval_metrics):
            if 'qa' in model_type:
                target = " ".join(input.split(" ")[int(target[0]): int(target[1])])
            if 'onebyone' in model_type and arg.beamsearch:
                predicted = result_dict['label_map'][eval_pos][0]
            elif 'tag' in model_type:
                predicted = " ".join([list(d.values())[0] for d in result_dict['label_map']])
            else:
                predicted = result[0] if len(result) > 0 else ''

            if arg.print:
                print('===eval===')
                print("input: ", input)
                print("target: ", target)
                print("predicted: ", predicted)
                print('==========')

            eval_metric.add_record(input, predicted, target)

    for eval_pos, eval_metric in enumerate(eval_metrics):
        if arg.outfile:
            argtype = "_dataset_" + valid.replace("/", "_").replace(".", "")
            if arg.beamsearch:
                argtype = "_beam_" + str(eval_pos)
            outfile_name = arg.model + argtype

            with open(outfile_name + "_predicted.csv", "w", encoding='utf8') as f:
                writer = csv.writer(f)
                records = eval_metric.get_record()
                for i, p in zip(records['input'], records['predicted']):
                    writer.writerow([i, p])
            print("write result at:", outfile_name)

            with open(outfile_name + "_score.csv", "w", encoding='utf8') as f:
                for i in eval_metric.cal_score(arg.metric):
                    f.write("TASK: " + str(i[0]) + " , " + str(eval_pos) + '\n')
                    f.write(str(i[1]) + '\n')
            print("write score at:", outfile_name)

        for i in eval_metric.cal_score(arg.metric):
            print("TASK: ", i[0], eval_pos)
            print(i[1])


if __name__ == "__main__":
    main()
