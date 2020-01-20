import argparse
import csv

import torch
import gen_once
import gen_onebyone
import classifier
import tag
from tqdm import tqdm
from utility.eval_metric import EvalMetric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--valid", required=True, type=str)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--type", type=str, choices=['once', 'onebyone', 'classify', 'tagRow', 'tagCol'])
    parser.add_argument("--metric", required=True, type=str, choices=['em', 'nlg', 'classification'])
    parser.add_argument("--print", action='store_true')
    parser.add_argument("--outfile", action='store_true')
    parser.add_argument("--beamsearch", action='store_true')
    parser.add_argument("--beamsize", type=int, default=3)
    parser.add_argument("--beamselect", type=int, default=0)
    parser.add_argument("--beamfiltersim", action='store_true')
    arg = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    package = torch.load(arg.model, map_location=device)

    maxlen = package['maxlen']
    type = arg.type if arg.type else package['type']
    config = package['model_config'] if 'model_config' in package else package['bert']
    type = type.lower()

    print("===model info===")
    print("maxlen", maxlen)
    print("type", type)
    print('==========')

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
    argtype = ""
    if arg.beamsearch:
        argtype = "_beam_" + str(arg.beamselect)
        if arg.beamfiltersim:
            argtype += "_filtersim_"
    outfile_name = arg.model + argtype + ".out"
    outfile_arr = []

    for i in tqdm(eval_dataset):
        tasks = i[0]
        task = i[1]
        input = i[2]
        target = i[3]
        if arg.beamsearch:
            result, possible = model.predict_beamsearch(input, topk=arg.beamsize, filtersim=arg.beamfiltersim)
            result = possible[arg.beamselect][0]
        else:
            result, outprob = model.predict(task=task, input=input)
        if arg.print:
            print('===eval===')
            print("input: ", input)
            print("target: ", target)
            print("result: ", result)
            if arg.beamsearch:
                print("possible: ", possible)
            print('==========')
        if arg.outfile:
            outfile_arr.append([input, target, result])
        eval_metric.add_record(result, target)

    if arg.outfile:
        with open(outfile_name, "w", encoding='utf8') as f:
            writer = csv.writer(f)
            for r in outfile_arr:
                writer.writerow(r)

    for i in eval_metric.cal_score(arg.metric):
        print("TASK: ", i[0])
        print(i[1])


if __name__ == "__main__":
    main()
