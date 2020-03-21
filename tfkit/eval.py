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
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--valid", required=True, type=str)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--type", type=str, choices=['once', 'onebyone', 'classify', 'tagRow', 'tagCol', 'qa'])
    parser.add_argument("--metric", required=True, type=str, choices=['emf1', 'nlg', 'classification'])
    parser.add_argument("--print", action='store_true')
    parser.add_argument("--outfile", action='store_true')
    parser.add_argument("--beamsearch", action='store_true')
    parser.add_argument("--beamsize", type=int, default=3)
    parser.add_argument("--beamselect", type=int, default=0)
    parser.add_argument("--beamfiltersim", action='store_true')
    parser.add_argument("--topP", type=int, default=1)
    parser.add_argument("--topK", type=float, default=0.6)
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
    elif "twice" in type:
        eval_dataset = gen_once.get_data_from_file(arg.valid)
        model = gen_twice.BertTwice(model_config=config, maxlen=maxlen)
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
    elif 'qa' in type:
        eval_dataset = qa.get_data_from_file(arg.valid)
        model = qa.BertQA(model_config=config, maxlen=maxlen)

    model = model.to(device)
    model.load_state_dict(package['model_state_dict'], strict=False)

    prob_list = []
    eval_metric = EvalMetric()
    for i in tqdm(eval_dataset):
        tasks = i[0]
        task = i[1]
        input = i[2]
        target = i[3]
        if arg.beamsearch:
            result, result_dict = model.predict(input, beamsearch=True, beamsize=arg.beamsize,
                                                filtersim=arg.beamfiltersim, topP=topP, topK=topK)
            result = [result_dict['label_map'][arg.beamselect][0]]
            result_dict = "NONE"
        else:
            result, result_dict = model.predict(task=task, input=input)

        if 'qa' in type:
            target = " ".join(input.split(" ")[int(target[0]): int(target[1])])

        if 'prob_list' in result_dict:
            for plist in result_dict['prob_list']:
                prob_list.extend(plist)

        if arg.print:
            print('===eval===')
            print("input: ", input)
            print("target: ", target)
            print("result: ", result)
            # print("result_dict: ", result_dict)
            print('==========')
        if 'classify' in type:
            predicted = result
        elif 'tag' in type:
            predicted = [list(d.values())[0] for d in result_dict['label_map']]
        else:
            predicted = result[0] if len(result) > 0 else ''

        eval_metric.add_record(predicted, target)

    if arg.outfile:
        argtype = ""
        if arg.beamsearch:
            argtype = "_beam_" + str(arg.beamselect)
        outfile_name = arg.model + argtype
        plt.yscale('log', basey=2)
        plt.plot(np.mean(prob_list, axis=0))
        plt.savefig(outfile_name + '_prob_dist.png')
        with open(outfile_name + ".out", "w", encoding='utf8') as f:
            for output in eval_metric.get_record():
                f.write(output + "\n")
        print("write file at:", outfile_name)

    for i in eval_metric.cal_score(arg.metric):
        print("TASK: ", i[0])
        print(i[1])


if __name__ == "__main__":
    main()
