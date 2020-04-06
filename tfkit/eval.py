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
    parser.add_argument("--valid", required=True, type=str)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--type", type=str, choices=['once', 'onebyone', 'classify', 'tagRow', 'tagCol', 'qa'])
    parser.add_argument("--metric", required=True, type=str, choices=['emf1', 'nlg', 'classification'])
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
    if not arg.beamsearch:
        eval_metrics = [EvalMetric()]
    else:
        eval_metrics = [EvalMetric() for _ in range(arg.beamsize)]

    for i in tqdm(eval_dataset):
        tasks = i[0]
        task = i[1]
        input = i[2]
        target = i[3]

        predict_param = {'input': input, 'task': task}
        if arg.beamsearch and 'onebyone' in type:
            predict_param['beamsearch'] = True
            predict_param['beamsize'] = arg.beamsize
            predict_param['filtersim'] = arg.beamfiltersim
        elif 'onebyone' in type:
            predict_param['topP'] = arg.topP
            predict_param['topK'] = arg.topK

        result, result_dict = model.predict(**predict_param)

        for eval_pos, eval_metric in enumerate(eval_metrics):
            if 'onebyone' in type and arg.beamsearch:
                predicted = result_dict['label_map'][eval_pos][0]
            elif 'classify' in type:
                predicted = result
            elif 'tag' in type:
                predicted = [list(d.values())[0] for d in result_dict['label_map']]

            elif 'qa' in type:
                target = " ".join(input.split(" ")[int(target[0]): int(target[1])])
            else:
                predicted = result[0] if len(result) > 0 else ''

            if arg.print:
                print('===eval===')
                print("input: ", input)
                print("target: ", target)
                print("result: ", result)
                print('==========')

            eval_metric.add_record(input, predicted, target)

    for eval_pos, eval_metric in enumerate(eval_metrics):
        if arg.outfile:
            argtype = "_dataset-" + arg.valid.replace("/", "_").replace(".", "_")
            if arg.beamsearch:
                argtype = "_beam_" + str(eval_pos)
            outfile_name = arg.model + argtype
            with open(outfile_name + "_predicted.csv", "w", encoding='utf8') as f:
                writer = csv.writer(f)
                records = eval_metric.get_record()
                for i, p in zip(records['input'], records['predicted']):
                    writer.writerow([i, p])
            print("write file at:", outfile_name)

        for i in eval_metric.cal_score(arg.metric):
            print("TASK: ", i[0], eval_pos)
            print(i[1])


if __name__ == "__main__":
    main()
