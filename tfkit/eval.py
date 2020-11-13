import argparse
import sys

from tqdm import tqdm
import csv
from tfkit.utility.eval_metric import EvalMetric
from tfkit.utility.model_loader import load_trained_model, load_predict_parameter


def parse_eval_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="model path")
    parser.add_argument("--config", type=str, help='pre-trained model path after add token')
    parser.add_argument("--metric", required=True, type=str, choices=['emf1', 'nlg', 'clas'], help="evaluate metric")
    parser.add_argument("--valid", required=True, type=str, nargs='+', help="evaluate data path")
    parser.add_argument("--print", action='store_true', help="print each pair of evaluate data")
    parser.add_argument("--panel", action='store_true', help="enable panel to input argument")
    return vars(parser.parse_args(args))


def main(arg=None):
    eval_arg = parse_eval_args(sys.argv[1:]) if arg is None else parse_eval_args(arg)

    valid = eval_arg.get('valid')[0]
    model, model_type, model_class = load_trained_model(eval_arg.get('model'), pretrained_config=eval_arg.get('config'))
    eval_dataset = model_class.get_data_from_file(valid)
    predict_parameter = load_predict_parameter(model, eval_arg.get('panel'))

    if 'decodenum' in predict_parameter and predict_parameter['decodenum'] > 1:
        eval_metrics = [EvalMetric(model.tokenizer) for _ in range(predict_parameter['decodenum'])]
    else:
        eval_metrics = [EvalMetric(model.tokenizer)]

    print("PREDICT PARAMETER")
    print("=======================")
    print(predict_parameter)
    print("=======================")
    for i in tqdm(eval_dataset):
        tasks = i[0]
        task = i[1]
        input = i[2]
        target = i[3]

        predict_parameter.update({'input': input})
        if 'task' not in predict_parameter:
            predict_parameter.update({'task': task})
        result, result_dict = model.predict(**predict_parameter)
        for eval_pos, eval_metric in enumerate(eval_metrics):
            # predicted can be list of string or string
            # target should be list of string
            predicted = result
            if 'qa' in model_type:
                target = " ".join(input.split(" ")[int(target[0]): int(target[1])])
                if len(result) > 0:
                    predicted = result[0][0] if isinstance(result[0], list) else result[0]
                else:
                    predicted = ''
            elif 'onebyone' in model_type:
                target = " ".join(target[0])
                if len(result) < eval_pos:
                    print("Decode size smaller than decode num:", result_dict['label_map'])
                predicted = result[eval_pos]
            elif 'mask' in model_type:
                target = target[0].split(" ")
                predicted = result
            elif 'tag' in model_type:
                predicted = " ".join([list(d.values())[0] for d in result_dict[0]['label_map']])
                target = target[0].split(" ")
                predicted = predicted.split(" ")

            if eval_arg.get('print'):
                print('===eval===')
                print("input: ", input)
                print("target: ", target)
                print("predicted: ", predicted)
                print('==========')

            eval_metric.add_record(input, predicted, target)

    for eval_pos, eval_metric in enumerate(eval_metrics):
        argtype = "_dataset" + valid.replace("/", "_").replace(".", "")
        if 'decodenum' in predict_parameter and predict_parameter['decodenum'] > 1:
            argtype += "_num_" + str(eval_pos)
        if 'mode' in predict_parameter:
            para_mode = predict_parameter['mode'][0] if isinstance(predict_parameter['mode'], list) else \
                predict_parameter['mode'].lower()
            argtype += "_mode_" + str(para_mode)
        if 'filtersim' in predict_parameter:
            argtype += "_filtersim_" + str(predict_parameter['filtersim'])
        outfile_name = eval_arg.get('model') + argtype

        with open(outfile_name + "_predicted.csv", "w", encoding='utf8') as f:
            writer = csv.writer(f)
            records = eval_metric.get_record()
            writer.writerow(['input', 'predicted', 'targets'])
            for i, p, t in zip(records['input'], records['predicted'], records['targets']):
                writer.writerow([i, p, "[SEP]".join([onet for onet in t if len(onet) > 0])])
        print("write result at:", outfile_name)

        with open(outfile_name + "_each_data_score.csv", "w", encoding='utf8') as edsf:
            eds = csv.writer(edsf)
            with open(outfile_name + "_score.csv", "w", encoding='utf8') as f:
                for i in eval_metric.cal_score(eval_arg.get('metric')):
                    f.write("TASK: " + str(i[0]) + " , " + str(eval_pos) + '\n')
                    f.write(str(i[1]) + '\n')
                    eds.writerows(i[2])

        print("write score at:", outfile_name)

        for i in eval_metric.cal_score(eval_arg.get('metric')):
            print("TASK: ", i[0], eval_pos)
            print(i[1])


if __name__ == "__main__":
    main()
