import inspect

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
import inquirer


def load_model(model_path, model_type=None, model_dataset=None):
    """load model from dumped file"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torchpack = torch.load(model_path, map_location=device)

    print("===model info===")
    [print(key, ':', torchpack[key]) for key in torchpack.keys() if 'state_dict' not in key and 'models' not in key]
    print('==========')

    if 'tags' in torchpack and torchpack['tags'] > 1:
        if model_type is None:
            print("Pick which models to use in multi-task models")
            inquirer_res = inquirer.prompt(
                [inquirer.List('model_type', message="Select model", choices=torchpack['tags'])])
            model_type = inquirer_res['model_type']
        type_ind = torchpack['tags'].index(model_type)
    else:
        type_ind = 0

    print("loading model from dumped file")
    # get all loading parameter
    maxlen = torchpack['maxlen']
    config = torchpack['model_config'] if 'model_config' in torchpack else torchpack['bert']
    model_types = [torchpack['type']] if not isinstance(torchpack['type'], list) else torchpack['type']
    models_state = torchpack['models'] if 'models' in torchpack else [torchpack['model_state_dict']]
    type = model_types[type_ind] if model_type is None else model_type

    # load model
    if 'albert_chinese' in config:
        tokenizer = BertTokenizer.from_pretrained(config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config)
    pretrained = AutoModel.from_pretrained(config)

    if "once" in type:
        eval_dataset = gen_once.get_data_from_file(model_dataset) if model_dataset else None
        model = gen_once.Once(tokenizer, pretrained, maxlen=maxlen)
    elif "twice" in type:
        eval_dataset = gen_once.get_data_from_file(model_dataset) if model_dataset else None
        model = gen_twice.Twice(tokenizer, pretrained, maxlen=maxlen)
    elif "onebyone" in type:
        eval_dataset = gen_once.get_data_from_file(model_dataset) if model_dataset else None
        model = gen_onebyone.OneByOne(tokenizer, pretrained, maxlen=maxlen)
    elif 'classify' in type or 'clas' in type:
        eval_dataset = classifier.get_data_from_file(model_dataset) if model_dataset else None
        model = classifier.MtClassifier(torchpack['task'], tokenizer, pretrained)
    elif 'tag' in type:
        if model_dataset and "row" in type:
            eval_dataset = tag.get_data_from_file_row(model_dataset)
        elif model_dataset and "col" in type:
            eval_dataset = tag.get_data_from_file_col(model_dataset)
        else:
            eval_dataset = None
        model = tag.Tagger(torchpack['label'], tokenizer, pretrained, maxlen=maxlen)
    elif 'qa' in type:
        eval_dataset = qa.get_data_from_file(model_dataset) if model_dataset else None
        model = qa.QA(tokenizer, pretrained, maxlen=maxlen)

    model = model.to(device)
    model.load_state_dict(models_state[type_ind], strict=False)

    print("finish loading")
    if model_dataset:
        return model, eval_dataset
    else:
        return model


def load_predict_parameter(model, use_default=False):
    """use inquirer panel to let user input model parameter or just use default value"""

    print("Input parameter for predict function")
    arg_len = len(inspect.getfullargspec(model.predict).args)
    def_len = len(inspect.getfullargspec(model.predict).defaults)
    arg_w_def = zip(inspect.getfullargspec(model.predict).args[arg_len - def_len:],
                    inspect.getfullargspec(model.predict).defaults)

    inquirer_list = []
    for k, v in arg_w_def:
        if v is not None:
            if callable(v):
                msg = k
                inquirer_list.append(inquirer.List(k, message=msg, choices=v(model)))
            elif isinstance(v, list):
                msg = k
                inquirer_list.append(inquirer.List(k, message=msg, choices=v))
            elif isinstance(v, bool):
                msg = k
                inquirer_list.append(inquirer.List(k, message=msg, choices=[True, False]))
            else:
                if isinstance(v, float) and 0 < v < 1:  # probability
                    msg = k + " (between 0-1)"
                elif isinstance(v, float) or isinstance(v, int):  # number
                    msg = k + " (number)"
                else:
                    msg = k
                inquirer_list.append(inquirer.Text(k, message=msg, default=v))
    predict_parameter = inquirer.prompt(inquirer_list)

    if use_default:
        return dict(arg_w_def)
    else:
        return predict_parameter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--metric", required=True, type=str, choices=['emf1', 'nlg', 'clas'])
    parser.add_argument("--valid", required=True, type=str, nargs='+')
    parser.add_argument("--tag", type=str)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--print", action='store_true')
    parser.add_argument("--beamsearch", action='store_true')
    parser.add_argument("--beamsize", type=int, default=3)
    parser.add_argument("--beamfiltersim", action='store_true')
    parser.add_argument("--topP", type=int, default=1)
    parser.add_argument("--topK", type=float, default=0.6)
    arg = parser.parse_args()

    valid = arg.valid[0]
    model, eval_dataset = load_model(arg.model, model_dataset=valid)

    if not arg.beamsearch:
        eval_metrics = [EvalMetric(model.tokenizer)]
    else:
        eval_metrics = [EvalMetric(model.tokenizer) for _ in range(arg.beamsize)]

    predict_parameter = load_predict_parameter(model)
    for i in tqdm(eval_dataset):
        tasks = i[0]
        task = i[1]
        input = i[2]
        target = i[3]

        predict_parameter.update({'input': input, 'task': task})
        result, result_dict = model.predict(**predict_parameter)
        for eval_pos, eval_metric in enumerate(eval_metrics):
            if 'QA' in model.__class__.__name__:
                target = " ".join(input.split(" ")[int(target[0]): int(target[1])])
            if 'OneByOne' in model.__class__.__name__ and arg.beamsearch:
                predicted = result_dict['label_map'][eval_pos][0]
            elif 'Tagger' in model.__class__.__name__:
                predicted = " ".join([list(d.values())[0] for d in result_dict['label_map']])
                target = target.split(" ")
                predicted = predicted.split(" ")
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

        argtype = "_dataset_" + valid.replace("/", "_").replace(".", "")
        if arg.beamsearch:
            argtype = "_beam_" + str(eval_pos)
        outfile_name = arg.model + argtype

        with open(outfile_name + "_predicted.csv", "w", encoding='utf8') as f:
            writer = csv.writer(f)
            records = eval_metric.get_record()
            writer.writerow(['input', 'predicted', 'targets'])
            for i, p, t in zip(records['input'], records['predicted'], records['targets']):
                writer.writerow([i, p, "[SEP]".join([onet for onet in t if len(onet) > 0])])
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
