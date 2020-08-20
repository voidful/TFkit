import inspect

from transformers import *
import argparse
import torch
import gen_once
import gen_onebyone
import qa
import classifier
import tag
from tqdm import tqdm
from utility.eval_metric import EvalMetric
import csv
import inquirer
import nlp2


def load_model(model_path, pretrained_path=None, model_type=None, model_dataset=None):
    """load model from dumped file"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torchpack = torch.load(model_path, map_location=device)

    print("===model info===")
    [print(key, ':', torchpack[key]) for key in torchpack.keys() if 'state_dict' not in key and 'models' not in key]
    print('==========')

    if 'tags' in torchpack and len(torchpack['tags']) > 1:
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
    if pretrained_path is not None:
        config = pretrained_path
    else:
        config = torchpack['model_config'] if 'model_config' in torchpack else torchpack['bert']
    model_types = [torchpack['type']] if not isinstance(torchpack['type'], list) else torchpack['type']
    models_state = torchpack['models'] if 'models' in torchpack else [torchpack['model_state_dict']]
    type = model_types[type_ind]

    # load model
    if 'albert_chinese' in config:
        tokenizer = BertTokenizer.from_pretrained(config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config)
    pretrained = AutoModel.from_pretrained(config)

    type = type.lower()
    if "once" in type:
        eval_dataset = gen_once.get_data_from_file(model_dataset) if model_dataset else None
        model = gen_once.Once(tokenizer, pretrained, maxlen=maxlen)
    elif "onebyone" in type:
        eval_dataset = gen_once.get_data_from_file(model_dataset) if model_dataset else None
        model = gen_onebyone.OneByOne(tokenizer, pretrained, maxlen=maxlen)
    elif 'classify' in type or 'clas' in type:
        eval_dataset = classifier.get_data_from_file(model_dataset) if model_dataset else None
        model = classifier.MtClassifier(torchpack['task-label'], tokenizer, pretrained)
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


def load_predict_parameter(model, enable_arg_panel=False):
    """use inquirer panel to let user input model parameter or just use default value"""
    return nlp2.function_argument_panel(model.predict, disable_input_panel=(not enable_arg_panel), func_parent=model,
                                        ignore_empty=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="model path")
    parser.add_argument("--config", type=str, help='pre-trained model path after add token')
    parser.add_argument("--metric", required=True, type=str, choices=['emf1', 'nlg', 'clas'], help="evaluate metric")
    parser.add_argument("--valid", required=True, type=str, nargs='+', help="evaluate data path")
    parser.add_argument("--print", action='store_true', help="print each pair of evaluate data")
    parser.add_argument("--enable_arg_panel", action='store_true', help="enable panel to input argument")
    arg = parser.parse_args()

    valid = arg.valid[0]
    model, eval_dataset = load_model(arg.model, model_dataset=valid, pretrained_path=arg.config)
    predict_parameter = load_predict_parameter(model, arg.enable_arg_panel)
    if 'beamsearch' in predict_parameter and predict_parameter['beamsearch'] == 'True':
        eval_metrics = [EvalMetric(model.tokenizer) for _ in range(predict_parameter['beamsize'])]
    else:
        eval_metrics = [EvalMetric(model.tokenizer)]

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
            if 'QA' in model.__class__.__name__:
                target = " ".join(input.split(" ")[int(target[0]): int(target[1])])
            if 'OneByOne' in model.__class__.__name__ and predict_parameter['beamsearch']:
                predicted = result_dict['label_map'][eval_pos][0] if 'label_map' in result_dict else ''
            elif 'Tagger' in model.__class__.__name__:
                target = target.split(" ")
                if 'label_map' in result_dict:
                    predicted = " ".join([list(d.values())[0] for d in result_dict['label_map']])
                    predicted = predicted.split(" ")
                else:
                    predicted = [""] * len(target)
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
        if 'beamsearch' in predict_parameter and predict_parameter['beamsearch'] == 'True':
            argtype = "_beam_" + str(eval_pos)
        outfile_name = arg.model + argtype

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
                for i in eval_metric.cal_score(arg.metric):
                    f.write("TASK: " + str(i[0]) + " , " + str(eval_pos) + '\n')
                    f.write(str(i[1]) + '\n')
                    eds.writerows(i[2])

        print("write score at:", outfile_name)

        for i in eval_metric.cal_score(arg.metric):
            print("TASK: ", i[0], eval_pos)
            print(i[1])


if __name__ == "__main__":
    main()
