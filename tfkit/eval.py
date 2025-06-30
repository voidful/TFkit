import argparse
import csv
import logging
import sys
import time
from datetime import timedelta

import nlp2
import torch
from tqdm.auto import tqdm

from tfkit.utility.constants import SUPPORTED_METRICS, MODEL_EXTENSION
from tfkit.utility.eval_metric import EvalMetric
from tfkit.utility.model import load_trained_model, load_predict_parameter

transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.CRITICAL)


def parse_eval_args(args):
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate TFKit models")
    
    # Model specification
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", nargs='+', type=str, help="evaluation model path(s)")
    parser.add_argument("--config", type=str, help='pre-trained model config path after adding tokens')
    
    # Evaluation parameters
    parser.add_argument("--metric", required=True, type=str, choices=SUPPORTED_METRICS,
                       help=f"evaluation metric: {', '.join(SUPPORTED_METRICS)}")
    parser.add_argument("--valid", required=True, type=str, nargs='+', help="evaluation data path(s)")
    parser.add_argument("--tag", type=str, help="evaluation task tag for multi-task model selection")
    
    # Output options
    parser.add_argument("--print", action='store_true', help="print each pair of evaluation data")
    parser.add_argument("--panel", action='store_true', help="enable interactive panel for argument input")

    input_arg, model_arg = parser.parse_known_args(args)
    input_arg = {k: v for k, v in vars(input_arg).items() if v is not None}
    model_arg = {k.replace("--", ""): v for k, v in zip(model_arg[:-1:2], model_arg[1::2])}
    return input_arg, model_arg


def main(arg=None):
    with torch.no_grad():
        eval_arg, model_arg = parse_eval_args(sys.argv[1:]) if arg is None else parse_eval_args(arg)
        models_path = eval_arg.get('model', [])

        if nlp2.is_dir_exist(models_path[0]):
            models = [f for f in nlp2.get_files_from_dir(models_path[0]) if f.endswith(MODEL_EXTENSION)]
        else:
            models = models_path

        for model_path in models:
            start_time = time.time()
            valid = eval_arg.get('valid')[0]
            model, model_type, model_class, model_info, preprocessor = load_trained_model(model_path,
                                                                                          pretrained_config=eval_arg.get(
                                                                                              'config'),
                                                                                          tag=eval_arg.get('tag'))
            predict_parameter = load_predict_parameter(model, model_arg, eval_arg.get('panel'))

            eval_metrics = [EvalMetric(model.tokenizer)
                            for _ in range(int(predict_parameter.get('decodenum', 1)))]

            print("PREDICT PARAMETER")
            print("=======================")
            print(predict_parameter)
            print("=======================")

            get_data_item = preprocessor.read_file_to_data(valid)
            for chunk in tqdm(get_data_item):
                for i in chunk:
                    input = i['input']
                    target = i['target']
                    predict_parameter.update({'input': input})
                    result, result_dict = model.predict(**predict_parameter)
                    for eval_pos, eval_metric in enumerate(eval_metrics):
                        # predicted can be list of string or string
                        # target should be list of string
                        predicted = result
                        processed_target = target
                        if 'qa' in model_type:
                            processed_target = " ".join(input.split(" ")[int(target[0]): int(target[1])])
                            if len(result) > 0:
                                predicted = result[0][0] if isinstance(result[0], list) else result[0]
                            else:
                                predicted = ''
                        elif 'onebyone' in model_type or 'seq2seq' in model_type or 'clm' in model_type:
                            processed_target = target
                            if len(result) < eval_pos:
                                print("Decode size smaller than decode num:", result_dict['label_map'])
                            predicted = result[eval_pos]
                        elif 'once' in model_type:
                            processed_target = target
                            predicted = result[eval_pos]
                        elif 'mask' in model_type:
                            processed_target = target.split(" ")
                            predicted = result
                        elif 'tag' in model_type:
                            predicted = " ".join([list(d.values())[0] for d in result_dict[0]['label_map']])
                            processed_target = target[0].split(" ")
                            predicted = predicted.split(" ")

                        if eval_arg.get('print'):
                            print('===eval===')
                            print("input: ", input)
                            print("target: ", processed_target)
                            print("predicted: ", predicted)
                            print('==========')

                        eval_metric.add_record(input, predicted, processed_target, eval_arg.get('metric'))

                    for eval_pos, eval_metric in enumerate(eval_metrics):
                        argtype = f"_dataset{valid.replace('/', '_').replace('.', '_')}"
                        if 'decodenum' in predict_parameter and int(predict_parameter['decodenum']) > 1:
                            argtype += f"_num_{eval_pos}"
                        if 'mode' in predict_parameter:
                            para_mode = predict_parameter['mode'][0] if isinstance(predict_parameter['mode'], list) else \
                                predict_parameter['mode'].lower()
                            argtype += f"_mode_{para_mode}"
                        if 'filtersim' in predict_parameter:
                            argtype += f"_filtersim_{predict_parameter['filtersim']}"
                        outfile_name = f"{model_path}{argtype}"

                        with open(f"{outfile_name}_predicted.csv", "w", encoding='utf8') as f:
                            writer = csv.writer(f)
                            records = eval_metric.get_record(eval_arg.get('metric'))
                            writer.writerow(['input', 'predicted', 'targets'])
                            for i, p, t in zip(records['ori_input'], records['ori_predicted'], records['ori_target']):
                                writer.writerow([i, p, t])
                        print("write result at:", outfile_name)

                        with open(f"{outfile_name}_each_data_score.csv", "w", encoding='utf8') as edsf:
                            eds = csv.writer(edsf)
                            with open(f"{outfile_name}_score.csv", "w", encoding='utf8') as f:
                                for i in eval_metric.cal_score(eval_arg.get('metric')):
                                    f.write(f"TASK: {i[0]} , {eval_pos}\n")
                                    f.write(f"{i[1]}\n")
                                    eds.writerows(i[2])

                        print("write score at:", outfile_name)

                        for i in eval_metric.cal_score(eval_arg.get('metric')):
                            print("TASK: ", i[0], eval_pos)
                            print(i[1])

                    print(f"=== Execution time: {timedelta(seconds=(time.time() - start_time))} ===")


if __name__ == '__main__':
    main()
