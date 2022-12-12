import json
from collections import defaultdict

import nlp2
import numpy as np
import torch
from math import log
from torch.distributions import Categorical

from tfkit.utility import tok


class BasePredictor:
    def __init__(self, model, preprocessor):
        self.preprocessor = preprocessor
        self.model = model

    def wrap_input(self, input=''):
        pass

    def processing(self, input_args):
        pass

    def wrap_output(self, results, detail_dicts):
        return results, detail_dicts

    def predict(self, **input_args):
        """
        Wrap the input 
        :param input_args:
        :return: [answer list], {result detail dict}
        """
        results, detail_dicts = self.processing(self.wrap_input(**input_args))
        return self.wrap_output(results, detail_dicts)


class BaseTextGeneratePredictor(BasePredictor):
    def wrap_input(self, input='', topK=1, topP=0.85, mode=['greedy', 'topK', 'topP'], decodenum=1,
                   filtersim=False, reserved_len=0, task=None, handle_exceed='noop', eos_num=1, decode_maxlen=0,
                   merge_strategy=['prob', 'entropy', 'count'],
                   no_repeat=False):
        return {
            'filtersim': json.loads(str(filtersim).lower()),
            'topK': int(topK),
            'no_repeat': json.loads(str(no_repeat).lower()),
            'eos_num': int(eos_num),
            'topP': float(topP),
            'decodenum': int(decodenum),
            'decode_maxlen': int(decode_maxlen),
            'reserved_len': int(reserved_len),
            'mode': mode[0] if isinstance(mode, list) else mode.lower(),
            'merge_strategy': merge_strategy[0] if isinstance(merge_strategy, list) else merge_strategy.lower(),
            'handle_exceed': handle_exceed,
            'task': task,
            'input': input
        }


class ClassificationPredictor(BasePredictor):
    def wrap_input(self, input='', task='', topK=1, mode=['greedy', 'topK', 'topP'], handle_exceed='noop',
                   merge_strategy=['prob', 'entropy', 'count'],
                   reserved_len=0):
        return {
            'topK': int(topK),
            'mode': mode[0] if isinstance(mode, list) else mode.lower(),
            'task': task if len(task) > 0 else list(self.model.tasks_detail.keys())[0],
            'input': input,
            'reserved_len': int(reserved_len),
            'merge_strategy': merge_strategy[0] if isinstance(merge_strategy, list) else merge_strategy.lower(),
            'handle_exceed': handle_exceed,
        }

    def processing(self, input_args):
        with torch.no_grad():
            ret_result = []
            ret_detail = []
            proc = self.preprocessor(self.model.tokenizer, maxlen=self.model.maxlen,
                                     handle_exceed=input_args['handle_exceed'],
                                     reserved_len=input_args['reserved_len'])

            for items in proc.preprocess({"task": input_args['task'], "input": input_args['input']}):
                feature = proc.postprocess({**items, **{'task_dict': self.model.tasks}}, self.model.tokenizer,
                                           self.model.maxlen)
            predictions = self.model.forward(feature, eval=True)
            if input_args['topK'] < 2:
                ret_result.append(
                    [i[input_args['task']] for i in predictions['max_item'] if input_args['task'] in i][0])
                ret_detail.append(predictions)
            else:
                task_map = [i[input_args['task']] for i in predictions['label_prob'] if input_args['task'] in i][0]
                ret_result.append(sorted(task_map, key=task_map.get, reverse=True)[:input_args['topK']])
                ret_detail.append(predictions)

        # apply different strategy to merge result after sliding windows
        if input_args['merge_strategy'] == 'count':
            ret_result = max(ret_result, key=ret_result.count)
        else:
            results_prob = []
            results_entropy = []
            for detail in ret_detail:
                prob_map = detail['label_prob'][0][input_args['task']]
                result_value = [v for _, v in prob_map.items()]
                results_entropy.append(Categorical(probs=torch.tensor(result_value)).entropy().data.tolist())
                results_prob.append(max(result_value))
            min_entropy_index = results_entropy.index(min(results_entropy))
            max_prob_index = results_prob.index(max(results_prob))
            if input_args['merge_strategy'] == 'entropy':
                ret_result = ret_result[min_entropy_index]
            if input_args['merge_strategy'] == 'prob':
                ret_result = ret_result[max_prob_index]
        return ret_result, ret_detail


class QuestionAnsweringPredictor(BasePredictor):
    def wrap_input(self, input='', task='', topK=1, mode=['greedy', 'topK', 'topP'], handle_exceed='slide',
                   merge_strategy=['prob', 'entropy', 'count'],
                   reserved_len=0):
        return {
            'topK': int(topK),
            'mode': mode[0] if isinstance(mode, list) else mode.lower(),
            'input': input,
            'task': task,
            'reserved_len': int(reserved_len),
            'merge_strategy': merge_strategy[0] if isinstance(merge_strategy, list) else merge_strategy.lower(),
            'handle_exceed': handle_exceed,
        }

    def processing(self, input_args):
        with torch.no_grad():
            ret_result = []
            ret_detail = []
            proc = self.preprocessor(self.model.tokenizer, maxlen=self.model.maxlen,
                                     handle_exceed=input_args['handle_exceed'],
                                     reserved_len=input_args['reserved_len'])

            for items in proc.preprocess({"task": input_args['task'], "input": input_args['input']}):
                raw_input = items['raw_input']
                feature = proc.postprocess(items, self.model.tokenizer, self.model.maxlen)
            for k, v in feature.items():
                feature[k] = [v]

            result = self.model.forward(feature, eval=True)
            start_dict = [i['start'] for i in result['label_prob_all'] if 'start' in i][0]
            end_dict = [i['end'] for i in result['label_prob_all'] if 'end' in i][0]

            answers = []
            sorted_start = sorted(start_dict.items(), key=lambda item: item[1], reverse=True)[:50]
            sorted_end = sorted(end_dict.items(), key=lambda item: item[1], reverse=True)[:50]
            for start_index, start_prob in sorted_start:
                for end_index, end_prob in sorted_end:
                    if start_index > end_index:
                        continue
                    answers.append((start_index, end_index, start_prob + end_prob))
            answer_results = sorted(answers, key=lambda answers: answers[2],
                                    reverse=True)[:input_args['topK']]
            ret_result.append(
                ["".join(self.model.tokenizer.convert_tokens_to_string(raw_input[ans[0]:ans[1]])) for ans in
                 answer_results])
            ret_detail.append(result)

            # apply different strategy to merge result after sliding windows
            non_empty_result = []
            non_empty_detail = []
            for r, d in zip(ret_result, ret_detail):
                if len(r[0]) != 0:
                    non_empty_result.append(r)
                    non_empty_detail.append(d)

            results_prob = []
            results_entropy = []
            for detail in non_empty_detail:
                prob_start = detail['label_prob_all'][0]['start'][int(detail['label_map'][0]['start'])]
                prob_end = detail['label_prob_all'][0]['end'][int(detail['label_map'][0]['end'])]
                prob_sum = [sum(x) for x in zip(list(detail['label_prob_all'][0]['start'].values()),
                                                list(detail['label_prob_all'][0]['end'].values()))]
                results_entropy.append(Categorical(probs=torch.tensor(prob_sum)).entropy().data.tolist())
                results_prob.append(-np.log(prob_start) + -np.log(prob_end))
        return non_empty_result, non_empty_detail


class TaggingPredictor(BasePredictor):
    def wrap_input(self, input='', task='', topK=1, mode=['greedy', 'topK', 'topP'], handle_exceed='slide',
                   merge_strategy=['prob', 'entropy', 'count'],
                   neg="O",
                   start_contain="B",
                   end_contain="I",
                   minlen=1,
                   reserved_len=0):
        return {
            'topK': int(topK),
            'mode': mode[0] if isinstance(mode, list) else mode.lower(),
            'input': input,
            'task': task,
            'neg': neg,
            'minlen': minlen,
            'start_contain': start_contain,
            'end_contain': end_contain,
            'reserved_len': int(reserved_len),
            'merge_strategy': merge_strategy[0] if isinstance(merge_strategy, list) else merge_strategy.lower(),
            'handle_exceed': handle_exceed,
        }

    def processing(self, input_args):
        with torch.no_grad():
            ret_detail = []
            predicted_pos_prob = defaultdict(lambda: defaultdict(list))
            proc = self.preprocessor(self.model.tokenizer, maxlen=self.model.maxlen,
                                     handle_exceed=input_args['handle_exceed'],
                                     reserved_len=input_args['reserved_len'])
            for items in proc.preprocess({"task": input_args['task'], "input": input_args['input']}):
                token_word_mapping = items['token_word_mapping']
                feature = proc.postprocess({**items, **{'task_dict': self.model.labels}}, self.model.tokenizer,
                                           self.model.maxlen)
                for k, v in feature.items():
                    feature[k] = [v]
                feature['token_word_mapping'] = token_word_mapping
                result = self.model.forward(feature, eval=True)
                for token_pred, token_map in zip(result['label_prob_all'], result['token_word_mapping']):
                    token_prob = list(token_pred.values())[0]
                    max_label = max(token_prob, key=token_prob.get)
                    max_prob = token_prob[max_label]
                    max_entropy = Categorical(probs=torch.tensor(list(token_prob.values()))).entropy().data.tolist()
                    predicted_pos_prob[token_map['pos']]['char'] = token_map['word']
                    predicted_pos_prob[token_map['pos']]['labels'].append(max_label)
                    predicted_pos_prob[token_map['pos']]['prob'].append(max_prob)
                predicted_pos_prob[token_map['pos']]['entropy'].append(max_entropy)

            ret_detail.append(result)

            ret_result = []
            for key, value in predicted_pos_prob.items():
                if input_args['merge_strategy'] == 'count':
                    label = max(value['labels'], key=value['labels'].count)
                if input_args['merge_strategy'] == 'entropy':
                    min_entropy_index = value['entropy'].index(min(value['entropy']))
                    label = value['labels'][min_entropy_index]
                if input_args['merge_strategy'] == 'prob':
                    max_prob_index = value['prob'].index(max(value['prob']))
                    label = value['labels'][max_prob_index]
                ret_result.append({value['char']: label})

            output = []
            target_str = ["", ""]
            after_start = False
            for mapping in ret_result:
                for k, y in mapping.items():
                    if input_args['start_contain'] in y:
                        after_start = True
                        if len(target_str[0]) > 0:
                            if len(target_str[0]) > input_args['minlen']:
                                output.append(target_str)
                            target_str = ["", ""]
                        target_str[0] += k
                        target_str[1] = y
                    elif y is not input_args['neg'] and after_start:
                        target_str[0] += k
                        target_str[1] = y
                    else:
                        after_start = False
            if len(target_str[0]) > input_args['minlen'] and target_str not in output:
                output.append(target_str)
            output = [[ner, tag.replace(input_args['start_contain'], "").replace(input_args['end_contain'], "")] for
                      ner, tag in output]

        return output, ret_detail


class NonAutoRegressivePredictor(BaseTextGeneratePredictor):

    def processing(self, input_args):
        with torch.no_grad():
            proc = self.preprocessor(self.model.tokenizer, maxlen=self.model.maxlen,
                                     handle_exceed=input_args['handle_exceed'],
                                     reserved_len=input_args['reserved_len'])
            for items in proc.preprocess({"task": input_args['task'], "input": input_args['input']}):
                feature = proc.postprocess(items, self.model.tokenizer, self.model.maxlen)
            for k, v in feature.items():
                feature[k] = [v]
            predictions = self.model.forward(feature, eval=True, beamsearch=input_args['decodenum'] > 1,
                                             max_return=input_args['decodenum'])
            return predictions['max_item'], predictions


class AutoRegressivePredictor(BaseTextGeneratePredictor):

    def processing(self, input_args):
        previous = []
        if tok.UNIVERSAL_SEP in input_args['input']:
            previous = self.model.tokenizer.tokenize(input_args['input'].split(tok.UNIVERSAL_SEP)[-1])
            input_args['eos_num'] += 1
        sep_tok = tok.tok_sep(self.model.tokenizer)
        sequences = [[[], 1.0]]
        with torch.no_grad():
            while True:
                all_candidates = list()
                exceed = False
                for seq in sequences:
                    if sep_tok not in seq[0] or seq[0].count(sep_tok) < input_args['eos_num']:
                        tokens, score = seq
                        if not tokens:
                            tokens = previous

                        proc = self.preprocessor(self.model.tokenizer, maxlen=self.model.maxlen,
                                                 handle_exceed=input_args['handle_exceed'],
                                                 reserved_len=input_args['reserved_len'])
                        for items in proc.preprocess(
                                {"task": input_args['task'], "input": input_args['input'], 'previous': tokens}):
                            feature_dict = proc.postprocess_batch(
                                proc.postprocess(items, self.model.tokenizer, self.model.maxlen))

                        # check input exceed
                        if len(tokens) >= self.model.maxlen \
                                or ('prev' in feature_dict and len(feature_dict['prev']) >= self.model.maxlen) \
                                or ('input' in feature_dict and len(feature_dict['input']) >= self.model.maxlen):
                            exceed = True
                            all_candidates.append(seq)
                            continue
                        predictions = self.model.forward(feature_dict, eval=True, use_prev=False,
                                                         output_hidden_states=True,
                                                         beamsearch=input_args['decodenum'] > 1,
                                                         max_return=max(input_args['decodenum'],
                                                                        input_args['topK']))
                        label_prob = predictions['label_prob'] if 'label_prob' in predictions else \
                            [predictions['max_item']]
                        # topK topP
                        if 'top' in input_args['mode']:
                            prob_list = predictions['prob_list']
                            if 'topk' in input_args['mode']:
                                sample_list = prob_list[:input_args['topK']]
                                decode_range = max(input_args['decodenum'], input_args['topK'])
                                prob_norm = [float(i) / sum(sample_list) for i in sample_list]
                                choice_list = np.random.choice(sample_list, p=prob_norm,
                                                               size=decode_range,
                                                               replace=False)
                            else:
                                topP_list = np.cumsum(prob_list)
                                index_overP = [i for i, x in enumerate(topP_list) if x > input_args['topP']]
                                index_overP = 0 if len(index_overP) < 1 else index_overP[0]
                                sample_list = prob_list[:index_overP + 1]
                                prob_norm = [float(i) / sum(sample_list) for i in sample_list]
                                choice_list = np.random.choice(sample_list, p=prob_norm,
                                                               size=input_args['decodenum'])
                            for idx in range(input_args['decodenum']):
                                sampling_index = prob_list.index(choice_list[idx])
                                k, v = label_prob[sampling_index]
                                candidate = [tokens + [k], score + -log(v)]
                                all_candidates.append(candidate)

                        # greedy / beam search
                        else:
                            for k, v in label_prob[:50]:
                                if (input_args['no_repeat'] and len(tokens) > 0 and tokens[-1] == k) or len(k) < 1:
                                    continue
                                candidate = [tokens + [k], score + -log(v) if v > 0 else 0]
                                all_candidates.append(candidate)
                    else:
                        all_candidates.append(seq)

                ordered = sorted(all_candidates, key=lambda tup: tup[1])
                if input_args['filtersim']:  # default disable, it will took a lots of time on long sequence generation
                    nlp2.filter_jaccard_similar_text_from_list(ordered, input_args['decodenum'])
                sequences = ordered[:input_args['decodenum']]
                stop = 0
                for i in sequences:
                    # i[0], i[1] - sequence, sequence score
                    if (i[0].count(sep_tok) >= input_args['eos_num']) or \
                            len(i[0]) > self.model.maxlen or \
                            0 < input_args['decode_maxlen'] < len(i[0]):
                        stop += 1
                if stop == len(sequences) or exceed:
                    break

            for i in range(len(sequences)):
                if sep_tok in sequences[i][0]:  # remove sep token
                    sequences[i][0] = sequences[i][0][:-1]
                slide_len = len(previous) if len(previous) > 0 else 0
                sequences[i][0] = self.model.tokenizer.decode(
                    self.model.tokenizer.convert_tokens_to_ids(sequences[i][0][slide_len:]))

            self.model.clean_cache()
            return [i[0] for i in sequences], {'label_map': sequences}
