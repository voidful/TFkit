import sys
import os

from torch.distributions import Categorical

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import torch
import torch.nn as nn
from torch.nn.functional import softmax
from tfkit.model.qa.dataloader import get_feature_from_data
from tfkit.utility.loss import FocalLoss
import numpy as np


class Model(nn.Module):

    def __init__(self, tokenizer, pretrained, maxlen=128, dropout=0.1, **kwargs):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', self.device)
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.maxlen = maxlen

        self.dropout = nn.Dropout(dropout)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        # self.loss_fct = FocalLoss(ignore_index=-1)
        # self.loss_fct = GWLoss()

        self.pretrained = self.pretrained.to(self.device)
        self.qa_classifier = nn.Linear(self.pretrained.config.hidden_size, 2).to(self.device)
        self.loss_fct = self.loss_fct.to(self.device)

    def forward(self, batch_data, eval=False):
        inputs = torch.as_tensor(batch_data['input']).to(self.device)
        masks = torch.as_tensor(batch_data['mask']).to(self.device)
        targets = torch.as_tensor(batch_data['target']).to(self.device)
        start_positions, end_positions = targets.split(1, dim=1)
        start_positions = start_positions.squeeze(1)
        end_positions = end_positions.squeeze(1)

        output = self.pretrained(inputs, attention_mask=masks)[0]
        logits = self.qa_classifier(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if eval:
            result_dict = {
                'label_prob_all': [],
                'label_map': []
            }
            reshaped_start_logits = softmax(start_logits, dim=1)
            reshaped_end_logits = softmax(end_logits, dim=1)
            start_prob = reshaped_start_logits.data.tolist()[0]
            end_prob = reshaped_end_logits.data.tolist()[0]
            result_dict['label_prob_all'].append({'start': dict(zip(range(len(start_prob)), start_prob)),
                                                  'end': dict(zip(range(len(end_prob)), end_prob))})
            result_dict['label_map'].append({'start': start_prob.index(max(start_prob)),
                                             'end': end_prob.index(max(end_prob))})
            outputs = result_dict
        else:
            start_loss = self.loss_fct(start_logits, start_positions)
            end_loss = self.loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = total_loss

        return outputs

    def predict(self, input='', topk=1, task=None, handle_exceed='slide',
                merge_strategy=['minentropy', 'maxprob', 'maxcount']):
        merge_strategy = merge_strategy[0] if isinstance(merge_strategy, list) else merge_strategy
        topk = int(topk)
        self.eval()
        with torch.no_grad():
            ret_result = []
            ret_detail = []
            for feature_dict in get_feature_from_data(self.tokenizer, input, maxlen=self.maxlen,
                                                      handle_exceed=handle_exceed):
                raw_input = feature_dict['raw_input']
                for k, v in feature_dict.items():
                    feature_dict[k] = [v]
                result = self.forward(feature_dict, eval=True)
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
                                        reverse=True)[:topk]

                ret_result.append(
                    ["".join(self.tokenizer.convert_tokens_to_string(raw_input[ans[0]:ans[1]])) for ans in
                     answer_results])
                ret_detail.append(result)

            # apply different strategy to merge result after sliding windows
            non_empty_result = []
            non_empty_detail = []
            for r, d in zip(ret_result, ret_detail):
                if len(r[0]) != 0:
                    non_empty_result.append(r)
                    non_empty_detail.append(d)

            if len(non_empty_result) == 0:
                ret_result = [[''] * topk]
            else:
                if merge_strategy == 'maxcount':  # should not be empty on count
                    ret_result = max(non_empty_result, key=non_empty_result.count)
                else:
                    results_prob = []
                    results_entropy = []
                    for detail in non_empty_detail:
                        prob_start = detail['label_prob_all'][0]['start'][int(detail['label_map'][0]['start'])]
                        prob_end = detail['label_prob_all'][0]['end'][int(detail['label_map'][0]['end'])]
                        prob_sum = [sum(x) for x in zip(list(detail['label_prob_all'][0]['start'].values()),
                                                        list(detail['label_prob_all'][0]['end'].values()))]
                        results_entropy.append(Categorical(probs=torch.tensor(prob_sum)).entropy().data.tolist())
                        results_prob.append(-np.log(prob_start) + -np.log(prob_end))

                    min_entropy_index = results_entropy.index(max(results_entropy))
                    max_prob_index = results_prob.index(max(results_prob))
                    if merge_strategy == 'minentropy':
                        ret_result = non_empty_result[min_entropy_index]
                    if merge_strategy == 'maxprob':
                        ret_result = non_empty_result[max_prob_index]
            return ret_result, ret_detail
