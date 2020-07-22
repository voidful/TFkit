import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import torch
import torch.nn as nn
from torch.nn.functional import softmax, sigmoid
from classifier.data_loader import get_feature_from_data
from utility.loss import *


class MtClassifier(nn.Module):

    def __init__(self, tasks_detail, tokenizer, pretrained, maxlen=512, dropout=0.1):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', self.device)
        self.tokenizer = tokenizer
        self.pretrained = pretrained

        self.dropout = nn.Dropout(dropout)
        self.loss_fct = FocalLoss()
        self.loss_fct_mt = BCEFocalLoss()
        # self.loss_fct = FocalLoss()
        # self.loss_fct = GWLoss()

        self.tasks = dict()
        self.tasks_detail = tasks_detail
        self.classifier_list = nn.ModuleList()
        for task, labels in tasks_detail.items():
            self.classifier_list.append(nn.Linear(self.pretrained.config.hidden_size, len(labels)).to(self.device))
            self.tasks[task] = len(self.classifier_list) - 1
        self.maxlen = maxlen

        self.pretrained = self.pretrained.to(self.device)
        self.classifier_list = self.classifier_list.to(self.device)
        self.loss_fct = self.loss_fct.to(self.device)
        self.loss_fct_mt = self.loss_fct_mt.to(self.device)

    def forward(self, batch_data, eval=False):
        tasks = batch_data['task']
        inputs = torch.as_tensor(batch_data['input']).to(self.device)
        targets = torch.as_tensor(batch_data['target']).to(self.device)
        masks = torch.as_tensor(batch_data['mask']).to(self.device)

        result_dict = {
            'label_prob_all': [],
            'label_map': []
        }
        result_logits = []
        result_labels = []

        for p, zin in enumerate(zip(tasks, inputs, masks)):
            task, input, mask = zin
            task_id = self.tasks[task]
            task_lables = self.tasks_detail[task]

            output = self.pretrained(input.unsqueeze(0), mask.unsqueeze(0))[0]
            pooled_output = self.dropout(output)
            classifier_output = self.classifier_list[task_id](pooled_output)[0, 0]
            reshaped_logits = classifier_output.view(-1, len(task_lables))  # 0 for cls position
            result_logits.append(reshaped_logits)
            if eval is False:
                target = targets[p]
                result_labels.append(target)
            else:
                if 'multi_target' in task:
                    reshaped_logits = sigmoid(reshaped_logits)
                else:
                    reshaped_logits = softmax(reshaped_logits, dim=1)
                logit_prob = reshaped_logits[0].data.tolist()
                logit_label = dict(zip(task_lables, logit_prob))
                result_dict['label_prob_all'].append({task: logit_label})
                result_dict['label_map'].append({task: task_lables[logit_prob.index(max(logit_prob))]})

        if eval:
            outputs = result_dict
        else:
            loss = 0
            for logits, labels, task in zip(result_logits, result_labels, tasks):
                if 'multi_target' in task:
                    loss += self.loss_fct_mt(logits, labels)
                else:
                    loss += self.loss_fct(logits, labels)
            outputs = loss

        return outputs

    def get_all_task(self):
        return list(self.tasks.keys())

    def predict(self, input='', topk=1, task=get_all_task):
        topk = int(topk)
        self.eval()
        with torch.no_grad():
            feature_dict = get_feature_from_data(self.tokenizer, self.maxlen, self.tasks_detail[task], task, input)
            if len(feature_dict['input']) <= self.maxlen:
                for k, v in feature_dict.items():
                    feature_dict[k] = [v]
                result = self.forward(feature_dict, eval=True)
                if topk < 2:
                    return [i[task] for i in result['label_map'] if task in i], result
                else:
                    task_map = [i[task] for i in result['label_prob_all'] if task in i][0]
                    return sorted(task_map, key=task_map.get, reverse=True)[:topk], result
            else:
                return [], {}
