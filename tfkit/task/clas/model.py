import os
import sys

import torch
from tfkit.utility.predictor import ClassificationPredictor
from torch import nn

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

from torch import softmax, sigmoid
from tfkit.task.clas import Preprocessor
from tfkit.utility.loss import FocalLoss, BCEFocalLoss


class Model(nn.Module):

    def __init__(self, tokenizer, pretrained, tasks_detail, maxlen=512, dropout=0.1, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained = pretrained

        self.dropout = nn.Dropout(dropout)
        self.loss_fct = FocalLoss()
        self.loss_fct_mt = BCEFocalLoss()

        self.tasks = dict()
        self.tasks_detail = tasks_detail
        self.classifier_list = nn.ModuleList()
        for task, labels in tasks_detail.items():
            self.classifier_list.append(nn.Linear(self.pretrained.config.hidden_size, len(labels)))
            self.tasks[task] = len(self.classifier_list) - 1
        self.maxlen = maxlen

        self.pretrained = self.pretrained
        self.classifier_list = self.classifier_list
        self.loss_fct = self.loss_fct
        self.loss_fct_mt = self.loss_fct_mt

        predictor = ClassificationPredictor(self, Preprocessor)
        self.predictor = predictor
        self.predict = predictor.predict

    def get_all_task(self):
        """
        list all classification task
        :return: tasks list
        """
        return list(self.tasks.keys())

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        from https://github.com/UKPLab/sentence-transformers
        modify - mask from -1 to 0
        :param model_output:
        :param attention_mask:
        :return:
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        input_mask_expanded[input_mask_expanded < 0] = 0
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, batch_data, eval=False, **kwargs):
        # covert input to correct data type
        tasks = batch_data['task']
        tasks = [bytes(t).decode(encoding="utf-8", errors="ignore") for t in tasks]
        inputs = torch.as_tensor(batch_data['input'])
        targets = torch.as_tensor(batch_data['target'])
        masks = torch.as_tensor(batch_data['mask'])
        # define model output
        result_dict = {
            'max_item': [],
            'prob_list': [],
            'label_prob': []
        }

        result_logits = []
        result_labels = []
        for p, zin in enumerate(zip(tasks, inputs, masks)):
            task, input, mask = zin
            task_id = self.tasks[task]
            task_labels = self.tasks_detail[task]

            output = self.pretrained(input.unsqueeze(0), mask.unsqueeze(0))[0]
            pooled_output = self.dropout(self.mean_pooling(output, mask.unsqueeze(0)))
            classifier_output = self.classifier_list[task_id](pooled_output)
            reshaped_logit = classifier_output.view(-1, len(task_labels))  # 0 for cls position
            result_logits.append(reshaped_logit)
            if not eval:
                target = targets[p]
                result_labels.append(target)
            else:
                if 'multi_label' in task:
                    reshaped_logit = sigmoid(reshaped_logit)
                else:
                    reshaped_logit = softmax(reshaped_logit, dim=1)
                logit_prob = reshaped_logit[0].data.tolist()
                logit_label = dict(zip(task_labels, logit_prob))
                result_dict['label_prob'].append({task: logit_label})
                if 'multi_label' in task:
                    result_dict['max_item'].append({task: [k for k, v in logit_label.items() if v > 0.5]})
                else:
                    result_dict['max_item'].append({task: [task_labels[logit_prob.index(max(logit_prob))]]})

        if eval:
            outputs = result_dict
        else:
            loss = 0
            for logit, labels, task in zip(result_logits, result_labels, tasks):
                if 'multi_label' in task:
                    loss += self.loss_fct_mt(logit, labels.type_as(logit))
                else:
                    loss += self.loss_fct(logit, labels)
            outputs = loss

        return outputs
