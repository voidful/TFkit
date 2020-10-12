import sys
import os

import torch
from torch import nn

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

from torch.nn.functional import softmax, sigmoid
from tfkit.classifier.data_loader import get_feature_from_data
from tfkit.utility.loss import FocalLoss, BCEFocalLoss
from torch.distributions import Categorical


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

    # from https://github.com/UKPLab/sentence-transformers
    # Mean Pooling - Take attention mask into account for correct averaging
    # modify - mask from -1 to 0
    def mean_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        input_mask_expanded[input_mask_expanded < 0] = 0
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

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
            pooled_output = self.dropout(self.mean_pooling(output, mask.unsqueeze(0)))
            classifier_output = self.classifier_list[task_id](pooled_output)
            reshaped_logits = classifier_output.view(-1, len(task_lables))  # 0 for cls position
            result_logits.append(reshaped_logits)
            if eval is False:
                target = targets[p]
                result_labels.append(target)
            else:
                if 'multi_label' in task:
                    reshaped_logits = sigmoid(reshaped_logits)
                else:
                    reshaped_logits = softmax(reshaped_logits, dim=1)
                logit_prob = reshaped_logits[0].data.tolist()
                logit_label = dict(zip(task_lables, logit_prob))
                result_dict['label_prob_all'].append({task: logit_label})
                if 'multi_label' in task:
                    result_dict['label_map'].append({task: [k for k, v in logit_label.items() if v > 0.5]})
                else:
                    result_dict['label_map'].append({task: [task_lables[logit_prob.index(max(logit_prob))]]})

        if eval:
            outputs = result_dict
        else:
            loss = 0
            for logits, labels, task in zip(result_logits, result_labels, tasks):
                if 'multi_label' in task:
                    loss += self.loss_fct_mt(logits, labels.type_as(logits))
                else:
                    loss += self.loss_fct(logits, labels)
            outputs = loss

        return outputs

    def get_all_task(self):
        return list(self.tasks.keys())

    def predict(self, input='', topk=1, task=get_all_task, handle_exceed='slide',
                merge_strategy=['minentropy', 'maxcount', 'maxprob']):
        topk = int(topk)
        task = task[0] if isinstance(task, list) else task
        handle_exceed = handle_exceed[0] if isinstance(handle_exceed, list) else handle_exceed
        merge_strategy = merge_strategy[0] if isinstance(merge_strategy, list) else merge_strategy
        self.eval()
        with torch.no_grad():
            ret_result = []
            ret_detail = []
            for feature in get_feature_from_data(self.tokenizer, self.maxlen, self.tasks_detail[task], task, input,
                                                 handle_exceed=handle_exceed):
                for k, v in feature.items():
                    feature[k] = [v]
                result = self.forward(feature, eval=True)
                if topk < 2:
                    ret_result.append([i[task] for i in result['label_map'] if task in i][0])
                    ret_detail.append(result)
                else:
                    task_map = [i[task] for i in result['label_prob_all'] if task in i][0]
                    ret_result.append(sorted(task_map, key=task_map.get, reverse=True)[:topk])
                    ret_detail.append(result)

        # apply different strategy to merge result after sliding windows
        if merge_strategy == 'maxcount':
            ret_result = max(ret_result, key=ret_result.count)
        else:
            results_prob = []
            results_entropy = []
            for detail in ret_detail:
                prob_map = detail['label_prob_all'][0][task]
                result_value = [v for _, v in prob_map.items()]
                results_entropy.append(Categorical(probs=torch.tensor(result_value)).entropy().data.tolist())
                results_prob.append(max(result_value))
            min_entropy_index = results_entropy.index(min(results_entropy))
            max_prob_index = results_prob.index(max(results_prob))
            if merge_strategy == 'minentropy':
                ret_result = ret_result[min_entropy_index]
            if merge_strategy == 'maxprob':
                ret_result = ret_result[max_prob_index]

        return ret_result, ret_detail
