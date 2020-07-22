import sys
import os
from collections import defaultdict, OrderedDict

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import torch
import torch.nn as nn
from transformers import *
from torch.nn.functional import softmax
from nlp2 import *
from utility.loss import *
from tag.data_loader import get_feature_from_data


class Tagger(nn.Module):

    def __init__(self, labels, tokenizer, pretrained, maxlen=512, dropout=0.2):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', self.device)
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.dropout = nn.Dropout(dropout)
        self.tagger = nn.Linear(self.pretrained.config.hidden_size, len(labels))
        self.labels = labels
        self.maxlen = maxlen
        # self.loss_fct = nn.CrossEntropyLoss()
        self.loss_fct = FocalLoss()
        # self.loss_fct = GWLoss()

        self.pretrained = self.pretrained.to(self.device)
        self.loss_fct = self.loss_fct.to(self.device)

    def forward(self, batch_data, eval=False, separator=" "):
        inputs = batch_data["input"]
        masks = batch_data["mask"]

        # bert embedding
        token_tensor = torch.as_tensor(inputs, dtype=torch.long).to(self.device)
        mask_tensors = torch.as_tensor(masks).to(self.device)
        bert_output = self.pretrained(token_tensor, attention_mask=mask_tensors)
        res = bert_output[0]
        pooled_output = self.dropout(res)
        reshaped_logits = self.tagger(pooled_output)

        if eval:
            result_dict = {
                'label_prob_all': [],
                'label_map': []
            }

            ilogit = softmax(reshaped_logits[0], dim=1)
            result_labels = ilogit.data.tolist()
            result_items = []
            for pos_logit_prob in result_labels:
                max_index = pos_logit_prob.index(max(pos_logit_prob))
                result_items.append(
                    [self.labels[max_index], dict(zip(self.labels, pos_logit_prob))])

            mapping = batch_data['mapping']
            for map in json.loads(mapping[0]):
                char, pos = map['char'], map['pos']
                result_dict['label_map'].append({char: result_items[pos][0]})
                result_dict['label_prob_all'].append({char: result_items[pos][1]})

            outputs = result_dict
        else:
            targets = batch_data["target"]
            target_tensor = torch.as_tensor(targets, dtype=torch.long).to(self.device)
            loss = self.loss_fct(reshaped_logits.view(-1, len(self.labels)), target_tensor.view(-1))
            outputs = loss

        return outputs

    def predict(self, input='', neg="O", task=None):
        self.eval()
        with torch.no_grad():
            feature_dict = get_feature_from_data(tokenizer=self.tokenizer, labels=self.labels, input=input.strip(),
                                                 maxlen=self.maxlen)
            
            if len(feature_dict['input']) > self.maxlen:
                return [], {}

            for k, v in feature_dict.items():
                feature_dict[k] = [v]
            result = self.forward(feature_dict, eval=True)

            output = []
            target_str = ["", ""]
            for map in result['label_map']:
                for k, y in map.items():
                    if y is not neg:
                        target_str[0] += k
                        target_str[1] = y
                    else:
                        if len(target_str[0]) > 0:
                            output.append(target_str)
                        target_str = ["", ""]
            if len(target_str[0]) > 0:
                output.append(target_str)

            return output, result
