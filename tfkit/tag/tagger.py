import sys
import os
from collections import defaultdict

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import torch
import torch.nn as nn
from transformers import *
from torch.nn.functional import softmax
from nlp2 import *
from utility.loss import *
from tag.data_loader import get_feature_from_data


class BertTagger(nn.Module):

    def __init__(self, labels, model_config, maxlen=512, dropout=0.2):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        self.pretrained = AutoModel.from_pretrained(model_config)
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
        token_tensor = torch.tensor(inputs, dtype=torch.long).to(self.device)
        mask_tensors = torch.tensor(masks).to(self.device)
        bert_output = self.pretrained(token_tensor, attention_mask=mask_tensors)
        res = bert_output[0]
        pooled_output = self.dropout(res)
        # tagger
        reshaped_logits = self.tagger(pooled_output)
        result_items = []
        result_labels = []
        for ilogit in reshaped_logits:
            ilogit = softmax(ilogit)
            logit_prob = ilogit.data.tolist()
            result_item = []
            for pos_logit_prob in logit_prob:
                max_index = pos_logit_prob.index(max(pos_logit_prob))
                result_item.append(max_index)
            result_labels.append(ilogit)
            result_items.append(result_item)
        if eval:
            outputs = (result_items,)
        else:
            outputs = (result_labels,)
        # output
        if eval is False:
            targets = batch_data["target"]
            loss = 0
            target_tensor = torch.tensor(targets, dtype=torch.long).to(self.device)
            for logit, label in zip(reshaped_logits, target_tensor):
                loss += self.loss_fct(logit, label)
            outputs = (loss,) + outputs

        return outputs

    def predict(self, input, task=None):
        self.eval()
        output = ""
        with torch.no_grad():
            feature_dict = get_feature_from_data(tokenizer=self.tokenizer, labels=self.labels, input=input.strip(),
                                                 maxlen=self.maxlen)
            mapping = feature_dict['mapping']

            for k, v in feature_dict.items():
                feature_dict[k] = [v]
            result = self.forward(feature_dict, eval=True)
            result = result[0][0]
            result_map = []
            for map in json.loads(mapping):
                char, pos = map['char'], map['pos']
                if self.labels[result[pos]] is not "O":
                    output += char
                    result_map.append({char: self.labels[result[pos]]})
        return output, result_map
