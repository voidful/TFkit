import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import torch
import torch.nn as nn
from transformers import *
from torch.nn.functional import softmax, sigmoid
from qa.data_loader import get_feature_from_data
from utility.loss import *


class BertQA(nn.Module):

    def __init__(self, model_config, maxlen=512, dropout=0.1):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', self.device)
        if 'albert_chinese' in model_config:
            self.tokenizer = BertTokenizer.from_pretrained(model_config)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        self.pretrained = AutoModel.from_pretrained(model_config)
        self.maxlen = maxlen

        self.dropout = nn.Dropout(dropout)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        # self.loss_fct = FocalLoss()
        # self.loss_fct = GWLoss()

        self.pretrained = self.pretrained.to(self.device)
        self.qa_classifier = nn.Linear(self.pretrained.config.hidden_size, 2).to(self.device)
        self.loss_fct = self.loss_fct.to(self.device)

    def forward(self, batch_data, eval=False):
        inputs = torch.tensor(batch_data['input']).to(self.device)
        masks = torch.tensor(batch_data['mask']).to(self.device)

        targets = torch.tensor(batch_data['target']).to(self.device)
        start_positions, end_positions = targets.split(1, dim=1)
        start_positions = start_positions.squeeze(1)
        end_positions = end_positions.squeeze(1)

        output = self.pretrained(inputs, attention_mask=masks)[0]
        logits = self.qa_classifier(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if eval:
            reshaped_start_logits = softmax(start_logits)
            reshaped_end_logits = softmax(end_logits)
            start_prob = reshaped_start_logits.data.tolist()
            end_prob = reshaped_end_logits.data.tolist()
            max_start = []
            max_end = []
            for pos_logit_prob in start_prob:
                max_index = pos_logit_prob.index(max(pos_logit_prob))
                max_start.append(max_index)
            for pos_logit_prob in end_prob:
                max_index = pos_logit_prob.index(max(pos_logit_prob))
                max_end.append(max_index)
            outputs = (list(zip(max_start, max_end)),)
        else:
            outputs = ([start_logits, end_logits],)
        if eval is False:
            start_loss = self.loss_fct(start_logits, start_positions)
            end_loss = self.loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs

    def predict(self, input, task=None):
        self.eval()
        with torch.no_grad():
            feature_dict = get_feature_from_data(self.tokenizer, input, maxlen=self.maxlen)
            if len(feature_dict['input']) <= self.maxlen:
                raw_input = feature_dict['raw_input']
                for k, v in feature_dict.items():
                    feature_dict[k] = [v]
                result = self.forward(feature_dict, eval=True)
                start, end = result[0][0]
                return raw_input[start:end + 1], [start, end]
            else:
                return [""], []
