import os
import sys

from tfkit.utility.predictor import QuestionAnsweringPredictor

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import torch
import torch.nn as nn
from torch.nn.functional import softmax
from tfkit.task.qa.preprocessor import Preprocessor


class Model(nn.Module):

    def __init__(self, tokenizer, pretrained, maxlen=128, dropout=0.1, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.maxlen = maxlen

        self.dropout = nn.Dropout(dropout)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        # self.loss_fct = FocalLoss(ignore_index=-1)
        # self.loss_fct = GWLoss()

        self.pretrained = self.pretrained
        self.qa_classifier = nn.Linear(self.pretrained.config.hidden_size, 2)
        self.loss_fct = self.loss_fct

        predictor = QuestionAnsweringPredictor(self, Preprocessor)
        self.predictor = predictor
        self.predict = predictor.predict

    def forward(self, batch_data, eval=False, **kwargs):
        print("batch_data",batch_data)
        inputs = torch.as_tensor(batch_data['input'])
        masks = torch.as_tensor(batch_data['mask'])
        targets = torch.as_tensor(batch_data['target'])
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
