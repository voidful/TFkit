import os
import sys
from collections import Counter

import torch
from torch import nn
from torch.nn.functional import softmax

from tfkit.task.tag import Preprocessor
from tfkit.utility.loss import FocalLoss
from tfkit.utility.predictor import TaggingPredictor

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))


class Model(nn.Module):
    def __init__(self, tokenizer, pretrained, tasks_detail, maxlen=512, dropout=0.2, **kwargs):
        super().__init__()
        self.initialize_components(tokenizer, pretrained, tasks_detail, maxlen, dropout)

    def initialize_components(self, tokenizer, pretrained, tasks_detail, maxlen, dropout):
        labels = list(tasks_detail.values())[0]
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.dropout = nn.Dropout(dropout)
        self.tagger = nn.Linear(self.pretrained.config.hidden_size, len(labels))
        self.labels = labels
        self.maxlen = maxlen
        self.loss_fct = FocalLoss()

        self.pretrained = self.pretrained
        self.loss_fct = self.loss_fct

        predictor = TaggingPredictor(self, Preprocessor)
        self.predictor = predictor
        self.predict = predictor.predict

    def forward(self, batch_data, eval=False, separator=" ", **kwargs):
        inputs = batch_data["input"]
        masks = batch_data["mask"]

        bert_output = self.compute_bert_output(inputs, masks)

        if eval:
            outputs = self.compute_eval_output(batch_data, bert_output)
        else:
            outputs = self.compute_loss_output(batch_data, bert_output)

        return outputs

    def compute_bert_output(self, inputs, masks):
        token_tensor = torch.as_tensor(inputs, dtype=torch.long)
        mask_tensors = torch.as_tensor(masks)
        bert_output = self.pretrained(token_tensor, attention_mask=mask_tensors)
        res = bert_output[0]
        pooled_output = self.dropout(res)
        reshaped_logits = self.tagger(pooled_output)

        return reshaped_logits

    def compute_eval_output(self, batch_data, reshaped_logits):
        result_dict = {
            'label_prob_all': [],
            'label_map': []
        }

        ilogit = softmax(reshaped_logits[0], dim=1)
        result_labels = ilogit.data.tolist()
        start, end = batch_data['pos'][0]
        token_word_mapping = batch_data['token_word_mapping']

        for pos, logit_prob in enumerate(result_labels[1:]):  # skip cls and sep
            if start + pos >= len(token_word_mapping):
                break

            word, pos = self.compute_word_pos(token_word_mapping, start, pos)
            self.update_result_dict(result_dict, logit_prob, word, pos)

        result_dict['token_word_mapping'] = token_word_mapping[start:end]

        return result_dict

    @staticmethod
    def compute_word_pos(token_word_mapping, start, pos):
        word = token_word_mapping[start + pos]['word']
        pos = token_word_mapping[start + pos]['pos']

        return word, pos

    def update_result_dict(self, result_dict, logit_prob, word, pos):
        if len(result_dict['label_map']) > pos:
            self.update_existing_result(result_dict, logit_prob, word, pos)
        else:
            self.append_new_result(result_dict, logit_prob, word)

    def update_existing_result(self, result_dict, logit_prob, word, pos):
        O = Counter(result_dict['label_prob_all'][-1][word])
        N = Counter(dict(zip(self.labels, logit_prob)))
        mean_prob = {k: v / 2 for k, v in (O + N).items()}
        result_dict['label_prob_all'][-1] = {word: mean_prob}
        result_dict['label_map'][-1] = {
            word: max(mean_prob, key=mean_prob.get)}

    def append_new_result(self, result_dict, logit_prob, word):
        max_index = logit_prob.index(max(logit_prob))
        result_dict['label_map'].append({word: self.labels[max_index]})
        result_dict['label_prob_all'].append({word: dict(zip(self.labels, logit_prob))})

    def compute_loss_output(self, batch_data, reshaped_logits):
        targets = batch_data["target"]
        target_tensor = torch.as_tensor(targets, dtype=torch.long)
        loss = self.loss_fct(reshaped_logits.view(-1, len(self.labels)), target_tensor.view(-1))

        return loss
