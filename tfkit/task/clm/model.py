import os
import sys

from tfkit.task.clm import Preprocessor
from tfkit.utility.predictor import AutoRegressivePredictor

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import torch
from torch import nn
from torch.nn.functional import softmax


class Model(nn.Module):
    def __init__(self, tokenizer, pretrained, maxlen=512, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.vocab_size = max(self.pretrained.config.vocab_size, self.tokenizer.__len__())
        self.model = nn.Linear(self.pretrained.config.hidden_size, self.vocab_size)
        self.maxlen = maxlen
        predictor = AutoRegressivePredictor(self, Preprocessor)
        self.predictor = predictor
        self.predict = predictor.predict

    def clean_cache(self):
        self.encoder_outputs = None
        self.past_key_values = None

    def forward(self, batch_data, eval=False, beamsearch=False, max_return=1, **kwargs):
        inputs = batch_data['input']
        masks = batch_data['mask']
        tokens_tensor = torch.as_tensor(inputs)
        mask_tensors = torch.as_tensor(masks)

        outputs = self.pretrained(tokens_tensor, attention_mask=mask_tensors)
        prediction_scores = self.model(outputs[0])

        if eval:
            result_dict = {}
            start = batch_data['start'][0]
            softmax_score = softmax(prediction_scores[0][start], dim=-1).flatten()
            max_item_id = torch.argmax(softmax_score, -1).item()
            max_item_prob = softmax_score[max_item_id].item()
            result_dict['max_item'] = (self.tokenizer.convert_ids_to_tokens(max_item_id), max_item_prob)
            if max_return > 1:
                topK = torch.topk(softmax_score, max_return)
                prob_result = [(self.tokenizer.convert_ids_to_tokens(tid), prob) for prob, tid in
                               zip(topK.values.data.tolist(), topK.indices.data.tolist())]
                result_dict['prob_list'] = softmax_score.data.tolist()[:max_return]
                result_dict['label_prob'] = prob_result
            outputs = result_dict
        else:
            targets = batch_data['target']
            loss_tensors = torch.as_tensor(targets)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size),
                                      loss_tensors.view(-1))

            outputs = masked_lm_loss
        return outputs
