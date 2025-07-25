from collections import defaultdict

import torch
from torch import nn
from torch.nn.functional import softmax

from tfkit.task.once import Preprocessor
from tfkit.utility.base_model import BaseTFKitModel
from tfkit.utility.loss import *
from tfkit.utility.predictor import NonAutoRegressivePredictor
from tfkit.utility.tok import *


class Model(BaseTFKitModel):
    """Once generation model for non-autoregressive text generation."""
    
    def __init__(self, tokenizer, pretrained, maxlen=512, tasks_detail=None, **kwargs):
        super().__init__(tokenizer, pretrained, maxlen, **kwargs)
        self.model = nn.Linear(self.get_hidden_size(), self.get_vocab_size())
        self._setup_predictor(NonAutoRegressivePredictor, Preprocessor)

    def forward(self, batch_data, eval=False, max_return=1, **kwargs):
        inputs = batch_data['input']
        masks = batch_data['mask']
        starts = batch_data['start']
        ends = batch_data['end']
        tokens_tensor = torch.as_tensor(inputs)
        mask_tensors = torch.as_tensor(masks)

        output = self.pretrained(tokens_tensor, attention_mask=mask_tensors)
        sequence_output = output[0]
        prediction_scores = self.model(sequence_output)

        if eval:
            result_dict = {
                'max_item': [],
                'label_prob': defaultdict(list),
                'prob_list': []
            }
            start = batch_data['start'][0]
            stop = False
            topK_ids = [[]] * max_return
            topK_probs = [1] * max_return
            while start < self.maxlen and not stop:
                softmax_score = softmax(prediction_scores[0][start], dim=0)
                max_item_id = torch.argmax(softmax_score, -1).item()
                max_item_prob = softmax_score[max_item_id].item()
                if max_return > 1:
                    topK = torch.topk(softmax_score, max_return)
                    for k, (prob, tid) in enumerate(zip(topK.values.data.tolist(), topK.indices.data.tolist())):
                        topK_ids[k].append(tid)
                        topK_probs[k] *= prob
                else:
                    topK_ids[0].append(max_item_id)
                    topK_probs[0] *= max_item_prob

                if tok_sep_id(self.tokenizer) == max_item_id:
                    stop = True
                start += 1
            result_dict['prob_list'] = topK_probs
            result_dict['label_prob'] = [[self.tokenizer.decode(ids), prob] for ids, prob in
                                         zip(topK_ids, topK_probs)]
            result_dict['max_item'] = [i[0] for i in result_dict['label_prob']]
            outputs = result_dict
        else:
            targets = batch_data['target']
            negative_targets = batch_data['ntarget']
            loss_tensors = torch.as_tensor(targets)
            negativeloss_tensors = torch.as_tensor(negative_targets)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size),
                                      loss_tensors.view(-1))
            if not torch.all(negativeloss_tensors.eq(-1)).item():
                negative_loss_fct = NegativeCElLoss()
                negative_loss = negative_loss_fct(prediction_scores.view(-1, self.vocab_size),
                                                  negativeloss_tensors.view(-1))
                masked_lm_loss += negative_loss
            outputs = masked_lm_loss

        return outputs
