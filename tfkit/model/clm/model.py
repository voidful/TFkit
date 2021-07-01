import sys
import os

from tfkit.utility.predictor import Predictor

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import torch
from torch import nn
from tfkit.model.clm.dataloader import get_feature_from_data
from torch.nn.functional import softmax
from tfkit.utility.loss import NegativeCElLoss


class Model(nn.Module):
    def __init__(self, tokenizer, pretrained, maxlen=512, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.vocab_size = max(self.pretrained.config.vocab_size, self.tokenizer.__len__())
        self.model = nn.Linear(self.pretrained.config.hidden_size, self.vocab_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.maxlen = maxlen
        print('Using device:', self.device)
        self.model.to(self.device)
        predictor = Predictor(self, get_feature_from_data)
        self.predict = predictor.gen_predict

    def forward(self, batch_data, eval=False, **args):
        inputs = batch_data['input']
        targets = batch_data['target']
        negative_targets = batch_data['ntarget']
        masks = batch_data['mask']

        tokens_tensor = torch.as_tensor(inputs).to(self.device)
        mask_tensors = torch.as_tensor(masks).to(self.device)

        outputs = self.pretrained(tokens_tensor, attention_mask=mask_tensors)
        prediction_scores = self.model(outputs[0])

        if eval:
            result_dict = {
                'label_prob_all': [],
                'label_map': [],
                'prob_list': []
            }
            start = batch_data['start'][0]
            logit_prob = softmax(prediction_scores[0][start], dim=0)
            topK = torch.topk(logit_prob, 50)
            prob_result = [(self.tokenizer.convert_ids_to_tokens(id), prob) for prob, id in
                           zip(topK.values.data.tolist(), topK.indices.data.tolist())]
            result_dict['prob_list'].append(logit_prob.data.tolist())
            result_dict['label_prob_all'].append(prob_result)
            result_dict['label_map'].append(prob_result[0])
            outputs = result_dict
        else:
            loss_tensors = torch.as_tensor(targets).to(self.device)
            negativeloss_tensors = torch.as_tensor(negative_targets).to(self.device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size),
                                      loss_tensors.view(-1))
            if not torch.all(negativeloss_tensors.eq(-1)).item():
                negative_loss_fct = NegativeCElLoss(ignore_index=-1).to(self.device)
                negative_loss = negative_loss_fct(prediction_scores.view(-1, self.vocab_size),
                                                  negativeloss_tensors.view(-1))
                masked_lm_loss += negative_loss
            outputs = masked_lm_loss
        return outputs
