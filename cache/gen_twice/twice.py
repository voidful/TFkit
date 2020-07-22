import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import torch
import torch.nn as nn
from transformers import *
from torch.nn.functional import softmax
from gen_once.data_loader import get_feature_from_data
from utility.loss import *
from utility.tok import *


class Twice(nn.Module):
    def __init__(self, tokenizer, pretrained, maxlen=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.model = nn.Linear(self.pretrained.config.hidden_size, self.pretrained.config.vocab_size)
        self.maxlen = maxlen
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', self.device)
        self.model.to(self.device)

    def forward(self, batch_data, eval=False):
        inputs = batch_data['input']
        targets = batch_data['target']
        negative_targets = batch_data['ntarget']
        types = batch_data['type']
        masks = batch_data['mask']
        start = batch_data['start']

        tokens_tensor = torch.as_tensor(inputs).to(self.device)
        type_tensors = torch.as_tensor(types).to(self.device)
        mask_tensors = torch.as_tensor(masks).to(self.device)
        loss_tensors = torch.as_tensor(targets).to(self.device)
        negativeloss_tensors = torch.as_tensor(negative_targets).to(self.device)

        output_once = self.pretrained(tokens_tensor, attention_mask=mask_tensors)
        prediction_scores_once = self.model(output_once[0])

        prediction_score_twice = []
        for i in range(len(prediction_scores_once)):
            start_ind = start[i]
            predictions = prediction_scores_once[i][0]
            output_string_once = ""
            end = False
            while start_ind < self.maxlen:
                predicted_index = torch.argmax(predictions[start]).item()
                predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])
                if tok_sep(self.tokenizer) in predicted_token:
                    end = True
                if end is False:
                    output_string_once += predicted_token[0] + " "
                start_ind += 1
            feature_dict = get_feature_from_data(self.tokenizer, self.maxlen, output_string_once)
            if len(feature_dict['input']) > self.maxlen:
                prediction_score_twice = prediction_scores_once
                break
            output_twice = self.pretrained(torch.as_tensor([feature_dict['input']]).to(self.device),
                                           attention_mask=torch.as_tensor(
                                               torch.as_tensor([feature_dict['mask']]).to(self.device)).to(
                                               self.device))
            prediction_score_twice.append(self.model(output_twice[0]))
        if isinstance(prediction_score_twice, list):
            prediction_score_twice = torch.cat(prediction_score_twice, dim=0)

        if eval:
            result_dict = {
                'label_prob_all': [],
                'label_map': [],
                'prob_list': []
            }
            start_ind = start[i]
            end = False
            while start_ind < self.maxlen and not end:
                predicted_index = torch.argmax(prediction_score_twice[0][start]).item()
                predicted_token = self.tokenizer.decode([predicted_index])
                logit_prob = softmax(prediction_score_twice[0][start], dim=0).data.tolist()
                prob_result = {self.tokenizer.decode([id]): prob for id, prob in enumerate(logit_prob)}
                prob_result = sorted(prob_result.items(), key=lambda x: x[1], reverse=True)

                result_dict['prob_list'].append(sorted(logit_prob, reverse=True))
                result_dict['label_prob_all'].append(dict(prob_result))
                result_dict['label_map'].append(prob_result[0])
                if tok_sep(self.tokenizer) in predicted_token:
                    end = True
                start_ind += 1
            outputs = result_dict
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores_once.view(-1, self.pretrained.config.vocab_size),
                                      loss_tensors.view(-1))
            masked_lm_loss += loss_fct(prediction_score_twice.view(-1, self.pretrained.config.vocab_size),
                                       loss_tensors.view(-1))
            outputs = masked_lm_loss

        return outputs

    def predict(self, input='', task=None):
        self.model.eval()
        with torch.no_grad():
            feature_dict = get_feature_from_data(self.tokenizer, self.maxlen, input)
            if len(feature_dict['input']) > self.maxlen:
                return [], {}
            for k, v in feature_dict.items():
                feature_dict[k] = [v]
            predictions = self.forward(feature_dict, eval=True)
            output = "".join(self.tokenizer.convert_tokens_to_string([i[0] for i in predictions['label_map']]))
            return [output], predictions
