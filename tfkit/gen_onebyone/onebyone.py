import sys
import os

from transformers import activations

from tfkit.utility import tok_sep

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import torch
import torch.nn as nn
from gen_onebyone.data_loader import get_feature_from_data
from itertools import combinations
from torch.nn.functional import softmax
from math import log
from utility.loss import *
from utility.tok import *
import numpy as np


class OneByOne(nn.Module):
    def __init__(self, tokenizer, pretrained, maxlen=512, lossdrop=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained = pretrained

        self.model = nn.Linear(self.pretrained.config.hidden_size, self.tokenizer.__len__(), bias=False)
        self.dense = nn.Linear(self.pretrained.config.hidden_size, self.pretrained.config.hidden_size)
        self.transform_act_fn = activations.ACT2FN[self.pretrained.config.hidden_act]
        self.LayerNorm = nn.LayerNorm(self.pretrained.config.hidden_size, eps=self.pretrained.config.layer_norm_eps)
        self.bias = nn.Parameter(torch.zeros(self.tokenizer.__len__()))
        self.model.bias = self.bias

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.maxlen = maxlen
        self.lossdrop = lossdrop
        self.dropper = LossDropper()
        print('Using device:', self.device)
        self.model.to(self.device)

    def forward(self, batch_data, eval=False):
        inputs = batch_data['input']
        targets = batch_data['target']
        negative_targets = batch_data['ntarget']
        masks = batch_data['mask']

        tokens_tensor = torch.as_tensor(inputs).to(self.device)
        mask_tensors = torch.as_tensor(masks).to(self.device)

        outputs = self.pretrained(tokens_tensor, attention_mask=mask_tensors)
        sequence_output = self.dense(outputs[0])
        sequence_output = self.transform_act_fn(sequence_output)
        sequence_output = self.LayerNorm(sequence_output)
        prediction_scores = self.model(sequence_output)

        if eval:
            result_dict = {
                'label_prob_all': [],
                'label_map': [],
                'prob_list': []
            }
            start = batch_data['start'][0]
            logit_prob = softmax(prediction_scores[0][start], dim=0).data.tolist()
            prob_result = {self.tokenizer.convert_ids_to_tokens(id): prob for id, prob in enumerate(logit_prob)}
            prob_result = sorted(prob_result.items(), key=lambda x: x[1], reverse=True)
            result_dict['prob_list'].append(sorted(logit_prob, reverse=True))
            result_dict['label_prob_all'].append(prob_result)
            result_dict['label_map'].append(prob_result[0])
            outputs = result_dict
        else:
            loss_tensors = torch.as_tensor(targets).to(self.device)
            negativeloss_tensors = torch.as_tensor(negative_targets).to(self.device)
            loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.pretrained.config.vocab_size),
                                      loss_tensors.view(-1))

            negative_loss_fct = NegativeCElLoss(reduction='none', ignore_index=-1).to(self.device)
            negative_loss = negative_loss_fct(prediction_scores.view(-1, self.pretrained.config.vocab_size),
                                              negativeloss_tensors.view(-1))

            masked_lm_loss += negative_loss
            masked_lm_loss = masked_lm_loss.view(-1, len(targets))  # view by batch size
            masked_lm_loss = masked_lm_loss.sum(dim=0)
            if self.lossdrop:
                mask = self.dropper(masked_lm_loss)
                masked_lm_loss *= mask
            masked_lm_loss = masked_lm_loss.mean()
            outputs = masked_lm_loss
        return outputs

    def predict(self, input='', topK=1, topP=0.7, beamsearch=False, beamsize=3, filtersim=True, task=None):
        topK = int(topK)
        topP = float(topP)
        beamsize = int(beamsize)

        if beamsearch:
            return self.predict_beamsearch(input, beamsize=beamsize, filtersim=filtersim)
        else:
            self.eval()
            with torch.no_grad():
                output = []
                result_dict = {
                    'label_prob_all': [],
                    'label_map': [],
                    'prob_list': []
                }
                while True:
                    feature_dict = get_feature_from_data(self.tokenizer, self.maxlen, input, output)
                    if len(feature_dict['input']) > self.maxlen:
                        break
                    for k, v in feature_dict.items():
                        feature_dict[k] = [v]
                    predictions = self.forward(feature_dict, eval=True)
                    result_dict['label_prob_all'].append(predictions['label_prob_all'])
                    result_dict['label_map'].append(predictions['label_map'])
                    result_dict['prob_list'].append(predictions['prob_list'])

                    topK_list = [p for w, p in predictions['label_prob_all'][0]][:topK]
                    topP_list = np.cumsum(topK_list)
                    index_overK = [i for i, x in enumerate(topP_list) if x > topP]
                    index_overK = 0 if len(index_overK) < 1 else index_overK[0]
                    topP_list = list(topK_list[:index_overK + 1])
                    prob_norm = [float(i) / sum(topP_list) for i in topP_list]
                    sampling_index = topP_list.index(np.random.choice(topP_list, p=prob_norm))

                    predicted_token = predictions['label_prob_all'][0][sampling_index][0]

                    if tok_sep(self.tokenizer) in predicted_token or \
                            len(output) > 2 and output[-1] == output[-2] == predicted_token[0]:
                        break
                    output.append(predicted_token)

                output = self.tokenizer.convert_tokens_to_string(output)
                if len(output) < 1:
                    return [], {}
                return [output], result_dict

    def jaccard_similarity(self, list1, list2):
        s1 = set(list1)
        s2 = set(list2)
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def isSimilar(self, s, t):
        return self.jaccard_similarity(s, t) > 0.5

    def filterSimilar(self, d, topP):
        while True:
            filteredOne = False
            for s, t in combinations(d, 2):
                if self.isSimilar(s[0], t[0]) and len(d) - 1 >= topP:
                    d.remove(t)
                    filteredOne = True
                    break
            if not filteredOne:
                break

    def predict_beamsearch(self, input, beamsize=3, filtersim=True):
        self.eval()
        sequences = [[[], 1.0]]
        with torch.no_grad():
            while True:
                all_candidates = list()
                exceed = False
                for seq in sequences:
                    if tok_sep(self.tokenizer) not in seq[0]:
                        tokens, score = seq
                        feature_dict = get_feature_from_data(self.tokenizer, self.maxlen, input, tokens)
                        if len(feature_dict['input']) > self.maxlen:
                            exceed = True
                            all_candidates.append(seq)
                            continue
                        for k, v in feature_dict.items():
                            feature_dict[k] = [v]
                        predictions = self.forward(feature_dict, eval=True)
                        for k, v in predictions['label_prob_all'][0][:50]:
                            if len(tokens) > 0 and tokens[-1] == k or len(k) < 1:
                                continue
                            candidate = [tokens + [k], score + -log(v)]
                            all_candidates.append(candidate)
                    else:
                        all_candidates.append(seq)

                ordered = sorted(all_candidates, key=lambda tup: tup[1])
                if filtersim:
                    self.filterSimilar(ordered, beamsize)
                sequences = ordered[:beamsize]
                stop = 0
                for i in sequences:
                    if tok_sep(self.tokenizer) in i[0] or \
                            len(i[0]) > 3 and i[0][-1] == i[0][-2] == i[0][-3] or \
                            i[1] > 300:
                        stop += 1
                if stop == len(sequences) or exceed:
                    break

            for i in range(len(sequences)):
                if tok_sep(self.tokenizer) in sequences[i][0]:
                    sequences[i][0] = sequences[i][0][:sequences[i][0].index(tok_sep(self.tokenizer))]
                sequences[i][0] = "".join(self.tokenizer.convert_tokens_to_string(sequences[i][0]))
            result_dict = {
                'label_map': sequences
            }
            return [i[0] for i in sequences], result_dict
