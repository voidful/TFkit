import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import torch
import torch.nn as nn
from transformers import *
from gen_onebyone.data_loader import get_feature_from_data
from itertools import combinations
from torch.nn.functional import softmax
from math import log
from utility.loss import *
from utility.tok import *


class BertOneByOne(nn.Module):
    def __init__(self, model_config, maxlen=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        self.pretrained = AutoModel.from_pretrained(model_config)
        self.model = nn.Linear(self.pretrained.config.hidden_size, self.pretrained.config.vocab_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.maxlen = maxlen

        print('Using device:', self.device)
        self.model.to(self.device)

    def forward(self, batch_data, eval=False):
        inputs = batch_data['input']
        targets = batch_data['target']
        negative_targets = batch_data['ntarget']
        types = batch_data['type']
        masks = batch_data['mask']

        tokens_tensor = torch.tensor(inputs).to(self.device)
        loss_tensors = torch.tensor(targets).to(self.device)
        negativeloss_tensors = torch.tensor(negative_targets).to(self.device)
        type_tensors = torch.tensor(types).to(self.device)
        mask_tensors = torch.tensor(masks).to(self.device)

        outputs = self.pretrained(tokens_tensor,  attention_mask=mask_tensors)
        sequence_output = outputs[0]
        prediction_scores = self.model(sequence_output)
        outputs = (prediction_scores,)

        if eval is False:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.pretrained.config.vocab_size),
                                      loss_tensors.view(-1))

            negative_loss_fct = NegativeCElLoss().to(self.device)
            negative_loss = negative_loss_fct(prediction_scores.view(-1, self.pretrained.config.vocab_size),
                                              negativeloss_tensors.view(-1))
            masked_lm_loss += negative_loss
            outputs = (masked_lm_loss,) + outputs

        return outputs

    def predict(self, input, task=None):
        self.eval()
        predicted = 0
        with torch.no_grad():
            output = ""
            outputs = []
            output_prob_dict = []
            while True:
                feature_dict = get_feature_from_data(self.tokenizer, self.maxlen, input, output)
                # print(feature_dict)
                if len(feature_dict['input']) > self.maxlen:
                    break
                start = feature_dict['start']
                for k, v in feature_dict.items():
                    feature_dict[k] = [v]
                predictions = self.forward(feature_dict, eval=True)
                predictions = predictions[0][0]
                logit_prob = softmax(predictions[start]).data.tolist()
                prob_result = {self.tokenizer.ids_to_tokens[id]: prob for id, prob in enumerate(logit_prob)}
                prob_result = sorted(prob_result.items(), key=lambda x: x[1], reverse=True)
                output_prob_dict.append(prob_result)
                predicted_index = torch.argmax(predictions[start]).item()
                predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])
                if predicted_token[0] != "#":
                    predicted_token[0] = predicted_token[0].replace("#", "")
                output += predicted_token[0] + ' '

            return output, output_prob_dict

    def jaccard_similarity(self, list1, list2):
        s1 = set(list1)
        s2 = set(list2)
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def isSimilar(self, s, t):
        return self.jaccard_similarity(s, t) > 0.5

    def filterSimilar(self, d, topk):
        while True:
            filteredOne = False
            for s, t in combinations(d, 2):
                if self.isSimilar(s[0], t[0]) and len(d) - 1 >= topk:
                    d.remove(t)
                    filteredOne = True
                    break
            if not filteredOne:
                break

    def predict_beamsearch(self, input, topk=3, filtersim=False):
        self.eval()
        sequences = [[[], 1.0]]
        with torch.no_grad():
            while True:
                all_candidates = list()
                exceed = False
                for seq in sequences:
                    if tok_sep(self.tokenizer) not in seq[0]:
                        tokens, score = seq
                        feature_dict = get_feature_from_data(self.tokenizer, self.maxlen, input, " ".join(tokens))
                        if len(feature_dict['input']) > self.maxlen:
                            exceed = True
                            all_candidates.append(seq)
                            continue
                        for k, v in feature_dict.items():
                            feature_dict[k] = [v]
                        predictions = self.forward(feature_dict, eval=True)
                        predictions = predictions[0][0]
                        predictions = predictions[feature_dict['start']][0]
                        logit_prob = softmax(predictions).data.tolist()
                        prob_result = {self.tokenizer.ids_to_tokens[id]: prob for id, prob in enumerate(logit_prob)}

                        for k, v in sorted(prob_result.items(), key=lambda x: x[1], reverse=True)[:50]:
                            if k != "#":
                                k = k.replace("#", "")
                            if len(tokens) > 0 and tokens[-1] == k or len(k) < 1:
                                continue
                            candidate = [tokens + [k], score + -log(v)]
                            all_candidates.append(candidate)
                    else:
                        all_candidates.append(seq)

                ordered = sorted(all_candidates, key=lambda tup: tup[1])
                if filtersim:
                    self.filterSimilar(ordered, topk)
                sequences = ordered[:topk]
                stop = 0
                for i in sequences:
                    if tok_sep(self.tokenizer) in i[0]:
                        stop += 1
                if stop == len(sequences) or exceed:
                    break

            for i in range(len(sequences)):
                if tok_sep(self.tokenizer) in sequences[i][0]:
                    sequences[i][0] = sequences[i][0][:sequences[i][0].index(tok_sep(self.tokenizer))]
                sequences[i][0] = " ".join(sequences[i][0])
            top = sequences[0][0]
            return top, sequences
