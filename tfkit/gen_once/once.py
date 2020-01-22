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


class BertOnce(nn.Module):
    def __init__(self, model_config, maxlen=128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        self.pretrained = AutoModel.from_pretrained(model_config)
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

        tokens_tensor = torch.tensor(inputs).to(self.device)
        type_tensors = torch.tensor(types).to(self.device)
        mask_tensors = torch.tensor(masks).to(self.device)
        loss_tensors = torch.tensor(targets).to(self.device)
        negativeloss_tensors = torch.tensor(negative_targets).to(self.device)

        output = self.pretrained(tokens_tensor, attention_mask=mask_tensors)
        sequence_output = output[0]
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
        self.model.eval()
        output_prob_dict = []
        with torch.no_grad():
            feature_dict = get_feature_from_data(self.tokenizer, self.maxlen, input)
            start = feature_dict['start']
            for k, v in feature_dict.items():
                feature_dict[k] = [v]
            predictions = self.forward(feature_dict, eval=True)
            predictions = predictions[0][0]
            output = ""
            end = False
            while start < self.maxlen:
                predicted_index = torch.argmax(predictions[start]).item()
                predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])
                logit_prob = softmax(predictions[start]).data.tolist()
                prob_result = {self.tokenizer.ids_to_tokens[id]: prob for id, prob in enumerate(logit_prob)}
                prob_result = sorted(prob_result.items(), key=lambda x: x[1], reverse=True)
                output_prob_dict.append(prob_result)
                if tok_sep(self.tokenizer) in predicted_token:
                    end = True
                if end is False:
                    output += predicted_token[0] + " "
                start += 1

        return output, output_prob_dict
