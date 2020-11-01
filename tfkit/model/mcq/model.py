import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import torch
import torch.nn as nn
from tfkit.model.mcq.dataloader import get_feature_from_data


class Model(nn.Module):
    def __init__(self, tokenizer, pretrained, maxlen=512, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.model = nn.Linear(self.pretrained.config.hidden_size, 2)
        self.maxlen = maxlen
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', self.device)
        self.model.to(self.device)

    def batched_index_select(self, input, dim, index):
        for ii in range(1, len(input.shape)):
            if ii != dim:
                index = index.unsqueeze(ii)
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, batch_data, eval=False):
        inputs = batch_data['input']
        targets = batch_data['target']
        masks = batch_data['mask']
        targets_pos = batch_data['target_pos']
        tokens_tensor = torch.as_tensor(inputs).to(self.device)
        mask_tensors = torch.as_tensor(masks).to(self.device)
        loss_tensors = torch.as_tensor(targets).to(self.device)
        indices_tensor = torch.as_tensor(targets_pos).to(self.device)
        indices_tensor = indices_tensor[indices_tensor > 0].unsqueeze(0)
        output = self.pretrained(tokens_tensor, attention_mask=mask_tensors)
        sequence_output = output[0]
        prediction_scores = self.model(sequence_output)
        if eval:
            self.eval()
            result_dict = {
                'label_map': [],
                'label_max': []
            }
            prediction_scores = self.batched_index_select(prediction_scores, 1, indices_tensor)
            logit_prob = torch.softmax(prediction_scores, dim=1)
            maxprob_result = logit_prob[:, :, 1].argmax(1).item()
            result_dict['label_max'].append(str(maxprob_result))
            result_dict['label_map'] = logit_prob.data.tolist()
            return result_dict
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, 2), loss_tensors.view(-1))
            outputs = masked_lm_loss
        return outputs

    def predict(self, input='', task=None, handle_exceed='start_slice'):
        handle_exceed = handle_exceed[0] if isinstance(handle_exceed, list) else handle_exceed
        self.eval()
        with torch.no_grad():
            ret_result = []
            ret_detail = []
            for feature in get_feature_from_data(self.tokenizer, self.maxlen, input,
                                                 handle_exceed=handle_exceed):
                for k, v in feature.items():
                    feature[k] = [v]
                predictions = self.forward(feature, eval=True)
                ret_detail.append(predictions)
                ret_result.append(predictions['label_max'])
            non_empty_result = [r for r in ret_result if len(r) != 0]
            if len(non_empty_result) == 0:
                ret_result = ['-1']
            else:
                ret_result = max(non_empty_result, key=non_empty_result.count)
            return ret_result, ret_detail
