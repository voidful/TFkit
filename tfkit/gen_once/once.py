import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

from torch.nn.functional import softmax
from tfkit.gen_once.data_loader import get_feature_from_data
from tfkit.utility.loss import *
from tfkit.utility.tok import *


class Once(nn.Module):
    def __init__(self, tokenizer, pretrained, maxlen=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.model = nn.Linear(self.pretrained.config.hidden_size, self.tokenizer.__len__())
        self.maxlen = maxlen
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', self.device)
        self.model.to(self.device)

    def forward(self, batch_data, eval=False):
        inputs = batch_data['input']
        targets = batch_data['target']
        negative_targets = batch_data['ntarget']
        masks = batch_data['mask']

        tokens_tensor = torch.as_tensor(inputs).to(self.device)
        mask_tensors = torch.as_tensor(masks).to(self.device)
        loss_tensors = torch.as_tensor(targets).to(self.device)
        negativeloss_tensors = torch.as_tensor(negative_targets).to(self.device)

        output = self.pretrained(tokens_tensor, attention_mask=mask_tensors)
        sequence_output = output[0]
        prediction_scores = self.model(sequence_output)

        if eval:
            result_dict = {
                'label_prob_all': [],
                'label_map': [],
                'prob_list': []
            }
            start = batch_data['start'][0]
            end = False
            outputs = result_dict
            while start < self.maxlen and not end:
                predicted_index = torch.argmax(prediction_scores[0][start]).item()
                predicted_token = self.tokenizer.decode([predicted_index])
                logit_prob = softmax(prediction_scores[0][start], dim=0).data.tolist()
                prob_result = {self.tokenizer.decode([id]): prob for id, prob in enumerate(logit_prob)}
                prob_result = sorted(prob_result.items(), key=lambda x: x[1], reverse=True)

                result_dict['prob_list'].append(sorted(logit_prob, reverse=True))
                result_dict['label_prob_all'].append(dict(prob_result))
                result_dict['label_map'].append(prob_result[0])

                if tok_sep(self.tokenizer) in predicted_token:
                    end = True
                start += 1
                outputs = result_dict
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.pretrained.config.vocab_size),
                                      loss_tensors.view(-1))

            negative_loss_fct = NegativeCElLoss().to(self.device)
            negative_loss = negative_loss_fct(prediction_scores.view(-1, self.pretrained.config.vocab_size),
                                              negativeloss_tensors.view(-1))
            masked_lm_loss += negative_loss
            outputs = masked_lm_loss

        return outputs

    def predict(self, input='', task=None, handle_exceed='start_slice'):
        handle_exceed = handle_exceed[0] if isinstance(handle_exceed, list) else handle_exceed
        self.eval()
        with torch.no_grad():
            ret_result = []
            ret_detail = []
            feature = get_feature_from_data(self.tokenizer, self.maxlen, input, handle_exceed=handle_exceed)[-1]
            for k, v in feature.items():
                feature[k] = [v]
            predictions = self.forward(feature, eval=True)
            output = "".join(self.tokenizer.convert_tokens_to_string([i[0] for i in predictions['label_map']]))
            ret_result.append(output)
            ret_detail.append(predictions)
            return ret_result, ret_detail
