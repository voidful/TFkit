import torch
import torch.nn as nn
from torch.nn.functional import softmax
import tfkit.utility.tok as tok
from tfkit.model.mask import get_feature_from_data


class Model(nn.Module):
    def __init__(self, tokenizer, pretrained, maxlen=512, **kwargs):
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
        masks = batch_data['mask']
        tokens_tensor = torch.as_tensor(inputs).to(self.device)
        mask_tensors = torch.as_tensor(masks).to(self.device)
        loss_tensors = torch.as_tensor(targets).to(self.device)
        output = self.pretrained(tokens_tensor, attention_mask=mask_tensors)
        sequence_output = output[0]
        prediction_scores = self.model(sequence_output)

        if eval:
            result_dict = {
                'label_map': [],
                'label_prob': []
            }
            for tok_pos, text in enumerate(batch_data['input'][0]):
                if text == self.tokenizer.convert_tokens_to_ids([tok.tok_mask(self.tokenizer)])[0]:
                    logit_prob = softmax(prediction_scores[0][tok_pos], dim=0).data.tolist()
                    prob_result = sorted(logit_prob, reverse=True)[:100]
                    prob_result = [(self.tokenizer.decode([logit_prob.index(prob)]), prob) for prob in prob_result]
                    result_dict['label_prob'].append(prob_result[:10])
                    result_dict['label_map'].append(prob_result[0])
            return result_dict
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.pretrained.config.vocab_size),
                                      loss_tensors.view(-1))
            outputs = masked_lm_loss
        return outputs

    def predict(self, input='', task=None, handle_exceed='slide'):
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
                ret_result.append([i[0] for i in predictions['label_map']])
            non_empty_result = [r for r in ret_result if len(r) != 0]
            if len(non_empty_result) == 0:
                ret_result = [['']]
            else:
                ret_result = max(non_empty_result, key=non_empty_result.count)
            return ret_result, ret_detail
