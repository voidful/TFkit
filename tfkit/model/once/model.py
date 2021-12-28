import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

from torch.nn.functional import softmax
from tfkit.model.once.dataloader import get_feature_from_data
from tfkit.utility.loss import *
from tfkit.utility.tok import *


class Model(nn.Module):
    def __init__(self, tokenizer, pretrained, maxlen=512, tasks_detail=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.vocab_size = max(self.pretrained.config.vocab_size, self.tokenizer.__len__())
        self.model = nn.Linear(self.pretrained.config.hidden_size, self.vocab_size)
        self.maxlen = maxlen

    def forward(self, batch_data, eval=False, **args):
        inputs = batch_data['input']
        targets = batch_data['target']
        negative_targets = batch_data['ntarget']
        masks = batch_data['mask']
        starts = batch_data['start']
        ends = batch_data['end']

        tokens_tensor = torch.as_tensor(inputs)
        mask_tensors = torch.as_tensor(masks)
        loss_tensors = torch.as_tensor(targets)
        negativeloss_tensors = torch.as_tensor(negative_targets)

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
            stop = False
            outputs = result_dict
            while start < self.maxlen and not stop:
                predicted_index = torch.argmax(prediction_scores[0][start]).item()
                predicted_token = self.tokenizer.decode([predicted_index])
                logit_prob = softmax(prediction_scores[0][start], dim=0)
                topK = torch.topk(logit_prob, 50)
                prob_result = [(self.tokenizer.convert_ids_to_tokens(id), prob) for prob, id in
                               zip(topK.values.data.tolist(), topK.indices.data.tolist())]

                result_dict['prob_list'].append(sorted(logit_prob, reverse=True))
                result_dict['label_prob_all'].append(dict(prob_result))
                result_dict['label_map'].append(prob_result[0])

                if tok_sep(self.tokenizer) in predicted_token:
                    stop = True
                start += 1
                outputs = result_dict
        else:
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
