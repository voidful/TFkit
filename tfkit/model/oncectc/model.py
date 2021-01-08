import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

from torch.nn.functional import softmax
from tfkit.model.oncectc.dataloader import get_feature_from_data
from tfkit.utility.loss import *
from tfkit.utility.tok import *
from tfkit.utility.loss import SeqCTCLoss


class Model(nn.Module):
    def __init__(self, tokenizer, pretrained, maxlen=512, tasks_detail=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.maxlen = maxlen
        self.blank_token = "<BLANK>"
        self.tokenizer.add_tokens(self.blank_token)
        self.pretrained.resize_token_embeddings(len(tokenizer))
        self.blank_index = self.tokenizer.convert_tokens_to_ids([self.blank_token])[0]
        self.loss = SeqCTCLoss(blank_index=self.blank_index)
        self.model = nn.Linear(self.pretrained.config.hidden_size, self.tokenizer.__len__())
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', self.device)
        self.model.to(self.device)

    def forward(self, batch_data, eval=False):
        inputs = batch_data['input']
        targets = batch_data['target']
        targets_once = batch_data['target_once']
        masks = batch_data['mask']
        input_lengths = batch_data['input_length']
        target_lengths = batch_data['target_length']

        tokens_tensor = torch.as_tensor(inputs).to(self.device)
        mask_tensors = torch.as_tensor(masks).to(self.device)
        target_tensors = torch.as_tensor(targets).to(self.device)
        target_once_tensors = torch.as_tensor(targets_once).to(self.device)
        input_length_tensors = torch.as_tensor(input_lengths).to(self.device)
        target_length_tensors = torch.as_tensor(target_lengths).to(self.device)
        output = self.pretrained(tokens_tensor, attention_mask=mask_tensors)
        sequence_output = output[0]
        prediction_scores = self.model(sequence_output)

        batch_size = list(tokens_tensor.shape)[0]
        prediction_scores = prediction_scores.view(batch_size, -1, self.pretrained.config.vocab_size)
        if eval:
            pscore = prediction_scores.detach().cpu()
            result_dict = {
                'label_prob_all': [],
                'label_map': [],
                'prob_list': []
            }
            predicted_indexs = pscore.argmax(2).tolist()[0]
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indexs)
            output = []
            for pos, (predicted_index, predicted_token) in enumerate(zip(predicted_indexs, predicted_tokens)):
                if len(output) > 0 and predicted_index == output[-1]:
                    continue
                if predicted_token == self.blank_token:
                    continue
                if predicted_token == tok_pad(self.tokenizer):
                    continue
                if predicted_token == tok_sep(self.tokenizer):
                    break

                topK = torch.topk(softmax(prediction_scores[0][pos], dim=0), 50)
                logit_prob = softmax(prediction_scores[0][pos], dim=0).data.tolist()
                prob_result = [(self.tokenizer.convert_ids_to_tokens(id), prob) for prob, id in
                               zip(topK.values.data.tolist(), topK.indices.data.tolist())]
                result_dict['prob_list'].append(logit_prob)
                result_dict['label_prob_all'].append(prob_result)
                result_dict['label_map'].append(prob_result[0])

            outputs = result_dict
        else:
            ctc_lm_loss = self.loss(prediction_scores,
                                    input_length_tensors,
                                    target_tensors.view(batch_size, -1),
                                    target_length_tensors)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.pretrained.config.vocab_size),
                                      target_once_tensors.view(-1))

            outputs = ctc_lm_loss + masked_lm_loss

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
