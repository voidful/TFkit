import sys
import os
from collections import defaultdict

from tfkit.task.once import Preprocessor
from tfkit.utility.predictor import NonAutoRegressivePredictor

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

from torch.nn.functional import softmax
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
        self.vocab_size = max(self.pretrained.config.vocab_size, self.tokenizer.__len__())
        self.model = nn.Linear(self.pretrained.config.hidden_size, self.vocab_size)
        predictor = NonAutoRegressivePredictor(self, Preprocessor)
        self.predictor = predictor
        self.predict = predictor.predict

    def clean_cache(self):
        self.encoder_outputs = None
        self.past_key_values = None

    def forward(self, batch_data, eval=False, max_return=1, **kwargs):
        inputs = batch_data['input']
        masks = batch_data['mask']
        starts = batch_data['start']
        ends = batch_data['end']
        tokens_tensor = torch.as_tensor(inputs)
        mask_tensors = torch.as_tensor(masks)

        output = self.pretrained(tokens_tensor, attention_mask=mask_tensors)
        sequence_output = output[0]
        prediction_scores = self.model(sequence_output)
        batch_size = list(tokens_tensor.shape)[0]
        prediction_scores = prediction_scores.view(batch_size, -1, self.vocab_size)

        if eval:
            result_dict = {
                'max_item': [],
                'label_prob': defaultdict(list),
                'prob_list': []
            }
            start = batch_data['start'][0]
            topK_ids = [[]] * max_return
            topK_probs = [1] * max_return

            pscore = prediction_scores.detach().cpu()
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

                softmax_score = softmax(prediction_scores[0][pos], dim=0)
                max_item_id = torch.argmax(softmax_score, -1).item()
                max_item_prob = softmax_score[max_item_id].item()
                if max_return > 1:
                    topK = torch.topk(softmax_score, max_return)
                    for k, (prob, tid) in enumerate(zip(topK.values.data.tolist(), topK.indices.data.tolist())):
                        topK_ids[k].append(tid)
                        topK_probs[k] *= prob
                else:
                    topK_ids[0].append(max_item_id)
                    topK_probs[0] *= max_item_prob
                start += 1

            result_dict['prob_list'] = topK_probs
            result_dict['label_prob'] = [[self.tokenizer.decode(ids), prob] for ids, prob in
                                         zip(topK_ids, topK_probs)]
            result_dict['max_item'] = [i[0] for i in result_dict['label_prob']]
            outputs = result_dict
        else:
            targets = batch_data['target']
            negative_targets = batch_data['ntarget']
            input_lengths = batch_data['input_length']
            target_lengths = batch_data['target_length']

            target_tensors = torch.as_tensor(targets)
            input_length_tensors = torch.as_tensor(input_lengths)
            target_length_tensors = torch.as_tensor(target_lengths)

            loss_tensors = torch.as_tensor(targets)
            negativeloss_tensors = torch.as_tensor(negative_targets)
            ctc_lm_loss = self.loss(prediction_scores,
                                    input_length_tensors,
                                    target_tensors.view(batch_size, -1),
                                    target_length_tensors)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size),
                                      loss_tensors.view(-1))
            if not torch.all(negativeloss_tensors.eq(-1)).item():
                negative_loss_fct = NegativeCElLoss()
                negative_loss = negative_loss_fct(prediction_scores.view(-1, self.vocab_size),
                                                  negativeloss_tensors.view(-1))
                masked_lm_loss += negative_loss
            outputs = ctc_lm_loss + masked_lm_loss

        return outputs
