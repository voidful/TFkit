import os
import sys

from transformers import AutoModel

from tfkit.task.seq2seq import Preprocessor
from tfkit.utility.model import tie_encoder_decoder_weights
from tfkit.utility.predictor import AutoRegressivePredictor

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import torch
from torch import nn
from torch.nn.functional import softmax
from tfkit.utility.loss import NegativeCElLoss, SelfKDLoss
import copy


class Model(nn.Module):
    def __init__(self, tokenizer, pretrained, maxlen=512, selfkd=False, **kwargs):
        super().__init__()
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.selfkd = selfkd
        init_weight = None
        if hasattr(pretrained, 'decoder'):
            self.decoder_model = None
            decoder_hidden_size = pretrained.config.hidden_size
            if hasattr(pretrained, 'shared'):
                init_weight = copy.deepcopy(pretrained.shared.weight)
        else:
            decoder_config = copy.deepcopy(pretrained.config)
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True
            self.decoder_model = AutoModel.from_config(decoder_config)
            tie_encoder_decoder_weights(
                self.pretrained, self.decoder_model,
                self.decoder_model.base_model_prefix
            )
            decoder_hidden_size = decoder_config.hidden_size

        self.vocab_size = max(self.pretrained.config.vocab_size, self.tokenizer.__len__())
        self.model = nn.Linear(decoder_hidden_size, self.vocab_size, bias=False)
        if init_weight is not None:
            self.model.weight = init_weight
        self.encoder_hidden = None
        self.past_key_values = None
        predictor = AutoRegressivePredictor(self, Preprocessor)
        self.predictor = predictor
        self.predict = predictor.predict

    def forward(self, batch_data, eval=False, beamsearch=False, max_return=1, **kwargs):
        inputs = batch_data['input']
        prevs = batch_data['prev']
        encoder_mask = batch_data['encoder_mask']
        decoder_mask = batch_data['decoder_mask']

        input_tensors = torch.as_tensor(inputs)
        prev_tensors = torch.as_tensor(prevs)
        encoder_mask_tensors = torch.as_tensor(encoder_mask)
        decoder_mask_tensors = torch.as_tensor(decoder_mask)
        if self.decoder_model is not None:
            if eval and self.encoder_hidden is not None:
                encoder_hidden_states = self.encoder_hidden
            else:
                outputs = self.pretrained(input_tensors, attention_mask=encoder_mask_tensors)
                encoder_hidden_states = outputs[0]
                self.encoder_hidden = encoder_hidden_states
            # Decoder
            prediction = self.decoder_model(
                input_ids=prev_tensors,
                attention_mask=decoder_mask_tensors,
                encoder_hidden_states=encoder_hidden_states,
                output_hidden_states=self.selfkd,
                return_dict=True,
            )
            prediction_output = prediction['last_hidden_state']
            prediction_all_hidden = prediction.get('hidden_states')
        else:
            # if eval and self.encoder_hidden is not None and not beamsearch:
            #     prev_tensors = prev_tensors[..., -1:]
            #     prediction = self.pretrained(
            #         inputs_embeds=self.encoder_hidden,
            #         decoder_input_ids=prev_tensors,
            #         # decoder_attention_mask=decoder_mask_tensors,
            #         past_key_values=self.past_key_values,
            #         output_hidden_states=self.selfkd,
            #         use_cache=True,
            #         return_dict=True
            #     )
            # else:
            prediction = self.pretrained(
                input_ids=input_tensors,
                attention_mask=encoder_mask_tensors,
                decoder_input_ids=prev_tensors,
                decoder_attention_mask=decoder_mask_tensors,
                past_key_values=self.past_key_values,
                output_hidden_states=self.selfkd,
                use_cache=False,
                return_dict=True
            )
            prediction_output = prediction['last_hidden_state']
            prediction_all_hidden = prediction.get('decoder_hidden_states')

            # if eval and not beamsearch:
            #     self.encoder_hidden = prediction['encoder_last_hidden_state']
            #     self.past_key_values = prediction['past_key_values']

        prediction_scores = self.model(prediction_output)
        if eval:
            result_dict = {}
            softmax_score = softmax(prediction_scores[0][0], dim=0)
            max_item_id = torch.argmax(softmax_score, -1).item()
            max_item_prob = softmax_score[max_item_id].item()
            result_dict['max_item'] = (self.tokenizer.convert_ids_to_tokens(max_item_id), max_item_prob)
            if max_return > 1:
                topK = torch.topk(softmax_score, max_return)
                prob_result = [(self.tokenizer.convert_ids_to_tokens(tid), prob) for prob, tid in
                               zip(topK.values.data.tolist(), topK.indices.data.tolist())]
                result_dict['prob_list'] = softmax_score.data.tolist()[:max_return]
                result_dict['label_prob'] = prob_result
            outputs = result_dict
        else:
            targets = batch_data['target']
            negative_targets = batch_data['ntarget']
            loss_tensors = torch.as_tensor(targets)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size),
                               loss_tensors.view(-1))
            if self.selfkd:
                selfkdloss_fct = SelfKDLoss(ignore_index=-1)
                for decoder_hidden in prediction_all_hidden[:-1]:
                    student = self.model(decoder_hidden)
                    lm_loss += selfkdloss_fct(student.view(-1, self.vocab_size),
                                              prediction_scores.view(-1, self.vocab_size), loss_tensors.view(-1))

            negativeloss_tensors = torch.as_tensor(negative_targets)
            if not torch.all(negativeloss_tensors.eq(-1)).item():
                negative_loss_fct = NegativeCElLoss(ignore_index=-1)
                negative_loss = negative_loss_fct(prediction_scores.view(-1, self.vocab_size),
                                                  negativeloss_tensors.view(-1))
                lm_loss += negative_loss
            outputs = lm_loss
        return outputs


