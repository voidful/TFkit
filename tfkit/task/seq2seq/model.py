import copy

import torch
from torch import nn
from torch.nn.functional import softmax
from transformers import AutoModel

from tfkit.task.seq2seq import Preprocessor
from tfkit.utility.loss import NegativeCElLoss, SelfKDLoss
from tfkit.utility.model import tie_encoder_decoder_weights
from tfkit.utility.predictor import AutoRegressivePredictor


class Model(nn.Module):
    def __init__(self, tokenizer, pretrained, maxlen=512, selfkd=False, **kwargs):
        super().__init__()
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.selfkd = selfkd
        self.decoder_model, init_weight = self.initialize_decoder()
        self.vocab_size = max(self.pretrained.config.vocab_size, self.tokenizer.__len__())
        self.model = nn.Linear(self.decoder_hidden_size, self.vocab_size, bias=False)
        if init_weight is not None:
            self.model.weight = init_weight
        self.predictor = AutoRegressivePredictor(self, Preprocessor)
        self.predict = self.predictor.predict

    def initialize_decoder(self):
        init_weight = None

        if hasattr(self.pretrained, 'decoder'):
            decoder_model = None
            self.decoder_hidden_size = self.pretrained.config.hidden_size
            if hasattr(self.pretrained, 'shared'):
                init_weight = copy.deepcopy(self.pretrained.shared.weight)
        else:
            decoder_config = copy.deepcopy(self.pretrained.config)
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True
            decoder_model = AutoModel.from_config(decoder_config)
            tie_encoder_decoder_weights(self.pretrained, decoder_model, decoder_model.base_model_prefix)
            self.decoder_hidden_size = decoder_config.hidden_size

        return decoder_model, init_weight

    def forward(self, batch_data, eval=False, beamsearch=False, max_return=1, **kwargs):
        if self.decoder_model:
            prediction_output, prediction_all_hidden = self.decoder_forward(batch_data, eval)
        else:
            prediction_output, prediction_all_hidden = self.encoder_forward(batch_data, eval, beamsearch)

        prediction_scores = self.model(prediction_output)

        if eval:
            outputs = self.process_eval_output(prediction_scores, max_return)
        else:
            outputs = self.calculate_loss(batch_data, prediction_scores, prediction_all_hidden)
        return outputs

    def decoder_forward(self, batch_data, eval):
        input_tensors = torch.as_tensor(batch_data['input'])
        prev_tensors = torch.as_tensor(batch_data['prev'])
        encoder_mask_tensors = torch.as_tensor(batch_data['encoder_mask'])
        decoder_mask_tensors = torch.as_tensor(batch_data['decoder_mask'])

        if not eval:
            outputs = self.pretrained(input_tensors, attention_mask=encoder_mask_tensors)
        prediction = self.decoder_model(
            input_ids=prev_tensors,
            attention_mask=decoder_mask_tensors,
            output_hidden_states=self.selfkd,
            use_cache=False,
            return_dict=True,
        )
        prediction_output = prediction['last_hidden_state']
        prediction_all_hidden = prediction.get('hidden_states')
        return prediction_output, prediction_all_hidden

    def encoder_forward(self, batch_data, eval, beamsearch):
        input_tensors = torch.as_tensor(batch_data['input'])
        prev_tensors = torch.as_tensor(batch_data['prev'])
        encoder_mask_tensors = torch.as_tensor(batch_data['encoder_mask'])
        decoder_mask_tensors = torch.as_tensor(batch_data['decoder_mask'])

        prediction = self.pretrained(
            input_ids=input_tensors,
            attention_mask=encoder_mask_tensors,
            decoder_input_ids=prev_tensors,
            decoder_attention_mask=decoder_mask_tensors,
            output_hidden_states=self.selfkd,
            use_cache=False,
            return_dict=True
        )
        prediction_output = prediction['last_hidden_state']
        prediction_all_hidden = prediction.get('decoder_hidden_states')
        return prediction_output, prediction_all_hidden

    def process_eval_output(self, prediction_scores, max_return):
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

        return result_dict

    def calculate_loss(self, batch_data, prediction_scores, prediction_all_hidden):
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

        if 'btarget' in batch_data:
            backtran_tensors = torch.as_tensor(batch_data['btarget'])
            if not torch.all(backtran_tensors.eq(-1)).item():
                backtran_predation = self.pretrained(
                    input_ids=backtran_tensors,
                    output_hidden_states=True,
                    return_dict=True
                )
                backtran_hidden = backtran_predation['encoder_last_hidden_state']
                backtran_loss = F.cosine_similarity(self.encoder_hidden, backtran_hidden).mean()
                lm_loss += backtran_loss

        negativeloss_tensors = torch.as_tensor(negative_targets)
        if not torch.all(negativeloss_tensors.eq(-1)).item():
            negative_loss_fct = NegativeCElLoss(ignore_index=-1)
            negative_loss = negative_loss_fct(prediction_scores.view(-1, self.vocab_size),
                                              negativeloss_tensors.view(-1))
            lm_loss += negative_loss

        return lm_loss
