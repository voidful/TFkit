import sys
import os

from transformers import AutoModel
from typing import List

from tfkit.utility.predictor import Predictor

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import torch
from torch import nn
from tfkit.model.seq2seq.dataloader import get_feature_from_data
from torch.nn.functional import softmax
from torch.nn import functional as F
import copy


class Model(nn.Module):
    def __init__(self, tokenizer, pretrained, maxlen=512, selfkd=False, **kwargs):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.selfkd = selfkd
        init_weight = None
        print('Using device:', self.device)
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
            self._tie_encoder_decoder_weights(
                self.pretrained, self.decoder_model,
                self.decoder_model.base_model_prefix
            )
            decoder_hidden_size = decoder_config.hidden_size
            self.decoder_model.to(self.device)

        self.vocab_size = max(self.pretrained.config.vocab_size, self.tokenizer.__len__())
        self.model = nn.Linear(decoder_hidden_size, self.vocab_size, bias=False)
        if init_weight is not None:
            self.model.weight = init_weight
        self.model.to(self.device)
        self.encoder_hidden = None
        self.past_key_values = None
        predictor = Predictor(self, get_feature_from_data)
        self.predict = predictor.gen_predict

    def forward(self, batch_data, eval=False, beamsearch=False, return_topN_prob=1, **args):

        inputs = batch_data['input']
        prevs = batch_data['prev']
        encoder_mask = batch_data['encoder_mask']
        decoder_mask = batch_data['decoder_mask']

        input_tensors = torch.as_tensor(inputs).to(self.device)
        prev_tensors = torch.as_tensor(prevs).to(self.device)
        encoder_mask_tensors = torch.as_tensor(encoder_mask).to(self.device)
        decoder_mask_tensors = torch.as_tensor(decoder_mask).to(self.device)

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
        else:
            if eval and self.encoder_hidden is not None and not beamsearch:
                prev_tensors = prev_tensors[..., -1:]
                batch_data['start'][0] = 0
                prediction = self.pretrained(
                    inputs_embeds=self.encoder_hidden,
                    decoder_input_ids=prev_tensors,
                    decoder_attention_mask=decoder_mask_tensors,
                    past_key_values=self.past_key_values,
                    output_hidden_states=self.selfkd,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                prediction = self.pretrained(
                    input_ids=input_tensors,
                    attention_mask=encoder_mask_tensors,
                    decoder_input_ids=prev_tensors,
                    decoder_attention_mask=decoder_mask_tensors,
                    past_key_values=self.past_key_values,
                    output_hidden_states=self.selfkd,
                    use_cache=True,
                    return_dict=True
                )
            prediction_output = prediction['last_hidden_state']
            self.encoder_hidden = prediction['encoder_last_hidden_state']
            if eval and not beamsearch:
                self.past_key_values = prediction['past_key_values']

        prediction_scores = self.model(prediction_output)
        if eval:
            result_dict = {}
            start = batch_data['start'][0]
            softmax_score = softmax(prediction_scores[0][start], dim=0)
            max_item_id = torch.argmax(softmax_score, -1).item()
            max_item_prob = softmax_score[max_item_id].item()
            result_dict['max_item'] = (self.tokenizer.convert_ids_to_tokens(max_item_id), max_item_prob)
            if return_topN_prob > 1:
                topK = torch.topk(softmax_score, return_topN_prob)
                prob_result = [(self.tokenizer.convert_ids_to_tokens(id), prob) for prob, id in
                               zip(topK.values.data.tolist(), topK.indices.data.tolist())]
                result_dict['prob_list'] = softmax_score.data.tolist()[:return_topN_prob]
                result_dict['label_prob'] = prob_result
            outputs = result_dict
        else:
            targets = batch_data['target']
            backtran_targets = batch_data['btarget']
            loss_tensors = torch.as_tensor(targets).to(self.device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size),
                               loss_tensors.view(-1))

            backtran_tensors = torch.as_tensor(backtran_targets).to(self.device)
            if not torch.all(backtran_tensors.eq(-1)).item():
                backtran_predation = self.pretrained(
                    input_ids=backtran_tensors,
                    output_hidden_states=True,
                    return_dict=True
                )
                backtran_hidden = backtran_predation[
                    'encoder_last_hidden_state'] if 'encoder_last_hidden_state' in backtran_predation else prediction[
                    'last_hidden_state']
                backtran_loss = F.cosine_similarity(self.encoder_hidden, backtran_hidden).mean()
                lm_loss += backtran_loss
            outputs = lm_loss
        return outputs

    def _tie_encoder_decoder_weights(self, encoder, decoder, base_model_prefix):
        uninitialized_encoder_weights: List[str] = []
        if decoder.__class__ != encoder.__class__:
            print(
                f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
            )

        def tie_encoder_to_decoder_recursively(
                decoder_pointer: nn.Module,
                encoder_pointer: nn.Module,
                module_name: str,
                uninitialized_encoder_weights: List[str],
                depth=0,
        ):
            assert isinstance(decoder_pointer, nn.Module) and isinstance(
                encoder_pointer, nn.Module
            ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
            if hasattr(decoder_pointer, "weight"):
                assert hasattr(encoder_pointer, "weight")
                encoder_pointer.weight = decoder_pointer.weight
                if hasattr(decoder_pointer, "bias"):
                    assert hasattr(encoder_pointer, "bias")
                    encoder_pointer.bias = decoder_pointer.bias
                return

            encoder_modules = encoder_pointer._modules
            decoder_modules = decoder_pointer._modules
            if len(decoder_modules) > 0:
                assert (
                        len(encoder_modules) > 0
                ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

                all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
                encoder_layer_pos = 0
                for name, module in decoder_modules.items():
                    if name.isdigit():
                        encoder_name = str(int(name) + encoder_layer_pos)
                        decoder_name = name
                        if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                                encoder_modules
                        ) != len(decoder_modules):
                            # this can happen if the name corresponds to the position in a list module list of layers
                            # in this case the decoder has added a cross-attention that the encoder does not have
                            # thus skip this step and subtract one layer pos from encoder
                            encoder_layer_pos -= 1
                            continue
                    elif name not in encoder_modules:
                        continue
                    elif depth > 500:
                        raise ValueError(
                            "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                        )
                    else:
                        decoder_name = encoder_name = name
                    tie_encoder_to_decoder_recursively(
                        decoder_modules[decoder_name],
                        encoder_modules[encoder_name],
                        module_name + "/" + name,
                        uninitialized_encoder_weights,
                        depth=depth + 1,
                    )
                    all_encoder_weights.remove(module_name + "/" + encoder_name)

                uninitialized_encoder_weights += list(all_encoder_weights)

        # tie weights recursively
        tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights)
        if len(uninitialized_encoder_weights) > 0:
            print(
                f"The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}"
            )
        else:
            print("All encoder weights tied to the decoder")
