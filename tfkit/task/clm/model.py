import torch
from torch import nn
from torch.nn.functional import softmax

from tfkit.task.clm import Preprocessor
from tfkit.utility.base_model import BaseTFKitModel
from tfkit.utility.predictor import AutoRegressivePredictor


class Model(BaseTFKitModel):
    """Causal Language Model for text generation."""
    
    def __init__(self, tokenizer, pretrained, maxlen=512, **kwargs):
        super().__init__(tokenizer, pretrained, maxlen, **kwargs)
        self.model = self._resolve_output_head()
        self.uses_pretrained_head = self.model is not None
        if not self.uses_pretrained_head:
            self.model = nn.Linear(self.get_hidden_size(), self.get_vocab_size())

        self._setup_predictor(AutoRegressivePredictor, Preprocessor)

    def _resolve_output_head(self):
        """Return the pretrained language modeling head if available."""

        if hasattr(self.pretrained, "get_output_embeddings"):
            output_embeddings = self.pretrained.get_output_embeddings()
            if output_embeddings is not None:
                return output_embeddings
        if hasattr(self.pretrained, "lm_head"):
            return self.pretrained.lm_head
        if hasattr(self.pretrained, "cls"):
            return self.pretrained.cls
        return None

    def forward(self, batch_data, eval=False, beamsearch=False, max_return=1, **kwargs):
        inputs = batch_data['input']
        masks = batch_data['mask']
        tokens_tensor = torch.as_tensor(inputs)
        mask_tensors = torch.as_tensor(masks)
        model_kwargs = {
            'attention_mask': mask_tensors,
            'return_dict': True,
        }
        if eval:
            model_kwargs['use_cache'] = False

        if eval:
            outputs = self.pretrained(tokens_tensor, **model_kwargs)
            prediction_scores = outputs['logits'] if 'logits' in outputs else outputs[0]
        else:
            targets = batch_data['target']
            loss_tensors = torch.as_tensor(targets)

            if self.uses_pretrained_head:
                labels = loss_tensors.clone().long()
                labels[labels == -1] = -100
                model_kwargs['labels'] = labels
                outputs = self.pretrained(tokens_tensor, **model_kwargs)
                prediction_scores = outputs['logits'] if 'logits' in outputs else outputs[0]
                masked_lm_loss = outputs['loss']
            else:
                loss_tensors = loss_tensors.long()
                outputs = self.pretrained(tokens_tensor, **model_kwargs)
                hidden_states = outputs['last_hidden_state'] if 'last_hidden_state' in outputs else outputs[0]
                prediction_scores = self.model(hidden_states)
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size),
                                          loss_tensors.view(-1))

        if eval:
            result_dict = {}
            start = batch_data['start'][0]
            softmax_score = softmax(prediction_scores[0][start], dim=-1).flatten()
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
            outputs = masked_lm_loss
        return outputs
