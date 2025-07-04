import torch
import torch.nn as nn
from torch.nn.functional import softmax

from tfkit.task.qa.preprocessor import Preprocessor
from tfkit.utility.base_model import BaseTFKitModel
from tfkit.utility.constants import DEFAULT_MAXLEN, DEFAULT_DROPOUT
from tfkit.utility.predictor import QuestionAnsweringPredictor


class Model(BaseTFKitModel):
    """Question Answering model for extractive QA tasks."""

    def __init__(self, tokenizer, pretrained, maxlen: int = DEFAULT_MAXLEN, 
                 dropout: float = DEFAULT_DROPOUT, **kwargs):
        # QA models typically use smaller max length
        if maxlen == DEFAULT_MAXLEN:
            maxlen = 128
        super().__init__(tokenizer, pretrained, maxlen, **kwargs)
        
        self.dropout = nn.Dropout(dropout)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.qa_classifier = nn.Linear(self.get_hidden_size(), 2)
        
        self._setup_predictor(QuestionAnsweringPredictor, Preprocessor)

    def forward(self, batch_data, eval=False, **kwargs):
        inputs = torch.as_tensor(batch_data['input'])
        masks = torch.as_tensor(batch_data['mask'])
        targets = torch.as_tensor(batch_data['target'])
        start_positions, end_positions = targets.split(1, dim=1)
        start_positions = start_positions.squeeze(1)
        end_positions = end_positions.squeeze(1)

        output = self.pretrained(inputs, attention_mask=masks)[0]
        logits = self.qa_classifier(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if eval:
            result_dict = {
                'label_prob_all': [],
                'label_map': []
            }
            reshaped_start_logits = softmax(start_logits, dim=1)
            reshaped_end_logits = softmax(end_logits, dim=1)
            start_prob = reshaped_start_logits.data.tolist()[0]
            end_prob = reshaped_end_logits.data.tolist()[0]
            result_dict['label_prob_all'].append({'start': dict(zip(range(len(start_prob)), start_prob)),
                                                  'end': dict(zip(range(len(end_prob)), end_prob))})
            result_dict['label_map'].append({'start': start_prob.index(max(start_prob)),
                                             'end': end_prob.index(max(end_prob))})
            outputs = result_dict
        else:
            start_loss = self.loss_fct(start_logits, start_positions)
            end_loss = self.loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = total_loss

        return outputs
