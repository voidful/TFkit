import sys
import os
from collections import defaultdict

import nlp2
from torch.distributions import Categorical

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

from torch.nn.functional import softmax
from tfkit.utility.loss import *
from tfkit.model.tag.dataloader import get_feature_from_data


class Model(nn.Module):

    def __init__(self, tokenizer, pretrained, tasks_detail, maxlen=512, dropout=0.2, **kwargs):
        super().__init__()
        labels = list(tasks_detail.values())[0]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', self.device)
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.dropout = nn.Dropout(dropout)
        self.tagger = nn.Linear(self.pretrained.config.hidden_size, len(labels))
        self.labels = labels
        self.maxlen = maxlen
        self.loss_fct = FocalLoss()

        self.pretrained = self.pretrained.to(self.device)
        self.loss_fct = self.loss_fct.to(self.device)

    def forward(self, batch_data, eval=False, separator=" "):
        inputs = batch_data["input"]
        masks = batch_data["mask"]

        # bert embedding
        token_tensor = torch.as_tensor(inputs, dtype=torch.long).to(self.device)
        mask_tensors = torch.as_tensor(masks).to(self.device)
        bert_output = self.pretrained(token_tensor, attention_mask=mask_tensors)
        res = bert_output[0]
        pooled_output = self.dropout(res)
        reshaped_logits = self.tagger(pooled_output)

        if eval:
            result_dict = {
                'label_prob_all': [],
                'label_map': []
            }

            ilogit = softmax(reshaped_logits[0], dim=1)
            result_labels = ilogit.data.tolist()
            token_word_mapping = batch_data['token_word_mapping'][0]
            start, end = batch_data['pos'][0]
            for pos, logit_prob in enumerate(result_labels[1:]):  # skip cls and sep
                if pos >= len(token_word_mapping):
                    break
                word = token_word_mapping[start + pos]['word']
                max_index = logit_prob.index(max(logit_prob))
                result_dict['label_map'].append({word: self.labels[max_index]})
                result_dict['label_prob_all'].append({word: dict(zip(self.labels, logit_prob))})
            result_dict['token_word_mapping'] = token_word_mapping[start:end]
            outputs = result_dict
        else:
            targets = batch_data["target"]
            target_tensor = torch.as_tensor(targets, dtype=torch.long).to(self.device)
            loss = self.loss_fct(reshaped_logits.view(-1, len(self.labels)), target_tensor.view(-1))
            outputs = loss

        return outputs

    def predict(self, input='', neg="O", task=None, handle_exceed='slide',
                merge_strategy=['minentropy', 'maxprob', 'maxcount'], minlen=1, start_contain="B", end_contain="I"):
        handle_exceed = handle_exceed[0] if isinstance(handle_exceed, list) else handle_exceed
        merge_strategy = merge_strategy[0] if isinstance(merge_strategy, list) else merge_strategy
        self.eval()
        input = " ".join(nlp2.split_sentence_to_array(input))
        with torch.no_grad():
            ret_detail = []
            predicted_pos_prob = defaultdict(lambda: defaultdict(list))
            for feature_dict in get_feature_from_data(tokenizer=self.tokenizer, labels=self.labels, input=input.strip(),
                                                      maxlen=self.maxlen, handle_exceed=handle_exceed):
                for k, v in feature_dict.items():
                    feature_dict[k] = [v]
                result = self.forward(feature_dict, eval=True)
                for token_pred, token_map in zip(result['label_prob_all'], result['token_word_mapping']):
                    token_prob = list(token_pred.values())[0]
                    max_label = max(token_prob, key=token_prob.get)
                    max_prob = token_prob[max_label]
                    max_entropy = Categorical(probs=torch.tensor(list(token_prob.values()))).entropy().data.tolist()
                    predicted_pos_prob[token_map['pos']]['char'] = token_map['word']
                    predicted_pos_prob[token_map['pos']]['labels'].append(max_label)
                    predicted_pos_prob[token_map['pos']]['prob'].append(max_prob)
                    predicted_pos_prob[token_map['pos']]['entropy'].append(max_entropy)

                ret_detail.append(result)

            ret_result = []
            for key, value in predicted_pos_prob.items():
                if merge_strategy == 'maxcount':
                    label = max(value['labels'], key=value['labels'].count)
                if merge_strategy == 'minentropy':
                    min_entropy_index = value['entropy'].index(min(value['entropy']))
                    label = value['labels'][min_entropy_index]
                if merge_strategy == 'maxprob':
                    max_prob_index = value['prob'].index(max(value['prob']))
                    label = value['labels'][max_prob_index]
                ret_result.append({value['char']: label})

            output = []
            target_str = ["", ""]
            after_start = False
            for mapping in ret_result:
                for k, y in mapping.items():
                    if start_contain in y:
                        after_start = True
                        if len(target_str[0]) > 0:
                            if len(target_str[0]) > minlen:
                                output.append(target_str)
                            target_str = ["", ""]
                        target_str[0] += k
                        target_str[1] = y
                    elif y is not neg and after_start:
                        target_str[0] += k
                        target_str[1] = y
                    else:
                        after_start = False
            if len(target_str[0]) > minlen and target_str not in output:
                output.append(target_str)
            output = [[ner, tag.replace(start_contain, "").replace(end_contain, "")] for ner, tag in output]
            return output, ret_detail
