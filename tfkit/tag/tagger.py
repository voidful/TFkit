import operator
import sys
import os
from collections import defaultdict

from torch.distributions import Categorical

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

from torch.nn.functional import softmax
from nlp2 import *
from tfkit.utility.loss import *
from tfkit.tag.data_loader import get_feature_from_data


class Tagger(nn.Module):

    def __init__(self, labels, tokenizer, pretrained, maxlen=512, dropout=0.2):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', self.device)
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.dropout = nn.Dropout(dropout)
        self.tagger = nn.Linear(self.pretrained.config.hidden_size, len(labels))
        self.labels = labels
        self.maxlen = maxlen
        # self.loss_fct = nn.CrossEntropyLoss()
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
            result_items = []
            for pos_logit_prob in result_labels:
                max_index = pos_logit_prob.index(max(pos_logit_prob))
                result_items.append(
                    [self.labels[max_index], dict(zip(self.labels, pos_logit_prob))])

            mapping = json.loads(batch_data['mapping'][0])
            start, end = batch_data['pos'][0]
            for map_token in mapping:
                char, pos = map_token['char'], map_token['pos']
                if pos > end:
                    break
                result_dict['label_map'].append({char: result_items[pos - start][0]})
                result_dict['label_prob_all'].append({char: result_items[pos - start][1]})

            outputs = result_dict
        else:
            targets = batch_data["target"]
            target_tensor = torch.as_tensor(targets, dtype=torch.long).to(self.device)
            loss = self.loss_fct(reshaped_logits.view(-1, len(self.labels)), target_tensor.view(-1))
            outputs = loss

        return outputs

    def predict(self, input='', neg="O", task=None, handle_exceed='slide',
                merge_strategy=['minentropy', 'maxcount', 'maxprob']):
        handle_exceed = handle_exceed[0] if isinstance(handle_exceed, list) else handle_exceed
        merge_strategy = merge_strategy[0] if isinstance(merge_strategy, list) else merge_strategy

        self.eval()
        with torch.no_grad():
            ret_detail = []
            predicted_pos_prob = defaultdict(lambda: defaultdict(list))
            for feature_dict in get_feature_from_data(tokenizer=self.tokenizer, labels=self.labels, input=input.strip(),
                                                      maxlen=self.maxlen, handle_exceed=handle_exceed):
                start, end = feature_dict['pos']
                for k, v in feature_dict.items():
                    feature_dict[k] = [v]
                result = self.forward(feature_dict, eval=True)
                pos_to_char = json.loads(feature_dict['mapping'][0])
                for ind, mapping in enumerate(result['label_prob_all']):
                    for map_pos, map_dict in mapping.items():
                        max_label = max(map_dict, key=map_dict.get)
                        max_prob = map_dict[max_label]
                        max_entropy = Categorical(probs=torch.tensor(list(map_dict.values()))).entropy().data.tolist()
                        predicted_pos_prob[str(ind + start)]['char'] = pos_to_char[ind]['char']
                        predicted_pos_prob[str(ind + start)]['labels'].append(max_label)
                        predicted_pos_prob[str(ind + start)]['prob'].append(max_prob)
                        predicted_pos_prob[str(ind + start)]['entropy'].append(max_entropy)
                ret_detail.append(result)

            ret_result = []
            for key, value in sorted(predicted_pos_prob.items()):
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
            for mapping in ret_result:
                for k, y in mapping.items():
                    if y is not neg:
                        target_str[0] += k
                        target_str[1] = y
                    else:
                        if len(target_str[0]) > 0:
                            output.append(target_str)
                        target_str = ["", ""]
            if len(target_str[0]) > 0:
                output.append(target_str)

            return output, ret_detail
