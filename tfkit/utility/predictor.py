import json

import torch

from tfkit.utility import tok, batch_reduce_pad
import numpy as np
from math import log
from itertools import combinations


def _jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def _is_similar(s, t):
    return _jaccard_similarity(s, t) > 0.5


def _filter_similar(d, top_k):
    while True:
        filtered = False
        for s, t in combinations(d, 2):
            if _is_similar(s[0], t[0]) and len(d) - 1 >= top_k:
                d.remove(t)
                filtered = True
                break
        if not filtered:
            break


class Predictor:
    def __init__(self, model, get_feature_from_data):
        self.get_feature_from_data = get_feature_from_data
        self.model = model

    def gen_predict(self, input='', topK=1, topP=0.85, mode=['greedy', 'topK', 'topP'], decodenum=1,
                    filtersim=False, reserved_len=0, task=None, handle_exceed='noop', eos_num=1, decode_maxlen=0,
                    no_repeat=False):
        filtersim = json.loads(str(filtersim).lower())
        topK = int(topK)
        eos_num = int(eos_num)
        topP = float(topP)
        decodenum = int(decodenum)
        decode_maxlen = int(decode_maxlen)
        mode = mode[0] if isinstance(mode, list) else mode.lower()
        previous = []
        if tok.UNIVERSAL_SEP in input:
            previous = self.model.tokenizer.tokenize(input.split(tok.UNIVERSAL_SEP)[-1])
            eos_num += 1
        sep_tok = tok.tok_sep(self.model.tokenizer)
        sequences = [[[], 1.0]]
        with torch.no_grad():
            while True:
                all_candidates = list()
                exceed = False
                for seq in sequences:
                    if sep_tok not in seq[0] or seq[0].count(sep_tok) < eos_num:
                        tokens, score = seq
                        if not tokens:
                            tokens = previous
                        feature_dict = \
                            self.get_feature_from_data(self.model.tokenizer, self.model.maxlen, input, tokens,
                                                       reserved_len=reserved_len,
                                                       handle_exceed=handle_exceed)[-1]
                        # check input exceed
                        if len(tokens) >= self.model.maxlen or feature_dict['start'] >= self.model.maxlen:
                            exceed = True
                            all_candidates.append(seq)
                            continue

                        feature_dict = batch_reduce_pad([feature_dict])[0]
                        for k, v in feature_dict.items():
                            feature_dict[k] = [v]
                        predictions = self.model.forward(feature_dict, eval=True, use_prev=True,
                                                         beamsearch=decodenum > 1,
                                                         return_topN_prob=max(decodenum, topK))
                        label_prob = predictions['label_prob'] if 'label_prob' in predictions else \
                            [predictions['max_item']]
                        # topK topP
                        if 'top' in mode:
                            prob_list = predictions['prob_list']
                            if 'topk' in mode:
                                sample_list = prob_list[:topK]
                                decode_range = max(decodenum, topK)
                                prob_norm = [float(i) / sum(sample_list) for i in sample_list]
                                choice_list = np.random.choice(sample_list, p=prob_norm,
                                                               size=decode_range,
                                                               replace=False)
                            else:
                                topP_list = np.cumsum(prob_list)
                                index_overP = [i for i, x in enumerate(topP_list) if x > topP]
                                index_overP = 0 if len(index_overP) < 1 else index_overP[0]
                                sample_list = prob_list[:index_overP + 1]
                                prob_norm = [float(i) / sum(sample_list) for i in sample_list]
                                choice_list = np.random.choice(sample_list, p=prob_norm,
                                                               size=decodenum)
                            for idx in range(decodenum):
                                sampling_index = prob_list.index(choice_list[idx])
                                k, v = label_prob[sampling_index]
                                candidate = [tokens + [k], score + -log(v)]
                                all_candidates.append(candidate)

                        # greedy / beam search
                        else:
                            for k, v in label_prob[:50]:
                                if (no_repeat and len(tokens) > 0 and tokens[-1] == k) or len(k) < 1:
                                    continue
                                candidate = [tokens + [k], score + -log(v) if v > 0 else 0]
                                all_candidates.append(candidate)
                    else:
                        all_candidates.append(seq)

                ordered = sorted(all_candidates, key=lambda tup: tup[1])
                # ordered = all_candidates
                if filtersim:  # default disable, it will took a lots of time on long sequence generation
                    _filter_similar(ordered, decodenum)
                sequences = ordered[:decodenum]
                stop = 0
                for i in sequences:
                    # i[0] - sequence,i[1] - sequence score
                    if (i[0].count(sep_tok) >= eos_num) or \
                            len(i[0]) > self.model.maxlen or \
                            0 < decode_maxlen < len(i[0]):
                        stop += 1
                if stop == len(sequences) or exceed:
                    break

            for i in range(len(sequences)):
                if sep_tok in sequences[i][0]:  # remove sep token
                    sequences[i][0] = sequences[i][0][:-1]
                slide_len = len(previous) if len(previous) > 0 else 0
                sequences[i][0] = self.model.tokenizer.decode(
                    self.model.tokenizer.convert_tokens_to_ids(sequences[i][0][slide_len:]))

            result_dict = {
                'label_map': sequences
            }
            self.model.encoder_hidden = None
            self.model.past_key_values = None
            return [i[0] for i in sequences], [result_dict]
