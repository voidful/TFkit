from collections import OrderedDict

import nlp2
from tqdm import tqdm


def tok_begin(tokenizer):
    return tokenizer._cls_token or tokenizer._bos_token or 'cls'


def tok_sep(tokenizer):
    return tokenizer._sep_token or tokenizer._eos_token or 'sep'


def tok_mask(tokenizer):
    return tokenizer._mask_token or 'msk'


def get_topP_unk_token(tokenizer, file_paths: list, topP: float):
    unk_count_dict = OrderedDict()
    for path in file_paths:
        for input_sent in tqdm(nlp2.read_files_yield_lines(path)):
            for tok in nlp2.split_sentence_to_array(input_sent):
                if tokenizer._unk_token in tokenizer.tokenize(tok):
                    unk_count_dict[tok] = unk_count_dict.get(tok, 0) + 1
    top_range = int(len(unk_count_dict) * (topP / 100))
    return list(unk_count_dict.keys())[:top_range]
