from collections import OrderedDict

import nlp2
from tqdm import tqdm


def tok_begin(tokenizer):
    if isinstance(tokenizer._cls_token, str):
        return tokenizer._cls_token
    elif isinstance(tokenizer._bos_token, str):
        return tokenizer._bos_token
    return 'cls'


def tok_sep(tokenizer):
    if isinstance(tokenizer._sep_token, str):
        return tokenizer._sep_token
    elif isinstance(tokenizer._eos_token, str):
        return tokenizer._eos_token
    return 'sep'


def tok_mask(tokenizer):
    if isinstance(tokenizer._mask_token, str):
        return tokenizer._mask_token
    return 'msk'


def tok_pad(tokenizer):
    if isinstance(tokenizer._pad_token, str):
        return tokenizer._pad_token
    return 'pad'


def handle_exceed(tokenizer, seq, maxlen, mode=['remove', 'slide', 'start_slice', 'end_slice']):
    mode = mode[0] if isinstance(mode, list) else mode
    t_seq = tokenizer.tokenize(seq)
    if mode == 'remove':
        ret_list = [t_seq] if len(t_seq) <= maxlen else []
        return ret_list, [[0, 0]]
    if mode == 'slide':
        return nlp2.sliding_windows(t_seq, maxlen)
    if mode == 'start_slice':
        return [t_seq[:maxlen]], [[0, maxlen]]
    if mode == 'end_slice':
        return [t_seq[len(t_seq) - maxlen:]], [[max(0, len(t_seq) - 512), len(t_seq)]]


def get_topP_unk_token(tokenizer, file_paths: list, topP: float):
    unk_count_dict = OrderedDict()
    for path in file_paths:
        for input_sent in tqdm(nlp2.read_files_yield_lines(path)):
            for tok in nlp2.split_sentence_to_array(input_sent):
                if tokenizer._unk_token in tokenizer.tokenize(tok):
                    unk_count_dict[tok] = unk_count_dict.get(tok, 0) + 1
    top_range = int((len(unk_count_dict) + 1) * topP * 100)
    return list(unk_count_dict.keys())[:top_range]


def get_freqK_unk_token(tokenizer, file_paths: list, freqK: int):
    unk_count_dict = OrderedDict()
    for path in file_paths:
        for input_sent in tqdm(nlp2.read_files_yield_lines(path)):
            for tok in nlp2.split_sentence_to_array(input_sent):
                if tokenizer._unk_token in tokenizer.tokenize(tok):
                    unk_count_dict[tok] = unk_count_dict.get(tok, 0) + 1
    return [key for key, value in unk_count_dict.items() if value >= freqK]
