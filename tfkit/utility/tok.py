from collections import OrderedDict

import nlp2
from tqdm import tqdm


def tok_begin(tokenizer):
    if isinstance(tokenizer.cls_token, str):
        return tokenizer.cls_token
    elif isinstance(tokenizer.bos_token, str):
        return tokenizer.bos_token
    return 'cls'


def tok_sep(tokenizer):
    if isinstance(tokenizer.sep_token, str):
        return tokenizer.sep_token
    elif isinstance(tokenizer.eos_token, str):
        return tokenizer.eos_token
    return 'sep'


def tok_mask(tokenizer):
    if isinstance(tokenizer.mask_token, str):
        return tokenizer.mask_token
    return 'msk'


def tok_pad(tokenizer):
    if isinstance(tokenizer.pad_token, str):
        return tokenizer.pad_token
    return 'pad'


def handle_exceed(tokenizer, seq, maxlen, mode=['remove', 'slide', 'start_slice', 'end_slice'], keep_after_sep=True):
    mode = mode[0] if isinstance(mode, list) else mode
    seq = seq.replace("[MASK]", tok_mask(tokenizer)).replace("[SEP]", tok_sep(tokenizer)).replace("[CLS]",
                                                                                                  tok_begin(tokenizer))
    sep_split = seq.split(tok_sep(tokenizer))
    ext_seq = [tok_sep(tokenizer)] + tokenizer.tokenize(sep_split[1]) if len(sep_split) > 1 and keep_after_sep else []
    t_seq = tokenizer.tokenize(sep_split[0])
    if mode == 'remove':
        if len(t_seq) <= maxlen:
            return [t_seq], [[0, len(t_seq)]]
        else:
            return [], [[0, 0]]
    if mode == 'slide':
        return nlp2.sliding_windows(t_seq, maxlen - len(ext_seq), append_seq=ext_seq)
    if mode == 'start_slice':
        slices = t_seq[:maxlen - len(ext_seq)]
        slices.extend(ext_seq)
        return [slices], [[0, maxlen - len(ext_seq)]]
    if mode == 'end_slice':
        start_pos = len(t_seq) + len(ext_seq) - maxlen
        slices = t_seq[start_pos:]
        slices.extend(ext_seq)
        return [slices], [[max(0, start_pos), len(t_seq)]]


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
