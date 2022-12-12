from collections import OrderedDict

import nlp2
from tqdm import tqdm
from transformers import AutoTokenizer

UNIVERSAL_SEP = "///"


def tok_begin(tokenizer):
    if tokenizer.special_tokens_map.get('bos_token'):
        return tokenizer.special_tokens_map.get('bos_token')
    elif tokenizer.special_tokens_map.get('cls_token'):
        tokenizer.special_tokens_map.get('cls_token')
    return 'cls'


def tok_begin_id(tokenizer):
    return tokenizer.convert_tokens_to_ids(tok_begin(tokenizer))


def tok_sep(tokenizer):
    if tokenizer.special_tokens_map.get('sep_token'):
        return tokenizer.special_tokens_map.get('sep_token')
    elif tokenizer.special_tokens_map.get('eos_token'):
        tokenizer.special_tokens_map.get('eos_token')
    return 'sep'


def tok_sep_id(tokenizer):
    return tokenizer.convert_tokens_to_ids(tok_sep(tokenizer))


def tok_mask(tokenizer):
    if tokenizer.special_tokens_map.get('mask_token'):
        return tokenizer.special_tokens_map.get('mask_token')
    return 'msk'


def tok_mask_id(tokenizer):
    return tokenizer.convert_tokens_to_ids(tok_mask(tokenizer))


def tok_pad(tokenizer):
    if tokenizer.special_tokens_map.get('pad_token'):
        return tokenizer.special_tokens_map.get('pad_token')
    return 'pad'


def tok_pad_id(tokenizer):
    return tokenizer.convert_tokens_to_ids(tok_pad(tokenizer))


def get_all_tok_from_config(config):
    tokenizer = AutoTokenizer.from_pretrained(config)
    return list(tokenizer.get_vocab().keys())


def handle_exceed(tokenizer, seq, maxlen, mode=['noop', 'remove', 'slide', 'start_slice', 'end_slice'],
                  keep_after_sep=True):
    if isinstance(seq, list):
        return seq, [[len(seq)]]
    mode = mode[0] if isinstance(mode, list) else mode
    sep_tok = tok_sep(tokenizer)
    sep_split = seq.split(sep_tok)
    ext_seq = [sep_tok] + tokenizer.tokenize(sep_tok.join(sep_split[1:])) \
        if len(sep_split) > 1 and keep_after_sep else []
    t_seq = tokenizer.tokenize(sep_split[0])
    if mode == 'noop':
        return [t_seq + ext_seq], [[0, len(t_seq + ext_seq)]]
    if mode == 'remove':
        if len(t_seq + ext_seq) <= maxlen:
            return [t_seq + ext_seq], [[0, len(t_seq + ext_seq)]]
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
