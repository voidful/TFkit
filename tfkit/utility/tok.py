def tok_begin(tokenizer):
    return tokenizer._cls_token or tokenizer._bos_token or 'cls'


def tok_sep(tokenizer):
    return tokenizer._sep_token or tokenizer._eos_token or 'sep'


def tok_mask(tokenizer):
    return tokenizer._mask_token or 'msk'
