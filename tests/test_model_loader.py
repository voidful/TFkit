from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tfkit.utility import model as model_utils
from tfkit.utility.model import load_pretrained_model, load_pretrained_tokenizer


def _make_config(**overrides):
    defaults = {
        "is_encoder_decoder": False,
        "architectures": [],
        "is_decoder": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_load_pretrained_model_prefers_seq2seq(monkeypatch):
    config = _make_config(is_encoder_decoder=True)

    auto_config = MagicMock()
    auto_config.from_pretrained.return_value = config
    monkeypatch.setattr(model_utils, "AutoConfig", auto_config)

    seq2seq_loader = MagicMock()
    seq2seq_instance = object()
    seq2seq_loader.from_pretrained.return_value = seq2seq_instance
    monkeypatch.setattr(model_utils, "AutoModelForSeq2SeqLM", seq2seq_loader)

    causal_loader = MagicMock()
    monkeypatch.setattr(model_utils, "AutoModelForCausalLM", causal_loader)

    base_loader = MagicMock()
    monkeypatch.setattr(model_utils, "AutoModel", base_loader)

    result = load_pretrained_model("mock-model", ["seq2seq"])  # type: ignore[arg-type]

    assert result is seq2seq_instance
    seq2seq_loader.from_pretrained.assert_called_once()
    causal_loader.from_pretrained.assert_not_called()
    base_loader.from_pretrained.assert_not_called()


def test_load_pretrained_model_prefers_causal(monkeypatch):
    config = _make_config(architectures=["CustomForCausalLM"])

    auto_config = MagicMock()
    auto_config.from_pretrained.return_value = config
    monkeypatch.setattr(model_utils, "AutoConfig", auto_config)

    seq2seq_loader = MagicMock()
    monkeypatch.setattr(model_utils, "AutoModelForSeq2SeqLM", seq2seq_loader)

    causal_loader = MagicMock()
    causal_instance = object()
    causal_loader.from_pretrained.return_value = causal_instance
    monkeypatch.setattr(model_utils, "AutoModelForCausalLM", causal_loader)

    base_loader = MagicMock()
    monkeypatch.setattr(model_utils, "AutoModel", base_loader)

    result = load_pretrained_model("mock-model", ["clm"])  # type: ignore[arg-type]

    assert result is causal_instance
    causal_loader.from_pretrained.assert_called_once()
    base_loader.from_pretrained.assert_not_called()


def test_load_pretrained_model_causal_fallback(monkeypatch):
    config = _make_config(architectures=["CustomForCausalLM"])

    auto_config = MagicMock()
    auto_config.from_pretrained.return_value = config
    monkeypatch.setattr(model_utils, "AutoConfig", auto_config)

    seq2seq_loader = MagicMock()
    monkeypatch.setattr(model_utils, "AutoModelForSeq2SeqLM", seq2seq_loader)

    causal_loader = MagicMock()
    causal_loader.from_pretrained.side_effect = ValueError("missing head")
    monkeypatch.setattr(model_utils, "AutoModelForCausalLM", causal_loader)

    base_loader = MagicMock()
    base_instance = object()
    base_loader.from_pretrained.return_value = base_instance
    monkeypatch.setattr(model_utils, "AutoModel", base_loader)

    result = load_pretrained_model("mock-model", ["clm"])  # type: ignore[arg-type]

    assert result is base_instance
    base_loader.from_pretrained.assert_called_once()
    assert config.is_decoder is True


def test_load_pretrained_model_trust_remote_code_env(monkeypatch):
    monkeypatch.setenv("TFKIT_TRUST_REMOTE_CODE", "false")

    config = _make_config()
    auto_config = MagicMock()
    auto_config.from_pretrained.return_value = config
    monkeypatch.setattr(model_utils, "AutoConfig", auto_config)

    base_loader = MagicMock()
    base_instance = object()
    base_loader.from_pretrained.return_value = base_instance
    monkeypatch.setattr(model_utils, "AutoModel", base_loader)

    result = load_pretrained_model("mock-model", ["clas"])  # type: ignore[arg-type]

    assert result is base_instance
    auto_config.from_pretrained.assert_called_once_with(
        "mock-model", trust_remote_code=False
    )
    base_loader.from_pretrained.assert_called_once()
    _, kwargs = base_loader.from_pretrained.call_args
    assert kwargs.get("trust_remote_code") is False


def test_load_pretrained_tokenizer_respects_env(monkeypatch):
    monkeypatch.setenv("TFKIT_TRUST_REMOTE_CODE", "0")

    tokenizer_loader = MagicMock()
    monkeypatch.setattr(model_utils, "AutoTokenizer", tokenizer_loader)

    load_pretrained_tokenizer("mock-tokenizer")

    tokenizer_loader.from_pretrained.assert_called_once_with(
        "mock-tokenizer", trust_remote_code=False
    )
