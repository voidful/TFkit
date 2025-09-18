from types import SimpleNamespace

import torch
from torch import nn

from tfkit.task.clm.model import Model as CLMModel
from tfkit.task.seq2seq.model import Model as Seq2SeqModel


class DummyTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size

    def convert_ids_to_tokens(self, idx):
        return f"token-{idx}"


class DummyCausalPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=5, hidden_size=4)
        self.output_layer = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.last_kwargs = None

    def get_output_embeddings(self):
        return self.output_layer

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        self.last_kwargs = kwargs
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.config.vocab_size)
        outputs = {
            "logits": logits,
            "last_hidden_state": torch.zeros(batch_size, seq_len, self.config.hidden_size),
        }
        if "labels" in kwargs:
            outputs["loss"] = torch.tensor(0.0)
        return outputs


class DummyEncoderPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=5, hidden_size=4)
        self.last_kwargs = None

    def get_output_embeddings(self):
        return None

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        self.last_kwargs = kwargs
        batch_size, seq_len = input_ids.shape
        hidden = torch.zeros(batch_size, seq_len, self.config.hidden_size)
        return {"last_hidden_state": hidden}


class DummySeq2SeqPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=3, hidden_size=4)
        self.decoder = nn.Module()
        self.output_layer = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def get_output_embeddings(self):
        return self.output_layer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        batch_size, seq_len = decoder_input_ids.shape
        hidden = torch.zeros(batch_size, seq_len, self.config.hidden_size)
        outputs = {
            "last_hidden_state": hidden,
            "decoder_hidden_states": (hidden,),
        }
        return outputs


def test_clm_model_uses_pretrained_head_for_loss():
    tokenizer = DummyTokenizer(vocab_size=5)
    pretrained = DummyCausalPretrained()
    model = CLMModel(tokenizer=tokenizer, pretrained=pretrained)

    batch = {
        "input": torch.zeros((1, 2), dtype=torch.long),
        "mask": torch.ones((1, 2), dtype=torch.long),
        "target": torch.tensor([[0, -1]]),
    }

    loss = model.forward(batch, eval=False)
    assert torch.is_tensor(loss)
    assert "labels" in pretrained.last_kwargs
    assert pretrained.last_kwargs["labels"].tolist() == [[0, -100]]

    eval_batch = {
        **batch,
        "start": [0],
    }
    result = model.forward(eval_batch, eval=True)
    assert isinstance(result, dict)
    assert "max_item" in result


def test_clm_model_falls_back_to_linear_head():
    tokenizer = DummyTokenizer(vocab_size=5)
    pretrained = DummyEncoderPretrained()
    model = CLMModel(tokenizer=tokenizer, pretrained=pretrained)

    batch = {
        "input": torch.zeros((1, 2), dtype=torch.long),
        "mask": torch.ones((1, 2), dtype=torch.long),
        "target": torch.tensor([[0, -1]]),
    }

    loss = model.forward(batch, eval=False)
    assert torch.is_tensor(loss)
    assert pretrained.last_kwargs == {}


def test_seq2seq_model_uses_pretrained_output_head():
    tokenizer = DummyTokenizer(vocab_size=3)
    pretrained = DummySeq2SeqPretrained()
    model = Seq2SeqModel(tokenizer=tokenizer, pretrained=pretrained)

    batch = {
        "input": torch.zeros((1, 1), dtype=torch.long),
        "prev": torch.zeros((1, 1), dtype=torch.long),
        "encoder_mask": torch.ones((1, 1), dtype=torch.long),
        "decoder_mask": torch.ones((1, 1), dtype=torch.long),
        "target": torch.zeros((1, 1), dtype=torch.long),
        "ntarget": torch.full((1, 1), -1),
    }

    loss = model.forward(batch, eval=False)
    assert torch.is_tensor(loss)
    assert model.model is pretrained.output_layer
