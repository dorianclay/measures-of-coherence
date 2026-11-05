import pytest
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

import tiny_model
from probing.activations import tokenize


@pytest.fixture(scope="module")
def tokenizers(tmp_path_factory):
    """Two tokenizers for one vocabulary with a BOS token: one adds it by default, one doesn't, like Gemma 4's pair."""
    path = tiny_model.build(tmp_path_factory.mktemp("tiny"))
    plain = AutoTokenizer.from_pretrained(path, bos_token="<|system|>")
    adding = AutoTokenizer.from_pretrained(path, bos_token="<|system|>")
    adding.backend_tokenizer.post_processor = TemplateProcessing(
        single="<|system|> $A", special_tokens=[("<|system|>", adding.bos_token_id)])
    return plain, adding


def first_tokens(tokenizer, bos):
    return tokenize(["umbrella", "no umbrella"], tokenizer, bos).input_ids[:, :2].tolist()


def test_tokenizer_mode_keeps_each_tokenizers_default(tokenizers):
    plain, adding = tokenizers
    assert all(row[0] != plain.bos_token_id for row in first_tokens(plain, "tokenizer"))
    assert all(row[0] == adding.bos_token_id for row in first_tokens(adding, "tokenizer"))


@pytest.mark.parametrize("which", [0, 1])
def test_add_gives_exactly_one_bos_whatever_the_tokenizer_does(tokenizers, which):
    tokenizer = tokenizers[which]
    assert all(row[0] == tokenizer.bos_token_id and row[1] != tokenizer.bos_token_id
               for row in first_tokens(tokenizer, "add"))


@pytest.mark.parametrize("which", [0, 1])
def test_remove_gives_no_bos_whatever_the_tokenizer_does(tokenizers, which):
    tokenizer = tokenizers[which]
    assert all(row[0] != tokenizer.bos_token_id for row in first_tokens(tokenizer, "remove"))
