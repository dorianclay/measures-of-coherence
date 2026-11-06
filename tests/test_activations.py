import pytest
from tokenizers.processors import TemplateProcessing
from transformers import AutoModelForCausalLM, AutoTokenizer

import tiny_model
from probing.activations import chat_text, extract, prompt_record, tokenize, tokenize_chat


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


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path = tiny_model.build(tmp_path_factory.mktemp("tiny-chat"))
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "right"
    return tokenizer


def read_token(tokenizer, statement, other, presentation):
    inputs, positions = tokenize_chat([statement, other], tokenizer, presentation)
    return tokenizer.decode(inputs.input_ids[0, positions[0]])


def test_chat_statement_reads_the_statements_last_token_though_shorter_prompts_are_padded(tiny):
    assert read_token(tiny, "no umbrella", "umbrella and no umbrella and umbrella", "chat-statement").strip() == \
        tiny.decode(tiny("no umbrella", add_special_tokens=False).input_ids[-1]).strip()


def test_chat_statement_finds_the_statement_when_the_template_repeats_it(tiny):
    inputs, positions = tokenize_chat(["user"], tiny, "chat-statement")
    assert positions[0] < inputs.attention_mask.sum() - 1


@pytest.mark.parametrize("presentation", ["chat-turn", "judge-turn"])
def test_turn_presentations_read_the_last_token_before_the_reply(tiny, presentation):
    assert read_token(tiny, "no umbrella", "umbrella and no umbrella and umbrella", presentation) == "<|assistant|>"


def test_judge_turn_asks_about_the_statement_and_chat_turn_does_not(tiny):
    assert "true or false" in chat_text("no umbrella", tiny, "judge-turn")
    assert "true or false" not in chat_text("no umbrella", tiny, "chat-turn")


def test_extract_reads_one_vector_per_statement_and_layer(tiny, tmp_path_factory):
    path = tiny_model.build(tmp_path_factory.mktemp("tiny-extract"))
    model = AutoModelForCausalLM.from_pretrained(path).eval()
    for presentation in ["raw", "chat-statement", "chat-turn", "judge-turn"]:
        out = extract(["umbrella", "no umbrella."], model, tiny, [1, 2], 2, presentation=presentation)
        assert out.shape == (2, 2, model.config.hidden_size)


def test_chat_statement_finds_a_statement_the_template_trims(tiny, tmp_path_factory):
    path = tiny_model.build(tmp_path_factory.mktemp("tiny-trim"))
    trimming = AutoTokenizer.from_pretrained(path, chat_template=tiny_model.CHAT_TEMPLATE.replace(
        "m['content']", "m['content'] | trim"))
    inputs, positions = tokenize_chat(["no umbrella\xa0"], trimming, "chat-statement")
    assert trimming.decode(inputs.input_ids[0, positions[0]]).strip() == \
        trimming.decode(trimming("no umbrella", add_special_tokens=False).input_ids[-1]).strip()


def test_prompt_record_keeps_the_prompt_and_the_token_read(tiny):
    record = prompt_record(tiny, "chat-turn")
    assert record["prompt_example"] == chat_text("The statement goes here", tiny, "chat-turn")
    assert tiny.convert_tokens_to_string([record["read_token"]]) in record["prompt_example"]
    assert prompt_record(tiny, "raw") == {}
