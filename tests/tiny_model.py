"""A tiny Qwen2-shaped model with random weights and a chat template, built locally so tests need no download."""

from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from coherence import umbrella

CHAT_TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
                 "{% if add_generation_prompt %}<|assistant|>{% endif %}")
SPECIAL = ["<|pad|>", "<|end|>", "<|user|>", "<|system|>", "<|assistant|>"]


def build(path: Path) -> Path:
    corpus = [s["prompt"][0]["content"] for s in umbrella.generate(seed=0, n=50)]
    corpus += ["umbrella", "no umbrella", "True", "False", "It will rain in Velmora tomorrow.", "0 50 100"] * 5
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<|pad|>", eos_token="<|end|>",
                                        chat_template=CHAT_TEMPLATE)
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                         tie_word_embeddings=True)
    Qwen2ForCausalLM(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path
