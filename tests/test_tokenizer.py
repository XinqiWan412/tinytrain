from tokenizer import enc, render_prompt, render_answer
from train_tokenizer import merge, PAT, train_bpe_slow, train_bpe_fast, iter_docs, load_words
import pytest, itertools
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data/raw/fineweb-edu/sample/10BT"


def test_tokenizer_basics():
    assert enc.n_vocab == 32768 and enc.eot_token == 32763
    assert len(enc.encode_ordinary("12345")) == 5
    question = "What is 37 + 45?"
    answer = "82"
    tmp = render_prompt(question) + render_answer(answer)
    out = enc.decode(tmp)
    check_out = """<|endoftext|><|im_start|>user\nWhat is 37 + 45?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n82<|im_end|>"""
    assert out == check_out


def test_merge():
    assert merge([97, 97, 97], (97, 97), 256) == [256, 97]
    assert merge([1, 2, 1, 2], (1, 2), 3) == [3, 3]


def test_split_pattern():
    assert PAT.findall("What is 37 + 45?") == ['What', ' is', ' ', '3', '7', ' +', ' ', '4', '5', '?']


def test_train_bpe_toy():
    merges, vocab = train_bpe_slow([list(b"aaabdaaabac")], [1], 3)
    assert merges == [(97, 97), (97, 98), (256, 257)]
    assert vocab[258] == b"aaab"
    merges, vocab = train_bpe_fast([list(b"aaabdaaabac")], [1], 3)
    assert merges == [(97, 97), (97, 98), (256, 257)]
    assert vocab[258] == b"aaab"
    merges, vocab = train_bpe_fast([list(b"aaaa")], [1], 3)
    assert merges == [(97, 97), (256, 256)]


@pytest.mark.slow
@pytest.mark.skipif(not DATA_DIR.exists(), reason="原始数据不存在")
def test_roundtrip_docs():
    docs = iter_docs(DATA_DIR, 100_000_000)
    texts = itertools.islice(docs, 1000)
    for text in texts:
        assert text == enc.decode(enc.encode_ordinary(text))


@pytest.mark.slow
@pytest.mark.skipif(not DATA_DIR.exists(), reason="原始数据不存在")
def test_bpe_fast_matches_slow():
    words, counts = load_words(DATA_DIR, 1_000_000)
    merges_f, vocab_f = train_bpe_fast(words, counts, 1000)
    merges_s, vocab_s = train_bpe_slow(words, counts, 1000)
    assert merges_f == merges_s
    assert vocab_f == vocab_s
