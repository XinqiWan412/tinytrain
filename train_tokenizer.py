import time
import regex as re
from collections import defaultdict, Counter
import pyarrow.parquet as pq
from pathlib import Path


SPLIT_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
PAT = re.compile(SPLIT_PATTERN)


def merge(word, pair, new_id):
    # 把 word 字符串里的 pair 给替换成 new_id
    # 就是很经典的 BPE 词表的 merge
    i = 0
    new = []
    len_word = len(word)
    # 
    while i < len_word:
        # 遍历 word 的每一个 id，如果找到 pair，替换成 new_id
        if i < len_word - 1 and (word[i], word[i+1]) == pair:
            # 如果 i 不是最后一个位置，且这两个字母等于 pair，放入 new_id
            new.append(new_id)
            i += 2
        else:
            # 如果不等于 pair，放 word[i]
            new.append(word[i])
            i += 1
    return new


def train_bpe_slow(words, counts, num_merges):
    # 一共合并 num_merges 次
    # 返回 merges, vocab
    # vocab 是 dict{token_id: token 对应的 bytes（字节序列）}
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    for i in range(num_merges):
        # 统计所有相邻对
        stats = defaultdict(int)
        for w, c in zip(words, counts):
            for a, b in zip(w, w[1:]):
                stats[(a, b)] += c
        
        # 挑出最多的那一对
        if not stats:
            break
        best_pair = min(stats, key=lambda p: (-stats[p], p))

        # 合并
        new_id = i + 256
        words = [merge(w, best_pair, new_id) for w in words]

        # 记录
        merges.append(best_pair)
        a, b = best_pair
        vocab[new_id] = vocab[a] + vocab[b]
    return merges, vocab


def iter_docs(data_dir, max_chars):
    # 从 parquet 里一篇一篇往外拿文档，总字符数到上限就停
    batch_size = 2000
    cnt = 0
    paths = sorted(Path(data_dir).glob("*.parquet"))
    assert paths, f"data_dir 里没有找到 .parquet：{data_dir}"

    for path in paths:
        pf = pq.ParquetFile(path)
        f = pf.iter_batches(batch_size=batch_size, columns=['text'])

        for batch in f:
            texts = batch.column("text").to_pylist()
            for text in texts:
                yield text
                cnt += len(text)
                if cnt >= max_chars:
                    return


def count_chunks(docs):
    # docs 是 iter_docs 返回的生成器
    # 用 regex 把所有文档内容提取成 chunks，放入 counter 字典返回
    counter = Counter()
    for doc in docs:
        chunks = PAT.findall(doc)
        counter.update(chunks)
    return counter


if __name__ == "__main__":
    # 玩具例子测试
    assert merge([97, 97, 97], (97, 97), 256) == [256, 97]
    assert merge([1, 2, 1, 2], (1, 2), 3) == [3, 3]
    assert PAT.findall("What is 37 + 45?") == ['What', ' is', ' ', '3', '7', ' +', ' ', '4', '5', '?']

    merges, vocab = train_bpe_slow([list(b"aaabdaaabac")], [1], 3)
    assert merges == [(97, 97), (97, 98), (256, 257)]
    assert vocab[258] == b"aaab"

    # 导入数据，切成 chunk，counter 计数
    data_dir = "data/raw/fineweb-edu/sample/10BT"
    max_chars = 1_000_000
    docs = iter_docs(data_dir, max_chars)
    counter = count_chunks(docs)
    print(counter.most_common(10))
    print(len(counter))

    # 把 counter 转换成 words 和 counts，准备训练 BPE    
    words = []
    counts = []
    for key, val in counter.items():
        tmp = list(key.encode("utf-8"))
        words.append(tmp)
        counts.append(val)

    t0 = time.perf_counter()
    merges, vocab = train_bpe_slow(words, counts, num_merges=300)
    t1 = time.perf_counter()

    for i in range(256, 286):
        print(repr(vocab[i].decode("utf-8", errors="replace")))
    
    for i in range(256, len(vocab)):
        # 新 token 的字节里不能有 ASCII 数字（48～57）
        assert not any(48 <= x <= 57 for x in vocab[i]), f"vocab[{i}] 在 48~57 之间：{repr(vocab[i])}"

    print(t1 - t0)  # 用时
