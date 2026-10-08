import time, json
import regex as re
import heapq
from collections import defaultdict, Counter
import pyarrow.parquet as pq
from pathlib import Path
import argparse


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


def train_bpe_fast(words, counts, num_merges, log_every=None):
    words = [list(w) for w in words]
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    stats = defaultdict(int)  # 存这一对 pair 一共出现了多少次
    where = defaultdict(set)  # 存这一对 pair 出现在哪些 word 中
    for idx, (w, c) in enumerate(zip(words, counts)):
        for a, b in zip(w, w[1:]):
            stats[(a, b)] += c  # 统计 pair 出现的次数
            where[(a, b)].add(idx)  # 记录 word 的位置
    
    h = [(-count, pair) for pair, count in stats.items()]
    heapq.heapify(h)  # 创建最小堆

    t0 = time.perf_counter()
    best_count = None

    # 开始 merge
    for i in range(num_merges):
        # 每轮 merge 都取出一个频率最大的 pair
        best_pair = None
        while h:
            count, pair = heapq.heappop(h)
            if stats[pair] == -count:
                # 如果和 stats 的记录相等，说明信息没过期，取出
                best_pair = pair
                best_count = -count
                break
            elif stats[pair] > 0:
                # 信息过期了，但 stats 里还大于 0，把更新的信息压回堆中
                heapq.heappush(h, (-stats[pair], pair))
            # stats[pair] 已经是 0，直接丢弃
        if best_pair is None:
            # 堆已经空了，结束 merge 环节
            break
        
        # 只更新 best_pair 所在的 word
        new_id = 256 + i
        new_pairs = set()
        # 遍历 best_pair 所在的每一个 word
        for idx in where.pop(best_pair):
            new_word = merge(words[idx], best_pair, new_id)  # merge 这个 word 中的 best_pair
            if len(new_word) == len(words[idx]):
                # 如果两者长度一样，说明 words[idx] 里已经没有 best_pair 了
                continue
            # 走到这里说明 merge 执行成功，将旧 word 的所有 pair 都在 stats 里减去
            # 将新 word 的所有 pair 都在 stats 里加上
            c = counts[idx]
            for a, b in zip(words[idx], words[idx][1:]):
                stats[(a, b)] -= c
            for a, b in zip(new_word, new_word[1:]):
                stats[(a, b)] += c
                if new_id in (a, b):
                    # new_id 所在的 pair 都是新的 pair，需要记录
                    where[(a, b)].add(idx)
                    new_pairs.add((a, b))
            words[idx] = new_word

        # 把 new_pairs 放进最小堆
        for pair in new_pairs:
            heapq.heappush(h, (-stats[pair], pair))
        
        merges.append(best_pair)
        a, b = best_pair
        vocab[new_id] = vocab[a] + vocab[b]

        if log_every is not None and (i + 1) % log_every == 0:
            t1 = time.perf_counter()
            print(f"第 {i+1} 次 merge，当前 pair 在全体语料中出现 {best_count} 次，新 token：{repr(vocab[new_id].decode(errors='replace'))}，用时：{t1 - t0:.1f}")

    return merges, vocab


def load_words(data_dir, max_chars):
    # 统计词块
    docs = iter_docs(data_dir, max_chars)
    counter = count_chunks(docs)

    words, counts = [], []
    for key, val in counter.items():
        tmp = list(key.encode("utf-8"))
        words.append(tmp)
        counts.append(val)
    return words, counts


def train_tokenizer():
    # 正式训练函数

    # 定常量
    max_chars = 100_000_000
    vocab_size = 32768
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    num_merges = vocab_size - 256 - len(special_tokens)

    # 统计词块
    data_dir = "data/raw/fineweb-edu/sample/10BT"
    words, counts = load_words(data_dir, max_chars)
    
    # 训练
    merges, vocab = train_bpe_fast(words, counts, num_merges, log_every=1000)
    assert len(merges) == num_merges, f"训练中途语料不足，要求合并 {num_merges} 次，只合并了 {len(merges)} 次"
    for i in range(256, len(vocab)):
        # 数字检查，新 token 的字节里不能有 ASCII 数字（48～57）
        assert not any(48 <= x <= 57 for x in vocab[i]), f"vocab[{i}] 包含 ASCII 数字：{repr(vocab[i])}"

    out = {
        "split_pattern": SPLIT_PATTERN,
        "special_tokens": special_tokens,
        "merges": merges,
    }

    with open("tokenizer_32k.json", "w", encoding="utf-8") as f:
        json.dump(out, f)


if __name__ == "__main__":
    # 整个过程是：从 parquet 里取出文本，切成词块并统计次数，
    # 然后反复做「找出最常见的相邻对，把它合并成新 token」，直到合并够次数。

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.train:
        train_tokenizer()
    else:
        parser.print_help()
    