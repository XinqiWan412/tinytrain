import pyarrow.parquet as pq
import tiktoken
import numpy as np
import os, json
from tqdm import tqdm


def write_shard(buf, shard_idx, out_dir):
    shard = np.concatenate(buf)
    if shard_idx == 0:
        prefix, n = "val", 0
    else:
        prefix, n = "train", shard_idx - 1
    fname = f"{prefix}_{n:06d}.bin"
    shard.tofile(os.path.join(out_dir, fname))
    print(f"wrote {fname}: {len(shard):,} tokens")
    return {"name": fname, "tokens": len(shard)}


def prepare_data(path, out_dir):
    # 输入 parquet，用分词器切成 token，dtype=uint16，输出 .bin 文件
    batch_size = 2000  # 一次从 .parquet 文件中读 2000 行
    pf = pq.ParquetFile(path)
    f = pf.iter_batches(batch_size=batch_size, columns=['text'])
    enc = tiktoken.get_encoding("gpt2")  # 加载分词器
    buf, count = [], 0
    val_size, shard_size, shard_idx = 10_000_000, 100_000_000, 0  # val 和 train 的 shard 尺寸不一样
    os.makedirs(out_dir, exist_ok=True)
    n_batches = (pf.metadata.num_rows + batch_size - 1) // batch_size
    shards = []
    for batch in tqdm(f, total=n_batches):
        texts = batch.column("text").to_pylist()
        batch_ids = enc.encode_ordinary_batch(texts)  # 并发对 text 做 encode
        for ids in batch_ids:
            ids.append(enc.eot_token)
            arr = np.array(ids, dtype=np.uint16)
            buf.append(arr)
            count += len(arr)
            limit = val_size if shard_idx == 0 else shard_size
            if count >= limit:
                # 每满 size 个 token，就写成一个 shard（.bin）文件
                shards.append(write_shard(buf, shard_idx, out_dir))
                shard_idx += 1
                buf, count = [], 0
    
    if buf:
        # 最后还有一片 shard 要塞进去
        shards.append(write_shard(buf, shard_idx, out_dir))
    meta = {
        "tokenizer": "gpt2",
        "vocab_size": enc.n_vocab,
        "eot_token": enc.eot_token,
        "dtype": "uint16",
        "shards": shards,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)


def check_data(path):
    # 校验分片是否正确
    m = np.memmap(path, dtype=np.uint16, mode="r")
    enc = tiktoken.get_encoding("gpt2")
    print(enc.decode(m[:100].tolist()))
    print((m == 50256).sum())
    print(m[-1] == 50256)


if __name__ == "__main__":
    data_path = "./data/raw/fineweb-edu/sample/10BT/000_00000.parquet"
    out_dir = "./data/tokenized/gpt2"
    # prepare_data(data_path, out_dir)
    check_data("./data/tokenized/gpt2/val_000000.bin")
