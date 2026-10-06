import pyarrow.parquet as pq
import numpy as np
import os, json
from tqdm import tqdm
from tokenizer import enc
from pathlib import Path


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


def prepare_data(data_dir, out_dir):
    # 输入 parquet，用分词器切成 token，dtype=uint16，输出 .bin 文件
    batch_size = 2000  # 一次从 .parquet 文件中读 2000 行
    paths = sorted(Path(data_dir).glob("*.parquet"))
    print(paths)
    assert len(paths) > 0, f"no parquet found in {data_dir}"
    buf, count = [], 0
    val_size, shard_size, shard_idx = 10_000_000, 100_000_000, 0  # val 和 train 的 shard 尺寸不一样
    os.makedirs(out_dir, exist_ok=True)
    shards = []
    for path in paths:
        pf = pq.ParquetFile(path)
        f = pf.iter_batches(batch_size=batch_size, columns=['text'])
        n_batches = (pf.metadata.num_rows + batch_size - 1) // batch_size
        for batch in tqdm(f, total=n_batches, desc=os.path.basename(path)):
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
        "tokenizer": "tokenizer_32k.json",
        "vocab_size": enc.n_vocab,
        "eot_token": enc.eot_token,
        "dtype": "uint16",
        "shards": shards,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)


def check_all(out_dir):
    # 校验全部输出分片
    with open(os.path.join(out_dir, "meta.json")) as fp:
        meta = json.load(fp)
    
    # 核对 token 总数
    meta_total, disk_total = 0, 0
    for s in meta["shards"]:
        meta_total += s["tokens"]  # 这个 shard 的 token 数
        disk_total += os.path.getsize(os.path.join(out_dir, s["name"])) // 2  # 这个 shard 的 token 数（用字节 / 2 计算）
    print(f"meta_total:{meta_total:,}, disk_total:{disk_total:,}")
    assert meta_total == disk_total, "meta_total != disk_total"

    # 分片个数
    disk_num = len(list(Path(out_dir).glob("*.bin")))  # our_dir 里 .bin 文件个数
    meta_num = len(meta["shards"])  # meta 里记录了几个分片
    print(f"disk_num:{disk_num}, meta_num:{meta_num}")
    assert disk_num == meta_num

    # decode 最后一个分片
    last = meta["shards"][-1]
    print(f"最后一个分片是：{last['name']}")
    m = np.memmap(os.path.join(out_dir, last["name"]), dtype=np.uint16, mode="r")
    mid = len(m) // 2
    m = m[mid:mid + 300]
    print(enc.decode(m.tolist()))


if __name__ == "__main__":
    data_dir = "./data/raw/fineweb-edu/sample/10BT/"
    out_dir = "./data/tokenized/bpe32k_new"
    # prepare_data(data_dir, out_dir)
    check_all(out_dir)
    