import os, json
import numpy as np
import torch
import tiktoken


class DataLoader:
    def __init__(self, data_dir, split, B, T, device):
        assert split in ("train", "val")
        self.data_dir = data_dir
        self.split = split  # 'train'或者'val'
        self.B = B  # batch
        self.T = T  # seq_len
        self.device = device
        self.shard_tokens = {}
        meta_path = os.path.join(self.data_dir, "meta.json")
        with open(meta_path, "r") as fp:
            meta = json.load(fp)
        self.dtype = np.dtype(meta["dtype"])
        names = []
        for shard in meta['shards']:
            if not shard['name'].startswith(f"{split}_"):
                continue
            path = os.path.join(self.data_dir, shard['name'])
            self.shard_tokens[path] = shard['tokens']  # 把meta里，完整路径对应的tokens信息保存下来
            names.append(path)  # 完整路径存到names里
        names.sort()
        self.shards = names  # 存放所有shard文件的路径
        assert self.shards, f"no shards found for split {split}"
        self.reset()

    def _load_shard(self):
        # 把shard_idx对应的shard读入self.tokens
        path = self.shards[self.shard_idx]
        self.tokens = np.memmap(path, dtype=self.dtype, mode="r")
        n_disk = len(self.tokens)
        n_meta = self.shard_tokens[path]
        assert n_disk == n_meta, (
            f"shard token count mismatch: " 
            f"file has {n_disk}, meta.json says {n_meta}; "
            f"shard={path}"
        )

    def reset(self):
        self.shard_idx = 0
        self.pos = 0
        self._load_shard()

    def next_batch(self):
        # 返回 x, y
        B, T = self.B, self.T
        if self.pos + B*T+1 > len(self.tokens):
            self.shard_idx = (self.shard_idx + 1) % len(self.shards)
            self.pos = 0
            self._load_shard()

        data = self.tokens[self.pos:self.pos + B*T+1]  # 抽出B*T+1个token
        x, y = data[:B*T], data[1:]
        x = x.reshape(B, T).astype(np.int64)
        x = torch.from_numpy(x)
        x = x.to(self.device)
        y = y.reshape(B, T).astype(np.int64)
        y = torch.from_numpy(y)
        y = y.to(self.device)        
        self.pos += B*T

        return x, y  # 已经做了格式转换、搬到 device 上


if __name__ == "__main__":
    data_dir = "./data/tokenized/gpt2"
    B = 4
    T = 512
    device = "cpu"
    # dataloader = DataLoader(data_dir, 'train', B, T, device)
    # x, y = dataloader.next_batch()
    # # 测试dataloader读取batch是否正确
    # print(x.shape, y.shape)  # 检查shape
    # print(x.dtype, y.dtype)  # 检查dtype
    # print(torch.equal(x[0, 1:], y[0, :-1]))  # 检查x,y是否成功错开1格

    # enc = tiktoken.get_encoding('gpt2')
    # print(enc.decode(x[0].tolist()[:100]))  # 检查内容是否正常


    # # val也检查一遍
    # dataloader = DataLoader(data_dir, 'val', B, T, device)
    # x, y = dataloader.next_batch()
    # # 测试dataloader读取batch是否正确
    # print(x.shape, y.shape)  # 检查shape
    # print(x.dtype, y.dtype)  # 检查dtype
    # print(torch.equal(x[0, 1:], y[0, :-1]))  # 检查x,y是否成功错开1格

    # enc = tiktoken.get_encoding('gpt2')
    # print(enc.decode(x[0].tolist()[:100]))  # 检查内容是否正常


    # 测试换页逻辑
    dataloader = DataLoader(data_dir, 'train', B, T, device)
    dataloader.pos = len(dataloader.tokens) - B*T
    dataloader.next_batch()
    print(dataloader.shard_idx)

    dataloader = DataLoader(data_dir, 'val', B, T, device)
    dataloader.pos = len(dataloader.tokens) - B*T
    dataloader.next_batch()
    print(dataloader.shard_idx)