# tinytrain

从零手写的 LLM 训练全流程：预训练 → SFT → GRPO。模型结构参考 Qwen3-0.6B，模型、数据加载、训练循环、对话模板、GRPO 都是自己实现的，没有用 transformers / trl。

在单张 RTX 4070 Ti Super（16GB）上先用 1150 万参数的小模型跑通全流程，再扩大到 1.12 亿参数，对比 AdamW 和 Muon 两种优化器；导出 HF 格式后与 transformers 对拍；预训练支持 DDP、手写的 ZeRO-1 / ZeRO-2、FSDP2（fully_shard），以及手写的张量并行（TP=2）和流水线并行（PP=2，GPipe / 1F1B），在 4×RTX 4090、4×A100 和 2×A100 上测过。

## 结果

### 1.12 亿参数全流程：AdamW vs Muon

| 阶段 | 指标 | AdamW | Muon |
| --- | --- | --- | --- |
| 预训练（34,000 步，约 22.3 亿 token） | val loss | 3.1466 | 3.1069 |
| SFT（0～99 两数相加，2000 步） | 测试集贪心准确率 | 100% | 100% |
| GRPO（从约 40% 的弱起点，500 步） | 结尾准确率（相对起点） | 94.2%（+54.2） | 84.6%（+43.8） |
|  | 10 次评估平均（相对起点） | 82.2%（+42.2） | 71.5%（+30.7） |

- 两条路线只有优化器不同（Muon 路线的 SFT、GRPO 也用 Muon）。Muon 的预训练 val loss 低 0.040，第 28,250 步就追上 AdamW 的终点，按时间算约省 14%
- GRPO 两条路线各只跑了一次，波动不小，关于优化器的结论以预训练为准
- 导出 HF 格式后用 transformers 的 Qwen3ForCausalLM 读回（fp32），logits 最大差约 2e-5，SFT 模型的贪心生成与 HF 完全一致

### 多卡：4×A100（NVLink），AdamW 路线前 300 步

|  | 单卡 | DDP | ZeRO-1 | ZeRO-2 | FSDP2 |
| --- | --- | --- | --- | --- | --- |
| 每步时间 | 682 ms | 177 ms | 183 ms | 186 ms | 190 ms |
| 扩展效率 | — | 96% | 93% | 92% | 90% |
| 每卡峰值显存（allocated） | 9,959 MiB | 10,396 MiB | 约 9,740 MiB | 约 9,310 MiB | 9,101 MiB |

- 五种做法的 val loss 相差不到 5e-4
- 显存按参数、梯度、优化器状态逐项与预计对得上。FSDP2 每卡比 DDP 少约 1.3 GB；这个模型的显存大头是激活和 logits（约 8.2 GB），模型状态只占约 1.7 GB，所以省得不多
- ZeRO-1 的 3 号卡比其余三张多约 440 MiB，原因待查
- 另外两组对比：卡间不支持 P2P 的 4×RTX 4090 上，DDP 的扩展效率只有 55%，瓶颈是梯度通信；Muon 路线下 ZeRO-1 比 DDP 还快（92% vs 84%），因为 Muon 的正交化被分摊到了各卡

### 多卡：2×A100（NVLink），TP 与 PP，AdamW 路线前 300 步

|  | 单卡 | DDP | TP | PP GPipe 6/6 | PP 1F1B 6/6 | PP 1F1B 7/5 |
| --- | --- | --- | --- | --- | --- | --- |
| 每步时间 | 688 ms | 352 ms | 622 ms | 430 ms | 434 ms | 406 ms |
| 扩展效率 | — | 98% | 55% | 80% | 79% | 85% |
| 峰值显存 allocated（0 号 / 1 号卡） | 9,959 MiB | 10,396 MiB | 7,462 MiB | 23,261 / 37,130 MiB | 6,768 / 6,511 MiB | 7,808 / 5,942 MiB |

- 6/6 指第一段 6 层、第二段 6 层；另测了 8/4，见 results.md。所有组的 val loss 相差不到 5e-4
- 这个模型单卡放得下，2 卡时 DDP 最好。PP 慢在 1/9 的气泡和两段不均衡，通信可以忽略。按参数个数估，lm_head 约等于 3.5 层，实测只相当于约 1.8 层，所以 8/4 反而比 6/6 慢；7/5 最快，接近只剩气泡时的上限（约 387 ms）
- 1F1B 与 GPipe 速度相同、训练结果逐位相同，显存却低得多：GPipe 要同时存 8 份激活，第二段每份还要多存约 1.7 GB 的 logits 等中间结果
- TP 只快 11%：lm_head、loss 和各层的小操作两张卡各算一遍，每份数据 48 次 all-reduce 又没和计算重叠

各版本的完整表格和分析见 [docs/results.md](docs/results.md)，多卡各做法的实现见 [docs/parallel.md](docs/parallel.md)。

## 快速复现

环境：单张 RTX 4070 Ti Super 16GB（bf16 autocast）；Python 3.12.14，PyTorch 2.14.0+cu130（CUDA 13.0），tiktoken 0.14.0、numpy 2.5.2、pyarrow 25.0.1、pytest 9.1.1；导出 HF 格式另需 transformers 5.19.0、safetensors 0.8.0。

```bash
# 1. 下载数据：fineweb-edu sample-10BT 的前三个分片
hf download HuggingFaceFW/fineweb-edu --repo-type dataset \
  --include "sample/10BT/000_00000.parquet" "sample/10BT/001_00000.parquet" "sample/10BT/002_00000.parquet" \
  --local-dir data/raw/fineweb-edu

# 2. 分词并切片，共约 22.9 亿 token（分词器 tokenizer_32k.json 已在仓库里）
python prepare_data.py

# 3. 预训练：单卡约 13 小时，预期 val loss 约 3.15；中断后在原命令后加 --resume
python pretrain.py --config configs/adamw_112m/pretrain.toml

# 4. SFT 并评估：预期 100%
python sft.py --config configs/adamw_112m/sft.toml
python eval_sft.py --ckpt checkpoints/adamw_112m/sft.pt

# 5. 弱 SFT，作为 GRPO 的起点：预期约 40%
python sft.py --config configs/adamw_112m/sft_weak.toml

# 6. GRPO：预期结尾约 94%；爆显存时加 --set grpo.chunk_size=64
python grpo.py --config configs/adamw_112m/grpo.toml

# 7. 导出 HF 格式并对拍；--gen 再比较贪心生成，只用于 SFT / GRPO 模型
python export_hf.py --ckpt checkpoints/adamw_112m/sft.pt --out hf/adamw_112m/sft --gen

# 8. 多卡预训练（单机）：默认 DDP；--set parallel.zero_stage=1 / 2 为 ZeRO-1 / ZeRO-2，=3 为 FSDP2（只支持 AdamW）；第二行是 TP（2 卡、只支持 AdamW），第三行是 PP（2 卡，按 7/5 切分）
torchrun --standalone --nproc_per_node=4 pretrain.py --config configs/adamw_112m/pretrain.toml
torchrun --standalone --nproc_per_node=2 pretrain.py --config configs/adamw_112m/pretrain.toml --set parallel.tp_size=2
torchrun --standalone --nproc_per_node=2 pretrain.py --config configs/adamw_112m/pretrain.toml --set parallel.pp_size=2 --set parallel.pp_first_stage_layers=7
```

- Muon 路线把路径里的 adamw_112m 换成 muon_112m
- 每次运行的参数都在 configs/ 的配置文件里；临时改参数用 `--set 节.键=值`（可写多次），完整配置会打在日志开头并存进 checkpoint
- 多卡的卡数限制、续训规则、在 CPU 上验证多进程的方法见 [docs/parallel.md](docs/parallel.md)

测试：`pytest -v` 跑 76 个快测试（CPU，约 150 秒，不需要数据），`pytest -m slow -v` 跑 2 个需要原始数据的慢测试。测试怎么设计、临时改坏能被哪些测试抓到，见 [docs/testing.md](docs/testing.md)。

## 项目结构

- `prepare_data.py`：把 parquet 分词、切成 .bin 分片并校验
- `train_tokenizer.py`：训练 byte-level BPE 分词器，存为 `tokenizer_32k.json`
- `tokenizer.py`：加载分词器，以及按对话模板拼 prompt 的 render 函数
- `dataloader.py`：预训练数据加载，支持多进程按 rank 跳着读和断点续读
- `model.py`：模型结构（参数名与 HF Qwen3 一致）、ModelConfig、load_model
- `trainer.py`：三个阶段共用的训练骨架 Trainer
- `pretrain.py`：预训练入口，含 PretrainTask、configure_optimizer，支持 torchrun 多卡
- `sft_data.py`：加法题的生成、切分和 x / y 构建
- `sft.py`：SFT 入口，含 SFTLoader、SFTTask
- `eval_sft.py`：测试集贪心准确率
- `grpo.py`：GRPO 入口，含 rollout、分块算 loss、GRPOTask
- `generate.py`：简易批量生成（temperature + top-k）
- `muon.py`：Newton-Schulz 正交化和 MuonWithAdamW 优化器
- `export_hf.py`：导出 HF 格式并与 transformers 对拍
- `config.py`：配置 dataclass 和 load_config（TOML + `--set`）
- `logger.py`：日志同时打印并写文件，多进程时只有 0 号进程输出
- `parallel/`
  - `dist_utils.py`：进程组初始化、并行布局（本卡的 DP、TP、PP 编号）和通信小工具
  - `zero.py`：手写 ZeRO-1 / ZeRO-2（ZeroOptimizer）
  - `fsdp.py`：FSDP2 分片，以及完整状态的收集和读回
  - `tp.py`：手写张量并行（f / g、两种并行线性层、apply_tp、TPOptimizer、按参数名拼回和切回）
  - `pp.py`：手写流水线并行（按段切分模型、两段的前向、GPipe 和 1F1B 调度、PPOptimizer）
- `configs/`：adamw_112m、muon_112m 两条路线，各 4 个配置文件（pretrain、sft、sft_weak、grpo）
- `tests/`：pytest 测试，见 [docs/testing.md](docs/testing.md)

各模块的设计和取舍见 [docs/notes.md](docs/notes.md)。

## 路线

| 版本 | 内容 |
| --- | --- |
| v0.1 | 小模型跑通预训练 → SFT → GRPO（gpt2 分词器） |
| v0.2 | 换成自训的 BPE 分词器（1150 万参数） |
| v0.3 | 扩大到 1.12 亿参数，AdamW / Muon 两条路线 |
| v0.4 | 训练框架化：pytest 测试、统一的 Trainer、合并的优化器、TOML 配置 |
| v0.5 | 导出 HF 格式，与 transformers 对拍 |
| v0.6 | DDP |
| v0.7 | 手写 ZeRO-1 / ZeRO-2 |
| v0.8 | FSDP2（fully_shard） |
| v0.9 | 手写 TP（张量并行）和 PP（流水线并行，GPipe / 1F1B） |
| 下一步 | 3D 并行（DP×TP×PP） |
| 之后 | 性能优化（MFU、torch.profiler、torch.compile、自写 Triton FlashAttention、分块 lm_head + CE、activation checkpointing）→ 序列打包 + varlen attention → 推理框架（KV cache、批量调度，接入 GRPO rollout，与 vLLM 对照） |

可选：零样本 benchmark（如 HellaSwag）、两种优化器扫 lr 与 GRPO 多种子、PPO 式 ratio + clip / DAPO、更多题型、MoE、GitHub Actions CI、FSDP2 的 MixedPrecisionPolicy。

复现旧版本时先 `git checkout <版本号>`。