# tinytrain

从零手写的 LLM 训练全流程：预训练 → SFT → GRPO。模型结构参考 Qwen3-0.6B，
在单张 RTX 4070 Ti Super（16GB）上先用 1150 万参数的小模型跑通，再把预训练扩大到 1.12 亿参数。
模型、数据加载、训练循环、对话模板、GRPO 都是自己实现的，没有用 transformers / trl。

## 结果

### 1.12 亿参数预训练（AdamW 基线）

| 项目 | 数值 |
|---|---|
| 参数量 | 112,482,048 |
| 训练数据 | fineweb-edu 约 22.3 亿 token（约 20 倍参数量），34,000 步 |
| val loss | 约 3.15（每字节损失 0.692） |
| 速度 | 约 4.9 万 tok/s，显存峰值约 11.7 GB，共约 13 小时 |

和下面 1150 万参数模型（val loss 4.2195，每字节损失 0.927）用的是同一份 val 数据，可以直接比较

### 1150 万参数全流程（v0.2）

这一版用于验证整个 pipeline 的正确性，数字用于说明各阶段跑通，不是为了说明模型效果好。复现这一版请先 `git checkout v0.2`。

| 阶段 | 设置 | 结果 |
|---|---|---|
| 预训练 | fineweb-edu 约 2 亿 token，3000 步 | val loss 4.2195，约 31.7 万 tok/s |
| SFT | 0~99 两数相加，9,500 题训练、500 题测试，2000 步 | 测试集贪心准确率 100% |
| SFT（弱起点） | 同上，200 步 | 37.4%，作为 GRPO 的起点 |
| GRPO | 每步 64 题 × 8 次采样，lr 2e-5，500 步，β=0.2 | 结尾 69.4%，10 次评估平均 59.5% |

对比 v0.1（gpt2 分词器）：预训练速度从 23.3 万提升到 31.7 万 tok/s，每字节损失 0.922 vs 0.927，效果持平；SFT 和 GRPO 的对比见下方复现步骤第 4～6 步

## 实现要点
- 模型结构：配置为 hidden 768、12 层、8 头 / 2 KV 头（head_dim 128）、FFN 2304，共 1.12 亿参数（v0.2 为 hidden 256、4 层、FFN 768，1150 万参数）
- 训练配置：AdamW（betas 0.9/0.95，weight decay 0.1 只加在 2 维参数上），warmup + 余弦衰减，梯度裁剪 1.0，梯度累积，bf16 autocast
- SFT：只对答案和 `<|im_end|>` 算 loss
- GRPO：每题采样 8 次，组内标准化算优势，loss 加对冻结参考模型的 KL 惩罚（k3 估计）
- 批量生成按 prompt 长度分组（eval_sft.py 和 grpo.py 的 rollout）：数字逐位切分后，prompt 长度随两个加数的位数变化，而同一批生成要求 prompt 等长
- 对话格式：与 Qwen3 一致
```
<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>
```
- 长时间训练：日志在 print 的同时追加写进文件、每行 flush；断点续训保存模型、优化器、步数和 DataLoader 位置（DataLoader 提供 state_dict / load_state_dict），checkpoint 先写临时文件再用 os.replace 替换，写到一半被打断也不会损坏旧文件
- 数据切分：一次处理目录下所有 parquet，分片编号跨文件连续；val 固定为第一个文件开头的 1000 万 token，增加数据后结果仍能和之前比较

## 项目结构
- `prepare_data.py`：把目录下所有 .parquet 切成 .bin 分片，内含 check_all 校验
- `dataloader.py`：用于预训练阶段的数据加载器
- `train_tokenizer.py`：训练自己的 BPE 分词器，结果存为 tokenizer_32k.json
- `tokenizer.py`：加载了特殊 token 的分词器 enc，以及两个输入 str、返回 token_ids(list[int])的 render 函数
- `model.py`：实现模型结构，内含 load_model 函数、ModelConfig
- `test_model.py`：用于测试 model.py 的模型各模块是否正确
- `pretrain.py`：预训练实现函数，内含 TrainConfig，支持 --log、--max_steps、--eval_interval、--resume
- `sft_data.py`：提供一些处理 question、prompt、answer 的函数，以及 SEED、N_TEST 参数
- `sft.py`：实现 sft 训练，内含 sft 阶段的数据加载器 SFTLoader
- `eval_sft.py`：提供 evaluate_accuracy 函数，用贪心采样，对测试集计算准确率
- `generate.py`：提供 generate 函数，简易的批量推理
- `grpo.py`：实现 grpo 训练
- `count_parameters.py`：按公式逐项估算模型参数量

## 复现

### 环境
- GPU：单张 RTX 4070 Ti Super 16GB，全程 bf16 autocast
- Python 3.12.14，PyTorch 2.14.0+cu130，CUDA 13.0
- 其他依赖：tiktoken 0.14.0、numpy 2.5.2、pyarrow 25.0.1

### 步骤

**1. 下载数据**：fineweb-edu sample-10BT 的前三个分片
```bash
hf download HuggingFaceFW/fineweb-edu --repo-type dataset \
  --include "sample/10BT/000_00000.parquet" "sample/10BT/001_00000.parquet" "sample/10BT/002_00000.parquet" \
  --local-dir data/raw/fineweb-edu
```

**2. 分词并切片**：`python prepare_data.py`

分词器来自 tokenizer_32k.json，由 train_tokenizer.py 训练得到，想重训可以运行 `python train_tokenizer.py --train`。

- 输入：`data/raw/fineweb-edu/sample/10BT/` 下的所有 parquet（000～002）
- 输出：`data/tokenized/bpe32k/` 下的 `val_000000.bin`（约 1000 万 token）、`train_000000~000022.bin`（每片约 1 亿 token）、`meta.json`，共约 22.9 亿 token；切完自动运行 check_all 校验

**3. 预训练**：`python pretrain.py --log logs/pretrain.log`
- 输入：上一步的 .bin 分片
- 输出：`checkpoints/final.pt`
- 1.12 亿参数，34,000 步，每步 65,536 token（B=16、T=512、累积 8 次），约 1.34 s/步，共约 13 小时
- 预期：val loss 约 3.15
- 长跑建议放在 tmux 里。日志追加写进 --log 指定的文件，可用 `tail -f` 实时查看；中断后在原命令后加 `--resume`，会从 `checkpoints/latest.pt`（每 250 步存一次）接着训练

> 第 4～6 步的预期数字来自 v0.2 的 1150 万参数模型，1.12 亿参数模型还没有跑过 SFT 和 GRPO。要复现这些数字，请先 `git checkout v0.2`。

**4. SFT**：`python sft.py`
- 输入：`checkpoints/final.pt`；题目由 sft_data.py 按 SEED=400 生成，不需要额外数据
- 输出：`checkpoints/sft.pt`（2000 步）
- 评估：`python eval_sft.py --ckpt checkpoints/sft.pt`，预期贪心准确率 100%（500 道测试题全对）
- 对比 v0.1：准确率从 92.4% 提升到 100%。gpt2 把整个和切成一个 token（如"82"），0~198 的每个和都是互不相关的 token，模型只能逐个记；自训 BPE 把数字逐位切开，每一位只需在 10 个数字里选，个位和进位的规律能推广到没见过的题。SFT 的 loss 只在答案和 `<|im_end|>` 上算，两版每道题的 token 数不同，loss 不能直接比

**5. 弱 SFT（GRPO 的起点）**：
`python sft.py --max_steps 200 --out checkpoints/sft_weak.pt`
- 预期：`eval_sft.py` 测得约 37.4%
- max_steps 从 500 下调到 200：换分词器后 SFT 学得快得多，500 步已远超目标；200 步让起点准确率接近 v0.1 的 33.4%，方便对比 GRPO 的效果

**6. GRPO**：`python grpo.py`
- 输入：`checkpoints/sft_weak.pt`
- 输出：`checkpoints/grpo_kl02.pt`
- 500 步，每 50 步打印一次测试集贪心准确率
- 预期：结尾准确率 69.4%，10 次评估平均 59.5%
- 对比 v0.1（相对各自起点的变化，单位为百分点；起点分别为 33.4% 和 37.4%）：结尾准确率 +1.2 → +32.0，10 次评估平均 −2.8 → +22.1
- 可能的原因：v0.1 里每个和是一个独立 token，某道题答对得到的奖励只提高那一个 token 的概率，对其他题几乎没有帮助；数字逐位切开后，奖励强化的是所有题共用的逐位相加和进位规则，一道题上的改进能推广到其他题


## 下一步
添加 Muon 优化器并与 AdamW 基线对比、实现多卡并行