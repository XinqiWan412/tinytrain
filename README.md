# tinytrain

从零手写的 LLM 训练全流程：预训练 → SFT → GRPO。模型结构参考 Qwen3-0.6B，
在单张 RTX 4070 Ti Super（16GB）上用 1150 万参数的小模型跑通。
模型、数据加载、训练循环、对话模板、GRPO 都是自己实现的，没有用 transformers / trl。

## 结果

当前版本验证整个 pipeline 的正确性，数字用于说明各阶段跑通，不是为了说明模型效果好。

| 阶段 | 设置 | 结果 |
|---|---|---|
| 预训练 | fineweb-edu 约 2 亿 token，3000 步 | val loss 4.2807，约 23.3 万 tok/s |
| SFT | 0~99 两数相加，9,500 题训练、500 题测试，2000 步 | 测试集贪心准确率 92.4% |
| SFT（弱起点） | 同上，500 步 | 33.4%，作为 GRPO 的起点 |
| GRPO | 每步 64 题 × 8 次采样，lr 2e-5，500 步，β=0.2 | 34.6% |

GRPO：不加 KL 时测试准确率持续下滑（33.4% → 25.8%）；加 KL（β=0.2）后稳在起点附近，
采样奖励从 0.165 涨到 0.187，但贪心准确率的提升在统计误差（约 ±2 个百分点）以内。

## 实现要点
- 模型结构：参考 Qwen3-0.6B，pre-norm、QK-norm、SwiGLU、GQA 等。配置为 hidden 256、4 层、4 头 / 2 KV 头、FFN 768，词表 32768（自训 BPE，含 5 个对话特殊 token），embedding 与 lm_head 共享权重
- 训练配置：AdamW（betas 0.9/0.95，weight decay 0.1 只加在 2 维参数上），warmup + 余弦衰减，梯度裁剪 1.0，梯度累积，bf16 autocast
- SFT：只对答案和 `<|im_end|>` 算 loss
- GRPO：每题采样 8 次，组内标准化算优势，loss 加对冻结参考模型的 KL 惩罚（k3 估计）
- 批量生成按 prompt 长度分组（eval_sft.py 和 grpo.py 的 rollout）：数字逐位切分后，prompt 长度随两个加数的位数变化，而同一批生成要求 prompt 等长
- 对话格式：与 Qwen3 一致
```
<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>
```

## 项目结构
- `prepare_data.py`：把 .parquet 数据组装成 .bin 格式的 shard 分片
- `dataloader.py`：用于预训练阶段的数据加载器
- `train_tokenizer.py`：训练自己的 BPE，尚未实现
- `tokenizer.py`：加载了特殊 token 的分词器 enc，以及两个输入 str、返回 token_ids(list[int])的 render 函数
- `model.py`：实现模型结构，内含 load_model 函数、ModelConfig
- `test_model.py`：用于测试 model.py 的模型各模块是否正确
- `pretrain.py`：预训练实现函数，内含 TrainConfig，后面 sft 和 grpo 阶段也用
- `sft_data.py`：提供一些处理 question、prompt、answer 的函数，以及 SEED、N_TEST 参数
- `sft.py`：实现 sft 训练，内含 sft 阶段的数据加载器 SFTLoader
- `eval_sft.py`：提供 evaluate_accuracy 函数，用贪心采样，对测试集计算准确率
- `generate.py`：提供 generate 函数，简易的批量推理
- `grpo.py`：实现 grpo 训练

## 复现

### 环境
- GPU：单张 RTX 4070 Ti Super 16GB，全程 bf16 autocast
- Python 3.12.14，PyTorch 2.14.0+cu130，CUDA 13.0
- 其他依赖：tiktoken 0.14.0、numpy 2.5.2、pyarrow 25.0.1

### 步骤

**1. 下载数据**：只用 fineweb-edu sample-10BT 的第一个分片
```bash
huggingface-cli download HuggingFaceFW/fineweb-edu --repo-type dataset \
  --include "sample/10BT/000_00000.parquet" --local-dir data/raw/fineweb-edu
```

**2. 分词并切片**：`python prepare_data.py`

分词器来自 tokenizer_32k.json，由 train_tokenizer.py 训练得到，想重训可以运行 `python train_tokenizer.py --train`。

- 输入：`data/raw/fineweb-edu/sample/10BT/000_00000.parquet`
- 输出：`data/tokenized/bpe32k/` 下的 `val_000000.bin`（约 1000 万 token）、`train_000000~000007.bin`（每片约 1 亿 token）、`meta.json`

**3. 预训练**：`python pretrain.py`
- 输入：上一步的 .bin 分片
- 输出：`checkpoints/final.pt`
- 3000 步，每步 65,536 token（B=16、T=512、累积 8 次），约 206ms/步，共约 10 分钟
- 预期：val loss 约 4.22
- 对比 v0.1：从 23.3 万 tok/s 提升到 31.7 万 tok/s，快了约 36%，每步从 280ms 降到 206ms，val loss 从 4.2807 降到 4.2195。两个分词器切出的 token 大小不同，loss 不能直接比；换算成每字节损失（loss / 每 token 字节数），v0.1 是 0.922，新版是 0.927，效果持平

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
训练 BPE 分词器、模型扩大至 1 亿、添加 Muon 优化器、实现多卡并行功能