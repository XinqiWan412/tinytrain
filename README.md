# tinytrain

从零手写的 LLM 训练全流程：预训练 → SFT → GRPO。模型结构参考 Qwen3-0.6B，
在单张 RTX 4070 Ti Super（16GB）上先用 1150 万参数的小模型跑通，再把全流程扩大到 1.12 亿参数。
模型、数据加载、训练循环、对话模板、GRPO 都是自己实现的，没有用 transformers / trl。

## 结果

### 1.12 亿参数全流程：AdamW vs Muon

参数量 112,482,048。两条路线的数据、步数、lr 和调度完全相同，唯一的区别是优化器（`--optimizer`）。Muon 路线的 SFT 和 GRPO 也用 Muon。


| 阶段   | 指标                  | AdamW          | Muon           |
| ---- | ------------------- | -------------- | -------------- |
| 预训练  | val loss（第 34000 步） | 3.1466         | 3.1069         |
|      | 每字节损失               | 0.692          | 0.683          |
|      | 速度                  | 约 4.90 万 tok/s | 约 4.76 万 tok/s |
|      | 显存                  | 约 11.7 GB      | 约 10.6 GB      |
|      | 总时长（按每步时间估算）        | 约 12.6 小时      | 约 13.0 小时      |
| SFT  | 2000 步，测试集贪心准确率     | 100%           | 100%           |
| 弱起点  | 步数，准确率              | 57 步，40.0%     | 75 步，40.8%     |
| GRPO | 结尾（相对起点）            | 94.2%（+54.2）   | 84.6%（+43.8）   |
|      | 10 次评估平均（相对起点）      | 82.2%（+42.2）   | 71.5%（+30.7）   |


- 预训练：Muon 最终 val loss 低 0.040；第 28250 步（3.1443）就达到了 AdamW 的最终值，少用约 17% 的步数，扣掉每步慢约 3%，按时间算约省 14%。这个 17% 是偏保守的估计：两条线的 lr 调度都按 34000 步排，Muon 在第 28250 步时 lr 还没退火完，如果按 28250 步排调度，可能更早追上
- SFT 和 GRPO：Muon 在短程训练的前期都比 AdamW 慢（弱起点要 75 步才到 40%，AdamW 57 步；GRPO 提升少约 10 个点，500 步结尾仍在上涨）。可能的原因：Muon 的更新幅度按 AdamW 稳定后的步长对齐，而 AdamW 在训练开头的实际步长更大，同样的 lr 下 Muon 在短程微调里更保守。这一点没有做实验验证；GRPO 只跑了一次，波动也不小，关于优化器的结论以预训练对比为准



### 1150 万参数全流程（v0.2）

这一版用于验证整个 pipeline 的正确性，数字用于说明各阶段跑通，不是为了说明模型效果好。复现这一版请先 `git checkout v0.2`。


| 阶段       | 设置                                  | 结果                             |
| -------- | ----------------------------------- | ------------------------------ |
| 预训练      | fineweb-edu 约 2 亿 token，3000 步      | val loss 4.2195，约 31.7 万 tok/s |
| SFT      | 0~99 两数相加，9,500 题训练、500 题测试，2000 步  | 测试集贪心准确率 100%                  |
| SFT（弱起点） | 同上，200 步                            | 37.4%，作为 GRPO 的起点              |
| GRPO     | 每步 64 题 × 8 次采样，lr 2e-5，500 步，β=0.2 | 结尾 69.4%，10 次评估平均 59.5%        |


对比 v0.1（gpt2 分词器）：

- 预训练：速度从 23.3 万提升到 31.7 万 tok/s，每字节损失 0.922 vs 0.927，效果持平
- SFT：准确率从 92.4% 提升到 100%。gpt2 把整个和切成一个 token（如"82"），0~198 的每个和都是互不相关的 token，模型只能逐个记；自训 BPE 把数字逐位切开，每一位只需在 10 个数字里选，个位和进位的规律能推广到没见过的题。SFT 的 loss 只在答案和 `<|im_end|>` 上算，两版每道题的 token 数不同，loss 不能直接比
- 弱起点：步数从 500 下调到 200，换分词器后 SFT 学得快得多，200 步让起点接近 v0.1 的 33.4%
- GRPO：相对各自起点的变化（单位为百分点；起点分别为 33.4% 和 37.4%），结尾准确率 +1.2 → +32.0，10 次评估平均 −2.8 → +22.1。可能的原因：v0.1 里每个和是一个独立 token，某道题答对得到的奖励只提高那一个 token 的概率，对其他题几乎没有帮助；数字逐位切开后，奖励强化的是所有题共用的逐位相加和进位规则，一道题上的改进能推广到其他题



## 实现要点

- 模型结构：配置为 hidden 768、12 层、8 头 / 2 KV 头（head_dim 128）、FFN 2304，共 1.12 亿参数（v0.2 为 hidden 256、4 层、FFN 768，1150 万参数）
- 训练配置：AdamW（betas 0.9/0.95，weight decay 0.1 只加在 2 维参数上）或 Muon + AdamW（--optimizer muon），warmup + 余弦衰减，梯度裁剪 1.0，梯度累积，bf16 autocast
- SFT：只对答案和 `<|im_end|>` 算 loss
- GRPO：每题采样 8 次，组内标准化算优势，loss 加对冻结参考模型的 KL 惩罚（k3 估计）
- GRPO 分块梯度累积：1.12 亿参数时，每步 512 条序列一次前向放不下。compute_loss 把它们分块（`--chunk_size`，默认 128），每块单独做参考模型前向、策略模型前向和 backward，梯度在 .grad 里累加；每块的 loss 和 KL 对块内各行求和后除以总行数，各块梯度之和与不分块时一致（v0.2 小模型上 1 块和 4 块的梯度范数 2.219704 vs 2.219820；现在由 tests/test_grpo.py 的分块一致性测试覆盖）
- 批量生成按 prompt 长度分组（eval_sft.py 和 grpo.py 的 rollout）：数字逐位切分后，prompt 长度随两个加数的位数变化，而同一批生成要求 prompt 等长
- 对话格式：与 Qwen3 一致

```
<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>
```

- 长时间训练：日志（logger.py，pretrain/sft/grpo 共用）在 print 的同时追加写进文件、每行 flush；断点续训保存模型、优化器、步数和 DataLoader 位置（DataLoader 提供 state_dict / load_state_dict），checkpoint 先写临时文件再用 os.replace 替换，写到一半被打断也不会损坏旧文件
- 数据切分：一次处理目录下所有 parquet，分片编号跨文件连续；val 固定为第一个文件开头的 1000 万 token，增加数据后结果仍能和之前比较
- Muon（muon.py）：动量（0.95，Nesterov）之后用 Newton-Schulz 迭代 5 次把动量矩阵近似正交化（只用矩阵乘，bf16），让各奇异方向的步长一致；更新系数用 Moonlight 的 0.2×√max(行, 列)，使更新幅度与 AdamW 对齐，lr、weight decay 和调度都直接沿用 AdamW；按参数形状分组：块内的 2 维矩阵（q/k/v/o 和 MLP 三个矩阵）用 Muon，embedding（与 lm_head 共享）和 1 维 norm 参数仍用 AdamW；configure_optimizer 返回优化器列表，训练循环通过 zero_grad_all / set_lr / step_all 统一操作



## 项目结构

- `prepare_data.py`：把目录下所有 .parquet 切成 .bin 分片，内含 check_all 校验
- `dataloader.py`：用于预训练阶段的数据加载器
- `train_tokenizer.py`：训练自己的 BPE 分词器，结果存为 tokenizer_32k.json
- `tokenizer.py`：加载了特殊 token 的分词器 enc，以及两个输入 str、返回 token_ids(list[int])的 render 函数
- `model.py`：实现模型结构，内含 load_model 函数、ModelConfig
- `pretrain.py`：预训练实现函数，内含 TrainConfig、train（返回每步 loss 和梯度范数）、save_checkpoint / load_checkpoint，支持 --log、--max_steps、--eval_interval、--resume、--optimizer、--ckpt_dir
- `sft_data.py`：提供一些处理 question、prompt、answer 的函数，以及 SEED、N_TEST 参数
- `sft.py`：实现 sft 训练，内含 sft 阶段的数据加载器 SFTLoader 和训练循环 train_sft（返回每步 loss 和梯度范数），支持 --init、--out、--log、--max_steps、--optimizer
- `eval_sft.py`：提供 evaluate_accuracy 函数，用贪心采样，对测试集计算准确率
- `generate.py`：提供 generate 函数，简易的批量推理
- `grpo.py`：实现 grpo 训练，内含 rollout、compute_loss 和训练循环 train_grpo（返回每步 loss、KL、梯度范数、平均奖励和有对有错的比例），支持 --init、--out、--log、--max_steps、--chunk_size、--optimizer
- `logger.py`：公共日志模块，setup_log 打开日志文件，log 同时打印并写入文件
- `muon.py`：newton_schulz 正交化函数和 Muon 优化器类
- `pytest.ini`：配置了测试目录，默认跳过标了 slow 的测试
- `tests/`：
  - `conftest.py`：公共 fixture，tiny_config（2 层、hidden 64、词表 256 的小模型配置）、sft_config（在 tiny_config 基础上把词表改成 32768、max_seq_len 改成 64，供用真分词器的 SFT / GRPO 测试使用）和 make_data_dir（在临时目录造随机 token 的 .bin 分片和 meta.json，可指定片数和每片 token 数）
  - `test_model.py`：模型各模块、参数量公式、初始化 loss
  - `test_tokenizer.py`：分词器和 BPE 训练（含 2 个慢测试）
  - `test_sft_data.py`：train / test 切分，构建 x, y 时只有答案和 `<|im_end|>` 参与 loss
  - `test_dataloader.py`：换片、state_dict 往返
  - `test_pretrain.py`：lr 调度；用假数据和小模型在 CPU 上跑 train()：AdamW / Muon 基线（每步 loss 和梯度范数）、单 batch overfit、load_model 读回 logits 一致、续训与一口气跑完逐位一致
  - `test_muon.py`：newton_schulz 正交化后的奇异值范围
  - `test_sft.py`：16 道题、B=4 在 CPU 上跑 train_sft 10 步（过程中换轮打乱两次），AdamW / Muon 基线
  - `test_grpo.py`：compute_loss 分块与不分块的 loss、KL、梯度一致；先 SFT 200 步把小模型训到采样时有对有错，再跑 4 步 train_grpo，AdamW / Muon 基线，并断言出现过有对有错的题



## 复现



### 环境

- GPU：单张 RTX 4070 Ti Super 16GB，全程 bf16 autocast
- Python 3.12.14，PyTorch 2.14.0+cu130，CUDA 13.0
- 其他依赖：tiktoken 0.14.0、numpy 2.5.2、pyarrow 25.0.1, pytest 9.1.1



### 测试

`pytest -v`：跑快测试（30 个，约 14 秒，不需要数据）
`pytest -m slow -v`：跑慢测试（2 个，需要第 1 步下载的原始数据）

训练循环的测试用假数据和小模型在 CPU 上以 fp32 运行，同一台机器上结果逐位可复现。基线记录的是当前代码每步的 loss 和梯度范数（相对误差 1e-5），只能说明行为没变，不能说明行为正确，所以另有 overfit 测试检查能否学会。有意改变训练行为时（如改 lr 调度）才重录基线，并在提交说明里写明。CPU 测试不覆盖 bf16 autocast 和训练速度。

SFT 和 GRPO 的基线用真分词器和小模型在 CPU 上跑，GRPO 另外记录每步的 KL 和平均奖励。GRPO 基线要求采样时有对有错：如果全对或全错，组内优势全为 0，策略梯度不起作用，基线就测不到它。临时把 compute_loss 里的优势取反时，GRPO 基线会失败而分块测试不受影响。

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
- 训完把 checkpoint 挪进这条路线的目录，后面几步都从这里读：

```bash
mkdir -p checkpoints/adamw_112m
mv checkpoints/final.pt checkpoints/latest.pt checkpoints/adamw_112m/
```

**4. SFT**：
`python sft.py --init checkpoints/adamw_112m/final.pt --out checkpoints/adamw_112m/sft.pt --log logs/sft.log`

- 题目由 sft_data.py 按 SEED=400 生成，不需要额外数据
- 2000 步，B=128，lr 3e-4（warmup 50 步）余弦降到 3e-5
- 评估：`python eval_sft.py --ckpt checkpoints/adamw_112m/sft.pt`，预期贪心准确率 100%（500 道测试题全对）

**5. 弱 SFT（GRPO 的起点）**：
`python sft.py --init checkpoints/adamw_112m/final.pt --max_steps 57 --out checkpoints/adamw_112m/sft_weak.pt --log logs/sft_weak.log`

- 预期：`python eval_sft.py --ckpt checkpoints/adamw_112m/sft_weak.pt` 测得约 40.0%
- 1.12 亿参数模型 SFT 学得更快，准确率在 50～80 步之间涨得很陡（50 步 23.6%，55 步 28.8%，60 步 43.0%，80 步 80.4%）；选 57 步让起点接近 v0.2 的 37.4%，方便对比 GRPO 的效果。因为这一段很陡，复现时起点准确率可能和 40.0% 有几个点的出入

**6. GRPO**：
`python grpo.py --init checkpoints/adamw_112m/sft_weak.pt --out checkpoints/adamw_112m/grpo_kl02.pt --log logs/grpo.log`

- 策略模型和参考模型都从 --init 读
- 500 步，每 50 步打印一次测试集贪心准确率
- 显存：每步要对 512 条序列做前向和反向，默认按 128 条一块做梯度累积；爆显存可以改小，如 `--chunk_size 64`，梯度和不分块时一致
- 预期：结尾准确率 94.2%，10 次评估平均 82.2%
- 对比 v0.2（相对各自起点的变化，单位为百分点；起点分别为 37.4% 和 40.0%）：结尾准确率 +32.0 → +54.2，10 次评估平均 +22.1 → +42.2。训练过程中没有出现 v0.2 前 50 步先掉的现象；结尾训练采样奖励约 0.8（v0.2 约 0.40），KL 约 0.2～0.5（v0.2 约 0.11）

**7. Muon 路线**：数据和第 1～2 步相同，各步命令加 `--optimizer muon`，输出放 `checkpoints/muon_112m/`

```bash
python pretrain.py --optimizer muon --ckpt_dir checkpoints/muon_112m --log logs/pretrain_muon.log
python sft.py --init checkpoints/muon_112m/final.pt --out checkpoints/muon_112m/sft.pt --optimizer muon --log logs/sft_muon.log
python sft.py --init checkpoints/muon_112m/final.pt --max_steps 75 --out checkpoints/muon_112m/sft_weak.pt --optimizer muon --log logs/sft_weak_muon.log
python grpo.py --init checkpoints/muon_112m/sft_weak.pt --out checkpoints/muon_112m/grpo_kl02.pt --optimizer muon --log logs/grpo_muon.log
```

- 预期：预训练 val loss 约 3.107，SFT 100%，弱起点约 40.8%，GRPO 结尾 84.6%、10 次评估平均 71.5%
- 中断后在预训练命令后加 `--resume`，会从 `checkpoints/muon_112m/latest.pt` 接着训练；优化器名和 checkpoint 里记录的不一致会直接报错



## 下一步

先把训练这一侧做成框架，推理框架放到之后。按顺序推进：

1. 训练框架化（多卡并行要改训练循环，先把循环统一起来再做会容易得多）
  - 把 pretrain.py、sft.py、grpo.py 里重复的训练循环（梯度累积、lr 调度、梯度裁剪、日志、checkpoint）抽成统一的 Trainer，各阶段只提供数据和 loss
  - 优化器改成类（如 MuonWithAdamW：内部按参数分组，对外是一个优化器）
  - 实验参数改用配置文件描述
2. 导出 HF 格式对拍：参数名已与 HF Qwen3 对齐，导出 config.json + safetensors，用 transformers 加载后比较同一输入的 logits，验证模型实现
3. 多卡并行：上云租 4 卡 5090，依次实现 DDP、ZeRO / FSDP、TP、PP
4. 性能优化
  - 先测 MFU，用 torch.profiler 定位瓶颈
  - 再依次尝试 torch.compile、把自写的 Triton FlashAttention 注册为 custom op 接入、分块计算 lm_head + cross-entropy（不物化完整 logits）
  - 激活显存优化，如 activation checkpointing
5. 序列打包：预训练和 SFT 改为打包 + 按文档边界的 varlen attention，避免跨文档的注意力和 padding 浪费



### 之后：推理框架

- 实现 KV cache、批量调度等，接入 GRPO 的 rollout，并与 vLLM 对照



### 可选

- 评估：给预训练底座测 HellaSwag 等零样本 benchmark，与同量级的外部模型对比
- 实验严谨性：两种优化器各自扫 lr，GRPO 多种子重复并加长训练
- RL 算法：GRPO 加 PPO 式的 ratio + clip、一批采样多次更新，尝试 DAPO 的改进
- 任务：在 a+b 之外加入更多题型，用于训练和评估
- 模型结构：支持 MoE
- CI：测试接入 GitHub Actions

