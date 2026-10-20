# 实现要点

多卡并行见 [parallel.md](parallel.md)，测试见 [testing.md](testing.md)。

## 模型

- 结构与 HF Qwen3 对齐：pre-norm、QK-norm、SwiGLU、GQA，embedding 与 lm_head 共享权重。主干 Model 外面再套一层 ModelForCausalLM（model + lm_head），参数名与 HF 一致，导出时不用改名
- 1.12 亿参数配置：hidden 768、12 层、8 头 / 2 KV 头（head_dim 128）、FFN 2304、词表 32768，共 112,482,048 参数，T=512
- v0.2 的小模型：hidden 256、4 层、4 头 / 2 KV 头（head_dim 64）、FFN 768，1150 万参数

## 分词器

- 自训的 byte-level BPE，词表 32768 = 256 个字节 + 32507 次合并 + 5 个特殊 token，用 1 亿字符训练，只做英文
- 预切分照搬 Qwen3 的正则，数字逐位切开（v0.2 加法任务大幅变好的原因，见 [results.md](results.md)）
- 特殊 token 依次为 `<|endoftext|>`、`<|im_start|>`、`<|im_end|>`、`<think>`、`</think>`，占最后 5 个 id（EOT = 32763）
- tokenizer_32k.json 存切分规则、特殊 token 和 merges（不存 vocab），tokenizer.py 从它构造 tiktoken.Encoding
- 每个 token 平均 4.55 字节（gpt2 为 4.64）
- 想重训可以运行 `python train_tokenizer.py --train`，会覆盖 tokenizer_32k.json

## 数据

- fineweb-edu sample-10BT 的前三个 parquet（000～002），共 2,292,410,661 token
- prepare_data.py 一次处理目录下所有 parquet，分片编号跨文件连续；每篇文档末尾追加 EOT，存成 uint16
- 输出在 data/tokenized/bpe32k/：val_000000.bin（约 1000 万 token）、train_000000～000022.bin（每片约 1 亿 token）、meta.json（分词器、词表大小、EOT、dtype、各分片 token 数）；切完自动运行 check_all 校验
- val 固定为第一个文件开头的 1000 万 token，以后增加数据，结果仍能和之前比较

## 训练

### 训练骨架

三个阶段共用 trainer.py 的 Trainer，它负责每步的公共流程：清梯度 → 逐份算 loss 并 backward → 梯度裁剪 → 设 lr → step → 记录、日志、eval、checkpoint。eval 在训练前、每 eval_interval 步和结束时各做一次。

各阶段写一个小类（PretrainTask、SFTTask、GRPOTask），用同名方法提供各自不同的部分：

- get_parts：这一步的数据分成几份，以及 rollout 奖励这类额外数字
- loss：每份缩放好的 loss
- evaluate：val loss 或准确率
- state_dict（只有预训练有）：DataLoader 的位置

每份 loss 怎么缩放、autocast 包在哪里由阶段决定，Trainer 只负责让各份梯度相加。

### 配置

- config.py 里的 dataclass（DataConfig、TrainConfig、SFTConfig、GRPOConfig、RunConfig、ParallelConfig；ModelConfig 在 model.py）规定字段、类型和默认值；每个脚本有一个顶层配置类，规定它有哪几节
- load_config 用 tomllib 读 TOML，文件只写和默认值不同的字段；不认识的节或键、类型对不上都直接报错。命令行的 `--set 节.键=值` 先合并进来，再一起检查
- 读完的完整配置（含默认值）打进日志开头，并存进 checkpoint 的 config 键

### 超参数

|  | 预训练 | SFT | GRPO |
| --- | --- | --- | --- |
| batch | B=16、T=512，累积 8 次，每步 65,536 token | B=128 | 每步 64 题 × 8 次采样 |
| 步数 | 34,000（约 22.3 亿 token） | 2000（弱起点见 GRPO 一节） | 500 |
| lr | 1e-3，warmup 1000 步，余弦降到 1e-4 | 3e-4，warmup 50 步，余弦降到 3e-5 | 2e-5，warmup 20 步，余弦降到 2e-6 |

三个阶段共用：AdamW betas (0.9, 0.95)，weight decay 0.1 只加在 2 维参数上，梯度裁剪 1.0，bf16 autocast。Muon 路线的 SFT、GRPO 也用 Muon，lr 不变。

### 长时间训练

- 日志（logger.py，三个阶段共用）在 print 的同时追加写进文件、每行 flush，可以用 `tail -f` 看；长跑放在 tmux 里
- 断点续训保存模型、优化器、步数和 DataLoader 位置（DataLoader 提供 state_dict / load_state_dict）。预训练每 250 步存一次 latest.pt，加 `--resume` 从它接着训练；优化器名和 checkpoint 里记录的不一致时直接报错
- checkpoint 先写临时文件，再用 os.replace 替换，写到一半被打断也不会损坏旧文件

## Muon

- 做法：动量（0.95，Nesterov）之后，用 Newton-Schulz 迭代 5 次把动量矩阵近似正交化（只用矩阵乘，bf16），让各奇异方向的步长一致；weight decay 先衰减再更新
- 更新系数用 Moonlight 的 0.2×√max(行, 列)，使更新幅度与 AdamW 对齐，所以 lr、weight decay 和调度都直接沿用 AdamW 的
- 分组：块内的 2 维矩阵（q/k/v/o 和 MLP 的三个矩阵）用 Muon；embedding（与 lm_head 共享）和 1 维的 norm 参数仍用 AdamW
- 两部分装在同一个优化器类 MuonWithAdamW 里：继承 torch.optim.Optimizer，参数组用 use_muon 标记，step 时按组分别走 Muon 和 AdamW 的更新。configure_optimizer 返回单个优化器，Trainer 直接调用它的 zero_grad / step
- AdamW 部分是手写的，运算顺序与 torch 的非 fused 实现一致。最初只差末位舍入，但经 Muon 放大后超出了基线的容差，对齐运算顺序后基线不变

## SFT

- 任务：0～99 两数相加，共 10,000 题，用 SEED=400 抽 500 题作测试集。题目由 sft_data.py 生成，不需要额外数据
- 对话格式与 Qwen3 一致，prompt 以 EOT 开头：

```
<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>
```

- 只对答案和 `<|im_end|>` 算 loss（构建 x, y 的函数与 GRPO 共用）
- 评估：eval_sft.py 对测试集贪心生成，截到 `<|im_end|>` 后解析答案（解析函数与 GRPO 的打分共用）
- 批量生成按 prompt 长度分组（eval_sft.py 和 GRPO 的 rollout 都是）：数字逐位切分后，prompt 长度随两个加数的位数变化，而同一批生成要求 prompt 等长

## GRPO

- 起点是弱 SFT：SFT 只跑几十步（sft_weak.toml 里 AdamW 57 步、Muon 75 步），贪心准确率约 40%，给 RL 留出提升空间
- 每题采样 8 次（temperature 1，不做 top-k），组内标准化算优势
- loss 加对冻结参考模型的 KL 惩罚（k3 估计，β=0.2，按条平均）。策略模型和参考模型都从 run.init 读，也就是弱起点
- 每 50 步测一次测试集贪心准确率
- 分块梯度累积：1.12 亿参数时，每步 512 条序列一次前向放不下。按 chunk_size（默认 128）分块，每块单独做参考模型前向、策略模型前向和 backward，梯度在 .grad 里累加。每块的 loss 和 KL 对块内各行求和后除以总行数，所以各块梯度之和与不分块时一致。爆显存时可以改小，如 `--set grpo.chunk_size=64`

## 导出 HF 格式

- export_hf.py 把 checkpoint 导出成 config.json + model.safetensors（fp32，约 430 MB），可用 AutoModelForCausalLM.from_pretrained 直接加载；输出放在 hf/ 下，不进 git
- ModelConfig 逐项转成 Qwen3Config：max_seq_len 对应 max_position_embeddings；rope_theta 按 transformers 5.x 写进 rope_parameters；tie_word_embeddings 必须显式写 true，Qwen3Config 默认是 false
- embedding 与 lm_head 共享时只存 embed_tokens，加载时由 transformers 绑回
- 导出后自动用 transformers 以 fp32 读回，确认没有漏读或多余的权重，再比较同一输入的 logits 和一段固定英文上的 loss，输入不带 padding。加 `--gen` 再对测试集前 8 题逐题比较贪心生成，只用于 SFT / GRPO 模型（预训练模型没学过对话格式）
- 为什么要对拍、它能查出哪些错误，见 [testing.md](testing.md)
- 只导出模型，不导出分词器，config 里也没有设 eos。接 vLLM 前，要先把分词器导出成 HF 的 tokenizer.json，并在 config 里设好 eos