# tinytrain

从零手写的 LLM 训练全流程：预训练 → SFT → GRPO。模型结构参考 Qwen3-0.6B，
在单张 RTX 4070 Ti Super（16GB）上先用 1150 万参数的小模型跑通，再把全流程扩大到 1.12 亿参数。预训练支持 DDP 多卡（在 4×RTX 4090 上验证过），以及手写的 ZeRO-1。
模型、数据加载、训练循环、对话模板、GRPO 都是自己实现的，没有用 transformers / trl。

## 结果

### 1.12 亿参数全流程：AdamW vs Muon

参数量 112,482,048。两条路线的数据、步数、lr 和调度完全相同，唯一的区别是优化器（`train.optimizer`）。Muon 路线的 SFT 和 GRPO 也用 Muon。


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



### 与 HF transformers 对拍

导出后用 transformers 的 Qwen3ForCausalLM 读回，fp32、CPU 上比较。

| 模型        | logits 最大差 | 一段英文上的 loss（自己 / HF） | 测试集前 8 题贪心生成 |
| --------- | ---------- | ------------------ | ------------- |
| AdamW 预训练 | 1.9e-5     | 2.3082 / 2.3082    | —             |
| Muon 预训练  | 1.7e-5     | 2.1467 / 2.1467    | —             |
| AdamW SFT | 2.1e-5     | 11.9223 / 11.9223  | 8/8 一致，8/8 答对 |
| Muon SFT  | 2.8e-5     | 11.1828 / 11.1828  | 8/8 一致，8/8 答对 |

- HF 生成时走 KV cache，generate.py 每步从头重算，两者生成的 token 完全相同
- 小模型测试里故意改错来验证对拍有效：GQA 的 repeat_interleave 换成 repeat，logits 最大差 0.862；QK-norm 挪到 RoPE 之后，最大差 0.164（norm 权重不随机化时测不出）
- SFT 模型在普通英文上的 loss 高于均匀分布的 ln 32768 ≈ 10.4，是只练加法 2000 步造成的遗忘，两边一致，与导出无关



### DDP：单卡与四卡

在 Runpod 上租一台 4×RTX 4090 的机器（卡间无 NVLink，也不支持 P2P），Muon 路线跑前 300 步。300 步全程在 warmup 内，学习率和 v0.3 那次完全相同。单卡用其中一张卡跑。

|                 | 单卡             | 四卡（DDP）         |
| --------------- | -------------- | --------------- |
| 每步时间            | 816 ms         | 372 ms          |
| 吞吐              | 约 8.02 万 tok/s | 约 17.6 万 tok/s |
| 每张卡显存           | 10,726 MiB     | 11,304 MiB      |
| 第 250 步 val loss | 6.3355         | 6.3356          |
| 第 300 步 val loss | 6.0104         | 6.0104          |

- 扩展效率：17.6 万 ÷（8.02 万 × 4）≈ 55%，4 张卡快了约 2.2 倍
- 瓶颈是梯度通信：每步要在 4 张卡之间平均约 450 MB 的 fp32 梯度，这台机器卡间不支持 P2P，要经 CPU 内存中转。单独测一次 450 MB 的 all-reduce 要 175 ms（每张卡有效收发约 3.9 GB/s），而四卡每步比只算不传多出约 150 ms，多出的时间基本都是它。DDP 在最后一份反向时提前开始通信，只藏住了其中一小部分：每张卡只累积 2 份，计算太少，藏不住通信
- loss：四卡与单卡相差万分之几；单卡与 v0.3（本地 4070 Ti Super）相差千分之几（第 250 步 val loss 6.3355 vs 6.3354），差别来自不同显卡上 bf16 计算的末位
- 四卡每张卡比单卡多用 578 MiB 显存，是 DDP 做梯度通信用的缓冲区



### ZeRO-1：与 DDP 逐位相同

本地用 gloo 起两个 CPU 进程验证（tests/test_zero.py）：同样的配置下，两进程 ZeRO-1 与两进程 DDP 各跑 10 步，final.pt 里的参数逐位相同、val_loss 相等，AdamW 和 Muon 都是；两进程中途续训与一口气跑完也逐位相同。每个参数的更新虽然只在负责它的那张卡上算，但用的梯度（DDP 已平均好）、公式和学习率都和 DDP 相同，算出的数原样广播出去，所以测试不留容差。

测试里各卡优化器状态的字节数相加正好等于 DDP 的总量（小模型 AdamW 1,512,032 字节、Muon 823,808 字节），每张卡都少于总量的 0.75。按同样的分配规则算 1.12 亿参数模型在 4 卡时的情况（未上云实测）：

|           | DDP 每张卡          | ZeRO-1 每张卡                                          |
| --------- | ---------------- | --------------------------------------------------- |
| 负责的元素数    | 全部 112,482,048   | 28,120,576 / 28,120,576 / 28,120,448 / 28,120,448 |
| AdamW 优化器状态 | 约 900 MB         | 约 225 MB                                            |
| Muon 优化器状态  | 约 551 MB         | 0 号约 213 MB，其余各约 113 MB                             |

- Muon 路线不均匀：分配按元素数，embedding 归 0 号卡，它走 AdamW，每个元素存两份状态，Muon 矩阵只存一份 buf
- 显存的大头是激活（上面 DDP 四卡时每张卡 11,304 MiB），每张卡省下的几百 MB 占比不大；通信反而比 DDP 多一轮参数广播（每个参数一次，112M 模型每步 134 次）。速度和实测显存等 ZeRO-2 写完后上云一起测



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
- 训练配置：AdamW（betas 0.9/0.95，weight decay 0.1 只加在 2 维参数上）或 Muon + AdamW（train.optimizer=muon），warmup + 余弦衰减，梯度裁剪 1.0，梯度累积，bf16 autocast
- 训练骨架：三个阶段共用 trainer.py 的 Trainer，它负责每步的公共流程（清梯度 → 逐份算 loss 并 backward → 梯度裁剪 → 设 lr → step → 记录、日志、eval、checkpoint）。各阶段写一个小类（PretrainTask、SFTTask、GRPOTask），用同名方法提供各自不同的部分：get_parts（这一步的数据分成几份，以及 rollout 奖励这类额外数字）、loss（每份缩放好的 loss）、evaluate（val loss 或准确率），预训练另有 state_dict（DataLoader 位置）。每份 loss 怎么缩放、autocast 包在哪里由阶段决定，Trainer 只负责让各份梯度相加
- 配置：config.py 里的 dataclass（DataConfig、TrainConfig、SFTConfig、GRPOConfig、RunConfig、ParallelConfig；ModelConfig 在 model.py）规定字段、类型和默认值，每个脚本有一个顶层配置类，规定它有哪几节；load_config 用 tomllib 读 TOML 配置文件，文件只写和默认值不同的字段，不认识的节或键、类型对不上都直接报错；命令行的 --set 节.键=值 先合并进来再一起检查；读完的完整配置（含默认值）打进日志开头，并存进 checkpoint 的 config 键
- SFT：只对答案和 `<|im_end|>` 算 loss
- GRPO：每题采样 8 次，组内标准化算优势，loss 加对冻结参考模型的 KL 惩罚（k3 估计）
- GRPO 分块梯度累积：1.12 亿参数时，每步 512 条序列一次前向放不下。compute_loss 把它们分块（`--chunk_size`，默认 128），每块单独做参考模型前向、策略模型前向和 backward，梯度在 .grad 里累加；每块的 loss 和 KL 对块内各行求和后除以总行数，各块梯度之和与不分块时一致（v0.2 小模型上 1 块和 4 块的梯度范数 2.219704 vs 2.219820；现在由 tests/test_grpo.py 的分块一致性测试覆盖）
- 批量生成按 prompt 长度分组（eval_sft.py 和 grpo.py 的 rollout）：数字逐位切分后，prompt 长度随两个加数的位数变化，而同一批生成要求 prompt 等长
- 导出 HF 格式（export_hf.py）：参数名与 HF Qwen3 对齐，ModelConfig 逐项转成 Qwen3Config（max_seq_len 对应 max_position_embeddings；rope_theta 按 transformers 5.x 写进 rope_parameters；tie_word_embeddings 必须显式写 true，Qwen3Config 默认是 false）。权重存成 safetensors，embedding 与 lm_head 共享时只存 embed_tokens，加载时由 transformers 绑回。导出后用 transformers 以 fp32 读回，确认没有漏读或多余的权重，再比较同一输入的 logits；输入不带 padding，生成对比逐题单条进行。这种对拍能查出"自己和自己比"的测试查不出的约定错误，比如 GQA 里 KV 头怎么对应到 Q 头、QK-norm 放在 RoPE 前还是后：写错了形状不变、照样能训练，基线也会照着错的结果录下来。小模型测试要先把 norm 权重随机化，因为权重全为 1 时 norm 和 RoPE 可以交换顺序，QK-norm 放错位置测不出来
- 对话格式：与 Qwen3 一致

```
<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>
```

- 长时间训练：日志（logger.py，pretrain/sft/grpo 共用）在 print 的同时追加写进文件、每行 flush；断点续训保存模型、优化器、步数和 DataLoader 位置（DataLoader 提供 state_dict / load_state_dict），checkpoint 先写临时文件再用 os.replace 替换，写到一半被打断也不会损坏旧文件
- 数据切分：一次处理目录下所有 parquet，分片编号跨文件连续；val 固定为第一个文件开头的 1000 万 token，增加数据后结果仍能和之前比较
- Muon（muon.py）：动量（0.95，Nesterov）之后用 Newton-Schulz 迭代 5 次把动量矩阵近似正交化（只用矩阵乘，bf16），让各奇异方向的步长一致；更新系数用 Moonlight 的 0.2×√max(行, 列)，使更新幅度与 AdamW 对齐，lr、weight decay 和调度都直接沿用 AdamW；按参数形状分组：块内的 2 维矩阵（q/k/v/o 和 MLP 三个矩阵）用 Muon，embedding（与 lm_head 共享）和 1 维 norm 参数仍用 AdamW；两部分装在同一个优化器类 MuonWithAdamW 里：它继承 torch.optim.Optimizer，参数组用 use_muon 标记，step 时按组分别走 Muon 和 AdamW 的更新（AdamW 部分手写，运算顺序与 torch 的非 fused 实现一致，tests/test_muon.py 里与 torch.optim.AdamW 逐步对拍）；configure_optimizer 返回单个优化器，Trainer 直接调用它的 zero_grad / step
- DDP（只用于预训练）：用 torchrun 启动时，rank、world_size 从环境变量读，GPU 用 nccl、CPU 用 gloo；直接用 python 启动时走原来的单进程路径，现有测试和基线不变。全局 batch 保持 65,536，每个进程的累积次数 = 全局 batch ÷（B×T×进程数），4 卡时每卡累积 2 次。数据按进程跳着读：DataLoader 的 pos 是所有进程共同的位置，读数据时再加上 rank×B×T 的偏移，每批前进 进程数×B×T，所有进程在同一步换片；state_dict 不用改，续训时各进程自动回到自己的位置。这样每一步各进程合起来读的正是单进程会读的那些 token，平均后的梯度和单进程相同，只差浮点加法顺序。梯度累积时最后一份之前用 no_sync 跳过同步（前向和反向一起包住）；日志里的 loss 和 val_loss 先跨进程平均；只有 0 号进程写日志、存 checkpoint，存的是包装前的模型，键不带 "module." 前缀，load_model、export_hf、sft.py 照常能读。正确性在本地用 gloo 起两个 CPU 进程验证：tests/test_ddp.py 用 subprocess 分别跑单进程和两进程，比较 final.pt；另测两进程中途续训与一口气跑完逐位相同。临时去掉 rank 偏移时，交替读取测试和等价测试失败；load_state_dict 里重复加偏移时，只有续训测试失败
- ZeRO-1（只用于预训练，配置 `[parallel] zero_stage = 1`，默认 0 即原来的做法）：每个参数整个归一张卡负责，不像常见做法那样拍平后按元素切，因为 Muon 的正交化要完整矩阵。partition_params 按元素数从大到小，每次放到当前最轻的卡上，只依赖参数大小和卡数，各进程算出的分配相同；112M 模型 2 卡、4 卡时各卡元素数几乎相等，8 卡时 embedding（2516 万）一个参数就超过平均值，0 号卡偏重。parallel/zero.py 的 ZeroOptimizer 包住原来的优化器：只用本卡负责的参数建里面的 AdamW / MuonWithAdamW（两者一行不改，Muon 的正交化也随之分摊到各卡）；param_groups 直接交出里面那个优化器的，Trainer 的 set_lr 照常生效；zero_grad 清全部参数的梯度（DDP 每步把平均后的梯度写进全部参数，没清的会累加，还会经 all-reduce 带坏别的卡）；step 后每个参数由负责它的卡广播给其他卡，每张卡都按同样顺序对每个参数调用。梯度同步仍由 DDP 做，Trainer 只改了存 checkpoint：所有进程先一起调 state_dict（各卡状态搬到 CPU 后用 gather_object 收集到 0 号进程，存成每卡一项的列表，并记下卡数），再由 0 号进程写文件；续训时每张卡取自己那一项，卡数不同时报错，load_checkpoint 不用改。临时让每张卡给全部参数建优化器时，等价比较照样通过，只有字节数检查失败；临时去掉广播时，从第 1 步起错误经梯度平均扩散到全部参数，等价测试和续训测试都失败


## 项目结构

- `prepare_data.py`：把目录下所有 .parquet 切成 .bin 分片，内含 check_all 校验
- `dataloader.py`：用于预训练阶段的数据加载器，支持按进程跳着读（rank、world_size，默认单进程）
- `train_tokenizer.py`：训练自己的 BPE 分词器，结果存为 tokenizer_32k.json
- `tokenizer.py`：加载了特殊 token 的分词器 enc，以及两个输入 str、返回 token_ids(list[int])的 render 函数
- `model.py`：实现模型结构，内含 load_model 函数、ModelConfig
- `pretrain.py`：预训练，内含 main、configure_optimizer、PretrainTask（每步取 grad_accum 份数据、提供 DataLoader 状态给 checkpoint）、train（薄外壳，返回每步 loss 和梯度范数）、load_checkpoint，支持 --config、--set、--resume；用 torchrun 启动时自动走 DDP 多卡，configure_optimizer 按 [parallel] zero_stage 决定是否用 ZeroOptimizer 包装
- `sft_data.py`：提供一些处理 question、prompt、answer 的函数，以及 SEED、N_TEST 参数
- `sft.py`：SFT，内含数据加载器 SFTLoader、SFTTask、train_sft（薄外壳，返回每步 loss 和梯度范数）和 main，支持 --config、--set、--resume
- `eval_sft.py`：提供 evaluate_accuracy 函数，用贪心采样，对测试集计算准确率
- `generate.py`：提供 generate 函数，简易的批量推理
- `export_hf.py`：把 checkpoint 导出成 Hugging Face 格式（config.json + model.safetensors，transformers 可按 Qwen3ForCausalLM 直接加载），导出后自动读回对拍 logits 最大差和一段文本上的 loss；加 --gen 时再比较测试题上的贪心生成，支持 --ckpt、--out、--gen
- `grpo.py`：GRPO，内含 rollout、build_rows（拼行补齐）、chunk_loss（算一块的 loss 和 KL）、compute_loss（分块算 loss 并 backward，供测试使用）、GRPOTask 和 train_grpo（薄外壳）和 main，支持 --config、--set、--resume
- `trainer.py`：通用训练骨架。get_lr、set_lr 两个小函数，以及 Trainer 类（清梯度、逐份 backward、梯度裁剪、lr 调度、记录、日志、按间隔 eval、可选的 checkpoint 保存和 tok/s）；DDP 时梯度累积只在最后一份同步、每步 loss 跨进程平均、存 checkpoint 时所有进程先一起取优化器状态（ZeRO 时要收集各卡状态），再由 0 号进程写文件（存包装前的模型）
- `logger.py`：公共日志模块，setup_log 打开日志文件，log 同时打印并写入文件；多进程时只有 0 号进程输出
- `muon.py`：newton_schulz 正交化函数和 MuonWithAdamW 优化器类（一个优化器内按参数组分别做 Muon 和 AdamW 更新）
- `parallel/`：并行相关代码的包，以后 TP、PP 也放这里
  - `dist_utils.py`：setup（读 torchrun 给的环境变量初始化进程组，GPU 用 nccl、CPU 用 gloo；没有这些变量时什么都不做）、get_rank、get_world_size、is_main、all_reduce_mean、cleanup
  - `zero.py`：partition_params（按元素数从大到小把整个参数分给当前最轻的卡）和 ZeroOptimizer（ZeRO-1 包装类：只给本卡负责的参数建里面的优化器，zero_grad 清全部参数，step 后广播参数，state_dict 把各卡状态收集到 0 号进程）
- `pytest.ini`：配置了测试目录，默认跳过标了 slow 的测试
- `tests/`：
  - `conftest.py`：公共 fixture，tiny_config（2 层、hidden 64、词表 256 的小模型配置）、sft_config（在 tiny_config 基础上把词表改成 32768、max_seq_len 改成 64，供用真分词器的 SFT / GRPO 测试使用）和 make_data_dir（在临时目录造随机 token 的 .bin 分片和 meta.json，可指定片数和每片 token 数）
  - `test_model.py`：模型各模块、参数量公式、初始化 loss
  - `test_tokenizer.py`：分词器和 BPE 训练（含 2 个慢测试）
  - `test_sft_data.py`：train / test 切分，构建 x, y 时只有答案和 `<|im_end|>` 参与 loss
  - `test_dataloader.py`：换片、state_dict 往返、按进程跳着读（world_size=2 的两个 loader 轮流取，与单进程连续取逐批相同）
  - `test_pretrain.py`：lr 调度；用假数据和小模型在 CPU 上跑 train()：AdamW / Muon 基线（每步 loss 和梯度范数）、单 batch overfit、load_model 读回 logits 一致、续训与一口气跑完逐位一致
  - `test_muon.py`：newton_schulz 正交化后的奇异值范围；MuonWithAdamW 的 AdamW 分支与 torch.optim.AdamW 逐步对拍
  - `test_sft.py`：16 道题、B=4 在 CPU 上跑 train_sft 10 步（过程中换轮打乱两次），AdamW / Muon 基线
  - `test_grpo.py`：compute_loss 分块与不分块的 loss、KL、梯度一致；先 SFT 200 步把小模型训到采样时有对有错，再跑 4 步 train_grpo，AdamW / Muon 基线，并断言出现过有对有错的题
  - `test_config.py`：正常读取、拼错键和节名报错、类型不对报错、--set 覆盖生效，以及 configs/ 下所有文件都能读通
  - `test_export_hf.py`：把 norm 权重随机化后的小模型导出，用 transformers 以 fp32 读回，检查没有漏读或多余的权重、logits 一致；没装 transformers 时自动跳过
  - `test_ddp.py`：用 subprocess 在 CPU 上分别跑 pretrain.py 单进程和 torchrun 两进程（gloo），比较两边 final.pt 的参数和 val_loss；另测两进程中途续训与一口气跑完逐位相同
  - `test_zero.py`：分配函数的手算例子；单进程下包装与不包装各训练 10 步逐位相同、单进程续训（AdamW / Muon）；用 subprocess 起 torchrun 两进程（gloo），ZeRO-1 与 DDP 的 final.pt 逐位相同、两进程中途续训与一口气跑完逐位相同、各卡优化器状态字节数相加等于总量且每卡少于总量的 0.75
- `config.py`：配置类  （DataConfig、TrainConfig、SFTConfig、GRPOConfig、RunConfig、ParallelConfig，以及三个脚本各自的顶层配置类），load_config 读 TOML 并支持 --set 覆盖
- `configs/`：adamw_112m、muon_112m 两条路线，每条 4 个配置文件（pretrain、sft、sft_weak、grpo），对应 v0.3 的 8 次运行


## 复现


### 环境

- GPU：单张 RTX 4070 Ti Super 16GB，全程 bf16 autocast
- Python 3.12.14，PyTorch 2.14.0+cu130，CUDA 13.0
- 其他依赖：tiktoken 0.14.0、numpy 2.5.2、pyarrow 25.0.1、pytest 9.1.1；导出 HF 格式另需 transformers 5.19.0、safetensors 0.8.0
- 多卡测速：Runpod 上的 4×RTX 4090 24GB（驱动 580，CUDA 13.0，卡间无 NVLink、不支持 P2P），软件版本与本地相同


### 测试

`pytest -v`：跑快测试（49 个，约 55 秒，不需要数据）
`pytest -m slow -v`：跑慢测试（2 个，需要第 1 步下载的原始数据）

训练循环的测试用假数据和小模型在 CPU 上以 fp32 运行，同一台机器上结果逐位可复现。基线记录的是当前代码每步的 loss 和梯度范数（相对误差 1e-5），只能说明行为没变，不能说明行为正确，所以另有 overfit 测试检查能否学会。有意改变训练行为时（如改 lr 调度）才重录基线，并在提交说明里写明。CPU 测试不覆盖 bf16 autocast 和训练速度。DDP 的等价测试也在 CPU 上跑：把 CUDA_VISIBLE_DEVICES 置空，用 torchrun 起两个进程走 gloo，和以后在多卡机器上的启动方式相同。ZeRO-1 的等价测试比较的是两进程 ZeRO-1 和两进程 DDP，两边都用 torchrun 启动，要求逐位相同；DDP 的等价测试比较的是 python 单进程和 torchrun 两进程，而 torchrun 会给每个进程设 OMP_NUM_THREADS=1，CPU 上矩阵乘的线程数不同，加法顺序就不同，末位会有差别，所以留了 1e-5 的容差。基线只保证在录它的机器上逐位一致，换一台机器时 Muon 和 GRPO 的基线可能在末几位对不上（在 Runpod 上实测有 4 个基线测试因此失败，其余全部通过）。

SFT 和 GRPO 的基线用真分词器和小模型在 CPU 上跑，GRPO 另外记录每步的 KL 和平均奖励。GRPO 基线要求采样时有对有错：如果全对或全错，组内优势全为 0，策略梯度不起作用，基线就测不到它。临时把 compute_loss 里的优势取反时，GRPO 基线会失败而分块测试不受影响。

### 步骤

每次运行的参数都写在 configs/ 下的配置文件里，脚本只用 --config 指定。临时改参数用 --set 节.键=值（可写多次），比如 --set train.max_steps=250。完整配置会打在日志开头，也会存进 checkpoint。

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

**3. 预训练**：
`python pretrain.py --config configs/adamw_112m/pretrain.toml`

- 输入：上一步的 .bin 分片
- 输出：`checkpoints/final.pt`
- 1.12 亿参数，34,000 步，每步 65,536 token（B=16、T=512、累积 8 次），约 1.34 s/步，共约 13 小时
- 预期：val loss 约 3.15
- 长跑建议放在 tmux 里。日志追加写进 run.log 指定的文件（logs/adamw_112m/pretrain.log），可用 `tail -f` 实时查看；中断后在原命令后加 `--resume`，会从 `checkpoints/adamw_112m/latest.pt`（每 250 步存一次）接着训练

**4. SFT**：
`python sft.py --config configs/adamw_112m/sft.toml`

- 题目由 sft_data.py 按 SEED=400 生成，不需要额外数据
- 2000 步，B=128，lr 3e-4（warmup 50 步）余弦降到 3e-5
- 评估：`python eval_sft.py --ckpt checkpoints/adamw_112m/sft.pt`，预期贪心准确率 100%（500 道测试题全对）

**5. 弱 SFT（GRPO 的起点）**：
`python sft.py --config configs/adamw_112m/sft_weak.toml`

- 预期：`python eval_sft.py --ckpt checkpoints/adamw_112m/sft_weak.pt` 测得约 40.0%
- 1.12 亿参数模型 SFT 学得更快，准确率在 50～80 步之间涨得很陡（50 步 23.6%，55 步 28.8%，60 步 43.0%，80 步 80.4%）；选 57 步让起点接近 v0.2 的 37.4%，方便对比 GRPO 的效果。因为这一段很陡，复现时起点准确率可能和 40.0% 有几个点的出入

**6. GRPO**：
`python grpo.py --config configs/adamw_112m/grpo.toml`

- 策略模型和参考模型都从 run.init 读
- 500 步，每 50 步打印一次测试集贪心准确率
- 显存：每步要对 512 条序列做前向和反向，默认按 128 条一块做梯度累积；爆显存可以改小，如 `--set grpo.chunk_size=64`，梯度和不分块时一致
- 预期：结尾准确率 94.2%，10 次评估平均 82.2%
- 对比 v0.2（相对各自起点的变化，单位为百分点；起点分别为 37.4% 和 40.0%）：结尾准确率 +32.0 → +54.2，10 次评估平均 +22.1 → +42.2。训练过程中没有出现 v0.2 前 50 步先掉的现象；结尾训练采样奖励约 0.8（v0.2 约 0.40），KL 约 0.2～0.5（v0.2 约 0.11）

**7. Muon 路线**：数据和第 1～2 步相同，使用 configs/muon_112m/ 下的配置文件

```bash
python pretrain.py --config configs/muon_112m/pretrain.toml
python sft.py --config configs/muon_112m/sft.toml
python sft.py --config configs/muon_112m/sft_weak.toml
python grpo.py --config configs/muon_112m/grpo.toml
```

- 预期：预训练 val loss 约 3.107，SFT 100%，弱起点约 40.8%，GRPO 结尾 84.6%、10 次评估平均 71.5%
- 中断后在预训练命令后加 `--resume`，会从 `checkpoints/muon_112m/latest.pt` 接着训练；优化器名和 checkpoint 里记录的不一致会直接报错

**8. 导出 HF 格式并对拍**：

```bash
python export_hf.py --ckpt checkpoints/adamw_112m/final.pt --out hf/adamw_112m/pretrain
python export_hf.py --ckpt checkpoints/adamw_112m/sft.pt --out hf/adamw_112m/sft --gen
```

- 输出目录里是 config.json 和 model.safetensors（fp32，约 430 MB），可用 `AutoModelForCausalLM.from_pretrained` 直接加载；hf/ 已加进 .gitignore
- 导出后自动读回，打印 logits 最大差和一段固定英文上两边的 loss；加 --gen 再对测试集前 8 题比较贪心生成，只用于 SFT / GRPO 模型（预训练模型没学过对话格式）
- 只导出模型，不导出分词器，config 里也没有设 eos
- 预期：logits 最大差约 2e-5，两边 loss 相同，SFT 生成 8/8 一致、8/8 答对
- Muon 路线把路径里的 adamw_112m 换成 muon_112m



**9. 多卡（DDP / ZeRO-1）**：

```bash
torchrun --standalone --nproc_per_node=4 pretrain.py --config configs/muon_112m/pretrain.toml
```

- 只支持单机多卡。卡数要同时整除 8（65,536 ÷ B×T）和 eval_steps（默认 20），所以默认配置下可以用 1、2、4 张；用 8 张卡时要把 eval_steps 改成 8 的倍数，比如 `--set data.eval_steps=24`
- 加 `--resume` 照样能续训；checkpoint 只由 0 号进程写，格式和单卡相同
- 本地没有多卡时，可以用 CPU 验证多进程的逻辑：`CUDA_VISIBLE_DEVICES= torchrun --standalone --nproc_per_node=2 pretrain.py --config 小配置`，tests/test_ddp.py 就是这么做的
- 复现上面的测速：加 `--set train.max_steps=300`，并用 `--set run.ckpt_dir=…`、`--set run.log=…` 指到临时目录；单卡那次用 python 启动，命令前加 `CUDA_VISIBLE_DEVICES=0`。前 300 步只用到第一片训练数据，云上只需传 train_000000.bin、val_000000.bin 和 meta.json
- 加 `--set parallel.zero_stage=1` 用 ZeRO-1，各卡只存自己负责的那些参数的优化器状态。checkpoint 的 optimizer 一项变成每卡一项的列表，续训时卡数必须和存的时候相同；zero_stage=0 和 1 存的 checkpoint 不能互相续训



## 下一步

先把训练这一侧做成框架，推理框架放到之后。按顺序推进：

1. 训练框架化（多卡并行要改训练循环，先把循环统一起来再做会容易得多）
  - 已完成：把 pretrain.py、sft.py、grpo.py 里重复的训练循环（梯度累积、lr 调度、梯度裁剪、日志、checkpoint）抽成统一的 Trainer，各阶段只提供数据和 loss
  - 已完成：把 Muon 和 AdamW 合并成一个优化器 MuonWithAdamW，configure_optimizer 返回单个对象
  - 已完成：实验参数改用配置文件描述
2. 导出 HF 格式对拍（已完成）：导出 config.json + safetensors，用 transformers 加载后比较 logits 和贪心生成，验证模型实现
3. 多卡并行：依次实现 DDP、ZeRO / FSDP、TP、PP
  - 已完成：DDP（只用于预训练），4×RTX 4090 上扩展效率约 55%，瓶颈是卡间不支持 P2P 时的梯度通信
  - 已完成：ZeRO-1（只切优化器状态，整个参数分给各卡；本地验证与 DDP 逐位相同）
  - 下一步：ZeRO-2（梯度只汇总到负责的卡，总通信量回到 DDP 的水平），写完后上云一次测 DDP、ZeRO-1、ZeRO-2 的速度和显存，并据实测决定要不要合并逐参数的小通信；再往后是 FSDP2（ZeRO-3）
4. 性能优化
  - 先测 MFU，用 torch.profiler 定位瓶颈
  - 再依次尝试 torch.compile、把自写的 Triton FlashAttention 注册为 custom op 接入、分块计算 lm_head + cross-entropy（不物化完整 logits）
  - 激活显存优化，如 activation checkpointing
5. 序列打包：预训练和 SFT 改为打包 + 按文档边界的 varlen attention，避免跨文档的注意力和 padding 浪费



### 之后：推理框架

- 实现 KV cache、批量调度等，接入 GRPO 的 rollout，并与 vLLM 对照
- 接 vLLM 前，先把分词器导出成 HF 的 tokenizer.json，并在 config 里设好 eos



### 可选

- 评估：给预训练底座测 HellaSwag 等零样本 benchmark，与同量级的外部模型对比
- 实验严谨性：两种优化器各自扫 lr，GRPO 多种子重复并加长训练
- RL 算法：GRPO 加 PPO 式的 ratio + clip、一批采样多次更新，尝试 DAPO 的改进
- 任务：在 a+b 之外加入更多题型，用于训练和评估
- 模型结构：支持 MoE
- CI：测试接入 GitHub Actions

