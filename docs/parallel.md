# 多卡并行

只用于预训练，只支持单机多卡。测速结果见 [results.md](results.md#多卡)，各测试能抓到哪些错误见 [testing.md](testing.md)。

## 用法

```bash
torchrun --standalone --nproc_per_node=4 pretrain.py --config configs/adamw_112m/pretrain.toml --set parallel.zero_stage=1
```

直接用 python 启动时走原来的单进程路径。用 `parallel.zero_stage` 选做法：

| zero_stage | 做法 | 模型 | 优化器 | 支持的优化器 | checkpoint 里的优化器状态 |
| --- | --- | --- | --- | --- | --- |
| 0（默认） | DDP | 包 DDP | 原来的优化器 | AdamW、Muon | 和单卡相同 |
| 1 | ZeRO-1 | 包 DDP | ZeroOptimizer 包住原来的优化器 | AdamW、Muon | 每卡一项的列表，续训时卡数必须相同 |
| 2 | ZeRO-2 | 不包 DDP，开头由 0 号卡广播一次参数 | 同上，另外负责梯度汇总和跨卡求范数 | AdamW、Muon | 同上 |
| 3 | FSDP2 | fully_shard 分片，不包 DDP | 分片之后再建的原来的优化器 | 只有 AdamW（配 muon 直接报错） | 按参数名存的完整状态，换卡数也能续训 |

张量并行另用 `parallel.tp_size` 打开（默认 1）：

```bash
torchrun --standalone --nproc_per_node=2 pretrain.py --config configs/adamw_112m/pretrain.toml --set parallel.tp_size=2
```

只支持 tp_size=2（KV 头只有 2 个）、zero_stage=0、AdamW，不满足时 main 一开头就报错；可以和 DP、PP 组合，见下面的 3D 用法。

流水线并行用 `parallel.pp_size` 打开（默认 1）：

```bash
torchrun --standalone --nproc_per_node=2 pretrain.py --config configs/adamw_112m/pretrain.toml --set parallel.pp_size=2 --set parallel.pp_first_stage_layers=7
```

只支持 pp_size=2、zero_stage=0，AdamW、Muon 都支持，不满足时 main 一开头就报错；可以和 DP、TP 组合，见下面的 3D 用法。

- `parallel.pp_first_stage_layers`：第一段放几层，默认 0 表示平分。这个模型 7/5 最快，见 [results.md](results.md)
- `parallel.pp_schedule`：`1f1b`（默认）或 `gpipe`，两者训练结果逐位相同，1F1B 更省显存

模型部分存成和单卡相同的格式；优化器状态每卡一项，续训时卡数和切点都要相同。

- 三种并行可以同时打开，卡数 = dp × tp × pp，DP 份数由卡数推出。8 卡 DP2×TP2×PP2：

```bash
torchrun --standalone --nproc_per_node=8 pretrain.py --config configs/adamw_112m/pretrain.toml --set parallel.tp_size=2 --set parallel.pp_size=2 --set parallel.pp_first_stage_layers=7 --set data.eval_steps=24
```

- 只要 tp_size 或 pp_size 大于 1，就不包 DDP，DP 的梯度同步由优化器完成，见 [3D 并行](#3d-并行dptppp)。

- DP 份数（卡数 ÷ (tp_size × pp_size)）要同时整除 65,536 ÷ (B×T)（默认是 8）和 eval_steps（默认 20）。纯 DP 时可以用 1、2、4 张卡；DP 份数为 8 时要把 eval_steps 改成 8 的倍数，比如 --set data.eval_steps=24
- 加 `--resume` 照样能续训。四种做法存的模型部分都和单卡格式相同，load_model、export_hf、sft.py 能直接读；但 zero_stage 0、1～2、3 这三类存的 checkpoint 不能互相续训，zero_stage=3 存的只能用 zero_stage=3 多卡续训；TP、PP、3D 存的 checkpoint 续训时，dp、tp、pp 三个大小都要和存的时候相同
- 本地没有多卡时，可以用 CPU 验证多进程的逻辑：`CUDA_VISIBLE_DEVICES= torchrun --standalone --nproc_per_node=2 pretrain.py --config 小配置`，tests/ 里的多进程测试都是这么做的
- 复现 results.md 里的测速：加 `--set train.max_steps=300`，并用 `--set run.ckpt_dir=…`、`--set run.log=…` 指到临时目录；单卡那次用 python 启动，命令前加 `CUDA_VISIBLE_DEVICES=0`。前 300 步只用到第一片训练数据，云上只需传 train_000000.bin、val_000000.bin 和 meta.json



## 公共部分

### 进程和数据

- 启动：parallel/dist_utils.py 的 setup 读 torchrun 给的环境变量，先 set_device 再初始化进程组，GPU 用 nccl、CPU 用 gloo；没有这些变量时什么都不做，所以直接用 python 启动时现有测试和基线不变
- 全局 batch 保持 65,536：每个进程的累积次数 = 全局 batch ÷（B×T×进程数），4 卡时每卡累积 2 次
- 数据按进程跳着读：DataLoader 的 pos 是所有进程共同的位置，读数据时再加上 rank×B×T 的偏移，每批前进 进程数×B×T，所有进程在同一步换片。state_dict 不用改，续训时各进程自动回到自己的位置。这样每一步各进程合起来读的，正是单进程会读的那些 token，平均后的梯度和单进程相同，只差浮点加法顺序
- 日志里的 loss 和 val_loss 先跨进程平均（all_reduce_mean 先求和再除以卡数，因为 gloo 不支持 AVG）；evaluate 每个进程测 eval_steps ÷ 卡数 批再平均，所以卡数要整除 eval_steps
- 只有 0 号进程写日志、存 checkpoint，存的是包装前的模型；训练结束时把各卡的峰值显存收集到 0 号进程打印
- 并行布局：parallel/dist_utils.py 的 compute_layout 算出本卡的 TP、DP、PP 编号和份数，按 Megatron 的惯例 TP 变化最快（放在编号相邻的卡上）、PP 变化最慢；DataLoader 用 DP 编号和份数读数据，只在纯 DP（DP 份数大于 1，tp_size、pp_size 都是 1）时包 DDP。不开 TP、PP 时 DP 编号就是 rank、份数就是卡数，和原来一样。进程组都由 init_parallel 建，三种组由同一个 _group_ranks 按"另外两个编号相同"归组：TP 组是 pp、dp 编号都相同的卡（编号相邻），PP 组是 dp、tp 编号都相同的卡，DP 组是 pp、tp 编号都相同的卡；每条流水线的首尾两段再组成 embedding 组。每个进程要对每个组都调用 new_group，自己不在的组也要调

### Trainer 里的接入点

Trainer 不认识具体做法，只看模型和优化器有没有某个方法：

| 接入点 | 谁用到 |
| --- | --- |
| 梯度累积时，最后一份之前用 no_sync 包住前向和反向 | DDP、ZeRO-1 |
| 模型有 set_requires_gradient_sync 时，每份之前设一次，只在最后一份打开 | FSDP2 |
| 模型是 PP 模型（有 pp_stage）时，用 pp_schedule 选出的调度函数代替逐份的前向反向，再把最后一段的 loss 在 PP 组里广播；evaluate 改用 pp_forward | PP |
| 优化器有 reduce_grads 就在逐份 backward 之后调用 | ZeRO-2、TP、PP |
| 优化器有 clip_grad_norm_ 就用它代替 torch 的 | ZeRO-2、TP、PP（ZeRO-1 里直接调 torch 的） |
| 存 checkpoint 前，所有进程先一起取优化器状态或完整状态，再由 0 号进程写文件 | ZeRO-1、ZeRO-2、FSDP2、TP、PP |
| 梯度范数是 DTensor 时，记录前先 full_tensor() | FSDP2 |

## DDP（zero_stage = 0）

### 做法

- 多进程时用 DDP 包装模型，梯度在最后一份反向时边算边 all-reduce
- 存包装前的模型，键不带 "module." 前缀

### 正确性标准

python 单进程与 torchrun 两进程（gloo，CPU）各跑 10 步，final.pt 参数 allclose（1e-5），val_loss 相等；两进程中途续训与一口气跑完逐位相同。

留 1e-5 的容差，是因为 torchrun 会给每个进程设 OMP_NUM_THREADS=1，而 python 直接跑时是多线程。CPU 上矩阵乘的线程数不同，加法顺序就不同，末位会有差别。要逐位比较，两边必须用同样的方式启动。

### 踩过的坑

- evaluate 初版除错了批数，也没接 all_reduce_mean 的返回值。单进程测不出来，多进程时才不对
- torch 2.14 弃用了 broadcast_buffers，改用 forward_sync_buffers=False

## ZeRO-1（zero_stage = 1）

### 做法

- 每个参数整个归一张卡负责，不像常见做法那样拍平后按元素切，因为 Muon 的正交化要完整矩阵。这样 Muon 的正交化也随之分摊到了各卡
- 分配（partition_params）：按元素数从大到小，每次放到当前最轻的卡上。只依赖参数大小和卡数，各进程算出的分配相同。1.12 亿参数的模型 2 卡、4 卡时各卡元素数几乎相等；8 卡时 embedding（2516 万）一个参数就超过平均值，0 号卡偏重
- parallel/zero.py 的 ZeroOptimizer 包住原来的优化器：只用本卡负责的参数建里面的 AdamW / MuonWithAdamW，两者一行不改；step 之后，每个参数由负责它的卡广播给其他卡，每张卡都按同样顺序对每个参数调用
- 梯度同步仍由 DDP 做
- checkpoint：各卡状态搬到 CPU 后用 gather_object 收集到 0 号进程，存成每卡一项的列表，并记下卡数；续训时每张卡取自己那一项，卡数不同时报错

### 正确性标准

两进程 ZeRO-1 与两进程 DDP 各跑 10 步，final.pt 参数逐位相同、val_loss 相等，AdamW 和 Muon 都是；两进程中途续训与一口气跑完逐位相同；各卡优化器状态的字节数相加正好等于 DDP 的总量（小模型 AdamW 1,512,032 字节、Muon 823,808 字节），每张卡都少于总量的 0.75。

能做到逐位相同，是因为每个参数的更新虽然只在负责它的卡上算，但梯度、公式和学习率都和 DDP 相同。两边都用 torchrun 启动，所以不留容差。

### 踩过的坑

- zero_grad 要清全部参数的梯度，不能只清本卡负责的：DDP 每步会把平均后的梯度写进全部参数，没清的会累加，还会经 all-reduce 带坏别的卡
- param_groups 要直接交出里面那个优化器的（写成 @property），Trainer 的 set_lr 才能生效；返回拷贝时 lr 会一直停在 max_lr
- state_dict 里有集合通信，所有进程都要调，不能只在 0 号进程调

## ZeRO-2（zero_stage = 2）

### 做法

- 在 ZeRO-1 的基础上把梯度也切开，不再包 DDP，梯度累积也就不需要 no_sync
- reduce_grads：反向全部算完后，对每个参数做一次 reduce，把各卡梯度加到负责它的卡上，由它除以卡数；其他卡把这份梯度设成 None
- clip_grad_norm_：跨卡算总范数，算法与 torch 的 clip_grad_norm_ 一致。先用 _foreach_norm 算本卡负责的参数的范数，按传入参数的顺序填进全 0 向量，跨卡相加后再求一次范数，最后用 clip_grads_with_norm_ 缩放本卡的梯度。每个位置只有一张卡有值，加 0 不引入误差
- 开头由 0 号卡广播一次参数；续训时要在读完 checkpoint 之后

### 正确性标准

和 ZeRO-1 一样，两进程与 DDP 逐位相同、续训逐位相同，不留容差。另外用 tests/zero2_worker.py 检查：汇总后负责的梯度等于 DDP 式的平均，不负责的梯度为 None。

- 两卡时逐位相同：ZeRO-2 先把各卡梯度加起来再除以 2，DDP 先各乘 1/2 再相加，两卡时两者结果相同；四卡时加法顺序不同，末位可能不同
- 不留容差很重要：临时让总范数只用本卡的梯度算时，最大差只有约 1.4e-4。AdamW 的 m/√v 和 Muon 的正交化都对梯度的整体缩放不敏感，所以错误被压得很小，1e-3 的容差就会漏掉

### 踩过的坑

- 忘了去掉 DDP 时，结果照样与 DDP 逐位相同，测试查不出来。所以不包 DDP 时打一行日志确认，测试检查这行日志

### 决定不做的事

看过 4×A100 上的实测（ZeRO-2 比 ZeRO-1 慢 6 ms）之后决定：

- 不把 134 次逐参数的小通信按卡合并：最多省几毫秒，不到每步的 3%
- 不给 ZeRO-2 加 hook 让梯度边算边发、边释放：提速的上限就是这 6 ms；而且每张卡要累积 2 份梯度，第一份算完时全部梯度就已在显存里，边算边释放也降不了峰值

## FSDP2（zero_stage = 3）

### 做法

- 用 PyTorch 自带的 fully_shard：每个参数按第 0 维切成卡数段，每张卡平时只存自己那一段；前向、反向算到某一层时临时 all-gather 凑齐，算完释放；梯度用 reduce-scatter 汇总并切开
- parallel/fsdp.py 的 shard_model 对每个 Block 调一次 fully_shard，最后对最外层 ModelForCausalLM 再调一次，要从里往外包
- fully_shard 会把参数换成 DTensor，所以要先分片、再建优化器；不包 DDP，并打一行日志确认
- fully_shard 不广播参数，靠各进程用同一个种子建出相同的模型
- checkpoint：所有进程一起调 get_full_state（get_model_state_dict / get_optimizer_state_dict，full_state_dict + cpu_offload），完整状态只收集到 0 号进程，由它写文件。优化器状态按参数名存，所以换卡数也能续训。续训时改用 load_full_state（set_model_state_dict / set_optimizer_state_dict），每张卡从完整状态里取自己那一段；它在分片、建优化器之后调用，所以 main 的顺序不用改

### 正确性标准

两进程 FSDP2 与两进程 DDP 各跑 10 步（AdamW），final.pt 参数 allclose（atol 1e-5），val_loss 近似相等；两进程中途续训与一口气跑完逐位相同；FSDP2 存的 final.pt 能被 load_model 读回；muon 配 zero_stage=3 时报错。

和 DDP 不再逐位相同，是因为在 DTensor 上求梯度裁剪的总范数时，各卡先算自己那一段再合起来，加法顺序不同（本地 10 步最大差约 1.8e-6）。1e-5 的容差仍能抓住 ZeRO-2 那种约 1.4e-4 的错误。

### 踩过的坑

- embedding 与 lm_head 共享权重，只能放在最外层这一组。单独包里面的 Model，会在算 lm_head 之前就把它释放回分片
- set_requires_gradient_sync 是开关，不是 with，设了就一直有效，所以每份之前都要设一次
- get_optimizer_state_dict 遇到还没走过一步的优化器，会先用全 0 梯度、lr=0 走一步来建出状态，使 AdamW 的步数多 1，之后的偏差修正都对不上。训练开始前那次存 checkpoint 正好撞上，所以优化器状态为空时直接存空字典
- 0 号进程直接存 model.state_dict() 时，训练照常跑完、不报错，但文件里存的是 DTensor 分片，只有测试能查出来

### 不支持 Muon

FSDP2 把每个矩阵按行切开，step 时每张卡只有几行动量，而 Muon 的正交化要完整矩阵。

## TP（tp_size = 2）

### 做法

- 手写 Megatron 式张量并行：每一层的矩阵乘切成两半，两张卡各存、各算一半，同一个 TP 组的卡读同一批数据
- 切法。nn.Linear 的权重是（输出, 输入），Megatron 说的"按列切"是按输出切，对应权重的第 0 维：

| 层 | 怎么切 | TP=2 时每卡 | 通信 |
| --- | --- | --- | --- |
| q_proj、k_proj、v_proj | 按输出切（ColumnParallelLinear） | 4 个 Q 头、1 个 KV 头 | 前向不用 |
| o_proj | 按输入切（RowParallelLinear） | 本卡那 4 个头对应的部分 | 输出后 all-reduce |
| gate_proj、up_proj | 按输出切 | 2304 里的 1152 | 前向不用 |
| down_proj | 按输入切 | 那 1152 对应的部分 | 输出后 all-reduce |
| embedding / lm_head、各个 RMSNorm、q_norm、k_norm | 不切，每卡一份完整的 | 全部 | q_norm、k_norm 的梯度要求和，见下 |

- 通信全在 parallel/tp.py 的两个 autograd Function 里：f（CopyToTP）前向原样传、反向把梯度在 TP 组里 all-reduce；g（ReduceFromTP）前向 all-reduce、反向原样传。f 放在按输出切的层的共用输入上（q/k/v 共用一次，gate/up 共用一次），g 放在 RowParallelLinear 的输出上，所以每层前向 2 次、反向 2 次 all-reduce。f 不放进 ColumnParallelLinear：q、k、v 各放一个时结果相同，但反向要做 3 次 all-reduce
- model.py 不知道 TP：Attention、MLP 入口留了 tp_copy（默认 nn.Identity，不切时计算和 state_dict 都不变）；Attention 按本卡 q_proj、k_proj 的权重形状推出头数。apply_tp 原地把每个 Block 的 7 个线性层换成并行版本、把 tp_copy 换成 f，参数名与单卡相同；按头数（不是维度）检查能否切分
- GQA 的头正好对得上：q_proj 按行连续切开，0 号卡拿到 Q 头 0～3 和 KV 头 0，而 repeat_interleave 本来就让 Q 头 0～3 共用 KV 头 0，所以每卡上每个 KV 头仍对应 4 个 Q 头
- 切开的参数打 tp_dim 标记（沿哪一维切），q_norm、k_norm 打 tp_sum_grad 标记，后面的函数都看标记，不认参数名
- 不切的参数分两种。embedding、lm_head、各层 RMSNorm 处理的是两卡相同的残差流，梯度经过 f 的反向已经加全；q_norm、k_norm 在 f 之后，每张卡都有完整的一份，但只用它处理本卡那几个头，梯度只有一部分。TPOptimizer 包住 AdamW：reduce_grads 把 q_norm、k_norm 的梯度在 TP 组里求和；clip_grad_norm_ 分两类算总范数，切开的参数各卡平方和相加，不切的只算一份
- 初始化：各进程用同一个种子建出完整模型，再各切各的，初始值和单卡完全一样
- checkpoint：gather_full_state_dict 按参数名把各卡的参数收集到 0 号进程，切开的沿 tp_dim 拼回，格式和单卡相同，load_model、export_hf、sft.py 能直接读。按名字收集，以后做 PP 时某张卡上没有某些层也照样成立。续训时 load_full_state_dict 按名字从完整权重切出本卡那一块，在切分、建优化器之后调用，main 的顺序不用改

### 正确性标准

两个 worker 在一步之内比较：切开的 MLP、整个小模型与完整版的输出和梯度，拼回、切回，TPOptimizer 的梯度求和、裁剪和一步更新，都在 1e-6 以内或逐位相同。torchrun 1 进程与 2 进程 TP 各跑 10 步，val_loss 相对误差 1e-5，参数 allclose（atol 1e-4），load_model 能读回；中途续训逐位相同；muon 配 tp_size=2、TP 配 ZeRO 时报错。

参数的容差放宽到 1e-4，是因为 TP 改变了前向的加法顺序，AdamW 又会把个别小梯度元素的末位差别放大（约 19 万个元素里只有 1 个超过 1e-5，最大 2.8e-5）。依据和对照组的选法见 [testing.md](testing.md#多进程测试)。

### 踩过的坑

- 去掉 q_norm、k_norm 的梯度求和时，形状全对、loss 照样下降，只有测试能查出来。连续训测试也会失败：两张卡上的 q_norm 各自更新、不再一致，而 checkpoint 只存了 0 号那份。TP 多了一条规矩：不切的参数在各卡上必须一直保持一致
- 去掉 f 时，单个 MLP 的输出和权重梯度都对，只有输入的梯度错
- 只按维度检查能否切分不够：k_proj 输出 256 维，tp_size=4 时能整除，但每卡只有半个头

### 测速

2×A100（NVLink）上每步 622 ms，只比单卡（688 ms）快 11%，比 DDP×2（352 ms）慢得多，和做之前的预计一致：lm_head、loss 和各层的小操作两张卡各算一遍，每份数据 48 次 all-reduce 又没和计算重叠。见 [results.md](results.md)。

### 这一轮的范围

- 不和 DP、ZeRO、FSDP2 组合：DP 组和 TP 组同时存在时，梯度要分别在各自的组里同步，留到 3D 并行那一轮
- 不做 Sequence Parallel：残差流上的激活每卡仍是完整的一份，要切它得用 Sequence Parallel，留作可选
- 不支持 Muon：正交化要完整矩阵，和 FSDP2 同一个原因

## PP（pp_size = 2）

### 做法

- 手写流水线并行：不切矩阵，切层。12 层分成两段，卡 0 放 embedding 和前 k 层，卡 1 放后 12−k 层、norm、lm_head 和 loss。一步里梯度累积的 8 份就是 8 个 micro-batch，依次流过两段。两段之间只在交界处传东西：前向传激活，反向传激活的梯度，形状 (B, T, 768)、fp32（autocast 下残差流仍是 fp32），每份每个方向约 25 MB
- 切分：每张卡用同一个种子建出完整模型，split_model 再把不属于本段的层、第一段的 norm 和 lm_head、第二段的 embed_tokens 换成没有参数的 nn.Identity。层名保持全局编号（第二段仍是 layers.6～11），所以 TP 那轮按名字拼回、切回 checkpoint 的函数可以直接用，load_full_state_dict 只多允许了一件事：PP 时 checkpoint 里有本段之外的键。在 .to(device) 之前切，每张卡只把自己那段搬上 GPU
- model.py 不知道 PP：两段的前向 forward_first、forward_last 写在 parallel/pp.py，照抄完整模型的前向，只遍历本段的层
- 两种调度：

| 调度 | 顺序 | 第一段同时存几份激活 |
| --- | --- | --- |
| GPipe | 8 份的前向全做完，再做反向 | 8 |
| 1F1B（默认） | 先灌满，再一前一后交替，最后排空 | 最多 2 |

  两种调度的反向都按第 0～7 份的顺序做，梯度累加顺序和单卡相同，所以结果逐位相同；气泡都是（段数−1）÷（份数＋段数−1）＝ 1/9
- 收发：send、recv 用默认组，对方的全局编号由 PP 组换算（dist.get_global_rank）。GPipe 任何时候只有一个方向在传，用阻塞的 send/recv 就够了；1F1B 交替时两段会同时给对方发东西，用 batch_isend_irecv 把"一发一收"打包提交。只把 send 换成 isend，在 NCCL 上仍可能互相等住
- 共享的 embedding / lm_head：两段各存一份（第二段上它叫 lm_head.weight），都打 pp_shared 标记。PPOptimizer 包住原来的优化器：reduce_grads 在 embedding 组里把两份梯度相加，让两份一直相同；clip_grad_norm_ 把各段的平方和在 PP 组里相加，共享参数只在第一段算一次。Muon 路线分组时，除了按名字认 embed_tokens，还要认 pp_shared，否则第二段的 lm_head.weight 会被分进 Muon 组
- loss：只有最后一段算得出 loss。broadcast_from_last 在 PP 组里把它广播出去，之后照常 all_reduce_mean：各卡的数值相同，平均后不变，到 3D 时正好在各份 DP 之间求平均。evaluate 用只做前向的 pp_forward，val_loss 同样广播后写进 checkpoint
- 数据：同一条流水线上的两段读同一批数据，第一段用输入、第二段用目标；DataLoader 照旧按 DP 编号读，这一轮 DP 只有 1 份

### 正确性标准

- 单进程里切出两段，用 detach().requires_grad_() 模拟交接，再用 h.backward(收到的梯度) 接着反向：两段前向的 loss、各参数梯度都与完整模型逐位相同，共享参数两段之和也逐位相同
- pp_worker 两进程 4 份数据：GPipe 与完整模型逐位相同（共享参数除外）；1F1B 与 GPipe 的各参数梯度（含共享参数）和 loss 逐位相同
- torchrun 1 进程与 PP 2 进程各跑 10 步：参数 allclose，AdamW 1e-5、Muon 1e-4，val_loss 用同样的相对容差，load_model 能读回；GPipe 与 1F1B 的完整训练逐位相同；中途续训逐位相同

和单卡只能 allclose，是因为共享参数梯度的加法分组变了：单卡时每份数据的 embedding 梯度和 lm_head 梯度在反向里先合在一起，是 (e0+l0)+(e1+l1)+…；PP 是两段各自累加完再相加，是 (e0+…+e7)+(l0+…+l7)。只有一份数据时只是 a+b 和 b+a 的区别，结果完全相同。只有这一处变了，所以 AdamW 不用像 TP 那样放宽到 1e-4。

### 踩过的坑

- 一处 recv 指定了 PP 组，对面却在默认组上发，测试卡死。进程组相当于不同的频道，发和收必须在同一个组上
- 第二段的共享参数叫 lm_head.weight，Muon 按名字分组会把它分进 Muon 组：第一段用 AdamW、第二段用 Muon 更新同一个参数，从第 1 步起两份就不一样了；而且要对 768×32768 的矩阵做正交化，CPU 小模型上每步从约 80 ms 变成约 350 ms
- 第一段如果留着 norm，它永远拿不到梯度、也不会更新，存 checkpoint 时又和第二段的同名参数撞名，最后存下的是谁全看收集顺序。所以不属于本段的部分必须换成占位
- 不广播 loss 时，0 号卡在第一段，日志里的 loss 是第一段的 0 和真实 loss 的平均，只剩一半
- 本地测试都在 CPU 上跑，autocast 不生效，测不到 GPU 上发过来的激活是不是 fp32、和第二段准备的缓冲区对不对得上；上云前先在单卡 GPU 上确认了 forward_first 在 autocast 下输出 fp32。同样，gloo 对小消息可能先缓冲，1F1B 就算写成阻塞收发，在 CPU 上也不一定卡住，这一点只能靠分析
- 切点：起初按参数个数估计 lm_head 约等于 3.5 层，实测只相当于约 1.8 层，8/4 反而比 6/6 慢，见 [results.md](results.md)

### 这一轮的范围

- 只做预训练，只切 2 段（pp_size 只能是 1 或 2）
- 不和 DP、TP、ZeRO、FSDP2 组合，留到 3D 并行
- 不做 interleaved 1F1B（每张卡放几段不相邻的层，用来缩小气泡）和 zero-bubble 一类的调度
- 默认切点仍是平分，没有改成 7/5：最快的切点和模型、卡型都有关，作为测速结论写在 results.md 里

## 3D 并行（DP×TP×PP）

### 做法

- 三种并行叠在一起，卡数 = dp × tp × pp。8 卡 DP2×TP2×PP2：8 张卡分成 2 份做数据并行，每份是一条 2 段的流水线，每段由 2 张卡做张量并行
- 进程组。编号规则不变（TP 变化最快、PP 最慢），每张卡恰好属于每种组里的一个，同一种组的成员另外两个编号都相同：

| 组 | 8 卡时的成员 | 组内相同的编号 | 在组里做什么 |
| --- | --- | --- | --- |
| TP 组 | [0,1] [2,3] [4,5] [6,7] | pp、dp | f / g 的 all-reduce；q_norm、k_norm 梯度求和；裁剪时平方和相加 |
| DP 组 | [0,2] [1,3] [4,6] [5,7] | pp、tp | 每步把所有梯度求平均 |
| PP 组 | [0,4] [1,5] [2,6] [3,7] | dp、tp | 两段之间收发激活和梯度；广播 loss；裁剪时平方和相加 |
| embedding 组 | 2 段时和 PP 组相同 | — | 共享的 embedding / lm_head 梯度求和 |

- DP 不用 DDP。只要 tp>1 或 pp>1 就不包 DDP，反向全部结束后由优化器对每个参数在 DP 组里 all-reduce 再除以 dp_size；纯 DP 时仍用 DDP。不用 DDP 有两个原因：PP 的调度不走 model.forward，DDP 在 forward 里为梯度同步做的准备不会发生；DDP 默认在全局组里平均，TP 两张卡上切开的参数形状一样、内容不同，会被当成同一个参数的副本平均到一起，不报错，结果却是错的（改坏验证见 [testing.md](testing.md#临时改坏记录)）
- 合并的优化器。TPOptimizer 和 PPOptimizer 合成 parallel/optimizer.py 的 ParallelOptimizer，tp>1 或 pp>1 时由 configure_optimizer 返回，按参数上的标记办事。上面 TP、PP 两节提到那两个类的地方，现在都由它完成：
  - reduce_grads：带 tp_sum_grad 的在 TP 组求和，带 pp_shared 的在 embedding 组求和，所有参数在 DP 组求平均。三件事都是线性的、作用在不同的组上，先后顺序不影响结果
  - clip_grad_norm_：每张卡只加该由自己算的那部分平方和（切开的参数每卡都加自己那一块，不切的只在 tp_rank=0 的卡上加，共享参数只在第一段加），再按需在 TP 组、PP 组里相加。不在 DP 组里加：DP 平均之后各副本的梯度完全相同
  - 优化器状态仍是每卡一项，收集到 0 号；checkpoint 里记下 dp、tp、pp 三个大小，续训时逐项核对
- 不用改就能组合的部分：
  - DataLoader 按 DP 编号读，同一条流水线上的 tp×pp 张卡读同一批数据，累积次数和 tok/s 按 dp_size 算
  - loss 先在 PP 组里从最后一段广播，再做全局平均；同一条流水线上的值都相同，所以全局平均等于各份 DP 的平均
  - checkpoint 按参数名拼回，各 DP 副本发来的分片按 tp_rank 存进字典，后到的覆盖先到的，值本来就相同
  - 激活收发的对象由 PP 组换算出全局编号，所以 TP 的 0 号卡只和下一段 TP 的 0 号卡交换
- 只改了一处模型切分：apply_tp 跳过 PP 切分留下的 nn.Identity 占位层
- 初始化：所有进程用同一个种子建完整模型，再切段、切 TP，同一 DP 组里的副本一开始就逐位相同，所以不像 ZeRO-2 那样开头广播参数

### 正确性标准

- 布局：8 卡时各组成员与上表一致。对 8 卡 2×2×2、4 卡 TP2、4 卡 PP2，每种组都覆盖每张卡恰好一次，任意两种组的交集只有自己。只查三种组共同的交集太弱：DP 组写成和 PP 组一样时，三者的交集仍然只有自己
- torchrun 1 进程对照 DP2×TP2、DP2×PP2、TP2×PP2、DP2×TP2×PP2 各跑 10 步：参数 allclose（atol 1e-4），val_loss 相对误差 1e-5，并检查 checkpoint 里 tp、pp、dp 的大小确实生效。容差沿用 TP 那一轮
- DP2×TP2×PP2 中途续训与一口气跑完逐位相同
- 原有 TP、PP 测试只改了类名和检查优化器状态格式的那条断言，全部通过

### 踩过的坑

- get_tp_group 漏写了 return，一直返回 None，TP 的通信全落到了全局组上。TP 单独用时卡数等于 tp_size，全局组碰巧就是 TP 组，原有测试全部通过；DP2×TP2 时，训练前的 val_loss 就差了约 8e-4（相对），因为两份不同数据算出的部分和被加到了一起，刚初始化时注意力和 MLP 的输出很小，所以只偏一点；TP2×PP2 时，第一段在全局组上等着做 all-reduce，第二段在等收激活，一直卡到超时。布局测试只测纯函数，抓不到；是等价测试抓到的，在 evaluate 里打印各卡运行时的组成员（dist.get_process_group_ranks）才定位到。教训：只有组和全局组不同的配置，才测得出"用错了组"
- 多进程测试超时后，torchrun 起的 worker 可能留在后台占着 CPU，之后的测试越跑越慢，看起来像是整体变慢了，要先清掉
- `pytest -k dp2_tp2` 按子串匹配，会把 dp2_tp2_pp2 也选上；只跑一个参数化用例要写完整编号

### 决定不做的事

- 不和 ZeRO、FSDP2 组合；有 TP 时仍只支持 AdamW；PP 仍只切 2 段
- DP 同步不和反向重叠，也不把小梯度合并成桶，这一轮先求对
- 不去掉 TP 两张卡之间重复的激活收发：同一段的两张 TP 卡发出的激活完全相同，Megatron 会先拆开传、到对面再拼起来，这里每张卡传整份，留作可选
- DP×PP 可以用 Muon，但等价测试只测了 AdamW

### 测速

待补，暂时租不到单机 8 卡。计划见 [results.md](results.md#3d-并行单机-8-卡adamw-路线v010测速待补)：dp=2 时每条流水线每步只分到 4 份，气泡从 1/9 涨到 1/5；把 B 改成 8 能分到 8 份，但每份的矩阵乘变小，哪种更快要实测。

## 待做

- FSDP2 支持 Muon：step 时先把每个矩阵的动量凑齐，再正交化
- 查 AdamW 路线下 ZeRO-1 的 3 号卡多出的约 440 MiB（用显存快照）
- 补测 3D 的 8 卡速度（用 v0.10 标签的代码）
- 可选：FSDP2 的 MixedPrecisionPolicy，参数以 bf16 收集，这一步的通信量减半，但数值和 autocast 不完全一样，作为单独的实验

## 性能优化

单卡性能优化（v0.11）见 [perf.md](perf.md)。torch.compile 目前只支持单卡，和这里各种并行的组合留到以后。

