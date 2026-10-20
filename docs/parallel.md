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

- 卡数要同时整除 8（65,536 ÷ B×T）和 eval_steps（默认 20），所以默认配置下可以用 1、2、4 张；用 8 张卡时要把 eval_steps 改成 8 的倍数，比如 `--set data.eval_steps=24`
- 加 `--resume` 照样能续训。四种做法存的模型部分都和单卡格式相同，load_model、export_hf、sft.py 能直接读；但 zero_stage 0、1～2、3 这三类存的 checkpoint 不能互相续训，zero_stage=3 存的只能用 zero_stage=3 多卡续训
- 本地没有多卡时，可以用 CPU 验证多进程的逻辑：`CUDA_VISIBLE_DEVICES= torchrun --standalone --nproc_per_node=2 pretrain.py --config 小配置`，tests/ 里的多进程测试都是这么做的
- 复现 results.md 里的测速：加 `--set train.max_steps=300`，并用 `--set run.ckpt_dir=…`、`--set run.log=…` 指到临时目录；单卡那次用 python 启动，命令前加 `CUDA_VISIBLE_DEVICES=0`。前 300 步只用到第一片训练数据，云上只需传 train_000000.bin、val_000000.bin 和 meta.json

## 公共部分

### 进程和数据

- 启动：parallel/dist_utils.py 的 setup 读 torchrun 给的环境变量，先 set_device 再初始化进程组，GPU 用 nccl、CPU 用 gloo；没有这些变量时什么都不做，所以直接用 python 启动时现有测试和基线不变
- 全局 batch 保持 65,536：每个进程的累积次数 = 全局 batch ÷（B×T×进程数），4 卡时每卡累积 2 次
- 数据按进程跳着读：DataLoader 的 pos 是所有进程共同的位置，读数据时再加上 rank×B×T 的偏移，每批前进 进程数×B×T，所有进程在同一步换片。state_dict 不用改，续训时各进程自动回到自己的位置。这样每一步各进程合起来读的，正是单进程会读的那些 token，平均后的梯度和单进程相同，只差浮点加法顺序
- 日志里的 loss 和 val_loss 先跨进程平均（all_reduce_mean 先求和再除以卡数，因为 gloo 不支持 AVG）；evaluate 每个进程测 eval_steps ÷ 卡数 批再平均，所以卡数要整除 eval_steps
- 只有 0 号进程写日志、存 checkpoint，存的是包装前的模型；训练结束时把各卡的峰值显存收集到 0 号进程打印

### Trainer 里的接入点

Trainer 不认识具体做法，只看模型和优化器有没有某个方法：

| 接入点 | 谁用到 |
| --- | --- |
| 梯度累积时，最后一份之前用 no_sync 包住前向和反向 | DDP、ZeRO-1 |
| 模型有 set_requires_gradient_sync 时，每份之前设一次，只在最后一份打开 | FSDP2 |
| 优化器有 reduce_grads 就在逐份 backward 之后调用 | ZeRO-2 |
| 优化器有 clip_grad_norm_ 就用它代替 torch 的 | ZeRO-2（ZeRO-1 里直接调 torch 的） |
| 存 checkpoint 前，所有进程先一起取优化器状态或完整状态，再由 0 号进程写文件 | ZeRO-1、ZeRO-2、FSDP2 |
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

## 待做

- FSDP2 支持 Muon：step 时先把每个矩阵的动量凑齐，再正交化
- 查 AdamW 路线下 ZeRO-1 的 3 号卡多出的约 440 MiB（用显存快照）
- 可选：FSDP2 的 MixedPrecisionPolicy，参数以 bf16 收集，这一步的通信量减半，但数值和 autocast 不完全一样，作为单独的实验

## 下一步：TP，之后 PP

- TP（张量并行）：q、k、v 和 gate、up 按列切，o 和 down 按行切，按行切之后做一次 all-reduce。模型只有 2 个 KV 头，按头切时 TP 最多切 2 份，再多就要复制 KV 头
- TP、PP 做完后，按上面同样的小标题（做法、正确性标准、踩过的坑、决定不做的事）补在这里