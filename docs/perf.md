# 单卡性能优化

只用于预训练，在本地单张 RTX 4070 Ti Super 16GB 上做。完整结果表见 [results.md](results.md#单卡性能优化v011)，各测试能抓到哪些错误见 [testing.md](testing.md)。

## 用法

所有优化都在配置的 `[perf]` 节里，默认全部关闭，所以 configs/ 里原有 8 个配置和 CPU 上的基线测试都不受影响。目前最快的组合写在 `configs/adamw_112m/pretrain_fast.toml`（compile + GQA + B=32）：

```bash
python pretrain.py --config configs/adamw_112m/pretrain_fast.toml
```

| 键 | 默认 | 作用 |
| --- | --- | --- |
| `peak_tflops` | 85.7 | 算 MFU 用的峰值。本机 cuBLAS bf16 实测值，换卡要改 |
| `profile` | false | 跳过前 5 步、预热 1 步、抓 2 步，导出 trace 并打印按类别汇总的 GPU 时间 |
| `compile` | false | `m.compile()` 原地编译；只支持单卡，多进程时报错 |
| `enable_gqa` | false | SDPA 传 `enable_gqa=True`，不再用 repeat_interleave 复制 K/V |
| `ce_chunk_size` | 0 | 每块多少行算 lm_head + CE，0 表示不分块 |
| `act_ckpt` | false | 逐个 Block 做 activation checkpointing |
| `attn_impl` | "sdpa" | 注意力实现：`sdpa` 或 `triton`（自写 FlashAttention） |

除 `peak_tflops`、`profile` 外，开关都是模块上的属性（Attention.enable_gqa、Model.act_ckpt 等），由 pretrain.py 在 compile 之前设置。属性要先设好再编译，否则会按旧值编译一次、改了以后再重编译。

## 量基线

### MFU

每 token 的计算量由 ModelConfig 算出（model.py 的 estimate_flops_per_token）：

- 矩阵乘：6N。每个参数每个 token 前向 2 次浮点运算（乘、加），反向算输入梯度和权重梯度各一份，合计 6。N 只算参与矩阵乘的权重：各层 q/k/v/o 和 gate/up/down，加上 lm_head（与 embedding 共享，但查表不算计算量，矩阵乘只在 lm_head 里发生），不算 RMSNorm 的权重
- 注意力：12·L·H·dh·T。QKᵀ 和乘 V 每 token 每头前向各 2·T·dh，乘 3（前向加反向）。H 用 query 头数，GQA 只是共用 K/V，点积仍按 query 头做；按惯例不扣 causal mask 省下的一半
- 1.12 亿参数、T=512 时：6 × 112,459,776 + 75,497,472 = 750,256,128，约 7.5 亿

MFU = 每 token 计算量 × tok/s ÷（peak_tflops × 卡数）。tokens_per_step 是所有卡加起来的，峰值是一张卡的，所以要乘卡数。只有 PretrainTask 提供 flops_per_token，SFT、GRPO 的日志不打 MFU。

测试：从实际建出来的模型里数所有 nn.Linear 的权重个数（lm_head 是 nn.Linear，embedding 不是，正好是 N），和公式的结果相等。

### profile

profiling.py 的 make_trace_handler 交给 torch.profiler 的 on_trace_ready：导出 chrome trace，再把 `key_averages()` 里 device_type 为 CUDA 的项按名字分类累加，打印每类每步多少毫秒、占比，以及前 15 个 kernel。

| 类别 | 名字里含（小写，按顺序匹配） |
| --- | --- |
| 注意力 | flash、fmha、attention |
| 矩阵乘 | gemm、nvjet、cutlass、xmma |
| CE | softmax、nll_loss、cross_entropy |
| 优化器（含梯度裁剪） | adam、multi_tensor |
| 其他（归一化 / 逐元素 / 拷贝） | 其余 |

- 注意力要放在矩阵乘前面：SDPA 退回 memory-efficient 后端时，kernel 名字（fmha_cutlassF…）同时含 fmha 和 cutlass
- 单独出现的 softmax 只可能是 CE 的 log_softmax，注意力里的 softmax 已经融合在 flash kernel 里
- 汇总时要跳过名字以 ProfilerStep 开头的项：它是 profiler 在 GPU 时间轴上画的步标记框，device_type 也是 CUDA，不跳过的话整步时间会被算两遍
- 判断 GPU 有没有在空等：每步 kernel 合计时间和普通步的 dt 比，接近就说明 CPU 提交跟得上

### 基线（AdamW，B=16，eager）

每步 1,334.9 ms，49,094 tok/s，MFU 43.0%，峰值 allocated 9,959 MiB。连跑两次的平均 dt 是 1,335.0 和 1,334.8 ms，噪声约 0.02%。

| 类别 | 每步 | 占比 |
| --- | --- | --- |
| 其他（归一化 / 逐元素 / 拷贝） | 712 ms | 53% |
| 矩阵乘 | 512 ms | 38% |
| 注意力 | 61 ms | 5% |
| CE | 36 ms | 3% |
| 优化器 | 13 ms | 1% |

- GPU 忙碌约 99%（kernel 合计 1,334 ms，dt 1,342 ms），所以不做 pin_memory / non_blocking
- 矩阵乘已经到顶：每步 6N × 65,536 ≈ 4.42e13 次，按 85.7 TFLOPS 要 516 ms，实测 512 ms。MFU 低的原因不在矩阵乘，而在比矩阵乘还多的逐元素小 kernel：RMSNorm（每层 4 个，含 q_norm、k_norm）、RoPE、SiLU 乘 up、残差相加、类型转换、拷贝，eager 下每个都要把整块激活读写一遍显存

## torch.compile

### 做法

- 用 `m.compile()` 原地编译，不用 `m = torch.compile(m)`。后者在外面包一层 OptimizedModule，state_dict 的键会多出 `_orig_mod.` 前缀，存下的 checkpoint 和 load_model、export_hf、SFT 都对不上，trainer 里的各种模型判断也是对外层做的；原地编译时模型还是原来的对象，只换掉了 forward
- 编译的是 ModelForCausalLM 的 forward，CE 也在里面，一起融合
- 只支持单卡，world_size > 1 时报错；和 DDP、TP、PP 的组合留到以后

### 正确性标准

compile 会改变数值，不能写成 CPU 上的对拍测试。在 GPU 上和 eager 各跑 300 步（同配置、同数据、同种子）：

- 断图 0 次；重编译 1 次，原因是 `GLOBAL_STATE changed: grad_mode`：fit 开头先在 no_grad 下 eval，编译一份，进入训练后再编译一份，之后 eval 和训练来回切换都命中缓存
- train loss 每步平均差 0.0005、最大差 0.0053；第 300 步 val loss 5.9384（eager）对 5.9391（compile）

### 实测

每步 1,335 → 796 ms（1.68 倍），MFU 43.0% → 72.1%，峰值显存 9,959 → 6,728 MiB。

compile 后的 profile：矩阵乘 511 ms（64%）、其他 184 ms（23%）、注意力 60 ms（8%）、CE 28 ms（4%）、优化器 13 ms（2%），GPU 仍然几乎不空闲。"其他"从 712 ms 降到 184 ms，原来几千个小 kernel 变成了 triton_poi_fused_… 这类融合 kernel（比如一整个 RMSNorm、SiLU 乘 up 连同它的反向）。

### 踩过的坑

- 测速要去掉前 10 步：第一次运行第 0 步要编译约 21 秒；Inductor 把编译结果缓存在磁盘上，第二次运行只要约 1.8 秒

## GQA 少一次拷贝（enable_gqa）

### 做法

- SDPA 传 `enable_gqa=True`，kernel 内部让第 i 个 query 头读第 i // 4 个 KV 头，不再先把 K/V 复制成 8 个头
- 前置检查：在 `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` 里跑一次带 enable_gqa 的前向加反向，确认 flash 后端支持。不支持时 SDPA 不报错，而是悄悄退回 math 实现，那样反而慢得多

### 正确性标准

开关前后整模型的 logits 和全部梯度 allclose（CPU、fp32，atol 1e-6）。把 repeat_interleave 换成 repeat（整组交替复制，query 头和 KV 头配错组）时测试失败。

### 实测

compile 下每步 796 → 791 ms，峰值显存 6,728 → 6,445 MiB。收益小，因为 compile 之后这次拷贝本来就只占"其他"里的一小部分。

## 加大每份的 B

全局 batch 保持 65,536，B 加大时累积次数相应减少，训练在数学上不变。

| compile + GQA | 每步 | MFU | 峰值显存 |
| --- | --- | --- | --- |
| B=16 | 791 ms | 72.5% | 6,445 MiB |
| B=32 | 774 ms | 74.1% | 10,884 MiB |
| B=64 | OOM | | |

显存账：B 从 16 加到 32 多了 4,439 MiB，这就是每 16 条序列的激活；剩下约 2,000 MiB 是参数、梯度和 Adam 状态（1.12 亿 × 16 字节 ≈ 1.8 GB）。B=64 约要 2,000 + 4 × 4,439 ≈ 19,800 MiB，放不下。

B=32 只快 2%：矩阵乘已经接近峰值，加大矩阵帮不上多少。省下的应该主要是梯度累加（推测：compile 后的 profile 里有一个每步约 945 次、共约 16 ms 的逐元素 kernel，次数和 134 个参数张量 × 7 次累加接近，B=32 时累加只剩 3 次）。

## 分块算 lm_head 和 CE（ce_chunk_size）

### 做法

- 把 (B·T, D) 的隐状态按行切块，每块算 lm_head 和 CE（reduction="sum"），在 float32 里累加，最后除以有效 token 数（targets != -100 的个数）。用 sum 而不是每块取 mean，是因为最后一块可能更小
- 每块用 `checkpoint(..., use_reentrant=False)` 包住。直接写 for 循环省不了显存：autograd 会把每块的 logits 都留给反向，循环结束时所有块的 logits 都还在。包上 checkpoint 后前向只存每块的输入，反向时重算这一块的 logits，用完就释放
- 这条路不返回 logits（第一个返回值是 None）；pretrain 和 evaluate 本来就不用 logits，SFT、GRPO、generate 不打开这个开关

### 正确性标准

块大小取 10、N=32（最后一块只有 2 行），loss 和全部梯度与 F.cross_entropy allclose（CPU、fp32，atol 1e-6）。每块改用 mean 时测试失败。

### 实测（每块 4096 行）

B=32 时每步 774 → 808 ms，峰值显存 10,884 → 9,863 MiB。省下的约 1 GB 正好是一整块 bf16 logits（16,384 × 32,768 × 2 字节）；多出的约 34 ms 是 lm_head 前向多做的那一遍（每步约 37 ms）。B=64 仍然 OOM，报错时申请的 256 MiB 正好是一块 logits。

### 踩过的坑

- 初版在分支之前先算了一次完整的 logits，分块的那条路也会算，只是没用上。eager 下白算一遍、多占一块显存；compile 下 Inductor 把它当作无用计算删掉了，所以改前改后峰值显存相同，单看测速发现不了

## activation checkpointing（act_ckpt）

### 做法

Model 的层循环在 `act_ckpt` 为真且需要梯度时（eval 在 no_grad 下，用不着），每个 Block 用 `checkpoint(block, x, cos, sin, use_reentrant=False)` 包住：前向只存每层的输入，反向时重算这一层。

### 正确性标准

开关前后 loss 和全部梯度逐位相同（CPU、fp32，torch.equal），重算走的是完全相同的运算。传进 checkpoint 的 x 改成 x.detach() 时测试失败：第 0～10 层收不到梯度，`.grad` 是 None。

### 实测（再加分块 CE）

| | 每步 | MFU | 峰值显存 |
| --- | --- | --- | --- |
| B=32 | 962 ms | 59.6% | 4,492 MiB |
| B=64 | 947 ms | 60.6% | 6,670 MiB |

B=32 时比不开慢 154 ms（19%），比粗估的 25%～30% 少一些；B=64 比 B=32 只快 15 ms，补不回重算的代价。

### 踩过的坑

- 测试初版写的是 `model.act_ckpt = True`，属性加在了外层的 ModelForCausalLM 上，里面的 Model 没变，两次跑的都是普通路径，怎么改坏都能通过。临时改坏时才发现，改成 `model.model.act_ckpt = True`

## 自写 Triton FlashAttention（attn_impl = "triton"）

### 做法

- kernels/flash_attn.py：Triton lesson 5、6 写的前向和双 kernel 反向（dkdv 一个、dq 一个）搬进来，只改外围：
  - 要求输入在内存里连续，形状 (B, H, T, D) 和 (B, H_kv, T, D)，于是 q 可以当作 (B·H, T, D)、k/v 当作 (B·H_kv, T, D)，所有张量共用一组 stride。Attention 调用前对 q、k、v 做 contiguous；compile 下 q、k 的这次拷贝会和 RoPE 融合
  - GQA：第 z 个 query 头读第 z // G 个 KV 头（z = b·H + h，H = H_kv·G，所以 z // G = b·H_kv + h // G）；dkdv 的 grid 按 KV 头开，在组内对 G 个 query 头循环累加，不用 atomic
  - 因果固定打开；块配置沿用 head_dim=128 时调出来的：前向 128/64/w8/s2，dkdv 64/64/w8/s2，dq 128/64/w8/s2
- 用 torch.library 注册两个 custom op：`tinytrain::flash_attn_fwd`（q, k, v → o, M）和 `tinytrain::flash_attn_bwd`（q, k, v, o, do, M → dq, dk, dv），各带 register_fake；register_autograd 在前向保存 q、k、v、o、M，反向调用 flash_attn_bwd。反向也要单独注册成 op，因为 compile 训练时也要把反向追踪成图
- 对外的接口是 `flash_attn(q, k, v)`，只返回 o

### 正确性标准

- kernel 与 fp32 SDPA（is_causal、enable_gqa）对拍 o、dq、dk、dv（rtol / atol 3e-2）：G=4、T=512；G=4、T=500（不整除块大小）；G=1。dkdv 的组内循环只跑 1 次时，G=4 的两组失败、G=1 通过
- torch.library.opcheck 前向、反向两个 op。fake 里 M 的 dtype 写成 bf16 时失败
- 经过 autograd 的结果与直接调用 launcher 逐位相同（反向不用 atomic，结果是确定的）
- `torch.compile(..., fullgraph=True)` 下前向加反向能跑，梯度与不编译时 allclose（1e-3）。fullgraph 下只要断图就直接报错
- 整模型（tiny_config，bf16 autocast）triton 与 sdpa 的 logits allclose（1e-2）。把 k、v 传反时 96% 的元素对不上
- GPU 测试整个文件在没有 CUDA 时跳过

### 实测（compile + GQA，B=32）

| attn_impl | 每步 | MFU | 峰值显存 |
| --- | --- | --- | --- |
| sdpa | 774 ms | 74.1% | 10,884 MiB |
| triton | 793 ms | 72.3% | 11,266 MiB |

- 断图 0；100 步 train loss 最大差 0.0001，第 100 步 val loss 7.2483 对 7.2482
- 慢 19 ms（2.4%）：注意力一共约 60 ms，自写 kernel 加上 contiguous 拷贝大约比 SDPA 慢三成
- 多出的 382 MiB 约等于每层一份 q 大小的张量（32 × 8 × 512 × 128 × 2 字节 = 32 MiB，乘 12 层是 384 MiB），推测是 contiguous 之后的那份拷贝被留给了反向，没有拆开验证

### 踩过的坑

- `_backward` 初版返回的是 kernel 用的 3D 形状，前向却 view 回了 4D；反向返回的梯度形状必须和输入一致
- 整模型测试初版比较的是 loss，交换 k、v 照样通过：刚初始化时 logits 都接近 0，loss 约等于 ln(词表大小)，注意力算错也几乎不动。改成比较 logits

### 决定不做的事

- 不重写 kernel（每个张量各带 4 维 stride、exp2）：compile 下 contiguous 的拷贝近乎免费，注意力只占每步约 8%，重写在结果表上看不出区别
- 不做单 kernel + atomic 的 dQ：反向从 7 次矩阵乘降到 5 次，但 atomic 有排队开销，而且 dQ 不再逐位确定；反向约 40 ms，就算快三成每步也只省约 1.5%
- attn_impl 默认仍是 sdpa，pretrain_fast.toml 不用 triton。这一项的目的是让注意力可以在配置里切换，并证明 custom op 和 compile 能一起用

## 结论

| 配置（AdamW，单卡） | 每步 | MFU | 峰值显存 |
| --- | --- | --- | --- |
| eager，B=16 | 1,335 ms | 43.0% | 9,959 MiB |
| compile + GQA，B=32（pretrain_fast.toml） | 774 ms | 74.1% | 10,884 MiB |

快 1.72 倍。默认打开的只有 compile、GQA 和 B=32（写在 pretrain_fast.toml 里，PerfConfig 的默认值不变）。分块 CE 和 activation checkpointing 在 1.12 亿参数上都不提速，默认关闭，留给放大模型：开了 checkpointing 之后 B=32 只要 4.5 GB，以后换成 4.7 亿参数、T=2048 时，参数、梯度和 Adam 状态本身就要约 7.5 GB，不开这两项多半跑不起来。

## 待做

- 租一次 A100，在同一台机器上测优化前后的单卡速度和 MFU（峰值要换成 A100 上实测的），租到 8 卡时顺便补测 3D
- compile 和多卡（DDP、TP、PP）的组合；Muon 和 compile 一起用还没测过
- 可选：用显存快照确认 triton 路线多出的 382 MiB 是什么