# 测试

## 运行

- `pytest -v`：89 个快测试，约 170 秒，不需要数据；其中 test_flash_attn.py 的 7 个需要 GPU，没有 CUDA 时自动跳过
- `pytest -m slow -v`：6 个慢测试。2 个需要原始数据；4 个是 3D 的两两组合和续训，不需要数据，约 30 秒
- `pytest -m "" -v`：全部跑
- pytest.ini 设了测试目录和 pythonpath，默认跳过标了 slow 的测试
- 慢主要是多次起 torchrun。试过用 pytest-xdist 并行跑，本机反而慢了约 4 倍（单进程测试默认用满所有核，几个进程的线程互相抢），所以全量仍串行跑

## 设计

### 训练循环的基线

- 训练循环的测试用假数据和小模型在 CPU 上以 fp32 运行，同一台机器上结果逐位可复现；不覆盖 bf16 autocast 和训练速度
- 基线记录的是当前代码每步的 loss 和梯度范数（GRPO 另外记录 KL 和平均奖励），写死在测试里，相对误差 1e-5
- 基线只能说明行为没变，不能说明行为正确，所以另有 overfit 测试检查能否学会
- 只有有意改变训练行为时（如改 lr 调度）才重录基线，并在提交说明里写明
- GRPO 基线要求采样时有对有错：如果全对或全错，组内优势全为 0，策略梯度不起作用，基线就测不到它。所以先用 SFT 200 步把小模型训到有对有错，再跑 4 步 GRPO，并断言出现过有对有错的题
- 基线只保证在录它的机器上逐位一致：换一台机器，Muon 和 GRPO 的基线可能在末几位对不上（在 Runpod 上实测有 4 个基线测试因此失败，其余全部通过）

### 与 HF 对拍

自己和自己比的测试查不出约定错误，比如 GQA 里 KV 头怎么对应到 Q 头、QK-norm 放在 RoPE 前还是后：写错了形状不变、照样能训练，基线也会照着错的结果录下来。所以导出后用 transformers 读回，比较 logits。

小模型测试要先把 norm 权重随机化：权重全为 1 时，norm 和 RoPE 可以交换顺序，QK-norm 放错位置测不出来。

### 多进程测试

都在 CPU 上跑：把 CUDA_VISIBLE_DEVICES 置空，用 torchrun 起 2～8 个进程走 gloo，和在多卡机器上的启动方式相同。

| 测试 | 比较的两边 | 容差 | 原因 |
| --- | --- | --- | --- |
| DDP 等价 | python 单进程 vs torchrun 两进程 | 1e-5 | torchrun 给每个进程设 OMP_NUM_THREADS=1，CPU 上矩阵乘的线程数不同，末位不同 |
| ZeRO-1 / ZeRO-2 等价 | torchrun 两进程 ZeRO vs torchrun 两进程 DDP | 逐位相同 | 两边启动方式相同，梯度、公式和学习率也相同 |
| FSDP2 等价 | torchrun 两进程 FSDP2 vs torchrun 两进程 DDP | 1e-5 | DTensor 上求梯度裁剪总范数时加法顺序不同 |
| TP 层级、整模型对照 | 两进程切开的 MLP / 小模型 vs 同一进程里的完整版，一步之内 | 1e-6（拼回、切回逐位相同） | all-reduce 改变加法顺序，还没被 AdamW 放大 |
| TP 等价 | torchrun 1 进程（tp_size=1）vs torchrun 2 进程（tp_size=2），10 步 | val_loss 相对 1e-5，参数 1e-4 | 见下 |
| PP 调度对照 | pp_worker 两进程的 GPipe、1F1B vs 同一进程里的完整模型，4 份数据，一步之内 | 逐位相同（GPipe 对完整模型时不比共享参数） | 每段的计算、梯度累加顺序都和完整模型相同；共享参数的加法分组变了，所以只在 GPipe 与 1F1B 之间比 |
| PP 等价 | torchrun 1 进程 vs torchrun 2 进程 PP（默认 1F1B），10 步 | AdamW 1e-5，Muon 1e-4（val_loss 用同样的相对容差） | 见下 |
| GPipe 与 1F1B | 两种调度各跑 10 步完整训练 | 逐位相同 | 计算和累加顺序都相同 |
| 3D 等价 | torchrun 1 进程 vs DP2×TP2、DP2×PP2、TP2×PP2（4 进程）、DP2×TP2×PP2（8 进程），10 步 | val_loss 相对 1e-5，参数 1e-4 | 沿用 TP；默认只跑 3D 那个，两两组合标 slow |
| 中途续训 | 中断再续 vs 一口气跑完（各做法各自比） | 逐位相同 | 同一种做法，计算完全相同 |

容差能不留就不留，因为容差会掩盖末位差别。ZeRO-2 那轮总范数不跨卡相加的错误只造成约 1.4e-4 的差，1e-3 的容差就会漏掉（1e-5 仍能抓住）。

TP 的参数容差 1e-4 是唯一放宽的地方。TP 改变了前向的加法顺序（o_proj、down_proj 先各算一半再相加），几乎每个梯度都带着末位差别，AdamW 又会把个别小梯度元素的差别放大。放宽之前，先逐个参数看了最大差和超过 1e-5 的元素个数：约 19 万个元素里只有 1 个超过（最大 2.8e-5，在第 1 层 k_proj），norm 权重最多差最后一位，是零星的噪声，而不是某个参数成片地差。精确的检查留在两个 worker 里（一步之内，1e-6），10 步的训练测试查的是接线；TP 最典型的错误（去掉 q_norm、k_norm 的梯度求和）差 6.4e-3，仍是容差的 60 多倍。

TP 等价的对照组用 torchrun 起 1 个进程，而不是直接用 python 跑，是为了让两边都单线程，差别只来自 TP 本身；直接用 python 跑时最大差 7.4e-6，混进了线程数的影响。比较参数时先找出最差的那个再报：只报第一个超标的参数会误导，改坏时 embedding 先报 1.5e-4，像是险些没抓住，最差的 q_norm 其实是 6.4e-3。

PP 的容差比 TP 小。PP 只改了共享参数（embedding / lm_head）这一处梯度的加法分组，其余参数每份数据的计算和单卡完全相同，所以 AdamW 用 1e-5 就能通过；Muon 的正交化会把末位差别放大，放宽一档到 1e-4。典型的错误（去掉共享参数的梯度求和）在 val_loss 上的相对差约 1e-3，比容差大两个数量级以上。

GPipe 与 1F1B 要求逐位相同，是因为两者之间最可能出的错正好只差末位：反向顺序不一致时，值都对，只是梯度的累加顺序变了，allclose 抓不到。

3D 测试除了 3D 本身，还要测两两组合，因为"用错了组"只有在组和全局组不同时才测得出来。TP 单独用时 TP 组就是全局组，get_tp_group 漏写 return、一直落到全局组上，原有 TP 测试全部通过。小配置正好够用：2 层能切成 1/1 两段，4 个头、2 个 KV 头能切给 TP=2，全局 64 个 token 在 dp=2 时每条流水线分到 2 份。8 进程的测试在本机每次约 8～10 秒。

CPU 上的多进程测试有两件事测不到。一是 autocast 在 CPU 上不生效，测不到 GPU 上两段之间传的激活是不是 fp32、和接收的缓冲区对不对得上，上云前另外在单卡 GPU 上确认过。二是 gloo 对小消息可能先缓冲，1F1B 里两段同时用阻塞的方式发送，在 CPU 上不一定卡住，所以收发要打包提交（batch_isend_irecv）这一点只能靠分析，测试保证不了。

### 性能优化的开关

- 每个开关默认关闭，原有基线一个都不用动。能精确比较的都写成开关前后的对拍：GQA、分块 CE 用 allclose（CPU、fp32，1e-6）；activation checkpointing 要求逐位相同，因为重算走的是完全相同的运算
- compile 会改变数值（300 步后 val loss 差约 7e-4），不写成单元测试，在 GPU 上跑 300 步对比 loss；compile 下是否断图用 TORCH_LOGS="graph_breaks,recompiles" 检查。自写 FA 的测试里用 fullgraph=True，只要断图就直接报错
- 自写 FA 对 fp32 的 SDPA 留 3e-2 的容差（输入是 bf16）；经过 autograd 与直接调用 launcher 之间要求逐位相同，反向不用 atomic，结果是确定的
- 开关是模块上的属性时，测试要设到真正读它的模块上：act_ckpt 在里面的 Model 上，enable_gqa、attn_impl 在每个 Attention 上。设到外层模型上，Python 不会报错，两次跑的都是原来的路径，怎么改坏都通过
- 比较整模型时比 logits，不比 loss：刚初始化时 logits 都接近 0，loss 约等于 ln(词表大小)，注意力算错 loss 也几乎不动

## 临时改坏记录

改坏一处，看哪些测试失败，用来确认测试真能抓到错误。

| 改坏了什么 | 结果 |
| --- | --- |
| 去掉 loss 除以 grad_accum_steps | 只有基线失败 |
| GRPO 分块 loss 改成除以块内行数（应除以总行数） | 分块一致性测试失败 |
| compute_loss 里的优势取反 | GRPO 基线失败，分块测试不受影响 |
| 手写 AdamW 把 eps 挪进根号 | 与 torch.optim.AdamW 对拍第 0 步就失败 |
| GQA 的 repeat_interleave 换成 repeat | HF 对拍 logits 最大差 0.862 |
| QK-norm 挪到 RoPE 之后 | HF 对拍 logits 最大差 0.164；norm 权重不随机化时测不出 |
| DataLoader 去掉 rank 偏移 | 交替读取测试和 DDP 等价测试失败；续训测试两边错得一样，仍通过 |
| DataLoader 的 load_state_dict 里重复加偏移 | 只有续训测试失败 |
| ZeRO-1 每张卡给全部参数建优化器 | 等价比较照样通过，只有字节数检查失败 |
| ZeRO 去掉 step 之后的参数广播 | 错误从第 1 步起经梯度平均扩散到全部参数，等价和续训测试共 4 个失败 |
| ZeRO-2 总范数只用本卡的梯度算 | 只有两进程与 DDP 的等价测试失败，最大差约 1.4e-4；单进程不跨卡，续训两边错得一样 |
| ZeRO-2 不把不负责的梯度设成 None | 只有 zero2_worker 那个测试失败，训练结果不变 |
| ZeRO-2 忘了去掉 DDP | 结果照样与 DDP 逐位相同，只有检查日志"不包 DDP"的断言能抓到 |
| FSDP2 由 0 号进程直接存 model.state_dict() | 训练照常跑完、不报错，两个比较测试失败 |
| FSDP2 去掉"优化器状态为空时存空字典"的判断 | 只有等价测试失败（最大差 2.9e-3）；续训测试两边错得一样 |
| TP 层级测试里去掉 f | 只有输入梯度对不上，输出和三个权重梯度照样相同 |
| TP 去掉 q_norm、k_norm 的梯度求和 | 整模型对照、等价（6 个参数超标，最大差 6.4e-3，在第 0 层 q_norm）、续训三个测试都失败 |
| PP 第一段不把 norm 换成占位 | 切分测试失败在"两段的键不重叠" |
| PP 第二段的前向漏掉 norm | 两段前向与完整模型的 loss 不同（5.4728 变成 5.5376） |
| PP 共享参数只比第一段那份梯度 | 单进程反向对照失败：共享参数的梯度两段各有一部分，要相加才对 |
| GPipe 两段的反向都倒序 | pp_worker 失败，最大差 4.66e-10，只差末位；加入 1F1B 后，GPipe 与 1F1B 的完整训练也不再逐位相同（45 个参数，最大差 7.0e-7）；其余测试默认走 1F1B，照样通过 |
| PP 去掉 reduce_grads 里共享参数的求和 | 两个对单卡测试（AdamW、Muon）都在 val_loss 上就失败，相对差约 1e-3 |
| PP 的 Muon 分组不认 pp_shared | 只用小配置手动对照过，没有跑测试：第 0 步相同，从第 1 步起 loss 与单卡差 1e-3～1e-2，而且越来越大，每步耗时约变为 4 倍 |
| ParallelOptimizer 的 DP 同步改成在全局组求平均（DDP 的默认做法） | DP2×TP2 等价测试在 val_loss 上失败（5.56731 → 5.56788，相对差约 1e-4，是容差的 10 倍），训练不报错。不要在 3D 下这样改：两段的参数个数和形状不同，全局组上的 all-reduce 会一直卡到超时 |
| get_tp_group 漏写 return（实际发生过） | TP 单独用的测试全部通过；DP2×TP2 等价测试失败（val_loss 相对差约 4e-4，训练前的 eval 就已不同）；TP2×PP2 和 3D 卡到超时。布局测试只测纯函数，抓不到 |
| 每 token 计算量的公式漏掉 lm_head（D×V）那一项 | 与实际模型里 nn.Linear 权重个数的对照失败 |
| enable_gqa 关闭时的 repeat_interleave 换成 repeat | GQA 开关对拍失败 |
| 分块 CE 每块改用 mean | 分块 CE 对拍失败（loss 5.5743 vs 0.7007），最后一块只有 2 行却和其他块权重相同 |
| checkpoint 前把 x 改成 x.detach() | 初版测试（开关设在外层模型上）照样通过；改正后失败，第 0～10 层的梯度是 None |
| 自写 FA 的 dkdv 组内循环只跑 1 次 | G=4 的两组在 dk、dv 上失败，G=1 那组通过 |
| custom op 的 fake 里 M 的 dtype 写成 bf16 | 只有 opcheck 失败（test_faketensor），其余照样通过 |
| Attention 的 triton 分支把 k、v 传反 | 初版比较 loss 的测试照样通过；改为比较 logits 后 96% 的元素不对，最大差 1.2 |

从这些记录看出的规律：

- 续训测试抓不到"中断前后错得一样"的错（rank 偏移、总范数、空优化器），要和等价测试配合
- 单进程测试抓不到跨卡的错（总范数），多进程的逻辑要在多进程下测
- 只影响显存或中间状态、不影响训练结果的错（每卡建全部优化器、不设 None、忘了去掉 DDP），等价测试抓不到，要专门检查字节数、梯度是否为 None、日志
- 续训测试也有能抓到的跨卡错误：TP 去掉 q_norm、k_norm 的求和后，两张卡上的 q_norm 各自更新、不再一致，而 checkpoint 只存了 0 号那份，续训后两次运行就分开了。错误在各卡之间不对称时，续训测试能抓到；对称时（两边错得一样）抓不到
- 只看单层的输出和权重梯度，查不出漏掉 f 的错，要比输入梯度；在整个模型里，这个错会一路传到前面所有层
- 只差末位的错误（比如反向顺序不一致）只有要求逐位相同的测试能抓到。两种实现本应算同一个东西时，就要求逐位相同，不留容差
- CPU 上的多进程测试测不到只在 GPU 上才有的情况（autocast 下的数据类型、NCCL 收发互相等住），要单独检查，或者靠分析
- 用错进程组的错，只有在组和全局组不同的配置下才测得出来；组合测试不能只测"每种并行单独用"。组用错时多进程测试可能直接卡死，所以 run_subprocess 要设超时，超时后还要清掉残留的 worker
- 测试通过不代表它真的在测：开关没设到起作用的模块上，或者比较的量对错误不敏感（刚初始化时的 loss），怎么改坏都能通过。这一轮有两个测试都是在临时改坏时才发现不起作用的，所以每个新测试都要改坏一次

## 各测试文件

- `conftest.py`：公共 fixture。tiny_config（2 层、hidden 64、词表 256 的小模型配置）；sft_config（在 tiny_config 基础上把词表改成 32768、max_seq_len 改成 64，供用真分词器的 SFT / GRPO 测试使用）；make_data_dir（在临时目录造随机 token 的 .bin 分片和 meta.json，可指定片数和每片 token 数）
- `test_model.py`：模型各模块、参数量公式、初始化 loss、每 token 计算量公式；enable_gqa、分块 CE 开关前后 logits / loss 与梯度 allclose，activation checkpointing 开关前后梯度逐位相同
- `test_tokenizer.py`：分词器和 BPE 训练；2 个慢测试是 1000 篇文档的编解码往返、快慢两版 BPE 训练 1000 次合并的对拍
- `test_sft_data.py`：train / test 切分，构建 x, y 时只有答案和 `<|im_end|>` 参与 loss
- `test_dataloader.py`：换片、state_dict 往返、按进程跳着读（world_size=2 的两个 loader 轮流取，与单进程连续取逐批相同）
- `test_pretrain.py`：lr 调度；用假数据和小模型在 CPU 上跑 train()：AdamW / Muon 基线、单 batch overfit、load_model 读回 logits 一致、续训与一口气跑完逐位一致
- `test_muon.py`：newton_schulz 正交化后的奇异值范围；MuonWithAdamW 的 AdamW 分支与 torch.optim.AdamW 逐步对拍
- `test_sft.py`：16 道题、B=4 在 CPU 上跑 train_sft 10 步（过程中换轮打乱两次），AdamW / Muon 基线
- `test_grpo.py`：compute_loss 分块与不分块的 loss、KL、梯度一致；先 SFT 200 步再跑 4 步 train_grpo 的 AdamW / Muon 基线
- `test_config.py`：正常读取、拼错键和节名报错、类型不对报错、--set 覆盖生效，以及 configs/ 下所有文件都能读通
- `test_export_hf.py`：把 norm 权重随机化后的小模型导出，用 transformers 以 fp32 读回，检查没有漏读或多余的权重、logits 一致；没装 transformers 时自动跳过
- `test_ddp.py`：单进程与两进程比较 final.pt 的参数和 val_loss；两进程中途续训与一口气跑完逐位相同
- `test_zero.py`：分配函数的手算例子；单进程下包装（zero_stage 1、2）与不包装各训练 10 步逐位相同，ZeRO-1 单进程续训；两进程 ZeRO-1、ZeRO-2 分别与 DDP 逐位相同、中途续训逐位相同、各卡优化器状态字节数检查，ZeRO-2 时检查日志里确实没包 DDP；用 zero2_worker.py 检查梯度汇总
- `test_fsdp.py`：两进程 FSDP2 与 DDP allclose、检查日志里确实分了片、final.pt 能被 load_model 读回；中途续训逐位相同；muon 配 zero_stage=3 时报错
- `test_tp.py`：并行布局的手算例子；用 tp_worker.py、tp_model_worker.py 做层级和整模型对照；torchrun 1 进程与 2 进程 TP 比较 val_loss 和 final.pt（参数 1e-4）、load_model 读回；两进程中途续训逐位相同；muon 配 tp_size=2、TP 配 ZeRO 时报错。compare_state_dicts 失败时报出超标参数的个数和最大差所在的参数
- `test_pp.py`：并行布局（含 PP 编号）和 PP 组、embedding 组组员的手算例子；单进程里检查切分（两段的键不重叠、合起来等于完整模型，值不变，层名保持全局编号）、两段前向的 loss、模拟交接后各参数的梯度，都与完整模型逐位相同；用 pp_worker.py 对照 GPipe、1F1B 和完整模型；torchrun 1 进程与 2 进程 PP 比较 val_loss 和 final.pt（AdamW 1e-5、Muon 1e-4），检查日志里确实切分了、load_model 能读回；GPipe 与 1F1B 的完整训练逐位相同；两进程中途续训逐位相同
- `test_3d.py`：三种组（TP、DP、PP）和 embedding 组的成员，8 卡时与手算的表一致，几种布局下每种组覆盖每张卡恰好一次、任意两种组的交集只有自己；torchrun 1 进程对照 DP2×TP2、DP2×PP2、TP2×PP2、DP2×TP2×PP2 比较 val_loss 和 final.pt（参数 1e-4），并检查 checkpoint 里 tp、pp、dp 的大小；3D 中途续训逐位相同。只有 3D 等价测试是快测试
- `test_flash_attn.py`：需要 GPU。自写 FA 与 fp32 SDPA 对拍（G=4、T=512；G=4、T=500；G=1）；opcheck 前向、反向两个 custom op；经过 autograd 与直接调用 launcher 逐位相同；compile fullgraph 下前向加反向不断图、梯度与不编译时一致；整模型 attn_impl=triton 与 sdpa 的 logits 一致
- `zero2_worker.py`：不是测试文件，由 test_zero.py 用 torchrun 起两个进程运行
- `tp_worker.py`：不是测试文件，两进程上完整 MLP 与切开的 MLP 比较输出、输入梯度、各卡权重梯度
- `tp_model_worker.py`：不是测试文件，两进程上整模型对照：各参数梯度（q_norm、k_norm 两卡求和后）、拼回和切回、TPOptimizer 的梯度求和、裁剪总范数、一步 SGD 更新
- `pp_worker.py`：不是测试文件，两进程 4 份数据：GPipe 与完整模型的各参数梯度和 loss 逐位比较（共享参数除外），1F1B 与 GPipe 的各参数梯度（含共享参数）和 loss 逐位比较