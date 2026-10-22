# 测试

## 运行

- `pytest -v`：59 个快测试，约 100 秒，不需要数据
- `pytest -m slow -v`：2 个慢测试，需要原始数据
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

都在 CPU 上跑：把 CUDA_VISIBLE_DEVICES 置空，用 torchrun 起两个进程走 gloo，和在多卡机器上的启动方式相同。

| 测试 | 比较的两边 | 容差 | 原因 |
| --- | --- | --- | --- |
| DDP 等价 | python 单进程 vs torchrun 两进程 | 1e-5 | torchrun 给每个进程设 OMP_NUM_THREADS=1，CPU 上矩阵乘的线程数不同，末位不同 |
| ZeRO-1 / ZeRO-2 等价 | torchrun 两进程 ZeRO vs torchrun 两进程 DDP | 逐位相同 | 两边启动方式相同，梯度、公式和学习率也相同 |
| FSDP2 等价 | torchrun 两进程 FSDP2 vs torchrun 两进程 DDP | 1e-5 | DTensor 上求梯度裁剪总范数时加法顺序不同 |
| 中途续训 | 中断再续 vs 一口气跑完（各做法各自比） | 逐位相同 | 同一种做法，计算完全相同 |

容差能不留就不留，因为容差会掩盖末位差别。ZeRO-2 那轮总范数不跨卡相加的错误只造成约 1.4e-4 的差，1e-3 的容差就会漏掉（1e-5 仍能抓住）。

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

从这些记录看出的规律：

- 续训测试抓不到"中断前后错得一样"的错（rank 偏移、总范数、空优化器），要和等价测试配合
- 单进程测试抓不到跨卡的错（总范数），多进程的逻辑要在多进程下测
- 只影响显存或中间状态、不影响训练结果的错（每卡建全部优化器、不设 None、忘了去掉 DDP），等价测试抓不到，要专门检查字节数、梯度是否为 None、日志

## 各测试文件

- `conftest.py`：公共 fixture。tiny_config（2 层、hidden 64、词表 256 的小模型配置）；sft_config（在 tiny_config 基础上把词表改成 32768、max_seq_len 改成 64，供用真分词器的 SFT / GRPO 测试使用）；make_data_dir（在临时目录造随机 token 的 .bin 分片和 meta.json，可指定片数和每片 token 数）
- `test_model.py`：模型各模块、参数量公式、初始化 loss
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
- `zero2_worker.py`：不是测试文件，由 test_zero.py 用 torchrun 起两个进程运行