from torch.autograd import DeviceType


WAIT, WARMUP, ACTIVE = 5, 1, 2

def categorize(name):
    name = name.lower()
    categories = [
        ("attn", ("flash", "fmha", "attention")),
        ("matmul", ("gemm", "nvjet", "cutlass", "xmma")),
        ("CE", ("softmax", "nll_loss", "cross_entropy")),
        ("optimizer", ("adam", "multi_tensor")),
        ("other", ()),
    ]
    for typ, kws in categories:
        # 如果 name 里包含这一组的 key_word
        if any(k in name for k in kws):
            return typ
    return "other"


def make_trace_handler(path):
    def handler(prof):
        # 导出 trace
        prof.export_chrome_trace(path)
        # 只取 GPU kernel
        totals = {}
        kernels = []
        for e in prof.key_averages():
            if e.device_type != DeviceType.CUDA:
                continue
            if e.key.startswith("ProfilerStep"):
                continue
            cat = categorize(e.key)
            t = e.self_device_time_total  # 微秒，ACTIVE 步合计
            c = e.count  # 调用次数
            totals[cat] = totals.get(cat, 0) + t
            kernels.append((cat, t, c, e.key))
        
        # 换算成每步毫秒
        totals_ms = {k: v / ACTIVE / 1000 for k, v in totals.items()}
        total_ms = sum(totals_ms.values())

        # 先 log 五行：每类的毫秒数和占比
        for cat in ["attn", "matmul", "CE", "optimizer", "other"]:
            ms = totals_ms.get(cat, 0.0)
            pct = (ms / total_ms * 100) if total_ms > 0 else 0.0
            print(f"  {cat:10s}: {ms:8.2f} ms/step  ({pct:5.1f}%)")

        # 再 log 一行合计
        print(f"  {'total':10s}: {total_ms:8.2f} ms/step")

        # 按耗时大小，log 前 15 个 kernel
        kernels.sort(key=lambda x: -x[1])
        for cat, t, c, name in kernels[:15]:
            ms = t / ACTIVE / 1000
            per_step_count = c / ACTIVE
            print(f"  {cat:10s} {ms:8.2f} ms  {per_step_count:8.1f} calls  {name[:60]}")
    return handler