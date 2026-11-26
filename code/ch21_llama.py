"""第 21 章 LLaMA 的“标准配方”：原版块 → LLaMA 块，一个零件一个零件地换（Pre-norm + RMSNorm、SwiGLU、RoPE、去掉 bias）；
一层的参数怎么变；训练超参（AdamW β₂ = 0.95、余弦到 10%、权重衰减 0.1、裁剪 1.0、warmup 2000）。

讲义：beyond-transformer/notes/lessons/21-llama.md
运行：uv run python ch21_llama.py
只算一个玩具宽度的一层；给真实模型（GPT-2、Llama、Qwen）算参数量并对官方数是里程碑 T-M1。
"""

import math

D = 96


def layer_params(d=D, ffn="relu", norm="layer", bias=True, pos="learned"):
    """一层（注意力 + FFN + 两个 norm）的参数个数。ffn：relu（d → 4d → d）或 swiglu（d → 8d/3 取 64 倍数，三个矩阵）。"""
    attn = 4 * d * d + (4 * d if bias else 0)
    if ffn == "relu":
        f = 4 * d
        mlp = 2 * d * f + ((f + d) if bias else 0)
    else:
        f = int(math.ceil(8 * d / 3 / 64) * 64)
        mlp = 3 * d * f
    norms = 2 * (2 * d if norm == "layer" else d)
    return {"注意力": attn, "FFN": mlp, "norm": norms}


STEPS = [
    ("原版（Post-LN、LayerNorm、ReLU FFN、有 bias）", dict(ffn="relu", norm="layer", bias=True)),
    ("换成 Pre-norm + RMSNorm", dict(ffn="relu", norm="rms", bias=True)),
    ("FFN 换成 SwiGLU（d_ff = 8d/3）", dict(ffn="swiglu", norm="rms", bias=True)),
    ("去掉所有 bias", dict(ffn="swiglu", norm="rms", bias=False)),
]

TRAIN = {"优化器": "AdamW，β₁ = 0.9、β₂ = 0.95", "学习率调度": "余弦，最后降到峰值的 10%", "权重衰减": "0.1", "梯度裁剪": "1.0", "warmup": "2000 步"}


def cosine_to_10pct(t, peak=3e-4, warm=2000, total=100_000):
    """LLaMA §2.3：warmup 后余弦降到峰值的 10%。"""
    if t < warm:
        return peak * t / warm
    p = (t - warm) / (total - warm)
    return 0.1 * peak + 0.5 * (1 + math.cos(math.pi * p)) * 0.9 * peak


def main():
    print(f"第一步 一个零件一个零件地换（d = {D} 的一层）")
    for name, kw in STEPS:
        p = layer_params(**kw)
        print(f"  {name}：{p}，合计 {sum(p.values()):,}")
    print("  位置编码：原版加在嵌入上的正弦 / 学习位置表 → 每层注意力里的 RoPE（不占参数）")

    print("\n第二步 训练超参")
    for k, v in TRAIN.items():
        print(f"  {k}：{v}")
    for t in (0, 1000, 2000, 51000, 100000):
        print(f"  t = {t:6d}：lr = {cosine_to_10pct(t):.2e}")


if __name__ == "__main__":
    main()
