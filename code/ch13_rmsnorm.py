"""第 13 章 RMSNorm：LayerNorm 去掉“减平均值”只留“除以均方根”；两者各对什么不变（平移 / 缩放）；
一个向量过 LayerNorm 和 RMSNorm 的数；少算几步、少读一份参数省在哪。

讲义：beyond-transformer/notes/lessons/13-rmsnorm.md
运行：uv run python ch13_rmsnorm.py
a1 的 rmsnorm（PyTorch 模块）和 layer_norm_ablation 的训练留给自己做。
"""

import numpy as np

X = np.array([1.0, 2.0, 3.0, 6.0])


def layer_norm(x, g=None, b=None, eps=1e-5):
    g = np.ones_like(x) if g is None else g
    b = np.zeros_like(x) if b is None else b
    return g * (x - x.mean()) / np.sqrt(x.var() + eps) + b


def rms_norm(x, g=None, eps=1e-5):
    """a1 式：x / RMS(x) · g，RMS(x) = √(mean(x²) + ε)。"""
    g = np.ones_like(x) if g is None else g
    return x / np.sqrt((x * x).mean() + eps) * g


def invariances(x=X):
    """平移（每个分量加同一个数）和缩放（整体乘一个数）后，输出变不变。"""
    return {
        "LN 平移 +5": float(np.abs(layer_norm(x + 5) - layer_norm(x)).max()),
        "LN 缩放 ×3": float(np.abs(layer_norm(3 * x) - layer_norm(x)).max()),
        "RMS 平移 +5": float(np.abs(rms_norm(x + 5) - rms_norm(x)).max()),
        "RMS 缩放 ×3": float(np.abs(rms_norm(3 * x) - rms_norm(x)).max()),
    }


def op_count(d: int):
    """每个向量的逐元素运算（粗略）：LN 要求均值、减均值、平方、求和、除、乘 γ、加 β；RMS 少了求均值、减均值、加 β。"""
    ln = {"求均值": d, "减均值": d, "平方求和": 2 * d, "除以标准差": d, "乘 γ": d, "加 β": d}
    rms = {"平方求和": 2 * d, "除以 RMS": d, "乘 γ": d}
    return sum(ln.values()), sum(rms.values())


def param_bytes(d: int, n_norms: int, bytes_per: int = 2):
    """每个 norm 的参数：LN 有 γ、β 两份，RMSNorm 只有 γ。"""
    return 2 * d * n_norms * bytes_per, d * n_norms * bytes_per


def main():
    print("第一步 同一个向量过两种 norm")
    print(f"  x = {X.tolist()}：均值 {X.mean()}，标准差 {X.std():.4f}，均方根 {np.sqrt((X * X).mean()):.4f}")
    print(f"  LayerNorm：{np.round(layer_norm(X), 4).tolist()}（均值 0、方差 1）")
    print(f"  RMSNorm：  {np.round(rms_norm(X), 4).tolist()}（均方根 1，均值不是 0）")

    print("\n第二步 对什么不变")
    for k, v in invariances().items():
        print(f"  {k}：输出最多变化 {v:.4f}")

    print("\n第三步 省在哪")
    ln, rms = op_count(4096)
    print(f"  d = 4096 一个向量：LN 约 {ln:,} 次逐元素运算，RMSNorm 约 {rms:,} 次（少 {1 - rms / ln:.0%}）")
    a, b = param_bytes(4096, 65)
    print(f"  32 层（每层 2 个 norm + 最后 1 个）、bf16：LN 参数 {a / 1024:.0f} KB，RMSNorm {b / 1024:.0f} KB")


if __name__ == "__main__":
    main()
