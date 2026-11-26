"""第 20 章 换宽度为什么要换学习率（μP）：Adam 的一步每个权重大约挪 lr，输出却按宽度 n 累加，
所以宽度翻倍、输出的变化也翻倍；μP 把隐藏层的 Adam 学习率按 1/n 缩。DeepNorm：x ← LN(α·x + f(x))，
α = (2N)^{1/4}、子层初始化再乘 β = (8N)^{−1/4}，让很深的 Post-LN 也能训。

讲义：beyond-transformer/notes/lessons/20-mup-deepnorm.md
运行：uv run python ch20_mup_deepnorm.py
不同宽度下扫学习率、看最优学习率漂不漂（里程碑 T-M4 选做）留给自己做。
"""

import numpy as np


def adam_step_effect(n, lr=1e-3, seed=0):
    """一层 y = W x（宽 n）。x 的每个分量 O(1)。Adam 的第一步 ΔW ≈ lr·sign(g)，g = δ xᵀ，
    于是 Δy = ΔW x = lr·Σ_j |x_j|·sign(δ)：随 n 线性增长。返回 Δy 的均方根。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    delta = rng.normal(size=n)
    dW = lr * np.sign(np.outer(delta, x))
    return float(np.sqrt(((dW @ x) ** 2).mean()))


def init_output(n, seed=0):
    """标准初始化 W ~ N(0, 1/n)：前向输出的均方根和宽度无关。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    W = rng.normal(0, 1 / np.sqrt(n), (n, n))
    return float(np.sqrt(((W @ x) ** 2).mean()))


def mup_lr(n, base_n=256, base_lr=1e-3):
    """μP（Adam，隐藏层）：学习率 ∝ 1/宽度。"""
    return base_lr * base_n / n


def deepnorm(N):
    """DeepNet（decoder-only / encoder-only）：α = (2N)^{1/4}，β = (8N)^{−1/4}，N 是层数。"""
    return (2 * N) ** 0.25, (8 * N) ** -0.25


def main():
    print("第一步 宽度变了，前向不变、更新变了")
    for n in (256, 1024, 4096):
        print(f"  宽 {n:4d}：初始化输出均方根 {init_output(n):.3f}；Adam 一步（lr = 1e-3）后输出变化 {adam_step_effect(n):.3f}")

    print("\n第二步 μP：隐藏层学习率按 1/n 缩")
    for n in (256, 1024, 4096):
        lr = mup_lr(n)
        print(f"  宽 {n:4d}：lr = {lr:.2e}，输出变化 {adam_step_effect(n, lr):.3f}")

    print("\n第三步 DeepNorm 的两个系数")
    for N in (12, 100, 1000):
        a, b = deepnorm(N)
        print(f"  {N:4d} 层：α = {a:.3f}，β = {b:.3f}")


if __name__ == "__main__":
    main()
