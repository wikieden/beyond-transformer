"""第 30 章 DeepSeekMoE：专家切细（N 个专家每个切成 m 份、选 mK 个），组合数暴涨；共享专家每个 token 都走；
不加 loss、只调 bias 的负载均衡：b_i ← b_i + u·sign(平均负载 − 专家 i 的负载)，bias 只影响“选谁”，不进权重和梯度。

讲义：beyond-transformer/notes/lessons/30-deepseekmoe.md
运行：uv run python ch30_deepseekmoe.py
在自己的 MoE 上比较三种均衡方法、画负载直方图是里程碑 T-M6。
"""

from math import comb

import numpy as np


def combos(n, k):
    return comb(n, k)


def fine_grained(N=16, K=2, m=4, d_ff=1024, d=256):
    """每个专家切成 m 份（中间层 d_ff/m），激活 mK 份：参数和计算量不变，组合数从 C(N, K) 变成 C(mN, mK)。"""
    params_before = N * 3 * d * d_ff
    params_after = m * N * 3 * d * (d_ff // m)
    return combos(N, K), combos(m * N, m * K), params_before, params_after


def simulate(steps=300, T=512, E=8, k=2, u=0.01, seed=0, use_bias=True):
    """router 分数 = 每个专家固定的偏好 + 噪声；偏好让前几个专家天然更受欢迎。每步按 top-k 选专家、统计负载、更新 bias。
    返回：第一步和最后一步的负载（占比）、最大负载 / 平均负载（MaxVio + 1）。"""
    rng = np.random.default_rng(seed)
    pref = np.linspace(1.0, -1.0, E)
    b = np.zeros(E)
    hist = []
    for _ in range(steps):
        s = pref + rng.normal(0, 1.0, (T, E))
        chosen = np.argsort(-(s + b), axis=1)[:, :k]          # bias 只用来选
        load = np.bincount(chosen.ravel(), minlength=E) / (T * k)
        hist.append(load)
        if use_bias:
            b += u * np.sign(load.mean() - load)               # 负载低于平均 → bias 加大
    first, last = hist[0], np.mean(hist[-20:], 0)
    return first, last, float(last.max() / last.mean()), b


def gate_value(s, chosen):
    """门控权重仍然用原始分数 s（不带 bias）：bias 不改变“加权多少”，只改变“选谁”。"""
    g = np.zeros_like(s)
    g[chosen] = s[chosen] / s[chosen].sum()
    return g


def main():
    print("第一步 细粒度切分：组合数")
    c1, c2, p1, p2 = fine_grained()
    print(f"  16 个专家选 2：{c1:,} 种组合；每个切成 4 份、64 个选 8：{c2:,} 种；参数 {p1:,} → {p2:,}（不变）")

    print("\n第二步 共享专家")
    print("  DeepSeek-V2：每层 2 个共享专家 + 160 个路由专家选 6；共享专家放所有 token 都需要的通用知识，路由专家各自专精")

    print("\n第三步 不加 loss、只调 bias（8 个专家选 2，u = 0.01）")
    f0, l0, v0, _ = simulate(use_bias=False)
    f1, l1, v1, b = simulate(use_bias=True)
    print(f"  不调 bias：负载 {np.round(l0, 3).tolist()}，最大 / 平均 = {v0:.2f}")
    print(f"  调 bias 300 步：负载 {np.round(l1, 3).tolist()}，最大 / 平均 = {v1:.2f}")
    print(f"  学到的 bias：{np.round(b, 2).tolist()}（天然受欢迎的专家被减分）")
    s = np.array([0.6, 0.3, 0.1, 0.05])
    print(f"  门控权重只用原始分数：分数 {s.tolist()}、选 [0, 1] → {np.round(gate_value(s, [0, 1]), 3).tolist()}")


if __name__ == "__main__":
    main()
