"""第 4 章 学习率怎么排、梯度太大怎么办：warmup + cosine 的几个取值；warmup 为什么有用（Adam 早期的 v̂ 不准）；
梯度裁剪（按全局 L2 范数缩放）。

讲义：beyond-transformer/notes/lessons/04-lr-schedule.md
运行：uv run python ch04_lr_schedule.py
只把讲义里的几个取值代进公式算一遍；a1 的 get_lr_cosine_schedule、gradient_clipping 两个 adapter 留给自己写。
"""

import math

import numpy as np

AMAX, AMIN, TW, TC = 1e-3, 1e-4, 100, 1000


def lr_at(t, amax=AMAX, amin=AMIN, tw=TW, tc=TC):
    """a1 §4.4：t < Tw 线性升；Tw ≤ t ≤ Tc 余弦从 amax 降到 amin；t > Tc 停在 amin。"""
    if t < tw:
        return t / tw * amax
    if t <= tc:
        return amin + 0.5 * (1 + math.cos((t - tw) / (tc - tw) * math.pi)) * (amax - amin)
    return amin


def early_v_noise(seed=0, n=2000):
    """梯度 ~ N(0.1, 1)：前 k 步估计的 √v̂（Adam 用来除的分母）有多不稳。返回 k → (√v̂ 的 5% 分位, 95% 分位)。"""
    rng = np.random.default_rng(seed)
    out = {}
    for k in (1, 5, 50, 500):
        vals = []
        for _ in range(n):
            g = rng.normal(0.1, 1.0, k)
            v = 0.0
            for i, gi in enumerate(g, 1):
                v = 0.999 * v + 0.001 * gi * gi
            vals.append(math.sqrt(v / (1 - 0.999**k)))
        out[k] = (float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95)))
    return out


def clip(grads, max_norm=1.0, eps=1e-6):
    """所有参数的梯度拼起来算一个 L2 范数；超过 max_norm 就整体乘 max_norm / (norm + eps)。"""
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads))
    if norm <= max_norm:
        return grads, norm, 1.0
    s = max_norm / (norm + eps)
    return [g * s for g in grads], norm, s


def main():
    print("第一步 warmup + cosine（αmax = 1e-3、αmin = 1e-4、Tw = 100、Tc = 1000）")
    for t in (0, 50, 100, 325, 550, 775, 1000, 1200):
        print(f"  t = {t:4d}：lr = {lr_at(t):.6f}")

    print("\n第二步 为什么要 warmup：前几步的 √v̂ 有多不准（梯度 ~ N(0.1, 1)，真值约 1.005）")
    for k, (lo, hi) in early_v_noise().items():
        print(f"  只看了 {k:3d} 步：√v̂ 的 90% 范围 [{lo:.3f}, {hi:.3f}]")

    print("\n第三步 梯度裁剪")
    for gs in ([np.array([3.0, 4.0])], [np.array([0.3, 0.4])], [np.array([3.0, 0.0]), np.array([[0.0, 4.0], [0.0, 0.0]])]):
        out, n, s = clip(gs)
        print(f"  梯度 {[g.tolist() for g in gs]}：范数 {n:.2f}，缩放 {s:.6f} → {[np.round(g, 4).tolist() for g in out]}")


if __name__ == "__main__":
    main()
