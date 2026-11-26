"""第 12 章 Post-LN 为什么离不开 warmup：一个 L 层的残差网络（子层是线性层），初始化时各层权重的梯度大小；
Post-LN 的梯度集中在靠近输出的层、且不随层数变小，Pre-LN 的梯度随层数变小、各层更均匀。

讲义：beyond-transformer/notes/lessons/12-pre-ln.md
运行：uv run python ch12_pre_ln.py
在真实模型上训练 Post-LN 和 Pre-LN 并比较曲线是 a1 的 pre_norm_ablation 和里程碑 T-M4 的内容，这里只看初始化时的梯度。
"""

import numpy as np


def ln_fwd(x):
    mu = x.mean(-1, keepdims=True)
    sd = np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + 1e-6)
    return (x - mu) / sd, sd


def ln_bwd(dy, y, sd):
    """y = (x − μ)/σ 的反向：dx = (dy − mean(dy) − y·mean(dy·y)) / σ。"""
    return (dy - dy.mean(-1, keepdims=True) - y * (dy * y).mean(-1, keepdims=True)) / sd


def grad_norms(kind: str, L: int = 12, d: int = 64, batch: int = 32, seed: int = 0):
    """每层子层 f_l(x) = x W_lᵀ（W ~ N(0, 1/d)）。Post：x ← LN(x + f(x))；Pre：x ← x + f(LN(x))，最后再 LN。
    loss = ½‖输出 − 目标‖²（目标随机）。返回每层 W 的梯度 Frobenius 范数。"""
    rng = np.random.default_rng(seed)
    W = [rng.normal(0, 1 / np.sqrt(d), (d, d)) for _ in range(L)]
    x = rng.normal(size=(batch, d))
    target = rng.normal(size=(batch, d))
    cache = []
    for w in W:
        if kind == "post":
            z = x + x @ w.T
            y, sd = ln_fwd(z)
            cache.append((x, z, y, sd))
            x = y
        else:
            u, sd = ln_fwd(x)
            cache.append((x, u, sd))
            x = x + u @ w.T
    if kind == "pre":
        out, sdf = ln_fwd(x)
        g = ln_bwd(out - target, out, sdf)
    else:
        g = x - target
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        if kind == "post":
            xin, z, y, sd = cache[l]
            dz = ln_bwd(g, y, sd)
            grads[l] = float(np.linalg.norm(dz.T @ xin))
            g = dz + dz @ W[l]
        else:
            xin, u, sd = cache[l]
            grads[l] = float(np.linalg.norm(g.T @ u))
            g = g + ln_bwd(g @ W[l], u, sd)
    return grads


def last_layer_vs_depth(depths=(6, 12, 24, 48)):
    return {L: (grad_norms("post", L)[-1], grad_norms("pre", L)[-1]) for L in depths}


def main():
    print("第一步 两种放法（初始化时、12 层、d = 64）各层 W 的梯度范数")
    post, pre = grad_norms("post"), grad_norms("pre")
    for l in (0, 3, 6, 9, 11):
        print(f"  第 {l + 1:2d} 层：Post-LN {post[l]:7.3f}   Pre-LN {pre[l]:7.3f}")
    print(f"  最后一层 / 第一层：Post-LN {post[-1] / post[0]:.2f}，Pre-LN {pre[-1] / pre[0]:.2f}")

    print("\n第二步 最后一层的梯度随层数 L 怎么变")
    for L, (a, b) in last_layer_vs_depth().items():
        print(f"  L = {L:2d}：Post-LN {a:7.3f}   Pre-LN {b:7.3f}   Pre-LN × √L = {b * np.sqrt(L):7.3f}")


if __name__ == "__main__":
    main()
