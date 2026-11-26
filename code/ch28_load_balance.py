"""第 28 章 专家忙闲不均怎么办：Switch 的 top-1 路由、专家容量 = (token 数 / 专家数) × 容量系数，超出的 token 被丢掉；
负载均衡 loss = α · N · Σ f_i · P_i（f 不可导、P 可导），均匀时最小；它的梯度把“太忙”的专家的概率往下压。

讲义：beyond-transformer/notes/lessons/28-load-balance.md
运行：uv run python ch28_load_balance.py
完整训练里比较几种均衡方法的负载直方图是里程碑 T-M6。
"""

import math

import numpy as np


def switch_aux_loss(probs, alpha=1e-2):
    """probs：T × N 的 router softmax。f_i = 被 top-1 分到专家 i 的 token 比例；P_i = 所有 token 给专家 i 的平均概率。"""
    T, N = probs.shape
    top = probs.argmax(1)
    f = np.bincount(top, minlength=N) / T
    P = probs.mean(0)
    return float(alpha * N * (f * P).sum()), f, P


def capacity(tokens, experts, cf):
    """Switch 式 (3)：每个专家最多收 (token 数 / 专家数) × 容量系数 个 token（向下取整）。"""
    return int(math.floor(tokens / experts * cf))


def dropped(assign, n_exp, cap):
    """按 token 顺序分配，专家满了后来的就被丢掉（不过这一层，只走残差）。"""
    count = np.zeros(n_exp, int)
    drop = 0
    for e in assign:
        if count[e] < cap:
            count[e] += 1
        else:
            drop += 1
    return count, drop


def grad_P(f, alpha=1e-2):
    """aux loss 对 P_i 的梯度 = α·N·f_i：越忙的专家（f 大），它的概率被压得越狠。"""
    return alpha * len(f) * f


def main():
    N, T = 4, 8
    balanced = np.full((T, N), 0.1)
    for t in range(T):
        balanced[t, t % N] = 0.7
    skewed = np.full((T, N), 0.1)
    skewed[:, 0] = 0.7
    print("第一步 Switch 的均衡 loss（α = 0.01，4 个专家、8 个 token）")
    for name, p in (("均匀分配", balanced), ("全挤到专家 0", skewed)):
        l, f, P = switch_aux_loss(p)
        print(f"  {name}：f = {f.tolist()}，P = {np.round(P, 3).tolist()}，loss = {l:.4f}（均匀时最小 = α = 0.01）")
    _, f, _ = switch_aux_loss(skewed)
    print(f"  全挤到专家 0 时，loss 对 P 的梯度 = α·N·f = {grad_P(f).tolist()}：只压专家 0 的概率")

    print("\n第二步 专家容量与丢 token（32 个 token、4 个专家）")
    rng = np.random.default_rng(0)
    assign = rng.choice(4, 32, p=[0.4, 0.3, 0.2, 0.1])
    for cf in (1.0, 1.25, 2.0):
        cap = capacity(32, 4, cf)
        cnt, d = dropped(assign, 4, cap)
        print(f"  容量系数 {cf}：每个专家最多 {cap} 个；实际 {cnt.tolist()}，丢掉 {d} 个 token")


if __name__ == "__main__":
    main()
