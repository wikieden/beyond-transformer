"""第 18 章 训练为什么会突然崩：注意力 logit 随 W_Q、W_K 变大而平方增长，softmax 塌成 one-hot（熵坍缩）；
QK-norm 先把 q、k 归一化，logit 有上界；输出 logit 发散（log Z 漂走）和 z-loss = λ·log²Z。

讲义：beyond-transformer/notes/lessons/18-stability.md
运行：uv run python ch18_stability.py
三个宽度 × 六个学习率的“LR 敏感度”实验、加不加 QK-norm / z-loss 的对比是里程碑 T-M4。
"""

import numpy as np

D, N = 64, 16


def inputs(seed=0, n=N, d=D):
    """n 个已经过 LayerNorm 的 token（每个向量均值 0、方差 1）。"""
    x = np.random.default_rng(seed).normal(size=(n, d))
    return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)


def weights(seed=1, d=D):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1 / np.sqrt(d), (d, d)), rng.normal(0, 1 / np.sqrt(d), (d, d))


def rms(v):
    return v / np.sqrt((v * v).mean(-1, keepdims=True))


def attn_stats(scale, qk_norm=False, seed=0):
    """W_Q、W_K 都乘 scale（训练中权重变大）。返回最大 |logit| 和每行 softmax 的平均熵（nats）、平均最大权重。"""
    x = inputs(seed)
    wq, wk = weights()
    q, k = x @ (scale * wq).T, x @ (scale * wk).T
    if qk_norm:
        q, k = rms(q), rms(k)
    s = q @ k.T / np.sqrt(D)
    p = np.exp(s - s.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    ent = float(-(p * np.log(p + 1e-30)).sum(1).mean())
    return float(np.abs(s).max()), ent, float(p.max(1).mean())


def z_loss(logits, lam=1e-4):
    z = np.log(np.exp(logits - logits.max()).sum()) + logits.max()
    return float(z), float(lam * z * z)


def main():
    print("第一步 注意力 logit 随权重平方增长")
    for c in (1, 2, 4, 8):
        mx, ent, pm = attn_stats(c)
        print(f"  W_Q、W_K 乘 {c}：最大 |logit| {mx:7.2f}，平均熵 {ent:.3f}（均匀是 {np.log(N):.3f}），平均最大权重 {pm:.3f}")

    print("\n第二步 QK-norm：先把 q、k 的均方根归一到 1")
    for c in (1, 8):
        mx, ent, pm = attn_stats(c, qk_norm=True)
        print(f"  乘 {c}：最大 |logit| {mx:.2f}（上界 √d = {np.sqrt(D):.0f}），平均熵 {ent:.3f}")

    print("\n第三步 z-loss：把 log Z 拉回 0")
    for lg in ([30.0, 25.0, 20.0], [0.0, -5.0, -10.0], [-40.0, -45.0, -50.0]):
        z, zl = z_loss(np.array(lg))
        print(f"  logits {lg}：log Z = {z:.4f}，z-loss（λ = 1e-4）= {zl:.6f}；softmax 都是 {np.round(np.exp(np.array(lg) - z), 4).tolist()}")


if __name__ == "__main__":
    main()
