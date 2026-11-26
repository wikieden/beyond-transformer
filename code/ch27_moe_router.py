"""第 27 章 MoE 是什么：很多个 FFN 专家，每个 token 只走其中 k 个；router 打分 → top-k → 只在选中的专家上做 softmax，
输出是加权和；noisy top-k：打分加上可学幅度的噪声；总参数和激活参数的账。

讲义：beyond-transformer/notes/lessons/27-moe-router.md
运行：uv run python ch27_moe_router.py
完整的 MoE FFN（top-k router + 三种均衡方法）是里程碑 T-M6。
"""

import numpy as np

E, D = 4, 3
WG = np.array([[1.0, -0.5, 0.2, 0.0], [0.3, 1.0, -0.4, 0.2], [-0.2, 0.1, 1.0, 0.5]])   # D × E 的 router
TOKENS = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.3], [0.1, 0.0, 1.0], [0.5, 0.5, 0.5]])
EXPERT_SCALE = np.array([1.0, 2.0, -1.0, 0.5])   # 玩具专家：专家 e 的输出 = scale_e · x


def topk_gate(logits, k=2):
    """Mixtral 式（式 1）：Softmax(TopK(x·W_g))：只保留前 k 个分数，其余设成 −∞ 再 softmax。"""
    idx = np.argsort(-logits)[:k]
    g = np.zeros_like(logits)
    z = np.exp(logits[idx] - logits[idx].max())
    g[idx] = z / z.sum()
    return idx, g


def moe_forward(x, k=2):
    logits = x @ WG
    idx, g = topk_gate(logits, k)
    y = sum(g[e] * EXPERT_SCALE[e] * x for e in idx)
    return logits, idx, g, y


def noisy_logits(x, w_noise, noise):
    """稀疏门控 MoE 式 (4)：H(x) = x·W_g + 噪声 · Softplus(x·W_noise)。"""
    return x @ WG + noise * np.log1p(np.exp(x @ w_noise))


def param_count(d, d_ff, n_exp, k, layers, vocab, attn_per_layer, shared=0):
    """SwiGLU 专家每个 3·d·d_ff。返回 (总参数, 每个 token 激活的参数)。"""
    expert = 3 * d * d_ff
    router = d * n_exp
    per_layer_total = attn_per_layer + router + n_exp * expert + shared * expert
    per_layer_active = attn_per_layer + router + (k + shared) * expert
    emb = 2 * vocab * d
    return layers * per_layer_total + emb, layers * per_layer_active + emb


def main():
    print("第一步 router 打分、选 top-2、在选中的专家上 softmax")
    for t in TOKENS:
        lg, idx, g, y = moe_forward(t)
        print(f"  token {t.tolist()}：分数 {np.round(lg, 2).tolist()} → 专家 {sorted(idx.tolist())}，权重 {np.round(g, 3).tolist()}，输出 {np.round(y, 3).tolist()}")

    print("\n第二步 noisy top-k")
    t = TOKENS[3]
    wn = np.full((D, E), 0.3)
    for noise in (np.zeros(E), np.array([0.0, 0.6, -0.5, 0.3]), np.array([-1.0, 0.0, 0.0, 0.5])):
        h = noisy_logits(t, wn, noise)
        idx, _ = topk_gate(h)
        print(f"  噪声 {noise.tolist()}：分数 {np.round(h, 2).tolist()} → 选专家 {sorted(idx.tolist())}")

    print("\n第三步 总参数 vs 激活参数（Mixtral 8×7B 的配置：d = 4096、d_ff = 14336、8 个专家选 2、32 层）")
    attn = 4096 * 4096 * 2 + 4096 * 1024 * 2    # Q、O 是 d × d；K、V 是 d × (8 头 × 128)
    tot, act = param_count(4096, 14336, 8, 2, 32, 32000, attn)
    print(f"  总参数 {tot / 1e9:.1f}B，每个 token 激活 {act / 1e9:.1f}B（论文：47B / 13B）")


if __name__ == "__main__":
    main()
