"""第 25 章 MLA（多头潜在注意力）：K、V 先压成一个低维向量 c = W_DKV·h 存进 cache，用时再展开；
推理时 W_UK 可以吸收进 query（q·(W_UK c) = (W_UKᵀ q)·c），不用真的展开；可是 RoPE 夹在中间就吸收不了，
所以单独拆出一小段带 RoPE 的 key（decoupled RoPE）。cache 账：每个 token 每层 d_c + d_h^R 个数。

讲义：beyond-transformer/notes/lessons/25-mla.md
运行：uv run python ch25_mla.py
"""

import numpy as np

D, DC, DH = 16, 4, 8          # 隐层宽、压缩维数、每头维数（单头演示）


def mats(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W_DKV": rng.normal(0, 0.3, (DC, D)),    # 压缩：h → c
        "W_UK": rng.normal(0, 0.3, (DH, DC)),    # 展开成 key
        "W_UV": rng.normal(0, 0.3, (DH, DC)),    # 展开成 value
        "W_Q": rng.normal(0, 0.3, (DH, D)),
    }


def scores(hq, hs, M):
    """一个 query（隐状态 hq）对若干个历史隐状态 hs 的分数：先展开再点积 vs 把 W_UK 吸收进 query。"""
    q = M["W_Q"] @ hq
    c = np.array([M["W_DKV"] @ h for h in hs])                 # cache 里存的是 c（DC 维）
    expand = np.array([q @ (M["W_UK"] @ ci) for ci in c])      # 先展开 k = W_UK c
    q_abs = M["W_UK"].T @ q                                    # 吸收：q′ = W_UKᵀ q（只算一次）
    absorbed = c @ q_abs
    return expand, absorbed


def rot_matrix(pos, theta=0.5, d=DH):
    R = np.eye(d)
    for i in range(0, d, 2):
        a = pos * theta / (1 + i)
        c, s = np.cos(a), np.sin(a)
        R[i : i + 2, i : i + 2] = [[c, -s], [s, c]]
    return R


def absorbed_query_with_rope(hq, M, rel):
    """q 在位置 m、k 在位置 n：(R_m q)ᵀ R_n W_UK c = qᵀ R_{n−m} W_UK c。要吸收就得算 q′ = W_UKᵀ R_{n−m}ᵀ q，
    它依赖相对位置 rel = n − m：每个 key 位置都得重新算一遍（等于把 key 展开），吸收就没意义了。"""
    q = M["W_Q"] @ hq
    return M["W_UK"].T @ rot_matrix(rel).T @ q


def cache_elems(kind, n_h=128, d_h=128, layers=60, n_g=8, d_c=512, d_rope=64):
    """每个 token 的 KV cache 元素个数（DeepSeek-V2 表 1）。"""
    per = {"MHA": 2 * n_h * d_h, "GQA": 2 * n_g * d_h, "MQA": 2 * d_h, "MLA": d_c + d_rope}[kind]
    return per * layers


def main():
    M = mats()
    rng = np.random.default_rng(1)
    hs = rng.normal(size=(5, D))
    print("第一步 cache 里只存压缩向量 c")
    print(f"  隐层 {D} 维 → c {DC} 维；展开出的 k、v 各 {DH} 维（真实设定：5120 → 512，128 个头 × 128 维）")

    print("\n第二步 W_UK 吸收进 query：不用真的展开")
    e, a = scores(hs[-1], hs, M)
    print(f"  先展开再点积：{np.round(e, 4).tolist()}")
    print(f"  吸收进 query：{np.round(a, 4).tolist()}（最大差 {np.abs(e - a).max():.1e}）")

    print("\n第三步 RoPE 夹在中间就吸收不了")
    for rel in (0, -1, -4):
        print(f"  相对位置 {rel:2d}：要用的 q′ = W_UKᵀ R({rel})ᵀ q = {np.round(absorbed_query_with_rope(hs[-1], M, rel), 4).tolist()}")
    print("  q′ 随相对位置变：每个 key 都要一个不同的 q′，没法只算一次；DeepSeek-V2 另拆一小段 d_h^R = 64 维的 key 专门带 RoPE")

    print("\n第四步 cache 账（60 层，每个 token 的元素数）")
    for k in ("MHA", "GQA", "MQA", "MLA"):
        print(f"  {k}：{cache_elems(k):,}")
    print(f"  MLA 每层 {512 + 64} = 4.5 × d_h，相当于只有 2.25 组的 GQA；比 MHA 小 {cache_elems('MHA') / cache_elems('MLA'):.1f} 倍")


if __name__ == "__main__":
    main()
