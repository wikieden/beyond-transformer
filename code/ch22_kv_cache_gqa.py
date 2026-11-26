"""第 22 章 KV cache 与 MQA / GQA：逐个生成时，把前面算过的 K、V 存起来，每步只算新 token；
带 cache 和每步从头算结果一样，但计算量从 O(n²) 降到 O(n)；cache 的字节数 2·L·H_kv·d_head·n·字节；
几个 Q 头共用一组 K、V（GQA），cache 缩小 H / H_kv 倍。

讲义：beyond-transformer/notes/lessons/22-kv-cache-gqa.md
运行：uv run python ch22_kv_cache_gqa.py
在 a1 模型上写 KV cache 解码、加 GQA 并实测 cache 大小是里程碑 T-M5。
"""

import numpy as np

D, H = 8, 4
DH = D // H


def weights(seed=0, n_kv=H):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.5, (D, D)), rng.normal(0, 0.5, (D, n_kv * DH)), rng.normal(0, 0.5, (D, n_kv * DH))


def attend_last(q, K, V):
    """一个 query（H 个头）对 K、V（n × H_kv × d_head）做注意力；GQA 时第 h 个 Q 头用第 h // (H / H_kv) 组 K、V。"""
    n_kv = K.shape[1]
    g = H // n_kv
    out = []
    for h in range(H):
        k, v = K[:, h // g], V[:, h // g]
        s = k @ q[h] / np.sqrt(DH)
        w = np.exp(s - s.max())
        w /= w.sum()
        out.append(w @ v)
    return np.concatenate(out)


def generate(x, n_kv=H, use_cache=True):
    """x：n × D 的输入（已经是每步的隐状态）。返回每一步最后一个位置的注意力输出和做了多少次 K、V 投影。"""
    Wq, Wk, Wv = weights(n_kv=n_kv)
    outs, proj = [], 0
    Kc, Vc = [], []
    for t in range(len(x)):
        q = (x[t] @ Wq).reshape(H, DH)
        if use_cache:
            Kc.append((x[t] @ Wk).reshape(n_kv, DH))
            Vc.append((x[t] @ Wv).reshape(n_kv, DH))
            proj += 1
            K, V = np.array(Kc), np.array(Vc)
        else:
            K = (x[: t + 1] @ Wk).reshape(t + 1, n_kv, DH)
            V = (x[: t + 1] @ Wv).reshape(t + 1, n_kv, DH)
            proj += t + 1
        outs.append(attend_last(q, K, V))
    return np.array(outs), proj


def mean_pool_heads(W, n_kv):
    """GQA 论文 §2.1：把 MHA 的 H 个 K（或 V）头按组取平均，得到 H_kv 个头，再少量继续训练（uptraining）。"""
    d, hd = W.shape
    heads = W.reshape(d, H, hd // H)
    return heads.reshape(d, n_kv, H // n_kv, hd // H).mean(2).reshape(d, n_kv * (hd // H))


def cache_bytes(layers, n_kv, d_head, n_tokens, bytes_per=2, batch=1):
    """K 和 V 两份 × 层数 × KV 头数 × 每头维数 × token 数 × 每个数的字节数。"""
    return 2 * layers * n_kv * d_head * n_tokens * bytes_per * batch


def main():
    x = np.random.default_rng(1).normal(size=(6, D))
    print("第一步 带 cache 和从头算，结果一样")
    a, pa = generate(x, use_cache=True)
    b, pb = generate(x, use_cache=False)
    print(f"  6 步生成：最大差 {np.abs(a - b).max():.1e}；K、V 投影次数：带 cache {pa}，从头算 {pb}（= 1 + 2 + … + 6）")
    print(f"  生成 1000 步：带 cache 1000 次，从头算 {1000 * 1001 // 2:,} 次")

    print("\n第二步 cache 有多大（32 层、32 个 Q 头、d_head = 128、bf16）")
    for name, nkv in (("MHA（32 组 K、V）", 32), ("GQA（8 组）", 8), ("MQA（1 组）", 1)):
        gb = cache_bytes(32, nkv, 128, 4096) / 2**30
        print(f"  {name}：4096 个 token 的 cache {gb:.3f} GiB；batch 32 时 {gb * 32:.1f} GiB")

    print("\n第三步 GQA：几个 Q 头共用一组 K、V")
    for nkv in (4, 2, 1):
        print(f"  H = 4、H_kv = {nkv}：Q 头 → K/V 组 {[h // (H // nkv) for h in range(H)]}，cache 是 MHA 的 1/{H // nkv}")
    _, Wk, _ = weights()
    pooled = mean_pool_heads(Wk, 2)
    print(f"  从 MHA 改成 GQA（uptraining 的初始化）：W_K {Wk.shape} → 组内取平均 {pooled.shape}")

if __name__ == "__main__":
    main()
