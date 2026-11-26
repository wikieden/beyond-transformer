"""第 16 章 不加位置编码行不行：ALiBi 在分数上按距离减一个斜坡（每个头一个斜率）；
NoPE 什么都不加，只靠因果 mask：第一层均匀地看前面所有词，就能算出“我是第几个”。

讲义：beyond-transformer/notes/lessons/16-alibi-nope.md
运行：uv run python ch16_alibi_nope.py
a1 的 no_pos_emb 消融和五种位置编码的外推对比（里程碑 T-M3）留给自己做。
"""

import numpy as np


def slopes(h: int):
    """ALiBi §3（L331）：8 个头时斜率是 1/2¹、1/2²、…、1/2⁸；一般 H 个头是 2^{−8k/H}。"""
    return [2 ** (-8 * (k + 1) / h) for k in range(h)]


def alibi_bias(n: int, m: float):
    """位置 i 看位置 j（j ≤ i）：分数加 −m·(i − j)。"""
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    b = -m * (i - j).astype(float)
    b[j > i] = -np.inf
    return b


def softmax_row(s):
    s = s - s[np.isfinite(s)].max()
    e = np.where(np.isfinite(s), np.exp(s), 0.0)
    return e / e.sum()


SCORES = np.array([1.0, 0.5, 2.0, 0.8, 1.2])   # 最后一个位置对前 5 个位置（含自己）的原始分数


def alibi_weights(m: float, scores=SCORES):
    n = len(scores)
    return softmax_row(scores + alibi_bias(n, m)[n - 1])


def nope_position(n: int = 8):
    """第一层一个头：所有分数相同（均匀看），value 只有句首 token 是 1。因果 mask 下位置 i 看 i + 1 个词，
    输出 = 1/(i + 1)，单调对应位置，后面的 MLP 就能读出“我是第几个”（NoPE 论文定理 1 的思路）。"""
    v = np.zeros(n)
    v[0] = 1.0
    return [float(np.mean(v[: i + 1])) for i in range(n)]


def main():
    print("第一步 ALiBi 的斜率")
    print("  8 个头：" + "、".join(f"1/{round(1 / s)}" for s in slopes(8)))
    print("  16 个头的前 4 个：" + "、".join(f"{s:.4f}" for s in slopes(16)[:4]))

    print("\n第二步 分数减一个斜坡")
    print(f"  最后一个位置的原始分数 {SCORES.tolist()}（距离 4、3、2、1、0）")
    for m in (0.0, 0.5, 1 / 256):
        print(f"  斜率 {m:.4f}：偏置 {(np.round(alibi_bias(5, m)[4], 4) + 0.0).tolist()}，权重 {np.round(alibi_weights(m), 4).tolist()}")

    print("\n第三步 NoPE：因果 mask 本身就带着位置")
    print(f"  均匀注意力 + 句首标记：各位置的输出 {np.round(nope_position(), 4).tolist()}")


if __name__ == "__main__":
    main()
