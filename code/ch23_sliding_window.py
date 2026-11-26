"""第 23 章 滑动窗口注意力：每层只看前 W 个词，叠 k 层后最远能间接看到约 W·k 个；rolling buffer cache 把位置 i 存在 i mod W；
local / global 交替；logit soft-capping：cap·tanh(x / cap)。

讲义：beyond-transformer/notes/lessons/23-sliding-window.md
运行：uv run python ch23_sliding_window.py
在 a1 模型上加可配置的 SWA 交替模式、soft-cap 是里程碑 T-M5。
"""

import numpy as np


def window_mask(n, w):
    """位置 i 能看 j：i − w < j ≤ i（包括自己，一共 w 个）。"""
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return ((j <= i) & (j > i - w)).astype(int)


def reach(n, w, layers):
    """叠 layers 层后，最后一个位置最远能间接看到多早的位置（布尔矩阵连乘）。"""
    m = window_mask(n, w)
    r = np.eye(n, dtype=int)
    for _ in range(layers):
        r = ((m @ r) > 0).astype(int)
    first = int(np.argmax(r[-1] > 0))
    return n - 1 - first


def rolling_buffer(tokens, w):
    """把第 i 个 token 写到 cache 的 i mod w 号槽位，返回最后的 cache 内容。"""
    cache = [None] * w
    for i, t in enumerate(tokens):
        cache[i % w] = t
    return cache


def softcap(x, cap):
    return cap * np.tanh(np.asarray(x, float) / cap)


def kv_tokens(n, pattern, w_local, layers):
    """每层要存多少个 token 的 K、V：local 层最多 w_local 个，global 层 n 个。pattern 例如 'LG'（交替）。"""
    total = 0
    for l in range(layers):
        total += min(n, w_local) if pattern[l % len(pattern)] == "L" else n
    return total


def main():
    print("第一步 感受野随层数线性变宽（窗口 W = 4，含自己）")
    for k in (1, 2, 3, 5):
        print(f"  {k} 层：最后一个位置最远能看到 {reach(40, 4, k)} 个位置之前（≈ (W − 1)·k）")
    print(f"  Mistral 7B：W = 4096、32 层 → 理论跨度约 4096 × 32 = {4096 * 32:,}")

    print("\n第二步 rolling buffer cache（W = 4）")
    toks = ["The", "cat", "sat", "on", "the", "mat", "and"]
    print(f"  依次写入 {toks}：cache = {rolling_buffer(toks, 4)}（第 i 个存在 i mod 4 号槽）")

    print("\n第三步 local / global 交替省多少 cache（42 层、读到 8192 个 token）")
    for name, pat in (("全 global", "G"), ("Gemma 2：local 4096 / global 交替", "LG"), ("5 local : 1 global", "LLLLLG")):
        t = kv_tokens(8192, pat, 4096, 42)
        print(f"  {name}：所有层共存 {t:,} 个 token 的 K、V（全 global 的 {t / (8192 * 42):.0%}）")

    print("\n第四步 logit soft-capping")
    xs = [5.0, 20.0, 50.0, 100.0, 1000.0]
    print(f"  cap = 50（注意力）：{np.round(softcap(xs, 50), 2).tolist()}")
    print(f"  cap = 30（最后的输出层）：{np.round(softcap(xs, 30), 2).tolist()}")


if __name__ == "__main__":
    main()
