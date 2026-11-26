"""第 26 章 读源码：nanoGPT（GPT-2 式）和 nanochat（现代配方）的差异。把 nanochat 里几个“小零件”用 NumPy 算一遍：
ReLU² 激活、无参数 RMSNorm、输出 logit 的 soft-cap = 15、每层的窗口模式 SSSL、残差 / x0 系数的初始化、哪些层有 value embedding。

讲义：beyond-transformer/notes/lessons/26-nanochat.md
运行：uv run python ch26_nanochat.py
差异表写进自己的笔记、在 a1 模型上实现这些改动是里程碑 T-M5。
"""

import math

import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


def relu2(x):
    """nanochat 的 MLP 激活：F.relu(x).square()。"""
    return np.maximum(x, 0) ** 2


def rms_norm_noparam(x, eps=1e-6):
    """F.rms_norm(x, (d,))：没有可学的 g，也没有 bias。"""
    return x / np.sqrt((x * x).mean() + eps)


def softcap(x, cap=15.0):
    return cap * np.tanh(np.asarray(x, float) / cap)


def window_sizes(n_layer=12, seq=2048, pattern="SSSL"):
    """S = 四分之一上下文（向上取到 128 的倍数），L = 全上下文；按 pattern 平铺，最后一层强制 L。"""
    short = -(-seq // 4 // 128) * 128
    ws = [seq if pattern[i % len(pattern)] == "L" else short for i in range(n_layer)]
    ws[-1] = seq
    return ws


def resid_x0_init(n_layer=12):
    resid = [1.15 - 0.10 * i / max(n_layer - 1, 1) for i in range(n_layer)]
    x0 = [0.20 - 0.15 * i / max(n_layer - 1, 1) for i in range(n_layer)]
    return resid, x0


def has_ve(i, n_layer=12):
    return i % 2 == (n_layer - 1) % 2


def main():
    print("第一步 MLP 激活：GELU（nanoGPT）vs ReLU²（nanochat）")
    for x in (-2.0, -0.5, 0.5, 1.0, 2.0):
        print(f"  x = {x:+.1f}：GELU {gelu(x):+.4f}，ReLU² {relu2(x):.4f}")

    print("\n第二步 norm 不带参数")
    v = np.array([1.0, 2.0, 3.0, 6.0])
    print(f"  {v.tolist()} → {np.round(rms_norm_noparam(v), 4).tolist()}（nanoGPT 的 LayerNorm 有 γ、β）")

    print("\n第三步 输出 logit soft-cap = 15")
    print(f"  {[5, 15, 30, 100]} → {np.round(softcap([5, 15, 30, 100]), 2).tolist()}")

    print("\n第四步 每层的窗口（12 层、上下文 2048、模式 SSSL）")
    print(f"  {window_sizes()}")

    print("\n第五步 残差流上的两个可学标量（初始化）")
    r, x0 = resid_x0_init()
    print(f"  resid_lambda：第 1 层 {r[0]:.2f} → 第 12 层 {r[-1]:.2f}；x0_lambda：{x0[0]:.2f} → {x0[-1]:.2f}")
    print(f"  有 value embedding 的层：{[i for i in range(12) if has_ve(i)]}")


if __name__ == "__main__":
    main()
