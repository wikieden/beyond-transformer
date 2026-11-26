"""第 14 章 SwiGLU：门（Swish(xW)）× 内容（xV），再乘 W₂；为什么中间层取 8/3·d 能和普通 FFN 参数一样多；
Swish（SiLU）和 ReLU、GELU 长什么样；一个小例子看门怎么“关掉”一部分内容。

讲义：beyond-transformer/notes/lessons/14-swiglu.md
运行：uv run python ch14_swiglu.py
a1 的 positionwise_feedforward（SwiGLU 模块）和 swiglu_ablation 的训练对比留给自己做。
"""

import math

import numpy as np


def relu(x):
    return np.maximum(x, 0)


def silu(x):
    """Swish₁ = SiLU：x·σ(x)。"""
    return x / (1 + np.exp(-x))


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


X = np.array([1.0, -1.0])
W = np.array([[2.0, 0.0, -1.0], [0.0, 2.0, 1.0]])      # 门：d = 2 → 3
V = np.array([[1.0, 1.0, 1.0], [-1.0, 1.0, 0.5]])      # 内容：d = 2 → 3
W2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])   # 3 → 2


def swiglu(x=X, w=W, v=V, w2=W2):
    gate = silu(x @ w)
    content = x @ v
    return gate, content, (gate * content) @ w2


def params(d: int, d_ff: int, gated: bool) -> int:
    return (3 if gated else 2) * d * d_ff


def swiglu_dff(d: int, multiple: int = 64) -> int:
    """8/3·d，再向上取到 64 的倍数（让矩阵乘法对齐 GPU 的 tensor core）。"""
    return int(math.ceil(8 * d / 3 / multiple) * multiple)


def main():
    print("第一步 三种激活")
    for x in (-3.0, -1.0, 0.0, 1.0, 3.0):
        print(f"  x = {x:+.0f}：ReLU {relu(x):.4f}，GELU {gelu(x):.4f}，SiLU {silu(x):.4f}")

    print("\n第二步 门 × 内容")
    g, c, y = swiglu()
    print(f"  xW = {(X @ W).tolist()} → 门 SiLU(xW) = {np.round(g, 4).tolist()}")
    print(f"  内容 xV = {c.tolist()}；门 ⊙ 内容 = {(np.round(g * c, 4) + 0.0).tolist()}；输出 = {np.round(y, 4).tolist()}")

    print("\n第三步 参数预算")
    for d in (512, 768, 4096):
        dff = swiglu_dff(d)
        print(f"  d = {d}：普通 FFN（d_ff = 4d）{params(d, 4 * d, False):,}；SwiGLU（d_ff = {dff}）{params(d, dff, True):,}")


if __name__ == "__main__":
    main()
