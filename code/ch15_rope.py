"""第 15 章 RoPE：要 q_m·k_n 只依赖 m − n，就把 q、k 按位置转一个角度；二维的例子；
d 维拆成 d/2 个平面、每个平面一个频率 θ_i = 10000^{−2i/d}；按元素乘 cos、sin 的高效写法；远距离衰减。

讲义：beyond-transformer/notes/lessons/15-rope.md
运行：uv run python ch15_rope.py
a1 的 rope（PyTorch 模块）留给自己写；五种位置编码的外推对比是里程碑 T-M3。
"""

import numpy as np


def rot2(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


Q = np.array([1.0, 0.5])
K = np.array([0.8, -0.3])
THETA = 0.3


def score_2d(m, n, q=Q, k=K, theta=THETA):
    """位置 m 的 q 转 mθ，位置 n 的 k 转 nθ，再点积。"""
    return float(rot2(q, m * theta) @ rot2(k, n * theta))


def freqs(d=8, base=10000.0):
    return base ** (-np.arange(0, d, 2) / d)


def rope(x, pos, base=10000.0):
    """高效写法（RoFormer §3.4.2 式 34）：x ⊙ cos + rotate_half(x) ⊙ sin。相邻两维 (x₀, x₁)、(x₂, x₃)… 各成一个平面。"""
    d = len(x)
    th = np.repeat(freqs(d, base), 2) * pos
    xr = np.empty_like(x)
    xr[0::2], xr[1::2] = -x[1::2], x[0::2]
    return x * np.cos(th) + xr * np.sin(th)


def rope_matrix(x, pos, base=10000.0):
    """笨写法：拼一个分块对角的旋转矩阵再乘，用来核对。"""
    d = len(x)
    R = np.zeros((d, d))
    for i, t in enumerate(freqs(d, base)):
        c, s = np.cos(pos * t), np.sin(pos * t)
        R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = [[c, -s], [s, c]]
    return R @ x


def decay_curve(d=128, base=10000.0, dists=(0, 1, 10, 100, 1000, 10000)):
    """RoFormer §3.4.3：q、k 全是 1 时，(1/(d/2))·|Σ_i e^{i·rθ_i}| 随相对距离 r 的变化（上界的直观版）。"""
    th = freqs(d, base)
    return {r: float(np.abs(np.exp(1j * r * th).sum()) / len(th)) for r in dists}


def wavelengths(d=128, base=10000.0):
    """每个平面转一圈要多少个位置：2π / θ_i。"""
    return 2 * np.pi / freqs(d, base)


def main():
    print("第一步 二维：只依赖 m − n")
    for m, n in ((1, 0), (5, 4), (10, 9), (3, 0), (7, 4)):
        print(f"  m = {m:2d}、n = {n}（差 {m - n}）：q_m·k_n = {score_2d(m, n):.6f}")

    print("\n第二步 d 维：每两维一个平面，频率越来越慢")
    print(f"  d = 8：θ = {np.round(freqs(8), 6).tolist()}")
    wl = wavelengths()
    print(f"  d = 128：转一圈要 {wl[0]:.2f} 个位置（最快）……{wl[-1]:,.0f} 个位置（最慢）")

    print("\n第三步 高效写法和旋转矩阵一样")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    a, b = rope(x, 3), rope_matrix(x, 3)
    print(f"  位置 3：{np.round(a, 4).tolist()}；和矩阵写法最多差 {np.abs(a - b).max():.1e}")

    print("\n第四步 远距离衰减（q、k 全 1 时的平均相干程度）")
    for r, v in decay_curve().items():
        print(f"  相对距离 {r:5d}：{v:.4f}")


if __name__ == "__main__":
    main()
