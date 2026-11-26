"""第 6 章 attention 的起源：RNN 编码器-解码器把整句压成一个向量的瓶颈；Bahdanau 的加性打分
e = vᵀ tanh(W s + U h) 和点积打分 e = sᵀh；softmax 成权重 α，加权求和得到上下文 c。

讲义：beyond-transformer/notes/lessons/06-bahdanau.md
运行：uv run python ch06_bahdanau.py
"""

import numpy as np

H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])   # 原文 3 个词的注释向量 h_1..h_3（编码器输出）
S = np.array([0.0, 2.0])                               # 解码器上一步的状态 s_{i−1}
W = np.array([[1.0, 0.0], [0.0, 1.0]])
U = np.array([[1.0, 0.0], [0.0, -1.0]])
V = np.array([1.0, 1.0])


def softmax(e):
    z = np.exp(e - e.max())
    return z / z.sum()


def additive_scores(s=S, h=H, w=W, u=U, v=V):
    """Bahdanau 附录 A.1.2（L813）：e_j = vᵀ tanh(W s + U h_j)；α 是式 (6) 的 softmax。"""
    return np.array([v @ np.tanh(w @ s + u @ hj) for hj in h])


def dot_scores(s=S, h=H):
    return h @ s


def context(scores, h=H):
    a = softmax(scores)
    return a, a @ h


def param_count(d_s, d_h, d_a):
    """加性打分要学 W（d_a × d_s）、U（d_a × d_h）、v（d_a）；点积打分什么都不用学（要求 d_s = d_h）。"""
    return d_a * d_s + d_a * d_h + d_a


def bottleneck(lengths=(5, 20, 80), d=4):
    """固定长度向量：不管句子多长都只有 d 个数，平均每个词分到 d/n 个数。"""
    return {n: d / n for n in lengths}


def main():
    print("第一步 瓶颈：整句压成一个 4 维向量")
    for n, r in bottleneck().items():
        print(f"  句长 {n:2d}：每个词平均只分到 {r:.2f} 个数")

    print("\n第二步 加性打分")
    e = additive_scores()
    a, c = context(e)
    print(f"  W s = {(W @ S).tolist()}；e = {np.round(e, 4).tolist()}；α = {np.round(a, 4).tolist()}；c = {np.round(c, 4).tolist()}")

    print("\n第三步 点积打分")
    e2 = dot_scores()
    a2, c2 = context(e2)
    print(f"  e = sᵀh = {e2.tolist()}；α = {np.round(a2, 4).tolist()}；c = {np.round(c2, 4).tolist()}")

    print("\n第四步 参数量与计算方式")
    print(f"  d_s = d_h = 1000、d_a = 1000（原文设定）：加性打分要学 {param_count(1000, 2000, 1000):,} 个参数（双向 RNN 的 h 是 2000 维）")
    print("  点积打分：0 个参数；n 个 query × m 个 key 一次矩阵乘法就算完")


if __name__ == "__main__":
    main()
