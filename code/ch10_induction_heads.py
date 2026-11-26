"""第 10 章 残差流与 induction head：每层把结果“加”进同一条残差流；手搭一个两层、只有注意力的小模型：
第 1 层“上一个词 head”把前一个 token 写进残差流的另一块子空间，第 2 层 induction head 用“当前词”去找
“前一个词是它”的位置，把那个位置的词复制出来：[A][B] … [A] → [B]。

讲义：beyond-transformer/notes/lessons/10-induction-heads.md
运行：uv run python ch10_induction_heads.py
权重都是手搭的，演示机制；训练一个小模型、看 induction head 自己长出来留给里程碑 T-M2。
"""

import numpy as np

TOK = ["<s>", "A", "B", "C", "D"]
SEQ = [0, 1, 2, 3, 4, 1]       # <s> A B C D A → 下一个应该是 B；<s> 是句首标记，位置 0 没有“上一个词”
V = len(TOK)


def softmax_rows(s):
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def embed(seq=SEQ):
    """残差流 10 维：前 5 维 = 当前 token 的 one-hot；后 5 维先空着，留给第 1 层写“上一个 token”。"""
    x = np.zeros((len(seq), 2 * V))
    x[np.arange(len(seq)), seq] = 1
    return x


def prev_token_head(x, beta=10.0):
    """第 1 层：位置 i 的分数只在 j = i − 1 处高（位置信息手写），值 = 前 5 维，写进后 5 维。"""
    n = len(x)
    s = np.full((n, n), -1e9)
    for i in range(n):
        for j in range(i + 1):
            s[i, j] = beta if j == i - 1 else 0.0
    a = softmax_rows(s)
    out = np.zeros_like(x)
    out[:, V:] = a @ x[:, :V]
    return a, out


def induction_head(x, beta=10.0):
    """第 2 层：query = 当前 token（前 5 维），key = 每个位置的“上一个 token”（后 5 维），值 = 那个位置的当前 token。"""
    n = len(x)
    q, k, v = x[:, :V], x[:, V:], x[:, :V]
    s = beta * q @ k.T
    s[np.triu(np.ones((n, n), bool), 1)] = -1e9
    a = softmax_rows(s)
    return a, a @ v


def run(seq=SEQ):
    x0 = embed(seq)
    a1, w1 = prev_token_head(x0)
    x1 = x0 + w1                    # 残差流：加进去，不覆盖
    a2, w2 = induction_head(x1)
    return x0, a1, x1, a2, w2


def main():
    print("第一步 残差流：每层往同一条主干上加东西")
    x0, a1, x1, a2, w2 = run()
    print(f"  序列 {' '.join(TOK[t] for t in SEQ)}；x0 最后一行 {x0[-1].astype(int).tolist()}（前 5 维 = A）")

    print("\n第二步 第 1 层：上一个词 head")
    print(f"  注意力（每行最大的位置）：{a1.argmax(1).tolist()}，第 5 行权重 {np.round(a1[5], 4).tolist()}")
    print(f"  x1 = x0 + 写入：位置 2（B）的残差流 {np.round(x1[2], 3).tolist()}（后 5 维 ≈ A：它前一个是 A）")

    print("\n第三步 第 2 层：induction head")
    print(f"  最后一个位置（A）的注意力 {np.round(a2[-1], 4).tolist()}：盯住位置 2（前一个词是 A 的那个位置）")
    print(f"  复制出来的向量 {np.round(w2[-1], 3).tolist()} → 预测 {TOK[int(w2[-1].argmax())]}")

    print("\n第四步 换一个序列：<s> C A D B C")
    _, _, _, a2b, w2b = run([0, 3, 1, 4, 2, 3])
    print(f"  最后一个 C 的注意力 {np.round(a2b[-1], 4).tolist()} → 预测 {TOK[int(w2b[-1].argmax())]}")


if __name__ == "__main__":
    main()
