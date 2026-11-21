"""第 1 章 交叉熵 loss：−log p、式 16 取平均、式 17 从 logits 直接算、减最大值、困惑度。

讲义：beyond-transformer/notes/lessons/01-cross-entropy.md
运行：uv run python ch01_cross_entropy.py
"""

import numpy as np

LOGITS = np.array([0.7, -1.7, 1.7, -1.6, 2.0, -1.5, 2.9, -1.6])  # 前作第 22 章那 8 个分数，bites 是第 7 个（下标 6）


def nll(p):
    """一次猜测的 loss：−log(正确词的概率)。"""
    return -np.log(p)


def dataset_loss(probs):
    """式 16：所有句子、所有位置的 −log p 取平均。"""
    return float(np.mean([nll(p) for p in probs]))


def ce_from_logits(o, target):
    """式 17 化简：−o[target] + log Σ e^o，先减最大值防溢出。"""
    o = np.asarray(o, float)
    m = o.max()
    return float(-o[target] + m + np.log(np.exp(o - m).sum()))


def perplexity(losses):
    """式 18：exp(平均 loss)。"""
    return float(np.exp(np.mean(losses)))


def main():
    print("\n第三步 −log p：模型 B、A、C")
    for name, p in (("B", 0.9), ("A", 0.537), ("C", 0.05)):
        print(f"模型 {name}：p = {p}，扣分 {nll(p):.3f}，对照 1 − p = {1 - p:.3f}")
    print("log(0.1 × 0.01) =", round(np.log(0.1 * 0.01), 3), " log 0.1 + log 0.01 =", round(np.log(0.1) + np.log(0.01), 3))

    print("\n第四步 式 16：两句话 4 次猜测")
    probs = [0.5, 0.8, 0.4, 0.9]
    print("平均 loss =", round(dataset_loss(probs), 3))
    print("式 17：bites 的 loss =", round(ce_from_logits(LOGITS, 6), 3))
    print("[1000, 999, 998]，正确词是第一个：", round(ce_from_logits([1000, 999, 998], 0), 3))

    print("\n第五步 困惑度")
    print("两句话：", round(perplexity([nll(p) for p in probs]), 2))
    print("4 个词里瞎猜：", round(perplexity([np.log(4)]), 2))


if __name__ == "__main__":
    main()
