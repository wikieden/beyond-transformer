"""第 47 章 MAE 与 DINO：遮挡、只在被遮块上的 MSE、DINO 交叉熵、EMA 老师。

讲义：beyond-transformer/notes/lessons/47-mae-dino.md
运行：uv run python ch47_mae_dino.py
"""

import numpy as np


def random_mask(n, ratio, seed=0):
    """MAE 随机遮挡：打乱 n 个位置，前面的留给 encoder，后面的遮掉。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_mask = int(round(n * ratio))
    return np.sort(perm[: n - n_mask]), np.sort(perm[n - n_mask :])


def shuffle_and_restore(tokens, seed=0, ratio=0.75):
    """MAE 的实现：打乱 → 砍掉后面 → 接上 mask token → 按原顺序还原。返回还原后的序列（被遮处为 None）。"""
    rng = np.random.default_rng(seed)
    n = len(tokens)
    perm = rng.permutation(n)
    keep = perm[: n - int(round(n * ratio))]
    seq = [tokens[i] for i in keep] + [None] * (n - len(keep))  # encoder 输出 + mask token
    order = list(keep) + [i for i in perm if i not in set(keep)]
    restored = [None] * n
    for pos, tok in zip(order, seq):
        restored[pos] = tok
    return restored


def mae_loss(x, pred):
    """MAE loss：目标先按这一块自己的平均、标准差归一化，再算均方误差。"""
    x = np.asarray(x, dtype=float)
    target = (x - x.mean()) / x.std()
    return target, ((np.asarray(pred) - target) ** 2).mean()


def softmax(v):
    """softmax（先减最大值）。"""
    e = np.exp(v - v.max())
    return e / e.sum()


def dino_loss(t, s, c, tau_t, tau_s):
    """DINO 式 3：老师输出减中心 c、除以 τt 做 softmax；学生除以 τs 做 softmax；交叉熵 H(Pt, Ps)。"""
    p_t = softmax((np.asarray(t, float) - c) / tau_t)
    p_s = softmax(np.asarray(s, float) / tau_s)
    return p_t, p_s, -(p_t * np.log(p_s)).sum()


def ema(theta_t, theta_s, lam):
    """老师 = λ·老师 + (1 − λ)·学生。"""
    return lam * theta_t + (1 - lam) * theta_s


def main():
    print("\n第一步 遮挡：196 块，遮 75%")
    keep, drop = random_mask(196, 0.75)
    print("留给 encoder：", len(keep), "块；遮掉：", len(drop), "块")
    print("注意力格子：", len(keep) ** 2, "，不遮时", 196**2, "，比值 1 /", 196**2 // len(keep) ** 2)

    print("\n第二步 打乱、接 mask token、还原（8 个 token 演示）")
    restored = shuffle_and_restore(list("ABCDEFGH"))
    print(["□" if t is None else t for t in restored])

    print("\n第三步 MAE loss：被遮的一块 [2, 4, 6, 8]，decoder 猜 [-1, -0.5, 0.5, 1]")
    target, loss = mae_loss([2, 4, 6, 8], [-1, -0.5, 0.5, 1])
    print("归一化目标：", target, " MSE =", round(loss, 3))

    print("\n第四步 DINO：老师 [2, 1, 0]，c = [1, 0, 0]，τt = 0.5；学生 [1, 2, 0]，τs = 1")
    p_t, p_s, h = dino_loss([2, 1, 0], [1, 2, 0], np.array([1, 0, 0]), 0.5, 1.0)
    print("Pt =", p_t, " Ps =", p_s, " H =", round(h, 3))
    print("锐化对比：[2, 1, 0] 温度 1 →", softmax(np.array([2, 1, 0.0])), "；温度 0.5 →", softmax(np.array([2, 1, 0.0]) / 0.5))

    print("\n第五步 EMA：老师 1.0，学生 2.0，λ = 0.996")
    print(round(ema(1.0, 2.0, 0.996), 4))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
