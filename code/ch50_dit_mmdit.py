"""第 50 章 DiT 与 MM-DiT：token 数、adaLN、adaLN-Zero 的门、联合注意力、logit-normal 抽 t。

讲义：beyond-transformer/notes/lessons/50-dit-mmdit.md
运行：uv run python ch50_dit_mmdit.py
"""

import numpy as np


def n_tokens(I, p):
    """潜空间 I × I 按 p × p 切块：T = (I/p)²。"""
    return (I // p) ** 2


def layer_norm(x):
    """普通 LayerNorm：减平均、除标准差（演示用，不加 eps）。"""
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()


def ada_ln(x, gamma, beta):
    """adaLN：γ、β 由条件算出，γ ⊙ LN(x) + β。"""
    return np.asarray(gamma) * layer_norm(x) + np.asarray(beta)


def gated_residual(x, f_out, alpha):
    """adaLN-Zero：x + α ⊙ 子层输出；α = 0 时原样输出。"""
    return np.asarray(x, float) + alpha * np.asarray(f_out, float)


def joint_quadrants(n_img, n_txt):
    """MM-DiT 联合注意力的分数表大小，以及四块：图看图、图看文、文看图、文看文。"""
    n = n_img + n_txt
    return {"table": (n, n), "img_img": (n_img, n_img), "img_txt": (n_img, n_txt),
            "txt_img": (n_txt, n_img), "txt_txt": (n_txt, n_txt)}


def logit_normal_t(u):
    """logit-normal：t = sigmoid(u)。"""
    return 1 / (1 + np.exp(-np.asarray(u, float)))


def main():
    print("\n第一步 token 数：潜空间 32 × 32 × 4")
    for p in (8, 4, 2):
        print(f"p = {p}：T =", n_tokens(32, p))

    print("\n第二步 adaLN：x = [1, 2, 3]")
    x = np.array([1, 2, 3.0])
    print("LN(x) =", layer_norm(x))
    print("t 很大：", ada_ln(x, [0.5] * 3, [0, 0, 0.1]))
    print("t 很小：", ada_ln(x, [1.5] * 3, [0.2, 0, 0]))

    print("\n第三步 adaLN-Zero 的门：子层输出 [0.5, −0.5, 1]")
    for a in (0.0, 0.2):
        print(f"α = {a}：", gated_residual(x, [0.5, -0.5, 1], a))

    print("\n第五步 MM-DiT：4 个图像 token + 2 个文字 token")
    print(joint_quadrants(4, 2))
    print("logit-normal：u = −2..2 → t =", logit_normal_t([-2, -1, 0, 1, 2]))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
