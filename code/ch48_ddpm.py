"""第 48 章 DDPM：前向加噪、一步跳到第 t 步、训练猜噪声、生成的一步。

讲义：beyond-transformer/notes/lessons/48-ddpm.md
运行：uv run python ch48_ddpm.py
"""

import numpy as np


def schedule(T=1000, beta_1=1e-4, beta_T=0.02):
    """论文的设置：β 线性从 0.0001 涨到 0.02；α = 1 − β；ᾱ = α 连乘。"""
    betas = np.linspace(beta_1, beta_T, T)
    alphas = 1 - betas
    return betas, alphas, np.cumprod(alphas)


def forward_step(x, beta, eps):
    """式 2：先缩 √(1 − β)，再加 √β · ε。"""
    return np.sqrt(1 - beta) * x + np.sqrt(beta) * eps


def jump(x0, t, eps, alpha_bar):
    """式 4：x_t = √ᾱ_t · x0 + √(1 − ᾱ_t) · ε（t 从 1 数）。"""
    ab = alpha_bar[t - 1]
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps


def train_example(x0, t, eps, eps_hat, alpha_bar):
    """Algorithm 1 的一次：造 x_t，算猜噪声的平方误差，并用猜的噪声反推 x0。"""
    ab = alpha_bar[t - 1]
    xt = jump(x0, t, eps, alpha_bar)
    x0_hat = (xt - np.sqrt(1 - ab) * eps_hat) / np.sqrt(ab)
    return xt, (eps - eps_hat) ** 2, x0_hat


def reverse_step(xt, t, eps_hat, z, betas, alphas, alpha_bar):
    """Algorithm 2 的一步：减一小部分猜出的噪声、放大回来、再加 σ_t·z（σ_t² = β_t）。"""
    i = t - 1
    mean = (xt - betas[i] / np.sqrt(1 - alpha_bar[i]) * eps_hat) / np.sqrt(alphas[i])
    return mean + np.sqrt(betas[i]) * z


def main():
    betas, alphas, alpha_bar = schedule()

    print("\n第一步 前向两步：x0 = 0.8，β = 0.01、0.02，ε = 1.0、−1.0")
    x1 = forward_step(0.8, 0.01, 1.0)
    x2 = forward_step(x1, 0.02, -1.0)
    print(round(x1, 3), round(x2, 3))

    print("\n第二步 一步跳：ᾱ_2 = 0.99 × 0.98 =", round(0.99 * 0.98, 4))
    for t in (1, 100, 250, 500, 750, 1000):
        ab = alpha_bar[t - 1]
        print(f"t = {t:4d}：ᾱ = {ab:.5f}，原图系数 {np.sqrt(ab):.3f}，噪声系数 {np.sqrt(1 - ab):.3f}")

    print("\n第三步 训练一次：x0 = 0.8，t = 500，ε = −0.5，网络猜 −0.4")
    xt, loss, x0_hat = train_example(0.8, 500, -0.5, -0.4, alpha_bar)
    print(f"x_500 = {xt:.3f}，loss = {loss:.3f}，反推 x0 ≈ {x0_hat:.3f}")

    print("\n第四步 生成一步：从 x_500 到 x_499，z 取 0")
    print(round(reverse_step(xt, 500, -0.4, 0.0, betas, alphas, alpha_bar), 4))


if __name__ == "__main__":
    main()
