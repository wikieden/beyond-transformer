"""第 49 章 Latent Diffusion 与 Rectified Flow：潜空间大小、交叉注意力、直线路径、Euler 生成。

讲义：beyond-transformer/notes/lessons/49-latent-diffusion-flow.md
运行：uv run python ch49_latent_diffusion_flow.py
"""

import numpy as np

K = np.array([[0, 0], [1, 0], [0, 1.0]])  # "一只"、"红色"、"猫"：两维分别是 颜色、动物（编的）
V = K.copy()
Q = np.array([[2, 0], [0, 2.0]])  # 位置 A 问颜色，位置 B 问是什么


def latent_size(side, f, c):
    """side × side 的彩色图压成 (side/f) × (side/f) × c：返回边长、个数、比像素小几倍。"""
    h = side // f
    return {"h": h, "numbers": h * h * c, "ratio": side * side * 3 / (h * h * c)}


def softmax(x):
    """按行 softmax。"""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def cross_attention(Q, K, V):
    """交叉注意力：Q 来自图像位置，K、V 来自文字 token。"""
    w = softmax(Q @ K.T / np.sqrt(K.shape[1]))
    return w, w @ V


def rf_example(x0, x1, t, v_hat):
    """Rectified Flow 的一次训练：直线上的位置、目标速度、平方误差。"""
    xt = t * x1 + (1 - t) * x0
    target = x1 - x0
    return xt, target, (target - v_hat) ** 2


def euler(z, v, dt):
    """Euler 一步：按当前速度走 dt。"""
    return z + dt * v


def perfect_velocity(z, t, data):
    """完美速度场 E[X1 − X0 | X_t = z]：噪声是标准高斯，数据是给定的几个点。"""
    data = np.atleast_1d(np.asarray(data, float))
    logw = -((z - t * data) ** 2) / (2 * (1 - t) ** 2)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return float((w * (data - z)).sum() / (1 - t))


def generate(z0, data, n_steps):
    """从噪声 z0 出发，用完美速度场走 n_steps 步 Euler，到 t = 1。"""
    z = z0
    for i in range(n_steps):
        z = euler(z, perfect_velocity(z, i / n_steps, data), 1 / n_steps)
    return z


def main():
    print("\n第一步 压缩：512 × 512 × 3，f = 8，4 个通道")
    print(latent_size(512, 8, 4))

    print("\n第二步 交叉注意力：文字 一只 / 红色 / 猫")
    w, out = cross_attention(Q, K, V)
    print("权重：\n", w)
    print("输出：\n", out)

    print("\n第三步 RF 训练：X0 = −1（噪声），X1 = 0.8（图），t = 0.25，网络猜 1.5")
    xt, target, loss = rf_example(-1, 0.8, 0.25, 1.5)
    print(f"X_t = {xt:.2f}，目标速度 {target:.1f}，loss = {loss:.2f}")

    print("\n第四步 Euler：从 −0.55 按速度 1.8 走完剩下的 0.75")
    print(round(euler(-0.55, 1.8, 0.75), 3))
    print("两个数据点 ±1：z = 0、t = 0.5 处的完美速度 =", perfect_velocity(0.0, 0.5, [-1, 1]))
    for n in (1, 2, 10):
        print(f"从 z0 = 0.3 出发走 {n:2d} 步 →", round(generate(0.3, [-1, 1], n), 3))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
