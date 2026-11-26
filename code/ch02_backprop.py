"""第 2 章 梯度与反向传播：导数 = 斜率、链式法则、两层小网络的前向和反向、softmax + 交叉熵的梯度 p − y、梯度检查。

讲义：beyond-transformer/notes/lessons/02-backprop.md
运行：uv run python ch02_backprop.py
"""

import numpy as np


def slope(f, x, h=1e-4):
    """中心差分：(f(x + h) − f(x − h)) / 2h，导数的数值近似。"""
    return (f(x + h) - f(x - h)) / (2 * h)


def chain_example(x):
    """y = (3x + 1)²：外层 u² 的导数 2u，乘内层 3x + 1 的导数 3。"""
    u = 3 * x + 1
    return u**2, 2 * u * 3


X = np.array([1.0, 2.0])                       # 输入
W1 = np.array([[0.5, -1.0], [1.0, 0.5]])       # 第 1 层：h = relu(W1 x)
W2 = np.array([[1.0, -1.0], [0.5, 1.0], [-1.0, 0.0]])  # 第 2 层：o = W2 h，3 个类
Y = 1                                          # 正确类是第 1 类（从 0 数）


def softmax(o):
    e = np.exp(o - o.max())
    return e / e.sum()


def forward(x=X, w1=W1, w2=W2, y=Y):
    a = w1 @ x
    h = np.maximum(a, 0)
    o = w2 @ h
    p = softmax(o)
    return {"a": a, "h": h, "o": o, "p": p, "loss": float(-np.log(p[y]))}


def backward(x=X, w1=W1, w2=W2, y=Y):
    """从 loss 往回：do = p − onehot(y)；dW2 = do hᵀ；dh = W2ᵀ do；da = dh ⊙ [a > 0]；dW1 = da xᵀ。"""
    c = forward(x, w1, w2, y)
    do = c["p"].copy()
    do[y] -= 1
    dw2 = np.outer(do, c["h"])
    dh = w2.T @ do
    da = dh * (c["a"] > 0)
    dw1 = np.outer(da, x)
    return {"do": do, "dW2": dw2, "dh": dh, "da": da, "dW1": dw1}


def grad_check(eps=1e-5):
    """把 W1 每个元素各挪 ±eps，用数值斜率和反向传播的结果比较，返回最大相对误差。"""
    g = backward()["dW1"]
    num = np.zeros_like(W1)
    for i in range(2):
        for j in range(2):
            wp, wm = W1.copy(), W1.copy()
            wp[i, j] += eps
            wm[i, j] -= eps
            num[i, j] = (forward(w1=wp)["loss"] - forward(w1=wm)["loss"]) / (2 * eps)
    return float(np.max(np.abs(num - g) / np.maximum(1e-8, np.abs(num) + np.abs(g)))), num


def main():
    print("第一步 导数 = 斜率")
    for x in (1.0, 2.0):
        print(f"  f(x) = x² 在 x = {x}：数值斜率 {slope(lambda t: t * t, x):.4f}（公式 2x = {2 * x}）")

    print("\n第二步 链式法则")
    y, d = chain_example(1.0)
    print(f"  y = (3x + 1)²，x = 1：y = {y}，dy/dx = 2·4·3 = {d}，数值斜率 {slope(lambda t: chain_example(t)[0], 1.0):.4f}")

    print("\n第三步 两层小网络的前向")
    c = forward()
    for k in ("a", "h", "o", "p"):
        print(f"  {k} = {(np.round(c[k], 4) + 0.0).tolist()}")
    print(f"  loss = −log p[1] = {c['loss']:.4f}")

    print("\n第四步 反向传播")
    b = backward()
    for k in ("do", "dW2", "dh", "da", "dW1"):
        print(f"  {k} = {(np.round(b[k], 4) + 0.0).tolist()}")

    print("\n第五步 梯度检查")
    err, num = grad_check()
    print(f"  数值梯度 dW1 = {(np.round(num, 4) + 0.0).tolist()}；最大相对误差 {err:.2e}")


if __name__ == "__main__":
    main()
