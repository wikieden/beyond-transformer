"""第 3 章 优化器：SGD → Momentum → Adam → AdamW。在一个"窄长碗" f(x, y) = ½(10x² + y²) 上比较走法；
Adam 的偏差修正；AdamW 和 "Adam + L2" 的区别；每个参数要存几份状态。

讲义：beyond-transformer/notes/lessons/03-optimizers.md
运行：uv run python ch03_optimizers.py
这里只用标量和二维向量演示公式；a1 的 AdamW（torch.optim.Optimizer 子类）留给自己写（T-M0 之后的 §4.3）。
"""

import numpy as np

A = np.array([10.0, 1.0])          # f(x, y) = ½(10x² + y²)：x 方向陡、y 方向缓
START = np.array([1.0, 1.0])


def f(p):
    return float(0.5 * (A * p * p).sum())


def grad(p):
    return A * p


def run(method: str, steps: int = 50, lr: float = 0.15, beta: float = 0.9, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
    """返回走过的点。method：sgd / momentum / adam。"""
    p, v, m, s = START.copy(), np.zeros(2), np.zeros(2), np.zeros(2)
    path = [p.copy()]
    for t in range(1, steps + 1):
        g = grad(p)
        if method == "sgd":
            p = p - lr * g
        elif method == "momentum":
            v = beta * v + g
            p = p - lr * v
        else:
            m = b1 * m + (1 - b1) * g
            s = b2 * s + (1 - b2) * g * g
            mh, sh = m / (1 - b1**t), s / (1 - b2**t)
            p = p - lr * mh / (np.sqrt(sh) + eps)
        path.append(p.copy())
    return np.array(path)


def adam_first_step(g: float, b1: float = 0.9, b2: float = 0.999):
    """第 1 步：不修正时 m = 0.1g、v = 0.001g²；修正后 m̂ = g、v̂ = g²，步长 = lr · sign(g)。"""
    m, v = (1 - b1) * g, (1 - b2) * g * g
    return m, v, m / (1 - b1), v / (1 - b2)


def decay_compare(theta: float = 1.0, g: float = 0.0, lr: float = 0.01, wd: float = 0.1, steps: int = 100):
    """梯度恒为 0（只看衰减）。AdamW：θ ← θ − lr·λ·θ；Adam + L2：把 λθ 加进梯度，再被 √v̂ 除掉。"""
    tw, tl = theta, theta
    m = v = 0.0
    for t in range(1, steps + 1):
        tw = tw - lr * wd * tw                  # AdamW：衰减和梯度分开
        gl = g + wd * tl                        # Adam + L2：衰减混进梯度
        m = 0.9 * m + 0.1 * gl
        v = 0.999 * v + 0.001 * gl * gl
        tl = tl - lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return tw, tl


def state_bytes(n_params: int):
    """fp32 训练：参数 4 B + 梯度 4 B + Adam 的 m、v 各 4 B。"""
    return {"参数": 4 * n_params, "梯度": 4 * n_params, "m": 4 * n_params, "v": 4 * n_params}


def main():
    print("第一步 SGD 在窄长碗里")
    for lr in (0.05, 0.15, 0.21):
        pth = run("sgd", lr=lr)
        print(f"  lr = {lr}：50 步后 f = {f(pth[-1]):.3g}，x 方向的系数 1 − 10·lr = {1 - 10 * lr:+.2f}")

    print("\n第二步 Momentum")
    for name in ("sgd", "momentum"):
        pth = run(name, lr=0.02)
        print(f"  {name}（lr = 0.02）：50 步后 f = {f(pth[-1]):.5f}，点 {np.round(pth[-1], 4).tolist()}")

    print("\n第三步 Adam")
    m, v, mh, vh = adam_first_step(4.0)
    print(f"  g = 4：不修正 m = {m:.2f}、v = {v:.4f}；修正后 m̂ = {mh:.2f}、v̂ = {vh:.2f}，第一步走 lr·m̂/√v̂ = lr × {mh / np.sqrt(vh):.2f}")
    pth = run("adam", lr=0.1)
    print(f"  Adam（lr = 0.1）：前 3 步 {np.round(pth[1:4], 3).tolist()}，50 步后 f = {f(pth[-1]):.5f}")

    print("\n第四步 AdamW vs Adam + L2（梯度为 0，只看衰减，100 步）")
    for wd in (0.1, 0.01):
        tw, tl = decay_compare(wd=wd)
        print(f"  λ = {wd}：AdamW θ = {tw:.4f}；Adam + L2 θ = {tl:.4f}")

    print("\n第五步 每个参数存几份")
    b = state_bytes(100_000_000)
    print(f"  1 亿参数：" + "，".join(f"{k} {v / 1e9:.1f} GB" for k, v in b.items()) + f"，合计 {sum(b.values()) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
