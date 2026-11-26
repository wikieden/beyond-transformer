"""第 5 章 一次完整训练：从一长串 token 里切 batch（输入和目标错开一位）、每步做什么、算 token 账、
checkpoint 要存什么（只存参数、不存优化器状态，续训就和不断的训练对不上）、看哪几条曲线。

讲义：beyond-transformer/notes/lessons/05-training-loop.md
运行：uv run python ch05_training_loop.py
a1 的 data_loading、checkpointing、training_together 三个题留给自己写；这里用一个两参数的小模型演示流程。
"""

import numpy as np

STREAM = np.array([5, 9, 2, 7, 1, 8, 3, 6, 4, 0, 2, 9])   # 一长串 token（整份语料拼起来）


def windows(stream, starts, m):
    """每个起点 s：输入 x = stream[s : s+m]，目标 y = stream[s+1 : s+m+1]（错开一位）。"""
    x = np.stack([stream[s : s + m] for s in starts])
    y = np.stack([stream[s + 1 : s + m + 1] for s in starts])
    return x, y


def token_budget(batch, ctx, steps):
    return batch * ctx * steps


def make_data(seed=0, n=200):
    """玩具回归：y = 2x + 1 + 噪声，前 160 个训练、后 40 个验证。"""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 2 * x + 1 + 0.1 * rng.normal(size=n)
    return (x[:160], y[:160]), (x[160:], y[160:])


def train(steps, state=None, lr=0.1, beta=0.9, batch=16, seed=1, start=0):
    """带动量的 SGD 训练 w、b。state = (w, b, 动量 vw, vb, 随机数生成器)；返回新 state 和每步的日志。"""
    (xt, yt), (xv, yv) = make_data()
    if state is None:
        state = (0.0, 0.0, 0.0, 0.0, np.random.default_rng(seed))
    w, b, vw, vb, rng = state
    log = []
    for t in range(start, start + steps):
        idx = rng.integers(0, len(xt), batch)
        err = w * xt[idx] + b - yt[idx]
        gw, gb = float(2 * (err * xt[idx]).mean()), float(2 * err.mean())
        vw, vb = beta * vw + gw, beta * vb + gb
        w, b = w - lr * vw, b - lr * vb
        val = float(((w * xv + b - yv) ** 2).mean())
        log.append({"step": t + 1, "train": float((err**2).mean()), "val": val, "gnorm": float(np.hypot(gw, gb))})
    return (w, b, vw, vb, rng), log


def resume_demo():
    """不断地训 60 步，对比：训 30 步存档，完整续训 / 只存参数（动量清零）续训。"""
    full, _ = train(60)
    half, _ = train(30)
    w, b, vw, vb, rng = half
    rng_state = rng.bit_generator.state
    r1 = np.random.default_rng()
    r1.bit_generator.state = rng_state
    good, _ = train(30, state=(w, b, vw, vb, r1), start=30)
    r2 = np.random.default_rng()
    r2.bit_generator.state = rng_state
    bad, _ = train(30, state=(w, b, 0.0, 0.0, r2), start=30)
    return {"不断训": full[:2], "完整存档续训": good[:2], "只存参数续训": bad[:2]}


def main():
    print("第一步 切 batch：输入和目标错开一位")
    x, y = windows(STREAM, [1, 6], 4)
    print(f"  token 串 {STREAM.tolist()}，起点 [1, 6]、长度 4：\n  x = {x.tolist()}\n  y = {y.tolist()}")
    print(f"  长度 12 的串、窗口 4：可选起点 0 到 {len(STREAM) - 4 - 1}，共 {len(STREAM) - 4} 个")

    print("\n第二步 token 账")
    for b, c, s in ((32, 256, 5000), (64, 512, 10000)):
        print(f"  batch {b} × 上下文 {c} × {s} 步 = {token_budget(b, c, s):,} 个 token")

    print("\n第三步 训练循环与曲线（y = 2x + 1 的小模型）")
    (w, b, *_), log = train(60)
    for r in log[::15] + [log[-1]]:
        print(f"  第 {r['step']:2d} 步：train {r['train']:.4f}，val {r['val']:.4f}，梯度范数 {r['gnorm']:.4f}")
    print(f"  学到 w = {w:.3f}、b = {b:.3f}")

    print("\n第四步 checkpoint 要存什么")
    for k, (w, b) in resume_demo().items():
        print(f"  {k}：w = {w:.6f}，b = {b:.6f}")


if __name__ == "__main__":
    main()
