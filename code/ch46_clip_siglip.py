"""第 46 章 CLIP 与 SigLIP：N × N 分数表、对称交叉熵、zero-shot、sigmoid 损失。

讲义：beyond-transformer/notes/lessons/46-clip-siglip.md
运行：uv run python ch46_clip_siglip.py
"""

import numpy as np

I = np.array([[1, 0], [0, 1], [0.6, 0.8]])  # 狗图、猫图、鸟图（编的，已归一化）
T = np.array([[0.96, 0.28], [0.28, 0.96], [0.6, 0.8]])  # "一只狗"、"一只猫"、"一只鸟"


def softmax(x, axis):
    """沿 axis 做 softmax（先减最大值，第 1 章）。"""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def sigmoid(x):
    """σ(x) = 1 / (1 + e^(−x))。"""
    return 1 / (1 + np.exp(-x))


def clip_loss(S, scale):
    """CLIP 对称交叉熵：按行（图找文）、按列（文找图）各算对角线的 −log，再平均。"""
    logits = scale * S
    row = -np.log(np.diag(softmax(logits, axis=1)))
    col = -np.log(np.diag(softmax(logits, axis=0)))
    return {"row_p": np.diag(softmax(logits, axis=1)), "row": row.mean(), "col": col.mean(),
            "loss": (row.mean() + col.mean()) / 2}


def zero_shot(img, texts, scale):
    """zero-shot：图片向量和每个类名句子的向量比余弦，放大后 softmax。"""
    return softmax(scale * (texts @ np.asarray(img)), axis=0)


def siglip_loss(S, t, b):
    """SigLIP：每格 −log σ(z · (t·cos + b))，z 对角线 +1、其余 −1；总和除以 |B|。"""
    n = len(S)
    z = 2 * np.eye(n) - 1
    cell = -np.log(sigmoid(z * (t * S + b)))
    return {"logits": t * S + b, "cell": cell, "loss": cell.sum() / n}


def main():
    print("\n第二步 分数表 S = I · Tᵀ")
    S = I @ T.T
    print(S)
    print("放大 10 倍：")
    print(10 * S)

    print("\n第三步 对称交叉熵")
    r = clip_loss(S, 10)
    print("按行 softmax 的对角线：", r["row_p"])
    print(f"按行 {r['row']:.3f}，按列 {r['col']:.3f}，总 {r['loss']:.3f}")

    print("\n第四步 zero-shot：新照片 [0.96, 0.28]")
    print("狗、猫、鸟：", zero_shot([0.96, 0.28], T, 10))

    print("\n第五步 SigLIP：t = 10，b = −10")
    s = siglip_loss(S, 10, -10)
    print(s["logits"])
    print(s["cell"])
    print(f"loss = {s['loss']:.3f}")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
