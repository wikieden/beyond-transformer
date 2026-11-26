"""第 11 章 读原版源码（The Annotated Transformer）：SublayerConnection 其实是先 norm（和论文的 Post-LN 不同）；
它的 LayerNorm 用无偏标准差、eps 加在根号外；label smoothing 用 KL 散度和平滑后的目标分布；
学习率 rate() = d^{-0.5}·min(step^{-0.5}, step·warmup^{-1.5})。

讲义：beyond-transformer/notes/lessons/11-annotated-transformer.md
运行：uv run python ch11_annotated_transformer.py
"""

import numpy as np


def post_ln(x, sub):
    """论文：LayerNorm(x + Sublayer(x))。"""
    return layer_norm_torch(x + sub(x))


def pre_ln(x, sub):
    """Annotated 的 SublayerConnection：x + Sublayer(LayerNorm(x))。"""
    return x + sub(layer_norm_torch(x))


def layer_norm_annotated(x, eps=1e-6):
    """源码 L315–L328：(x − mean) / (std + eps)，std 是无偏的（除以 n − 1）。"""
    return (x - x.mean()) / (x.std(ddof=1) + eps)


def layer_norm_torch(x, eps=1e-5):
    """nn.LayerNorm：(x − mean) / √(var + eps)，var 是有偏的（除以 n）。"""
    return (x - x.mean()) / np.sqrt(x.var() + eps)


def smoothed_target(size=5, target=2, smoothing=0.1, padding_idx=0):
    """源码 L1160–L1166：先每格填 smoothing/(size − 2)，正确格填 1 − smoothing，padding 格清零。"""
    d = np.full(size, smoothing / (size - 2))
    d[target] = 1 - smoothing
    d[padding_idx] = 0
    return d


def kl_loss(pred, target_dist):
    """nn.KLDivLoss(reduction='sum')：Σ q (log q − log p)，q = 0 的格不算。"""
    m = target_dist > 0
    return float((target_dist[m] * (np.log(target_dist[m]) - np.log(pred[m]))).sum())


def confidence_curve(xs=range(1, 100), smoothing=0.1):
    """源码 L1232 的例子：预测 [0, x/d, 1/d, 1/d, 1/d]（d = x + 3），正确类 1。x 越大越“自信”。"""
    q = smoothed_target(5, 1, smoothing)
    out = []
    for x in xs:
        d = x + 3
        p = np.array([1e-12, x / d, 1 / d, 1 / d, 1 / d])
        out.append((x, x / d, kl_loss(p, q)))
    return out


def rate(step, model_size=512, factor=1.0, warmup=4000):
    """源码 L1058：AIAYN 式 (3)。"""
    step = max(step, 1)
    return factor * model_size**-0.5 * min(step**-0.5, step * warmup**-1.5)


def main():
    print("第一步 SublayerConnection：源码是先 norm")
    x = np.array([1.0, 2.0, 3.0, 4.0])
    sub = lambda v: 3 * v  # noqa: E731  一个让数变大的“子层”
    print(f"  x = {x.tolist()}，子层 = 乘 3")
    print(f"  论文 Post-LN：{np.round(post_ln(x, sub), 4).tolist()}（输出被 norm 住）")
    print(f"  源码 Pre-LN： {np.round(pre_ln(x, sub), 4).tolist()}（主干上的 x 原样保留）")

    print("\n第二步 源码的 LayerNorm 和 nn.LayerNorm 不完全一样")
    print(f"  [1, 2, 3, 4]：源码 {np.round(layer_norm_annotated(x), 4).tolist()}；nn.LayerNorm {np.round(layer_norm_torch(x), 4).tolist()}")

    print("\n第三步 label smoothing")
    q = smoothed_target()
    print(f"  词表 5、padding 0、正确类 2、smoothing 0.1：目标分布 {np.round(q, 4).tolist()}")
    for p in ([0.01, 0.05, 0.9, 0.02, 0.02], [0.001, 0.001, 0.996, 0.001, 0.001]):
        print(f"  预测 {p}：KL loss {kl_loss(np.array(p), q):.4f}")
    curve = confidence_curve()
    best = min(curve, key=lambda r: r[2])
    print(f"  正确类概率从 0.25 涨到 0.97：loss 在 x = {best[0]}（概率 {best[1]:.3f}）最小，再自信 loss 反而变大（x = 99：{curve[-1][2]:.4f}）")

    print("\n第四步 学习率 rate()（d = 512、warmup = 4000）")
    for s in (1, 1000, 4000, 16000, 100000):
        print(f"  step {s:6d}：{rate(s):.2e}")


if __name__ == "__main__":
    main()
