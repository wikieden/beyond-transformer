"""第 19 章 巨大激活与 attention sink：softmax 的权重必须加起来是 1，一个“这一步没什么要看的”头只好把权重
停在第一个 token 上（它的 value 很小，等于什么都不加）；滑窗推理把第一个 token 踢出去，权重被迫摊到别的 token 上，
输出就变了；StreamingLLM 留住开头几个 token 就修好了。

讲义：beyond-transformer/notes/lessons/19-attention-sink.md
运行：uv run python ch19_attention_sink.py
在自己训的小模型里观察 attention sink 是里程碑 T-M2 的一部分。
"""

import numpy as np

N, W, D = 24, 8, 4


def toy(seed=0, n=N, d=D):
    """位置 0 是 sink：分数高（巨大激活让它和所有 query 都对得上）、value 接近 0；其余 token 分数小、value 随机。"""
    rng = np.random.default_rng(seed)
    scores = rng.normal(0, 0.5, n)
    scores[0] = 6.0
    values = rng.normal(0, 1, (n, d))
    values[0] = 0.02 * values[0]
    return scores, values


def attend(scores, values, keep):
    s = scores[keep]
    w = np.exp(s - s.max())
    w /= w.sum()
    return w, w @ values[keep]


def compare(seed=0, n_sink=4):
    """最后一个位置：全看 / 只看最近 W 个 / 最近 W 个 + 开头 n_sink 个。"""
    s, v = toy(seed)
    full_w, full = attend(s, v, np.arange(N))
    _, win = attend(s, v, np.arange(N - W, N))
    keep = np.r_[np.arange(n_sink), np.arange(N - W, N)]
    _, sink = attend(s, v, keep)
    return {
        "sink 拿到的权重": float(full_w[0]),
        "全看的输出范数": float(np.linalg.norm(full)),
        "滑窗（踢掉开头）与全看的差": float(np.linalg.norm(win - full)),
        f"滑窗 + 开头 {n_sink} 个与全看的差": float(np.linalg.norm(sink - full)),
    }


def massive(seed=0, d=64):
    """巨大激活：某个 token 在少数几维上的值是其余的上千倍。"""
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 1, (6, d))
    h[0, 7], h[0, 41] = 900.0, -650.0
    med = np.median(np.abs(h))
    return float(np.abs(h).max()), float(med)


def main():
    print("第一步 巨大激活")
    mx, med = massive()
    print(f"  6 个 token × 64 维：最大的 |值| {mx:.0f}，中位数 {med:.3f}，相差 {mx / med:,.0f} 倍（只在第 0 个 token 的两维上）")

    print("\n第二步 attention sink 与滑窗")
    for k, val in compare().items():
        print(f"  {k}：{val:.4f}")


if __name__ == "__main__":
    main()
