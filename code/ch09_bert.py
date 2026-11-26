"""第 9 章 encoder-only 的 BERT：双向 mask 和因果 mask；MLM 遮 15%（其中 80% 换 [MASK]、10% 换随机词、10% 不变）；
loss 只在被选中的位置算；每个 token 带来的训练信号 MLM 只有 15%，而 decoder-only 每个位置都算。

讲义：beyond-transformer/notes/lessons/09-bert.md
运行：uv run python ch09_bert.py
"""

import numpy as np

SENT = ["[CLS]", "my", "dog", "is", "hairy", "and", "he", "likes", "to", "play", "fetch", "[SEP]"]
VOCAB = ["the", "cat", "apple", "run", "blue", "happy"]


def masks(n=5):
    """双向：全 1；因果：下三角（第 i 个只看 0..i）。"""
    return np.ones((n, n), int), np.tril(np.ones((n, n), int))


def mlm_corrupt(tokens, seed=3, rate=0.15):
    """选中 15% 的位置（不选 [CLS]/[SEP]）；选中的 80% 换 [MASK]、10% 换随机词、10% 保持原样。"""
    rng = np.random.default_rng(seed)
    cand = [i for i, t in enumerate(tokens) if not t.startswith("[")]
    k = max(1, round(rate * len(cand)))
    chosen = sorted(rng.choice(cand, k, replace=False).tolist())
    out, how = list(tokens), {}
    for i in chosen:
        r = rng.uniform()
        if r < 0.8:
            out[i], how[i] = "[MASK]", "换成 [MASK]"
        elif r < 0.9:
            out[i], how[i] = str(rng.choice(VOCAB)), "换成随机词"
        else:
            how[i] = "保持原样"
    return out, chosen, how


def corrupt_stats(n=100_000, seed=0):
    """大量 token 上统计：被选中、换 [MASK]、换随机、不变各占多少。"""
    rng = np.random.default_rng(seed)
    sel = rng.uniform(size=n) < 0.15
    r = rng.uniform(size=n)
    return {"被选中": sel.mean(), "换 [MASK]": (sel & (r < 0.8)).mean(), "换随机词": (sel & (r >= 0.8) & (r < 0.9)).mean(), "不变": (sel & (r >= 0.9)).mean()}


def mlm_loss(probs_at_chosen):
    """只在被选中的位置取 −log p(原词) 的平均。"""
    return float(np.mean([-np.log(p) for p in probs_at_chosen]))


def signal_per_sequence(n=512):
    """一条长 n 的序列能给多少个“猜词”的训练信号。"""
    return {"MLM（15%）": int(round(0.15 * n)), "decoder-only（每个位置）": n - 1}


def main():
    print("第一步 双向 vs 因果")
    bi, ca = masks()
    print(f"  双向：每行 {bi.sum(1).tolist()} 个能看的位置\n  因果：每行 {ca.sum(1).tolist()} 个")

    print("\n第二步 MLM 遮词")
    out, chosen, how = mlm_corrupt(SENT)
    print(f"  原句：{' '.join(SENT)}")
    print(f"  遮后：{' '.join(out)}")
    for i in chosen:
        print(f"  位置 {i}（{SENT[i]}）：{how[i]}")
    print("  10 万个 token 的统计：" + "，".join(f"{k} {v:.2%}" for k, v in corrupt_stats().items()))

    print("\n第三步 loss 只在被选中的位置算")
    print(f"  两个选中位置的原词概率 0.6、0.3：MLM loss = (−log 0.6 − log 0.3)/2 = {mlm_loss([0.6, 0.3]):.4f}")

    print("\n第四步 为什么现在都用 decoder-only")
    for k, v in signal_per_sequence().items():
        print(f"  序列长 512，{k}：{v} 个训练信号")


if __name__ == "__main__":
    main()
