"""第 24 章 大厂怎么选配置：LLaMA 3 用 IsoFLOPs 拟合 “最优 token 数 = A·C^α”，外推到 3.8e25 FLOPs 选出 405B / 15.6T；
token / 参数比从 Chinchilla 的约 20 一路涨到上千；FFN 宽度比、GQA 头数这些配置怎么读；长上下文分六段续训；
Qwen3 用 QK-norm 换掉 QKV bias。

讲义：beyond-transformer/notes/lessons/24-configs.md
运行：uv run python ch24_configs.py
"""

C_FLAG = 3.8e25


def optimal_tokens(c, a=0.29, alpha=0.53):
    """LLaMA 3 §3.2.1：N*(C) = A·C^α，N* 是最优训练 token 数（论文里这个 N 指 token，不是参数）。"""
    return a * c**alpha


def params_from(c, tokens):
    """训练 FLOPs ≈ 6 × 参数 × token（第 8 章）⇒ 参数 ≈ C / (6·token)。"""
    return c / (6 * tokens)


RUNS = {  # (参数, 训练 token)：公开报告里的数
    "Chinchilla 70B": (70e9, 1.4e12),
    "LLaMA 1 7B": (6.7e9, 1.0e12),
    "LLaMA 3 8B": (8e9, 15e12),
    "LLaMA 3 405B": (405e9, 15.6e12),
}

LLAMA3 = {  # 论文表 3（L322–L333）
    "8B": dict(layers=32, d=4096, ffn=14336, heads=32, kv=8),
    "70B": dict(layers=80, d=8192, ffn=28672, heads=64, kv=8),
    "405B": dict(layers=126, d=16384, ffn=53248, heads=128, kv=8),
}

QWEN3 = {"Qwen3-0.6B": (28, 16, 8, "是", "32K"), "Qwen3-8B": (36, 32, 8, "否", "128K"), "Qwen3-32B": (64, 64, 8, "否", "128K")}


def read_config(c):
    return {"FFN / d": c["ffn"] / c["d"], "每头维数": c["d"] // c["heads"], "Q 头 / KV 头": c["heads"] // c["kv"]}


def context_stages(start=8192, end=131072, stages=6):
    """长上下文续训：从 8K 分 6 段涨到 128K（每段乘同一个倍数，示意）。"""
    r = (end / start) ** (1 / (stages - 1))
    return [round(start * r**i) for i in range(stages)]


def main():
    print("第一步 LLaMA 3 的缩放律选型")
    for a, al in ((0.29, 0.53), (0.299, 0.537)):
        t = optimal_tokens(C_FLAG, a, al)
        print(f"  (A, α) = ({a}, {al})：C = 3.8e25 → 最优 token {t / 1e12:.2f}T，参数 ≈ C/(6·token) = {params_from(C_FLAG, t) / 1e9:.0f}B")
    print("  论文：402B 参数、16.55T token；实际训的是 405B、15.6T")

    print("\n第二步 token / 参数比")
    for k, (n, d) in RUNS.items():
        print(f"  {k}：{d / n:,.0f} 个 token / 参数，训练 FLOPs ≈ {6 * n * d:.2e}")

    print("\n第三步 读配置表")
    for k, c in LLAMA3.items():
        print(f"  LLaMA 3 {k}：{read_config(c)}")
    for k, (l, h, kv, tie, ctx) in QWEN3.items():
        print(f"  {k}：{l} 层，Q/KV 头 {h}/{kv}，共享嵌入 {tie}，上下文 {ctx}")

    print("\n第四步 长上下文分段续训（示意）")
    print(f"  6 段：{context_stages()}")


if __name__ == "__main__":
    main()
