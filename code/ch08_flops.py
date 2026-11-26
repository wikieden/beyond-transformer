"""第 8 章 数 FLOPs：一次矩阵乘法 2mnk；前向每个参数每个 token 约 2 次，反向约 4 次 → 训练 ≈ 6ND；
注意力的 QKᵀ 和 AV 不对应参数，每层每个 token 多 4·n·d 次；它和 24·d² 的比值是 n / (6d)。

讲义：beyond-transformer/notes/lessons/08-flops.md
运行：uv run python ch08_flops.py
只算一个玩具配置；给定配置算参数量、FLOPs、KV cache 并对官方数的 sizing.py 留给 T-M1。
"""

import numpy as np


def matmul_flops(m, n, k):
    """(m × k) @ (k × n)：输出 m·n 个数，每个数 k 次乘、k 次加。"""
    return 2 * m * n * k


def count_matmul(m=2, n=3, k=4):
    """真的数一遍：乘法和加法各多少次。"""
    a, b = np.ones((m, k)), np.ones((k, n))
    mul = add = 0
    out = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += a[i, t] * b[t, j]
                mul += 1
                add += 1
            out[i, j] = s
    return mul, add


TOY = dict(d=64, L=2, n=128, V=1000, f=256)   # 宽 64、2 层、序列 128、词表 1000、FFN 中间 256


def layer_forward_per_token(d, n, f):
    """一层、一个 token 的前向 FLOPs（不含 softmax 等逐元素运算）。"""
    return {
        "Q、K、V 投影": 2 * 3 * d * d,
        "输出投影": 2 * d * d,
        "FFN 两层": 2 * 2 * d * f,
        "注意力 QKᵀ": 2 * n * d,
        "注意力 AV": 2 * n * d,
    }


def toy_summary(d=64, L=2, n=128, V=1000, f=256):
    per = layer_forward_per_token(d, n, f)
    params_layer = 4 * d * d + 2 * d * f
    params = L * params_layer + V * d          # 嵌入和输出层共享
    fwd = L * sum(per.values()) + 2 * V * d    # 最后乘回词表
    attn = L * (per["注意力 QKᵀ"] + per["注意力 AV"])
    return {"参数": params, "前向/token": fwd, "2×参数": 2 * params, "注意力部分": attn, "比例": attn / fwd}


def attn_ratio(n, d):
    """f = 4d 时：每层参数矩阵 12d²，前向 24d²；注意力 4nd。比值 = n / (6d)。"""
    return 4 * n * d / (24 * d * d)


def main():
    print("第一步 一次矩阵乘法")
    mul, add = count_matmul()
    print(f"  (2 × 4) @ (4 × 3)：乘 {mul} 次、加 {add} 次，合计 {mul + add} = 2·2·3·4 = {matmul_flops(2, 3, 4)}")

    print("\n第二步 一层一个 token 的前向（d = 64、n = 128、f = 256）")
    per = layer_forward_per_token(64, 128, 256)
    for k, v in per.items():
        print(f"  {k}：{v:,}")
    print(f"  合计 {sum(per.values()):,}")

    print("\n第三步 前向 ≈ 2 × 参数，训练 ≈ 6 × 参数 × token")
    s = toy_summary()
    print(f"  玩具模型参数 {s['参数']:,}；每个 token 前向 {s['前向/token']:,} FLOPs，2 × 参数 = {s['2×参数']:,}")
    print(f"  训练 100 万 token ≈ 6 × {s['参数']:,} × 1e6 = {6 * s['参数'] * 1e6:.3e} FLOPs")

    print("\n第四步 注意力什么时候不能忽略：比值 n / (6d)")
    print(f"  玩具模型（n = 128、d = 64）：注意力占前向 {s['比例']:.1%}")
    for n, d in ((2048, 4096), (32768, 4096), (128, 64)):
        print(f"  n = {n:6d}、d = {d}：注意力 / 参数矩阵 = {attn_ratio(n, d):.3f}")


if __name__ == "__main__":
    main()
