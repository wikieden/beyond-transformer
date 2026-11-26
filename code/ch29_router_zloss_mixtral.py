"""第 29 章 MoE 训练稳定：router logit 一大，bf16 的舍入误差就被 exp 放大；router z-loss = (1/B)·Σ(log Σ_j e^{x_j})²，
系数 c_z = 0.001，把 router logit 压小；Mixtral 8×7B 的参数都花在哪。

讲义：beyond-transformer/notes/lessons/29-router-zloss-mixtral.md
运行：uv run python ch29_router_zloss_mixtral.py
"""

import numpy as np


def to_bf16(x):
    """把 float32 的低 16 位抹掉（按最近舍入），模拟 bfloat16：只剩 8 位有效数字（7 位尾数 + 隐含的 1）。"""
    a = np.asarray(x, np.float32).view(np.uint32)
    a = (a + 0x7FFF + ((a >> 16) & 1)) & 0xFFFF0000
    return a.view(np.float32).astype(float)


def bf16_error(logits):
    """router logit 存成 bf16 后的绝对误差，以及 exp 之后的相对误差。"""
    l = np.asarray(logits, float)
    b = to_bf16(l)
    return np.abs(b - l), np.abs(np.exp(b - l) - 1)


def router_z_loss(logits):
    """ST-MoE 式 (5)：L_z = (1/B) Σ_i (log Σ_j e^{x_ij})²。"""
    x = np.atleast_2d(logits)
    m = x.max(1, keepdims=True)
    lse = (m + np.log(np.exp(x - m).sum(1, keepdims=True))).ravel()
    return float((lse**2).mean())


def mixtral_breakdown(d=4096, d_ff=14336, n_exp=8, k=2, layers=32, vocab=32000, n_kv=8, d_head=128):
    attn = layers * (2 * d * d + 2 * d * n_kv * d_head)
    experts = layers * n_exp * 3 * d * d_ff
    router = layers * d * n_exp
    emb = 2 * vocab * d
    active = attn + router + emb + layers * k * 3 * d * d_ff
    return {"注意力": attn, "专家": experts, "router": router, "嵌入 + 输出": emb}, active


def main():
    print("第一步 bf16 舍入误差随 logit 变大")
    for lg in (0.37, 3.7, 37.3, 73.7):
        ab, rel = bf16_error([lg])
        print(f"  logit {lg}：bf16 存成 {to_bf16([lg])[0]}，误差 {ab[0]:.4f}，exp 之后相对误差 {rel[0]:.2%}")

    print("\n第二步 router z-loss（c_z = 0.001）")
    for name, lg in (("小 logit", [0.5, -0.2, 0.1, 0.3]), ("大 logit", [30.5, 29.8, 10.0, 5.0])):
        z = router_z_loss(lg)
        print(f"  {name} {lg}：L_z = {z:.4f}，乘 c_z 后 {0.001 * z:.4f}")

    print("\n第三步 Mixtral 8×7B 的参数都花在哪")
    parts, active = mixtral_breakdown()
    total = sum(parts.values())
    for k, v in parts.items():
        print(f"  {k}：{v / 1e9:.2f}B（{v / total:.1%}）")
    print(f"  总 {total / 1e9:.1f}B，每个 token 激活 {active / 1e9:.1f}B")


if __name__ == "__main__":
    main()
