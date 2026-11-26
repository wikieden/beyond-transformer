"""第 7 章 把 Transformer 写成伪代码：Formal Algorithms 的 Algorithm 1–14 和前作章节、a1 函数的对照；
Algorithm 3（单 query 注意力）算一个小例子；Algorithm 10（DTransformer）每一行的形状（论文是“列 = token”，a1 是“行 = token”）。

讲义：beyond-transformer/notes/lessons/07-formal-algorithms.md
运行：uv run python ch07_formal_algorithms.py
只跟踪形状、算一个单 query 的例子；完整的 a1 §3 前向留给 T-M0。
"""

import numpy as np

MAP = [  # (算法, 做什么, 前作章, a1 里的题)
    ("Algorithm 1", "Token embedding：查表取一列", "T22", "embedding"),
    ("Algorithm 2", "Positional embedding：位置也查表（或用 sin/cos）", "T14–T16", "rope（a1 换成 RoPE）"),
    ("Algorithm 3", "单 query 注意力：一个 token 看全部上下文", "T2–T9", "softmax、scaled_dot_product_attention"),
    ("Algorithm 4", "整段注意力 + Mask", "T11、T12、T17", "scaled_dot_product_attention"),
    ("Algorithm 5", "多头注意力", "T13", "multihead_self_attention"),
    ("Algorithm 6", "LayerNorm", "T19", "rmsnorm（a1 换成 RMSNorm）"),
    ("Algorithm 7", "Unembedding：乘回词表再 softmax", "T22", "transformer_lm 的输出层"),
    ("Algorithm 10", "DTransformer：GPT 式的整体前向", "T21、T24", "transformer_block、transformer_lm"),
    ("Algorithm 13", "DTraining：交叉熵 + 梯度下降", "第 1–4 章", "cross_entropy、adamw、training_together"),
    ("Algorithm 14", "DInference：逐个采样生成", "T22", "decoding"),
]


def single_query(e, ctx, wq, wk, wv):
    """Algorithm 3：q = Wq e；k_t = Wk e_t；v_t = Wv e_t；α = softmax(qᵀk / √d_attn)；输出 Σ α_t v_t。"""
    q = wq @ e
    k = ctx @ wk.T
    v = ctx @ wv.T
    s = k @ q / np.sqrt(len(q))
    a = np.exp(s - s.max())
    a = a / a.sum()
    return s, a, a @ v


E_CUR = np.array([1.0, 0.0])
CTX = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
WQ = np.array([[2.0, 0.0], [0.0, 2.0]])
WK = np.eye(2)
WV = np.array([[1.0, 0.0], [0.0, -1.0]])


def shape_trace(V=10, L=2, H=2, de=4, dmlp=16, n=5):
    """Algorithm 10 每一行之后的形状：论文写法（d × n，一列一个 token） vs a1 写法（n × d，一行一个 token）。"""
    rows = [("行 2 嵌入：W_e[:, x[t]] + W_p[:, t]", (de, n)), ("行 3 拼成 X", (de, n))]
    for l in range(1, L + 1):
        rows += [(f"第 {l} 层 行 5 layer_norm", (de, n)), (f"第 {l} 层 行 6 X + MHAttention（{H} 头，每头 {de // H} 维）", (de, n)),
                 (f"第 {l} 层 行 7 layer_norm", (de, n)), (f"第 {l} 层 行 8 MLP 中间（GELU）", (dmlp, n)), (f"第 {l} 层 行 8 X + MLP", (de, n))]
    rows += [("行 10 最后的 layer_norm", (de, n)), ("行 11 W_u X → logits", (V, n)), ("行 11 softmax → P", (V, n))]
    return [(name, shp, shp[::-1]) for name, shp in rows]


def main():
    print("第一步 算法 ↔ 前作章节 ↔ a1")
    for a, what, t, a1 in MAP:
        print(f"  {a:12s} {what}  | {t} | {a1}")

    print("\n第二步 Algorithm 3：单 query 注意力")
    s, a, out = single_query(E_CUR, CTX, WQ, WK, WV)
    print(f"  q = {(WQ @ E_CUR).tolist()}；分数 qᵀk/√2 = {np.round(s, 4).tolist()}；α = {np.round(a, 4).tolist()}；输出 = {np.round(out, 4).tolist()}")

    print("\n第三步 Algorithm 10 的形状（V = 10、L = 2、H = 2、d_e = 4、d_mlp = 16、序列 5）")
    for name, paper, a1 in shape_trace():
        print(f"  {name:42s} 论文 {paper}  a1 {a1}")


if __name__ == "__main__":
    main()
