"""第 51 章 视觉接 LLM：LLaVA 投影、拼进序列、BLIP-2 压缩、Qwen2-VL 的 token 数和 M-RoPE 编号。

讲义：beyond-transformer/notes/lessons/51-vision-llm.md
运行：uv run python ch51_vision_llm.py
"""

import numpy as np

Z = np.array([[1, 0, 2], [0, 1, 1.0]])  # 两个小块的图片特征（编的，3 维）
W = np.array([[1, 0], [0, 1], [0.5, 0.5]])  # 投影：3 维 → 词向量 2 维


def build_sequence(img_tokens, text_emb):
    """图片 token 排在文字 token 前面，拼成一条序列。"""
    return np.concatenate([img_tokens, text_emb])


def qwen2vl_tokens(H, W, patch=14, merge=2):
    """Qwen2-VL：按原尺寸切 14 × 14，相邻 2 × 2 合并成 1 个，再加首尾两个特殊 token。"""
    n = (H // patch) * (W // patch)
    return n // (merge * merge) + 2


def mrope_ids(segments):
    """M-RoPE 编号。segments 是列表：("text", 个数) 或 ("image", 行, 列)。返回每个 token 的 (时间, 高, 宽)。"""
    ids, start = [], 0
    for seg in segments:
        if seg[0] == "text":
            for i in range(seg[1]):
                ids.append((start + i,) * 3)
        else:
            _, rows, cols = seg
            for r in range(rows):
                for c in range(cols):
                    ids.append((start, start + r, start + c))
        start = max(max(i) for i in ids) + 1  # 下一段从最大编号 + 1 开始
    return ids


def main():
    print("\n第二步 LLaVA：H_v = Z_v · W")
    H_v = Z @ W
    print(H_v)
    text = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # "这张 图 里" 的词向量（编的）
    print("拼成序列（前 2 行是图片 token）：\n", build_sequence(H_v, text))

    print("\n第三步 BLIP-2：257 × 1024 =", 257 * 1024, "→ 32 × 768 =", 32 * 768)

    print("\n第四步 Qwen2-VL token 数")
    for h, w in [(224, 224), (448, 448), (448, 224), (1344, 896)]:
        print(f"{h} × {w} →", qwen2vl_tokens(h, w))

    print("\n第五步 M-RoPE：文字 3 个 + 图片 2 × 2 + 文字 1 个")
    for tok, i in zip(["描述", "这张", "图", "左上", "右上", "左下", "右下", "。"],
                      mrope_ids([("text", 3), ("image", 2, 2), ("text", 1)])):
        print(tok, i)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
