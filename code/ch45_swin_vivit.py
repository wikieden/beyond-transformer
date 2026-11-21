"""第 45 章 Swin 与 ViViT：窗口注意力、错开窗口、patch merging、视频切小管子。

讲义：beyond-transformer/notes/lessons/45-swin-vivit.md
运行：uv run python ch45_swin_vivit.py
"""

import numpy as np


def omega(h, w, C, M):
    """Swin 式 1、2：全局 MSA 和窗口 W-MSA 的计算量（不算 softmax）。"""
    proj = 4 * h * w * C * C
    return {"proj": proj, "global_attn": 2 * (h * w) ** 2 * C, "window_attn": 2 * M * M * h * w * C}


def score_cells(side, M):
    """side × side 的特征图：全局注意力和 M × M 窗口注意力各要算多少个分数格子。"""
    n = side * side
    windows = (side // M) ** 2
    return {"global": n * n, "window": windows * (M * M) ** 2}


def window_id(r, c, M, shift=0):
    """小块 (r, c) 落在哪个窗口：窗口往右下挪 shift，等于小块坐标减 shift。"""
    r, c = r - shift, c - shift
    return (r // M, c // M)


def patch_merge(x):
    """2 × 2 个相邻块首尾拼接：h × w × C → h/2 × w/2 × 4C。"""
    h, w, C = x.shape
    x = x.reshape(h // 2, 2, w // 2, 2, C)
    return x.transpose(0, 2, 1, 3, 4).reshape(h // 2, w // 2, 4 * C)


def video_tokens(T, H, W, t, h, w):
    """ViViT tubelet：三个方向各切几段、一共多少 token、Model 1 和 Model 2 的分数格子数。"""
    nt, nh, nw = T // t, H // h, W // w
    n = nt * nh * nw
    return {"nt": nt, "nh": nh, "nw": nw, "tokens": n, "model1": n * n, "model2": nt * (nh * nw) ** 2 + nt**2}


def main():
    print("\n第一步 计算量：h = w = 56，C = 96，M = 7")
    o = omega(56, 56, 96, 7)
    print(o)
    print(f"全局合计 {(o['proj'] + o['global_attn']) / 1e8:.2f} 亿，窗口合计 {(o['proj'] + o['window_attn']) / 1e8:.2f} 亿")
    print("注意力那一项差", o["global_attn"] // o["window_attn"], "倍")

    print("\n第二步 分数格子：8 × 8，M = 4")
    print(score_cells(8, 4))

    print("\n第三步 错开窗口：(3, 3) 和 (4, 4)")
    for shift in (0, 2):
        print(f"shift = {shift}：", window_id(3, 3, 4, shift), window_id(4, 4, 4, shift))

    print("\n第四步 patch merging：8 × 8 × 3 → ?")
    x = np.arange(8 * 8 * 3, dtype=float).reshape(8, 8, 3)
    print(patch_merge(x).shape)

    print("\n第五步 视频：32 帧 224 × 224，tubelet 2 × 16 × 16")
    print(video_tokens(32, 224, 224, 2, 16, 16))


if __name__ == "__main__":
    main()
