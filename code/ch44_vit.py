"""第 44 章 ViT：图片切成小块，当成一句话。

讲义：beyond-transformer/notes/lessons/44-vit.md
运行：uv run python ch44_vit.py
"""

import numpy as np

IMG = np.array([[1, 2, 0, 1], [3, 4, 1, 0], [0, 0, 5, 6], [1, 1, 7, 8]], dtype=float)  # 讲义里编的 4 × 4 灰度图
E = np.array([[0.25, 1], [0.25, 1], [0.25, -1], [0.25, -1]])  # 第 1 列算平均亮度，第 2 列算上半减下半
CLS = np.zeros(2)  # [class] 向量，一开始设成 0
POS = np.array([[0, 0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])  # 编的位置编码，N + 1 = 5 行


def patchify(img, P):
    """把图片按 P × P 切块，每块按行展平，从左到右、从上到下排好。"""
    H, W = img.shape[:2]
    return np.array([img[i : i + P, j : j + P].reshape(-1) for i in range(0, H, P) for j in range(0, W, P)])


def seq_stats(size, P, C=3):
    """size × size 的图切成 P × P 小块：块数 N、每块几个数、加 [class] 后序列长度、注意力分数表格子数。"""
    N = (size // P) ** 2
    return {"N": N, "per_patch": P * P * C, "seq": N + 1, "scores": (N + 1) ** 2}


def embed(patches, E, cls, pos):
    """式 1：每块乘 E，最前面放 [class]，再加位置编码。"""
    return np.vstack([cls, patches @ E]) + pos


def hard_distill_loss(p, y, y_t):
    """DeiT 式 3：真实标签和老师标签的交叉熵各占一半。"""
    return 0.5 * -np.log(p[y]) + 0.5 * -np.log(p[y_t])


def main():
    print("\n第一步 切小块：4 × 4 的图，P = 2")
    patches = patchify(IMG, 2)
    print(patches)
    for P in (32, 16, 14, 8):
        print(f"224 × 224 彩色图，P = {P}：", seq_stats(224, P))

    print("\n第二步 线性投影：每块乘 E")
    print(patches @ E)
    print("加 [class] 和位置编码后的 z0（5 × 2）：")
    print(embed(patches, E, CLS, POS))

    print("\n第五步 硬蒸馏：学生概率 [0.5, 0.1, 0.4]，真实标签 猫(0)，老师说 狐(2)")
    p = np.array([0.5, 0.1, 0.4])
    print(f"loss = {hard_distill_loss(p, 0, 2):.3f}")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
