"""第 17 章 训练时只见过 L 个 token，怎么读 s·L 个：位置插值（PI）把位置压回 m/s；NTK-aware 改底数；
YaRN 的 NTK-by-parts 按每个平面的波长分别处理，再加注意力温度 √(1/t) = 0.1 ln s + 1。

讲义：beyond-transformer/notes/lessons/17-long-context.md
运行：uv run python ch17_long_context.py
在小模型上训练、比较 PI / YaRN 推理期缩放的 loss（里程碑 T-M3）留给自己做。
"""

import math

import numpy as np

D, BASE, L_TRAIN = 128, 10000.0, 4096


def theta(d=D, base=BASE):
    return base ** (-np.arange(0, d, 2) / d)


def wavelength(d=D, base=BASE):
    return 2 * np.pi / theta(d, base)


def angle_range(m_max, th):
    """位置从 0 到 m_max，最慢那个平面转过的角度（弧度）。"""
    return m_max * th[-1]


def ntk_base(s, d=D, base=BASE):
    """NTK-aware：b′ = b · s^{d/(d−2)}，让最慢的平面也正好压慢 s 倍，最快的几乎不动。"""
    return base * s ** (d / (d - 2))


def by_parts(s, d=D, base=BASE, L=L_TRAIN, alpha=1.0, beta=32.0):
    """YaRN 式 (17)–(18)：r = L / λ（训练长度里转了几圈）。r < α：完全插值（θ/s）；r > β：不动；中间线性过渡。"""
    th = theta(d, base)
    r = L / wavelength(d, base)
    gamma = np.clip((r - alpha) / (beta - alpha), 0, 1)
    return (1 - gamma) * th / s + gamma * th, r, gamma


def yarn_temp(s):
    """YaRN 式 (15)：√(1/t) = 0.1 ln s + 1。"""
    return 0.1 * math.log(s) + 1


def main():
    s = 8
    th = theta()
    print(f"第一步 问题：训练 {L_TRAIN}，要读 {s * L_TRAIN}（s = {s}）")
    print(f"  最慢平面在训练长度内转了 {angle_range(L_TRAIN, th):.3f} 弧度；读到 {s * L_TRAIN} 时转到 {angle_range(s * L_TRAIN, th):.3f}（没见过的角度）")

    print("\n第二步 PI：位置除以 s")
    print(f"  最慢平面：{angle_range(s * L_TRAIN, th / s):.3f} 弧度（回到见过的范围）；最快平面相邻两个位置的角度差 {th[0]:.3f} → {th[0] / s:.3f}（分辨率掉了 s 倍）")

    print("\n第三步 NTK-aware：改底数")
    b2 = ntk_base(s)
    th2 = theta(base=b2)
    print(f"  底数 10000 → {b2:,.0f}；最快平面 θ 不变（{th2[0]:.3f}），最慢平面 {th[-1]:.2e} → {th2[-1]:.2e}（÷ {th[-1] / th2[-1]:.2f}）")

    print("\n第四步 YaRN：按波长分段 + 温度")
    th3, r, g = by_parts(s)
    for i in (0, 16, 24, 32, 40, 48, 63):
        kind = "不动" if g[i] == 1 else ("完全插值" if g[i] == 0 else "过渡")
        print(f"  第 {i:2d} 个平面：波长 {wavelength()[i]:9.1f}，训练长度里转 r = {r[i]:8.2f} 圈 → {kind}（θ 乘 {th3[i] / th[i]:.3f}）")
    print(f"  注意力温度：s = 8 时 √(1/t) = {yarn_temp(8):.4f}，s = 32 时 {yarn_temp(32):.4f}")


if __name__ == "__main__":
    main()
