# 超越原版 Transformer

**在线阅读：<https://wikieden.github.io/beyond-transformer/>** · [讲义目录](https://wikieden.github.io/beyond-transformer/lessons/index.html) · [全部动画](https://wikieden.github.io/beyond-transformer/animations/index.html) · [复习卡片](https://wikieden.github.io/beyond-transformer/review.html)

> 在线阅读：<https://wikieden.github.io/beyond-transformer/> · 前作：《[从零读懂 Transformer](https://wikieden.github.io/transformer-visual/)》（24 章）
> 计划 51 章 · 8 段 · 每章一个动画（GIF + 可交互页面）· 第 1 章正文完成（练习进行中）

## 这本书讲什么

《从零读懂 Transformer》讲完了 2017 年的原版。可今天的大模型早就不是原版的样子了：训练要用 AdamW 和精心排过的学习率，LayerNorm 换成了 RMSNorm，位置编码换成了 RoPE，attention 加了 GQA、滑动窗口和 FlashAttention，FFN 变成了 MoE，有的模型干脆用 Mamba、线性注意力替掉一部分 attention，图片和视频也被切成小块送进同一套结构。这本书沿着这条路线，把 2017 年到现在每一处改动讲清楚：**改了什么、为什么改、代价是什么**。

- **适合谁**：读完前作（或已经能看懂 AIAYN §3 的公式），想知道 LLaMA、Qwen、DeepSeek、Mixtral、Mamba、ViT、DiT 这些模型到底在原版上改了什么的人。
- **读完能做什么**：能读懂现代大模型的架构论文和技术报告，能在自己写的 CS336 作业 1 模型上动手加 GQA、换位置编码、加 MoE、写分块注意力和线性注意力（每段结尾都有一个动手里程碑）。
- **怎么读**：段 ②（第 1–5 章）和段 A（第 6–11 章）先读，做完 CS336 作业 1 再往后读；段 B–F 基本按顺序；段 V（视觉）可以和 C–F 穿插。

## 讲解的特点

和前作一样的讲法，再加上"现代模型"需要的几样东西：

- **一次一小步**：白话类比 → 公式（每个符号说清楚怎么读）→ 小数字亲手算一遍。用到前作的知识时标出章号（如 T21），忘了可以点回去。
- **每章一个动画**：关键原理做成可交互页面，能单步、能拖滑块（比如拖序列长度看 KV cache 怎么涨、拖窗口大小看感受野怎么变宽）。
- **对照原文**：每章对应具体论文的章节和本地文本行号，引用原句；章节号没把握的先标"待核对"，写正文时核对。
- **讲清"为什么换"**：每个新零件都和它替换掉的旧零件对比（Post-LN → Pre-LN、正弦位置编码 → RoPE、dense FFN → MoE、softmax attention → 线性注意力），说明换来了什么、付出了什么。
- **学完就动手**：每段结尾对应一个里程碑（T-M0 到 T-M8），都在自己写的模型上改，用实验验证讲过的结论。
- **保留真实的学习过程**：练习与作答记录、复习卡片（编号 B01 起）、口诀、延伸阅读，格式和前作一致。

## 示例动画

本书的动画还没开始做，下面是前作的动画，本书会沿用同样的风格（▶ 播放、→ 单步、↺ 从头开始，有滑块的可以拖动参数）。每个都标了和本书哪一章衔接。

<table>
<tr>
<td width="33%" valign="top"><img src="https://wikieden.github.io/transformer-visual/animations/09-qkv.gif" width="100%" alt="Q、K、V：图书馆找书"><br><b>Q、K、V：图书馆找书</b><br>原版第 9 章<br><a href="https://wikieden.github.io/transformer-visual/lessons/09-qkv.html">讲义</a> · <a href="https://wikieden.github.io/transformer-visual/animations/09-qkv.html">▶ 交互</a></td>
<td width="33%" valign="top"><img src="https://wikieden.github.io/transformer-visual/animations/16-multi-clock.gif" width="100%" alt="多档时钟：位置编码"><br><b>多档时钟：位置编码</b><br>原版第 16 章，本书 RoPE 一章会接着用<br><a href="https://wikieden.github.io/transformer-visual/lessons/16-multi-clock.html">讲义</a> · <a href="https://wikieden.github.io/transformer-visual/animations/16-multi-clock.html">▶ 交互</a></td>
<td width="33%" valign="top"><img src="https://wikieden.github.io/transformer-visual/animations/19-layernorm.gif" width="100%" alt="LayerNorm 与 RMSNorm"><br><b>LayerNorm 与 RMSNorm</b><br>原版第 19 章，本书第 13 章展开<br><a href="https://wikieden.github.io/transformer-visual/lessons/19-layernorm.html">讲义</a> · <a href="https://wikieden.github.io/transformer-visual/animations/19-layernorm.html">▶ 交互</a></td>
</tr>
<tr>
<td width="33%" valign="top"><img src="https://wikieden.github.io/transformer-visual/animations/20-ffn.gif" width="100%" alt="FFN 与 SwiGLU 闸门"><br><b>FFN 与 SwiGLU 闸门</b><br>原版第 20 章，本书第 14 章复盘<br><a href="https://wikieden.github.io/transformer-visual/lessons/20-ffn.html">讲义</a> · <a href="https://wikieden.github.io/transformer-visual/animations/20-ffn.html">▶ 交互</a></td>
<td width="33%" valign="top"><img src="https://wikieden.github.io/transformer-visual/animations/23-complexity.gif" width="100%" alt="为什么 attention 是 n²"><br><b>为什么 attention 是 n²</b><br>原版第 23 章，本书段 E、F 的出发点<br><a href="https://wikieden.github.io/transformer-visual/lessons/23-complexity.html">讲义</a> · <a href="https://wikieden.github.io/transformer-visual/animations/23-complexity.html">▶ 交互</a></td>
<td width="33%" valign="top"><img src="https://wikieden.github.io/transformer-visual/animations/24-wrap-up.gif" width="100%" alt="数参数与 warmup 曲线"><br><b>数参数与 warmup 曲线</b><br>原版第 24 章，本书第 1–5 章接着讲训练<br><a href="https://wikieden.github.io/transformer-visual/lessons/24-wrap-up.html">讲义</a> · <a href="https://wikieden.github.io/transformer-visual/animations/24-wrap-up.html">▶ 交互</a></td>
</tr>
</table>

## 怎么排的

- **先补"模型怎么学会的"**（第 1–5 章）：原版那本只说过"训练时一点点调"，这一段讲清楚 loss（损失）、梯度、优化器、学习率，对应 CS336 a1 §4。
- **再补原理深化**（第 6–11 章），然后**动手做 CS336 a1**（里程碑 T-M0、T-M1）。
- **阶段 B–F 都在 a1 的模型上改**：每段讲完有一个里程碑（T-M2 到 T-M8）。里程碑的完成标准见 `transformer-deep-dive/notes/study-plan.md`，代码放 `transformer-deep-dive/solutions/`。
- **视觉延展 V**（第 44–51 章）可以和 C–F 穿插，也可以放到最后。

| 段 | 章 | 一句话 | 段末动手 |
|---|---|---|---|
| ② 模型怎么学会的 | 1–5 | loss、梯度、AdamW、学习率、训练循环 | 读懂 a1 §4–§5 |
| A 原理深化 | 6–11 | attention 的起源、伪代码对照、数 FLOPs、BERT、残差流与 induction head、读原版源码 | **T-M0** 做 a1 §3；**T-M1** 写 sizing.py |
| B 零件升级 | 12–20 | Pre-LN、RMSNorm、SwiGLU、RoPE、ALiBi / NoPE、长文本外推、训练稳定性 | T-M2、T-M3、T-M4 |
| C 现代 dense 配方 | 21–26 | LLaMA 配方、KV cache 与 GQA、滑窗、MLA、读 nanochat 源码 | T-M5 |
| D MoE | 27–30 | 路由、负载均衡、细粒度专家 | T-M6 |
| E 高效注意力 | 31–36 | online softmax、FlashAttention、稀疏与长序列 | T-M7 |
| F 超越注意力 | 37–43 | 线性注意力、RetNet、GLA、Mamba、DeltaNet、混合架构 | T-M8 |
| V 视觉 Transformer | 44–51 | ViT、CLIP、MAE、扩散、DiT、视觉接 LLM | 接 nvidia-cosmos3 |

---

---

本仓库由私有学习仓库里的脚本 `tools/publish-transformer/build.py` 自动生成，内容摘自讲义目录，不要直接修改这里的文件。
