"""Transformer 04 — Transformer 模块：残差连接、层归一化与前馈网络.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python tf04_block.py <out_dir>

Real computations: a 32-layer random tanh stack with and without residual connections + LayerNorm
(output scale and sensitivity to a tiny input perturbation, layer by layer); a LayerNorm worked
example; parameter counts of attention (4d^2) and the feed-forward network (8d^2).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED

# 1. deep stacks: plain tanh vs residual + LayerNorm (Pre-LN)
D, L = 64, 32
rng = np.random.default_rng(0)
X = rng.standard_normal((512, D))
WS = [rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(L)]
EPS = rng.standard_normal(X.shape) * 1e-4


def ln(x):
    return (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)


def run(x, res):
    out = [x]
    for W in WS:
        x = x + np.tanh(ln(x) @ W) if res else np.tanh(x @ W)
        out.append(x)
    return out


PA, PA2, RB, RB2 = run(X, False), run(X + EPS, False), run(X, True), run(X + EPS, True)
STD_P = np.array([a.std() for a in PA])
STD_R = np.array([a.std() for a in RB])
SENS_P = np.array([np.linalg.norm(b - a) / np.linalg.norm(EPS) for a, b in zip(PA, PA2)])
SENS_R = np.array([np.linalg.norm(b - a) / np.linalg.norm(EPS) for a, b in zip(RB, RB2)])
print(f"plain std {STD_P[-1]:.3f} sens {SENS_P[-1]:.3f} | residual std {STD_R[-1]:.2f} sens {SENS_R[-1]:.2f}")
assert 0.1 < STD_P[-1] < 0.15 and 0.12 < SENS_P[-1] < 0.2
assert SENS_R.min() >= 0.99 and 3 < STD_R[-1] < 4

# 2. LayerNorm worked example
V = np.array([2.0, 4.0, 6.0, 8.0])
MU, SD = V.mean(), V.std()
VN = (V - MU) / SD
assert MU == 5 and abs(SD - 5 ** 0.5) < 1e-12

# 3. parameters per block (weights only, d_ff = 4d)
DM = 512
P_ATT, P_FFN = 4 * DM * DM, 2 * DM * 4 * DM
FRAC_FFN = P_FFN / (P_ATT + P_FFN)
assert abs(FRAC_FFN - 2 / 3) < 1e-12

SCENES = [
    ("title", [
        "前几期我们讲了注意力和位置编码。这一期把它们组装成一个完整的变换器模块，看看另外三个关键部件，残差连接、层归一化和前馈网络。",
    ]),
    ("block", [
        "一个变换器模块包含两个子层，第一个是多头注意力，负责在词元之间交换信息；第二个是前馈网络，对每个词元单独做非线性变换。",
        "每个子层的输入先做层归一化，子层的输出再加回原来的输入，这就是残差连接。大模型就是把这样的模块重复堆叠几十层。",
    ]),
    ("deep", [
        "为什么需要残差连接呢？我们做一个实验，把三十二层随机初始化的网络堆起来，看信号怎样逐层传递。",
        f"没有残差连接时，输出的标准差从一降到{dec_zh(STD_P[-1])}，对输入施加一个微小的扰动，传到最后一层只剩{pct_zh(SENS_P[-1])}。反向传播时梯度沿同一条路径往回走，前面的层就很难学到东西。",
        f"加上残差连接以后，每一层都至少保留一条直通的路径，扰动在三十二层中始终没有衰减，到最后放大到了{dec_zh(SENS_R[-1], 1)}倍。不过数值的尺度也一路变大，标准差涨到了{dec_zh(STD_R[-1], 1)}，这正是需要层归一化的原因。",
    ]),
    ("layernorm", [
        "层归一化对每个词元的向量单独处理，减去均值，再除以标准差，最后乘上可学习的缩放，再加上偏移。",
        f"比如向量二，四，六，八，均值是{int_zh(MU)}，标准差是根号五，约为{dec_zh(SD)}，归一化以后变成负一点三四，负零点四五，零点四五和一点三四。",
        "无论残差流里的数值长到多大，送进每个子层的输入都保持在标准的尺度。这种先归一化的写法叫做前置层归一化，从第二代生成式预训练模型开始被广泛采用，训练更稳定。",
    ]),
    ("ffn", [
        "最后是前馈网络。它先把每个词元的向量升到四倍的维度，经过一个非线性激活函数，再降回原来的维度。",
        f"算一下参数量，注意力有四个维度乘维度的矩阵，前馈网络的两个矩阵加起来是八倍的维度平方，所以前馈网络约占一个模块参数的{pct_zh(FRAC_FFN)}，是模型存储知识的重要地方。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，残差连接让信号和梯度可以直接穿过很多层，深层网络才能训练。",
        "第二，层归一化让每个子层的输入保持稳定的尺度。",
        "第三，前馈网络对每个词元单独做变换，占了模块的大部分参数。",
        "下一期，我们讨论掩码，以及编码器和解码器三种不同的架构。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 4, "Transformer 模块", "残差连接 · 层归一化 · 前馈网络")


def box(d, xy, text, col, w=260, h=56):
    x, y = xy
    d.rounded_rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 12, outline=col, width=3, fill=(12, 19, 34))
    vk.text_c(d, (x, y), text, F_SMALL, col)


def draw_block(d, cx, show_res=True):
    ys = [600, 520, 440, 360, 280, 200, 120]
    vk.arrow(d, (cx, 650), (cx, 80), (71, 85, 105), 3)
    box(d, (cx, ys[0]), "输入词元向量 x", TEXT)
    box(d, (cx, ys[1]), "LayerNorm", AMBER)
    box(d, (cx, ys[2]), "多头注意力", CYAN)
    vk.text_c(d, (cx, ys[3]), "⊕", vk.font(36), GREEN)
    box(d, (cx, ys[4] - 10), "LayerNorm", AMBER)
    box(d, (cx, ys[5] - 20), "前馈网络 FFN", VIOLET)
    vk.text_c(d, (cx, ys[6] - 30), "⊕", vk.font(36), GREEN)
    if show_res:
        y0, y1 = ys[0] - 28, ys[3]
        d.line([(cx, y0 - 8), (cx + 170, y0 - 8), (cx + 170, y1), (cx + 22, y1)], fill=GREEN, width=3)
        y0, y1 = ys[3] - 26, ys[6] - 30
        d.line([(cx, y0), (cx + 210, y0), (cx + 210, y1), (cx + 22, y1)], fill=GREEN, width=3)
        d.text((cx + 180, 470), "残差", font=F_SMALL, fill=GREEN)
        d.text((cx + 220, 200), "残差", font=F_SMALL, fill=GREEN)


def sc_block(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "一个 Transformer 模块（Pre-LN）")
    draw_block(d, 430, show_res=seg >= 1)
    tv.fpaste(c, r"$h=x+\mathrm{Attn}(\mathrm{LN}(x))$", 950, 260, 28)
    tv.fpaste(c, r"$y=h+\mathrm{FFN}(\mathrm{LN}(h))$", 950, 330, 28)
    if seg >= 1:
        vk.text_c(d, (950, 430), "× N 层（GPT-3：96 层）", F_BODY, MUTED)


def sc_deep(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "32 层随机网络：信号能传多远？")
    tv.fpaste(c, r"$\text{普通: }x\leftarrow\tanh(xW)\qquad\text{残差: }x\leftarrow x+\tanh(\mathrm{LN}(x)\,W)$".replace(r"\text{普通: }", "").replace(r"\text{残差: }", r"\quad|\quad "), 640, 115, 22)
    if seg == 0:
        return
    n = L + 1 if seg >= 2 else max(2, int(1 + L * vk.ease(su)))
    layers = np.arange(L + 1)
    series = [("普通网络", SENS_P, RED)]
    if seg >= 2:
        series.append(("残差 + LayerNorm", SENS_R, CYAN))
    tv.line_chart(d, (110, 190, 610, 560), layers, series, ylim=(0, 5), upto=n, xlabel="层数", ylabel="对输入扰动的敏感度",
                  xticks=[(0, "0"), (8, "8"), (16, "16"), (24, "24"), (32, "32")], legend_xy=(130, 230))
    series2 = [("普通网络", STD_P, RED)] + ([("残差", STD_R, CYAN)] if seg >= 2 else [])
    tv.line_chart(d, (700, 190, 1200, 560), layers, series2, ylim=(0, 4), upto=n, xlabel="层数", ylabel="数值标准差",
                  xticks=[(0, "0"), (8, "8"), (16, "16"), (24, "24"), (32, "32")])
    if seg >= 1:
        d.text((610, 640), f"普通：敏感度 {SENS_P[-1]:.2f}，标准差 {STD_P[-1]:.2f}", font=F_SMALL, fill=RED, anchor="rm")
    if seg >= 2:
        d.text((700, 640), f"残差：敏感度 {SENS_R[-1]:.1f}，标准差 {STD_R[-1]:.1f}", font=F_SMALL, fill=CYAN)


def sc_layernorm(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "层归一化 LayerNorm")
    tv.fpaste(c, r"$\mathrm{LN}(x)=\gamma\odot\frac{x-\mu}{\sigma}+\beta,\qquad \mu=\frac{1}{d}\sum_i x_i,\quad \sigma=\sqrt{\frac{1}{d}\sum_i (x_i-\mu)^2}$", 640, 150, 26)
    if seg >= 1:
        tv.bar_chart(d, (150, 260, 560, 520), ["x₁", "x₂", "x₃", "x₄"][:0] or ["1", "2", "3", "4"], list(V), [CYAN] * 4, fmt="{:.0f}", vmax=9)
        vk.text_c(d, (355, 580), f"均值 {MU:.0f}，标准差 √5 ≈ {SD:.2f}", F_SMALL, TEXT)
        vk.arrow(d, (600, 400), (680, 400), MUTED, 3)
        x0, y0, y1 = 720, 260, 520
        mid = (y0 + y1) / 2
        d.line([(x0, mid), (x0 + 420, mid)], fill=(71, 85, 105), width=2)
        for j, v in enumerate(VN):
            bx = x0 + 20 + j * 102
            top = mid - v * 80
            d.rectangle([bx, min(top, mid), bx + 60, max(top, mid)], fill=VIOLET)
            d.text((bx + 30, top - 16 if v > 0 else top + 16), f"{v:+.2f}", font=vk.font(20, mono=True), fill=VIOLET, anchor="mm")
        vk.text_c(d, (930, 580), "归一化后：均值 0，标准差 1", F_SMALL, TEXT)
    if seg >= 2:
        vk.text_c(d, (640, 640), "Pre-LN：LayerNorm 放在每个子层的输入（GPT-2 起广泛使用）", F_SMALL, AMBER)


def sc_ffn(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "前馈网络 FFN")
    tv.fpaste(c, r"$\mathrm{FFN}(x)=W_2\,\mathrm{GELU}(W_1x+b_1)+b_2,\qquad W_1\in\mathbb{R}^{4d\times d},\ W_2\in\mathbb{R}^{d\times 4d}$", 640, 140, 24)
    for x, h, lab, col in [(260, 120, "d", CYAN), (520, 360, "4d", VIOLET), (780, 120, "d", CYAN)]:
        d.rounded_rectangle([x - 40, 380 - h / 2, x + 40, 380 + h / 2], 10, outline=col, width=3)
        vk.text_c(d, (x, 380), lab, vk.font(26, mono=True), col)
    vk.arrow(d, (305, 380), (475, 380), MUTED, 3)
    vk.arrow(d, (565, 380), (735, 380), MUTED, 3)
    vk.text_c(d, (390, 355), "升维", F_SMALL, MUTED)
    vk.text_c(d, (650, 355), "GELU → 降维", F_SMALL, MUTED)
    if seg >= 1:
        cx, cy, r = 1040, 380, 110
        a = FRAC_FFN * 360
        d.pieslice([cx - r, cy - r, cx + r, cy + r], -90, -90 + a, fill=VIOLET)
        d.pieslice([cx - r, cy - r, cx + r, cy + r], -90 + a, 270, fill=CYAN)
        vk.text_c(d, (cx, cy + r + 30), f"FFN 8d² ≈ {FRAC_FFN:.0%}", F_SMALL, VIOLET)
        vk.text_c(d, (cx, cy + r + 60), "注意力 4d²", F_SMALL, CYAN)
        vk.text_c(d, (520, 620), f"d = {DM}：注意力 {P_ATT:,} 参数，FFN {P_FFN:,} 参数", F_SMALL, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("残差连接", "信号和梯度直接穿过很多层"), ("层归一化", "子层输入保持稳定的尺度"),
                      ("前馈网络", "逐词元变换，约占 2/3 参数")], seg, "下一期：掩码与三种架构 —— 编码器、解码器")


RENDER = {"title": sc_title, "block": sc_block, "deep": sc_deep, "layernorm": sc_layernorm, "ffn": sc_ffn,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "block")
