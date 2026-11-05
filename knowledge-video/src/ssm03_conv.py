"""State Space Models 03 — 状态空间模型：从连续系统到卷积.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python ssm03_conv.py <out_dir>

Real computations (ssmx, numpy): a diagonal continuous system x' = A x + B u, y = C x with A = -[0.3, 1, 3] and a
complex pair -0.2 +- 2i (N = 5), discretised with zero-order hold at step 0.1; an input of 60 random values run
(a) as a recurrence h_t = A_bar h_{t-1} + B_bar u_t, (b) as a causal convolution with the kernel
K_t = C A_bar^t B_bar; largest difference; kernel values; recurrence cost per step (N) vs convolution cost (T per
output, or T log T with FFT).
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import ssmx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "State Space Models"

A = np.array([-0.3, -1.0, -3.0, -0.2 + 2j, -0.2 - 2j])
Bc = np.ones(5, complex)
C = np.array([1.0, 0.5, 0.5, 0.4, 0.4], complex)
DT = 0.1
Ab, Bb = ssmx.ssm_discretize(A, Bc, DT)
U = np.random.default_rng(3).normal(size=60)
Y1 = ssmx.ssm_recurrent(Ab, Bb, C, U)
Y2, K = ssmx.ssm_conv(Ab, Bb, C, U)
DIFF = float(np.abs(Y1 - Y2).max())
print(DIFF, np.round(np.abs(Ab), 4), np.round(K[:5], 4))
assert DIFF < 1e-12 and round(abs(Ab[0]), 3) == 0.970 and round(abs(Ab[2]), 3) == 0.741

SCENES = [
    ("title", [
        "状态空间模型来自控制理论。它最早描述的是连续时间的系统，比如一个弹簧或者一个电路。这一期看它怎样变成一个处理序列的神经网络层，以及它最特别的地方：同一个模型，既可以一步一步地递推，也可以一次卷积算完。",
    ]),
    ("continuous", [
        "连续的状态空间模型由三个矩阵描述：状态矩阵决定状态怎样随时间自己变化，输入矩阵决定输入怎样影响状态，输出矩阵决定怎样从状态读出结果。",
        "为了处理离散的序列，要选一个步长，把连续系统离散化。常用的零阶保持方法，把状态矩阵变成它乘以步长后的指数。如果状态矩阵是对角的，离散化就是对每个对角元素各取一次指数。",
    ]),
    ("recur", [
        f"我们的例子有五个状态维度：三个实数衰减率，和一对复数，复数让状态可以振荡。步长取零点一，三个实数维度每一步分别保留约{pct_zh(abs(Ab[0]))}、{pct_zh(abs(Ab[1]))}和{pct_zh(abs(Ab[2]))}。",
        "递推的算法和循环网络一样：每一步，状态乘以离散后的状态矩阵，加上输入的贡献，再读出输出。每一步的代价固定，适合生成。",
    ]),
    ("conv", [
        "但因为这个系统是线性的，而且参数不随时间变化，把递推展开，每个输出就是过去所有输入的加权和，权重只取决于两者相隔多远。这正是一个卷积，卷积核由三个矩阵算出来。",
        f"我们用六十个随机输入验证：一步一步递推的结果，和一次卷积的结果，最大只差约十的负十六次方。卷积可以在整段序列上并行计算，用快速傅里叶变换还能更快，适合训练。",
    ]),
    ("history", [
        "二零二一年的结构化状态空间模型，正是利用了这两种视角：训练时用卷积并行计算，生成时用递推逐步计算，并配合特殊的状态矩阵结构，在长序列基准上大幅超过了之前的方法。后来的对角化版本把状态矩阵简化成对角的，效果几乎一样好。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，状态空间模型由三个矩阵描述，离散化后得到逐步递推的形式。",
        "第二，线性且参数不随时间变化时，递推等价于一个卷积，我们的实验验证了两者完全一致。",
        "第三，卷积用于并行训练，递推用于高效生成。下一期，我们看状态能记住多久。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 3, "状态空间模型", "从连续系统到卷积", SERIES)


def sc_continuous(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "连续系统与离散化")
    tv.fpaste(c, r"$x'(t)=A\,x(t)+B\,u(t),\qquad y(t)=C\,x(t)$", 640, 220, 40)
    if seg >= 1:
        tv.fpaste(c, r"$\bar A=e^{\Delta A},\qquad \bar B=(\bar A-I)A^{-1}B$", 640, 340, 38)
        tv.fpaste(c, r"$h_t=\bar A\,h_{t-1}+\bar B\,u_t,\qquad y_t=C\,h_t$", 640, 450, 38)
        vk.text_c(d, (640, 560), "对角 A：每个元素各取一次指数", F_BODY, GREEN)


def sc_recur(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "例子：5 个状态维度，步长 0.1")
    f = vk.font(22, mono=True)
    d.text((140, 180), "A", font=f, fill=MUTED)
    d.text((440, 180), "离散后每步保留的比例", font=F_SMALL, fill=MUTED)
    for i, (a, ab) in enumerate(zip(A, Ab)):
        y = 220 + i * 50
        d.text((140, y), f"{a.real:+.1f}{'' if a.imag == 0 else f'{a.imag:+.0f}i'}", font=f, fill=CYAN if a.imag == 0 else VIOLET)
        d.rectangle([440, y + 4, 440 + abs(ab) * 300, y + 30], fill=GREEN)
        d.text((760, y), f"{abs(ab):.3f}", font=f, fill=TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 520), "递推：每一步代价固定（∝ 状态维度）→ 适合生成", F_BODY, AMBER)


def sc_conv(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "展开递推：一个卷积")
    tv.fpaste(c, r"$y_t=\sum_{s\leq t}K_{t-s}\,u_s,\qquad K_j=C\,\bar A^{\,j}\,\bar B$", 640, 175, 36)
    tv.line_chart(d, (120, 250, 600, 560), list(range(40)), [("卷积核 K", K[:40], GREEN)], xticks=[(0, "0"), (20, "20"), (39, "39")], xlabel="间隔 j")
    d.text((120, 225), "卷积核（衰减 + 振荡）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        tv.line_chart(d, (700, 250, 1180, 560), list(range(60)), [("递推", Y1, CYAN), ("卷积", Y2, AMBER)], xticks=[(0, "0"), (30, "30"), (59, "59")], xlabel="时间步", legend_xy=(1000, 250))
        d.text((700, 225), f"两种算法的输出：最大差别 {DIFF:.0e}", font=F_SMALL, fill=MUTED)
        vk.text_c(d, (640, 650), "卷积：整段并行（FFT：T log T）→ 适合训练", F_SMALL, AMBER)


def sc_history(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "S4 与 S4D")
    vk.text_c(d, (640, 230), "S4（Gu et al., 2021）：训练用卷积、生成用递推 + 结构化的状态矩阵", F_SMALL, CYAN)
    vk.text_c(d, (640, 300), "长序列基准（Long Range Arena）大幅领先", F_SMALL, TEXT)
    vk.text_c(d, (640, 390), "S4D（2022）：对角状态矩阵，效果几乎一样", F_SMALL, GREEN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("SSM", "三个矩阵，离散化后递推"), ("等价", "线性 + 时不变 → 卷积"),
                      ("两种算法", "卷积训练，递推生成")], seg, "下一期：长程记忆")


RENDER = {"title": sc_title, "continuous": sc_continuous, "recur": sc_recur, "conv": sc_conv, "history": sc_history, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "ssm-conv")
