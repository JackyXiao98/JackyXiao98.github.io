"""Optimization 08 — 新一代优化器：Shampoo、SOAP 与 Muon.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] OP_CACHE=<dir> python op08_muon.py <out_dir>

Real computations:
- Newton-Schulz orthogonalisation (5 quintic iterations, coefficients 3.4445, -4.7750, 2.0315, as in Muon)
  of a random 64 x 256 Gaussian matrix with a decaying spectrum: singular values before / after.
- opx.optimizers (JAX): the addition Transformer, 600 steps, batch 256, Adam (lr 1e-3 / 3e-3 / 1e-2 / 3e-2)
  vs Muon (momentum 0.95 + Nesterov + Newton-Schulz on the 2-D hidden matrices, lr 0.005 / 0.01 / 0.02 / 0.05;
  embeddings, unembedding and gains with AdamW 3e-3); steps until test exact-match >= 99%.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import opx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Optimization"
CACHE = os.environ.get("OP_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "opcache"))


def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / np.linalg.norm(G)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X


rng = np.random.default_rng(0)
U, _ = np.linalg.qr(rng.normal(size=(64, 64)))
Vt, _ = np.linalg.qr(rng.normal(size=(256, 64)))
SV0 = np.geomspace(1, 1e-2, 64)
G = U @ np.diag(SV0) @ Vt.T
SV1 = np.linalg.svd(newton_schulz(G), compute_uv=False)
print(SV0.max() / SV0.min(), SV1.min(), SV1.max())
assert abs(SV0.max() / SV0.min() - 100) < 1e-6 and SV1.min() > 0.6 and SV1.max() < 1.25

O = opx.cached(CACHE, "opt", opx.optimizers)
T99 = {(o, lr): opx._first(O[o][lr]["hist"]) for o in ("adam", "muon") for lr in O[o]}
A100 = {(o, lr): float(O[o][lr]["hist"][5, 2]) for o in ("adam", "muon") for lr in O[o]}
BA, BM = 1e-3, 0.02
print(T99, A100)
assert T99[("adam", BA)] == 340 and min(v for (o, l), v in T99.items() if o == "adam" and v) == 340
assert T99[("muon", BM)] == 140 and all(T99[("muon", l)] <= 220 for l in O["muon"])
assert O["adam"][BA]["hist"][5, 0] == 100 and abs(A100[("adam", BA)] - 0.108) < 1e-3 and abs(A100[("muon", BM)] - 0.91) < 1e-3
ST = O["adam"][BA]["hist"][:, 0]

SCENES = [
    ("title", [
        "亚当已经用了十年。最近几年，一批新的优化器开始在大模型训练中挑战它。这一期看其中的代表，香波、它的变体，以及穆昂。",
    ]),
    ("diag", [
        "亚当对每个参数单独缩放步长，相当于用一个对角矩阵做预条件。它看不到参数之间的关系。而神经网络的权重是矩阵，同一行、同一列的参数紧密相关。",
        "香波优化器为每个权重矩阵的行和列，各维护一个统计矩阵，用它们的逆根来修正梯度，是一种完整得多的矩阵预条件。它的改进版把亚当放到香波找到的坐标系里运行。",
    ]),
    ("muon", [
        "穆昂走了一条更简单的路。它先像动量方法一样累积梯度，然后把这个更新矩阵正交化，也就是保留它的方向，把所有奇异值都变成一。",
        "直觉是，梯度矩阵往往被少数几个方向主导，正交化以后，那些被忽视的方向也能得到同样大小的更新。",
    ]),
    ("ns", [
        "精确的正交化需要奇异值分解，太慢。穆昂用牛顿舒尔茨迭代来近似，只需要几次矩阵乘法。",
        f"我们拿一个奇异值从一到零点零一、相差一百倍的随机矩阵，迭代五次以后，所有奇异值都落在{dec_zh(SV1.min(), 2)}到{dec_zh(SV1.max(), 2)}之间，已经接近正交。",
    ]),
    ("experiment", [
        "我们在加法模型上比较亚当和穆昂，穆昂只用在中间的权重矩阵上，嵌入和输出层仍用亚当。两者都试了四个学习率。",
        f"第一百步时，最好的亚当测试准确率只有{pct_zh(A100[('adam', BA)])}，穆昂已经是{pct_zh(A100[('muon', BM)])}。达到百分之九十九，亚当最快要{int_zh(T99[('adam', BA)])}步，穆昂只要{int_zh(T99[('muon', BM)])}步，四个学习率里最慢的也只要{int_zh(max(T99[('muon', l)] for l in O['muon']))}步。",
    ]),
    ("scale", [
        "在大模型上，二零二五年的一项研究在穆昂里加入了权重衰减等调整，报告在计算最优的训练中，它的计算效率大约是亚当W的两倍，并用它训练了一个混合专家模型。之后，一个万亿参数的模型也用改进的穆昂完成了训练。",
        "不过，新优化器的优势在不同规模和设置下差别很大，亚当W仍是最稳妥的默认选择，这个领域还在快速变化。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，亚当是对角预条件，香波和它的变体用矩阵统计量做更完整的预条件。",
        "第二，穆昂把动量更新正交化，用牛顿舒尔茨迭代近似，只需几次矩阵乘法。",
        "第三，在加法模型上，穆昂比亚当快一倍多，在大模型上也显示出优势。最后一期，我们看训练的稳定性。感谢大家的观看。",
    ]),
]
assert T99[("adam", BA)] / T99[("muon", BM)] > 2


def sc_title(c, u, seg, su):
    tv.title_card(c, 8, "新一代优化器", "Shampoo、SOAP 与 Muon", SERIES)


def sc_diag(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "从对角预条件到矩阵预条件")
    d.rounded_rectangle([120, 170, 600, 300], 14, outline=AMBER, width=2, fill=(12, 19, 34))
    vk.text_c(d, (360, 210), "Adam：对角预条件", F_BODY, AMBER)
    vk.text_c(d, (360, 260), "每个参数单独缩放，看不到参数间的关系", F_SMALL, TEXT)
    if seg >= 1:
        d.rounded_rectangle([680, 170, 1160, 300], 14, outline=CYAN, width=2, fill=(12, 19, 34))
        vk.text_c(d, (920, 210), "Shampoo（Gupta et al., 2018）", F_BODY, CYAN)
        vk.text_c(d, (920, 260), "行、列各一个统计矩阵 → 逆根修正梯度", F_SMALL, TEXT)
        tv.fpaste(c, r"$W\leftarrow W-\eta\,L^{-1/4}\,G\,R^{-1/4},\quad L=\sum GG^{\top},\ R=\sum G^{\top}G$", 640, 400, 30)
        vk.text_c(d, (640, 500), "SOAP（Vyas et al., 2024）：在 Shampoo 的坐标系里运行 Adam", F_SMALL, VIOLET)


def sc_muon(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "Muon（Jordan et al., 2024）：把更新正交化")
    tv.fpaste(c, r"$M_t=\beta M_{t-1}+G_t,\qquad W\leftarrow W-\eta\,\mathrm{Orth}(M_t)$", 640, 220, 34)
    tv.fpaste(c, r"$M=U\Sigma V^{\top}\ \ \Rightarrow\ \ \mathrm{Orth}(M)=UV^{\top}$", 640, 320, 34)
    vk.text_c(d, (640, 400), "保留方向，所有奇异值变成 1", F_SMALL, AMBER)
    if seg >= 1:
        vk.text_c(d, (640, 500), "梯度常被少数方向主导 → 正交化后被忽视的方向也得到同样大小的更新", F_SMALL, GREEN)


def sc_ns(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "Newton–Schulz 迭代：只用矩阵乘法")
    tv.fpaste(c, r"$X\leftarrow aX+(bXX^{\top}+c(XX^{\top})^2)X$", 640, 190, 32)
    if seg >= 1:
        tv.line_chart(d, (160, 270, 1120, 590), np.arange(64), [("迭代前：1 → 0.01", SV0, RED), ("迭代 5 次后", SV1, GREEN)], ylim=(0, 1.3),
                      xlabel="奇异值（从大到小）", legend_xy=(700, 300))
        vk.text_c(d, (640, 665), f"5 次迭代后奇异值都在 {SV1.min():.2f} – {SV1.max():.2f} 之间", F_BODY, GREEN)


def sc_experiment(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "加法模型：Adam vs Muon（各试 4 个学习率）")
    series = [(f"Adam（η = {BA:g}）", O["adam"][BA]["hist"][:, 2] * 100, AMBER)]
    if seg >= 1:
        series.append((f"Muon（η = {BM:g}）", O["muon"][BM]["hist"][:, 2] * 100, GREEN))
    tv.line_chart(d, (140, 170, 760, 560), ST, series, ylim=(0, 105), xlabel="训练步数", ylabel="测试准确率 %", legend_xy=(820, 520),
                  xticks=[(s, str(s)) for s in (0, 200, 400, 600)])
    d.text((820, 200), "Muon 只用于中间的权重矩阵", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        rows = [("第 100 步", f"{A100[('adam', BA)]:.0%}", f"{A100[('muon', BM)]:.0%}"), ("到 99%", f"{T99[('adam', BA)]} 步", f"{T99[('muon', BM)]} 步")]
        d.text((990, 260), "Adam", font=F_SMALL, fill=AMBER)
        d.text((1110, 260), "Muon", font=F_SMALL, fill=GREEN)
        for i, (a, b, cc) in enumerate(rows):
            d.text((820, 305 + i * 45), a, font=F_SMALL, fill=TEXT)
            d.text((990, 305 + i * 45), b, font=F_SMALL, fill=AMBER)
            d.text((1110, 305 + i * 45), cc, font=F_SMALL, fill=GREEN)


def sc_scale(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "大模型上的 Muon")
    items = [("Moonlight（Moonshot AI, 2025）", "加入权重衰减等调整 · 计算最优训练中约 2 倍于 AdamW 的效率", CYAN),
             ("Kimi K2（2025）", "1 万亿参数 MoE · MuonClip（Muon + QK-Clip）", VIOLET)]
    for i, (a, b, col) in enumerate(items):
        y = 170 + i * 130
        d.rounded_rectangle([100, y, 1180, y + 100], 14, outline=col, width=2, fill=(12, 19, 34))
        d.text((130, y + 32), a, font=F_BODY, fill=col, anchor="lm")
        d.text((130, y + 72), b, font=F_SMALL, fill=TEXT, anchor="lm")
    if seg >= 1:
        vk.text_c(d, (640, 500), "优势随规模与设置差别很大 · AdamW 仍是最稳妥的默认", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 6, [("预条件", "Adam 对角 · Shampoo / SOAP 矩阵"), ("Muon", "正交化动量更新（Newton–Schulz）"),
                      ("实验", f"到 99%：{T99[('adam', BA)]} → {T99[('muon', BM)]} 步")], seg, "最后一期：训练的稳定性", label_w=430)


RENDER = {"title": sc_title, "diag": sc_diag, "muon": sc_muon, "ns": sc_ns, "experiment": sc_experiment, "scale": sc_scale, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "op-muon")
