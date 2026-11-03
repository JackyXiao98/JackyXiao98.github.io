"""Interpretability 07 — 完整地理解一个算法：模加法与顿悟.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MI_CACHE=<dir> python mi07_grokking.py <out_dir>

Real computations (mix.grokking, JAX): (a + b) mod 59 with logits = W_out ReLU(W_in (E[a] + E[b])),
d = 128, 256 hidden neurons; full-batch AdamW (lr 1e-3, weight decay 0.3) on 30% of the 3,481 pairs
(1,044 train, 2,437 test), 40,000 steps. For each hidden neuron, its input as a function of one operand,
(E W_in)[a, h] for a = 0..58: share of its Fourier power (frequencies 1..29) in the strongest frequency;
median over the 256 neurons.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import mix
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Interpretability"
CACHE = os.environ.get("MI_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "micache"))

G = mix.cached(CACHE, "grok", mix.grokking)
H = G["hist"]
ST = np.array([h["step"] for h in H])
TR = np.array([h["train"] for h in H])
TE = np.array([h["test"] for h in H])
NS = np.array([h["neuron_share"] for h in H])
at = lambda A, s: float(A[list(ST).index(s)])
first = lambda A, t: int(ST[np.argmax(A >= t)])
TOPN = np.argsort(G["final_share"])[::-1][:3]
print(G["n_train"], G["n_test"], first(TR, 1.0), at(TE, 250), TE[ST <= 5000].max(), first(TE, 0.99), at(NS, 0), NS[-1])
assert G["P"] == 59 and G["n_train"] == 1044 and G["n_test"] == 2437 and first(TR, 1.0) == 250
assert abs(at(TE, 250) - 0.294) < 0.001 and TE[ST <= 5000].max() < 0.32 and first(TE, 0.99) == 29000 and TE[-1] > 0.99
assert abs(at(NS, 0) - 0.13) < 0.005 and abs(NS[-1] - 0.74) < 0.005 and G["final_share"][TOPN].min() > 0.98

SCENES = [
    ("title", [
        "前面我们都只是理解了模型的一小部分。这一期挑战一个完整的例子，从权重里读出一个小模型用的整个算法，同时看一个奇怪的训练现象，顿悟。",
    ]),
    ("task", [
        "任务是模加法，两个零到五十八之间的数相加，再除以五十九取余数。所有可能的题目一共三千四百八十一道，我们只拿出百分之三十训练，剩下的用来测试。",
        "模型很小，把两个数的嵌入向量相加，经过一个有二百五十六个神经元的隐藏层，再输出五十九个类别。训练时加了权重衰减。",
    ]),
    ("grok", [
        f"训练准确率在第二百五十步就达到了百分之百，但这时测试准确率只有{pct_zh(at(TE, 250))}。模型基本上是把训练题背了下来。",
        "接下来的五千步里，测试准确率几乎不动，一直在百分之三十左右。如果在这里停下来，我们会以为模型就是学不会。",
        "继续训练，测试准确率开始缓慢上升，在第两万九千步超过百分之九十九。训练集上早就全对，模型却在之后很久才真正学会规律，这就是顿悟。",
    ]),
    ("fourier", [
        "模型学会的是什么算法？二零二三年的一项研究，在一个做模加法的小变换器里发现，它把每个数表示成几组不同频率的正弦和余弦，再利用三角函数的和角公式，算出两数之和对应的波，最后读出余数。",
        "我们检查自己的模型。把每个隐藏神经元的输入，看成第一个加数的函数，做傅里叶分解，看它有多少能量集中在单一的频率上。",
        f"训练开始时，这个比例的中位数只有{dec_zh(at(NS, 0), 2)}；训练结束时升到{dec_zh(NS[-1], 2)}，最好的神经元超过零点九八，几乎就是一条完美的正弦波。",
    ]),
    ("why", [
        "为什么会顿悟？一种解释是，背下所有训练题和学会通用的算法，都能让训练误差为零。但权重衰减偏爱更简洁、权重更小的解，波形的算法比死记硬背更简洁。",
        "所以在训练误差已经为零之后，权重衰减仍在慢慢地把模型推向通用的算法，测试准确率也就跟着上来了。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，顿悟是指训练集早已全对，测试准确率在很久之后才上升。",
        "第二，模加法模型学会的是用正弦波表示数字，隐藏神经元变成了单一频率的波。",
        "第三，权重衰减把模型从死记硬背推向更简洁的通用算法。下一期，我们回到加法模型，画出它完整的电路。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 7, "模加法与顿悟", "从权重里读出整个算法", SERIES)


def sc_task(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "任务：(a + b) mod 59")
    d.text((400, 180), "例", font=F_BODY, fill=MUTED, anchor="lm")
    vk.text_c(d, (660, 180), "(40 + 30) mod 59 = 11", vk.font(32, mono=True), TEXT)
    vk.text_c(d, (640, 240), f"共 59 × 59 = 3,481 道 · 训练 {G['n_train']:,}（30%）· 测试 {G['n_test']:,}", F_SMALL, MUTED)
    if seg >= 1:
        for i, (t, col) in enumerate([("E[a] + E[b]", CYAN), ("256 个神经元（ReLU）", VIOLET), ("59 个类别", GREEN)]):
            x = 120 + i * 370
            d.rounded_rectangle([x, 330, x + 320, 420], 14, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 160, 375), t, F_SMALL, col)
            if i < 2:
                vk.arrow(d, (x + 325, 375), (x + 365, 375), MUTED, 3)
        vk.text_c(d, (640, 500), "AdamW · 权重衰减 0.3 · 全批量 · 40,000 步", F_SMALL, MUTED)


def sc_grok(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "训练集全对之后很久，才真正学会")
    n = [list(ST).index(250) + 1, list(ST).index(5000) + 1, len(ST)][min(seg, 2)]
    tv.line_chart(d, (140, 170, 820, 560), ST + 50, [("训练准确率", TR * 100, CYAN), ("测试准确率", TE * 100, GREEN)], xlog=True, ylim=(0, 105),
                  upto=n, xlabel="训练步数（对数）", ylabel="%", legend_xy=(520, 480), xticks=[(50, "0"), (1050, "1000"), (10050, "10000"), (40050, "40000")])
    lines = [(f"250 步：训练 100%，测试 {at(TE, 250):.0%}", CYAN), ("5000 步：测试仍约 30%", MUTED), ("29000 步：测试 > 99%", GREEN)]
    for i, (t, col) in enumerate(lines[:seg + 1]):
        d.text((860, 230 + i * 55), t, font=F_SMALL, fill=col)
    if seg >= 2:
        d.text((860, 420), "顿悟（grokking）", font=F_BODY, fill=AMBER)
        d.text((860, 465), "Power et al., 2022", font=F_SMALL, fill=MUTED)


def sc_fourier(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "算法：用正弦波表示数字（Nanda et al., 2023）")
    if seg == 0:
        tv.fpaste(c, r"$\cos(\omega a)\cos(\omega b)-\sin(\omega a)\sin(\omega b)=\cos(\omega(a+b)),\quad \omega=\frac{2\pi k}{59}$", 640, 260, 30)
        vk.text_c(d, (640, 360), "和角公式 → 两数之和对应的波 → 读出余数", F_BODY, AMBER)
        return
    a = np.arange(59)
    cols = [CYAN, VIOLET, GREEN]
    for j, h in enumerate(TOPN):
        g = G["G"][:, h]
        g = g / np.abs(g).max()
        tv.line_chart(d, (110, 160 + j * 140, 560, 270 + j * 140), a, [("", g, cols[j])], ylim=(-1.1, 1.1))
        d.text((575, 200 + j * 140), f"神经元 {h}", font=F_SMALL, fill=cols[j])
        d.text((575, 230 + j * 140), f"频率 {G['final_freq'][h]}", font=F_SMALL, fill=MUTED)
    d.text((110, 590), "横轴：第一个加数 a = 0 … 58（训练结束时）", font=F_SMALL, fill=MUTED)
    if seg >= 2:
        tv.line_chart(d, (800, 180, 1220, 520), ST + 50, [("单一频率占比（中位数）", NS, AMBER)], xlog=True, ylim=(0, 1), legend_xy=(815, 190),
                      xlabel="训练步数（对数）", xticks=[(50, "0"), (1050, "1000"), (10050, "10000"), (40050, "40000")])
        d.text((800, 600), f"开始 {at(NS, 0):.2f} → 结束 {NS[-1]:.2f}", font=F_BODY, fill=AMBER)


def sc_why(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "为什么会顿悟？")
    for i, (a, b, col) in enumerate([("死记硬背", "训练误差 = 0 · 权重大", RED), ("通用算法（波）", "训练误差 = 0 · 更简洁", GREEN)]):
        x = 140 + i * 540
        d.rounded_rectangle([x, 190, x + 460, 310], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 230, 230), a, F_BODY, col)
        vk.text_c(d, (x + 230, 275), b, F_SMALL, TEXT)
    vk.arrow(d, (605, 250), (675, 250), AMBER, 4)
    vk.text_c(d, (640, 360), "权重衰减偏爱更简洁的解", F_BODY, AMBER)
    if seg >= 1:
        vk.text_c(d, (640, 470), "训练误差已经为零后，权重衰减仍在慢慢推向通用算法", F_BODY, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("顿悟", "训练早已全对，测试很久后才上升"), ("算法", "神经元变成单一频率的正弦波"),
                      ("原因", "权重衰减偏爱简洁的通用算法")], seg, "下一期：电路与归因图")


RENDER = {"title": sc_title, "task": sc_task, "grok": sc_grok, "fourier": sc_fourier, "why": sc_why, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "mi-grokking")
