"""Optimization 06 — 初始化与归一化：让信号在深层网络里活下来.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] OP_CACHE=<dir> python op06_init.py <out_dir>

Real computations:
- opx.signal: a 50-layer MLP (width 256, no biases) at initialisation, input N(0, 1); standard deviation of the
  activations after each layer for ReLU with weights N(0, g^2 / 256), g = 1 (Xavier-style), sqrt(2) (He), 2;
  and tanh with g = 1 and g = 0.5.
- opx.postln (JAX): 6-layer addition Transformer, Adam lr 1e-2, 800 steps, pre-LN vs post-LN, without warmup
  and with 200 warmup steps; test exact-match accuracy.
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

SG = opx.cached(CACHE, "signal", opx.signal)
PL = opx.cached(CACHE, "postln", opx.postln, lr=0.01)
FIN = {k: float(PL[k]["hist"][-1, 2]) for k in PL}
T99 = {k: opx._first(PL[k]["hist"]) for k in PL}
print({k: v[[10, 50]] for k, v in SG.items()}, FIN, T99)
assert SG["relu_xavier"][50] < 1e-7 and 2e7 < SG["relu_large"][50] < 3e7 and 0.5 < SG["relu_he"][50] < 0.9 and SG["relu_he"][1:].min() > 0.5
assert abs(SG["relu_xavier"][10] - 0.0233) < 1e-3 and abs(SG["relu_large"][10] - 23.86) < 0.01
assert T99[(False, 200)] == 340 and FIN[(False, 0)] < 0.5 and FIN[(True, 0)] < 0.02 and abs(FIN[(True, 200)] - 0.80) < 0.01
assert abs(FIN[(False, 0)] - 0.435) < 0.001

SCENES = [
    ("title", [
        "前面几期讲的是怎样走下坡。但如果网络一开始的状态就不对，再好的优化器也走不动。这一期看初始化和归一化。",
    ]),
    ("signal", [
        "信号每经过一层，大小都会乘上一个因子。网络有几十层时，这个因子哪怕只比一小一点，信号也会指数级地消失；比一大一点，就会指数级地爆炸。梯度在反向传播时也是一样。",
        f"我们构造一个五十层的网络，看信号在每一层的标准差。权重用一除以根号宽度的标准差时，因为激活函数会把一半的值变成零，每层信号都会变小，十层后只剩{dec_zh(SG['relu_xavier'][10], 3)}，五十层后几乎为零。",
        f"权重放大一倍，十层后信号就放大到约{int_zh(round(SG['relu_large'][10]))}倍，五十层后达到两千多万倍。",
    ]),
    ("he", [
        "解决办法是选好权重的大小，让每层信号的方差保持不变。对于这种整流线性单元，标准差应该是根号二除以根号宽度，这叫何恺明初始化；对于双曲正切这类激活函数，则用泽维尔初始化。",
        f"用了何恺明初始化，五十层以后信号的标准差仍然在{dec_zh(SG['relu_he'][50], 2)}左右，既没有消失也没有爆炸。",
    ]),
    ("norm", [
        "初始化只管开头，训练中权重变了，信号又可能失控。归一化层在每一层把激活重新调整到均值为零、方差为一，再乘上可学习的缩放。变换器里用的是层归一化。",
        "层归一化放在哪里也很重要。最初的变换器把它放在残差相加之后，叫后归一化；现在几乎都放在每个子层的输入处，残差通路保持干净，叫前归一化。",
    ]),
    ("experiment", [
        "我们训练一个六层的加法模型，学习率取得比较大，比较两种放法，以及有没有预热。",
        f"后归一化不预热时完全训练不起来，准确率只有{pct_zh(FIN[(True, 0)])}；加了预热，八百步后也只有{pct_zh(FIN[(True, 200)])}。前归一化不预热时训练不稳定，最后{pct_zh(FIN[(False, 0)])}；加了预热，第{int_zh(T99[(False, 200)])}步就达到百分之九十九。",
        "这和研究的结论一致，后归一化对预热和学习率非常敏感，前归一化稳定得多，所以今天的大模型几乎都采用前归一化。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，在深层网络里，信号每层乘上一个因子，很容易指数级地消失或爆炸。",
        "第二，何恺明和泽维尔初始化选好权重大小，让各层信号的方差保持不变。",
        "第三，层归一化在训练中保持信号稳定，前归一化比后归一化稳定得多。下一期，我们看超参数怎样从小模型迁移到大模型。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 6, "初始化与归一化", "让信号在深层网络里活下来", SERIES)


LAY = np.arange(51)


def sc_signal(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "50 层 ReLU 网络：每层激活的标准差（对数）")
    series = []
    if seg >= 1:
        series.append(("权重标准差 1/√n", SG["relu_xavier"], RED))
    if seg >= 2:
        series.append(("权重标准差 2/√n", SG["relu_large"], AMBER))
    if series:
        tv.line_chart(d, (140, 170, 820, 580), LAY, series, ylog=True, ylim=(-9, 8), xlabel="层数", ylabel="标准差（对数）",
                      legend_xy=(160, 470), xticks=[(0, "0"), (10, "10"), (25, "25"), (50, "50")])
        d.text((870, 230), "第 10 层", font=F_SMALL, fill=MUTED)
        d.text((870, 270), f"{SG['relu_xavier'][10]:.4f}", font=vk.font(24, mono=True), fill=RED)
        if seg >= 2:
            d.text((870, 310), f"{SG['relu_large'][10]:.1f}", font=vk.font(24, mono=True), fill=AMBER)
            d.text((870, 380), "第 50 层", font=F_SMALL, fill=MUTED)
            d.text((870, 420), f"{SG['relu_xavier'][50]:.0e}", font=vk.font(24, mono=True), fill=RED)
            d.text((870, 460), f"{SG['relu_large'][50]:.1e}", font=vk.font(24, mono=True), fill=AMBER)
    else:
        vk.text_c(d, (640, 300), "每层乘一个因子：< 1 → 指数级消失 · > 1 → 指数级爆炸", F_BODY, TEXT)


def sc_he(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "He 初始化与 Xavier 初始化")
    tv.fpaste(c, r"$\mathrm{ReLU}:\ W\sim\mathcal{N}(0,\,2/n)\qquad \tanh:\ W\sim\mathcal{N}(0,\,1/n)$", 640, 190, 32)
    if seg >= 1:
        tv.line_chart(d, (140, 260, 820, 600), LAY, [("1/√n", SG["relu_xavier"], RED), ("He：√2/√n", SG["relu_he"], GREEN), ("2/√n", SG["relu_large"], AMBER)],
                      ylog=True, ylim=(-9, 8), xlabel="层数", legend_xy=(160, 500), xticks=[(0, "0"), (25, "25"), (50, "50")])
        d.text((870, 380), f"He：第 50 层 {SG['relu_he'][50]:.2f}", font=F_BODY, fill=GREEN)


def sc_norm(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "层归一化（LayerNorm）放在哪里？")
    tv.fpaste(c, r"$\mathrm{LN}(x)=\gamma\odot\frac{x-\mu}{\sigma}$", 640, 180, 34)
    if seg >= 1:
        for i, (t, f, col) in enumerate([("后归一化（Post-LN，原始 Transformer）", r"$x\leftarrow\mathrm{LN}(x+F(x))$", RED),
                                         ("前归一化（Pre-LN，今天的默认）", r"$x\leftarrow x+F(\mathrm{LN}(x))$", GREEN)]):
            x = 120 + i * 540
            d.rounded_rectangle([x, 270, x + 500, 430], 14, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 250, 305), t, F_SMALL, col)
            tv.fpaste(c, f, x + 250, 375, 30)
        vk.text_c(d, (640, 500), "前归一化：残差通路保持干净", F_SMALL, MUTED)


def sc_experiment(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "6 层加法模型 · Adam 学习率 0.01 · 800 步")
    if seg >= 1:
        keys = [(True, 0), (True, 200), (False, 0), (False, 200)]
        labs = ["后归一化\n不预热", "后归一化\n预热", "前归一化\n不预热", "前归一化\n预热"]
        cols = [RED, AMBER, AMBER, GREEN]
        x0, y1, gw = 180, 560, 230
        d.line([(x0, y1), (x0 + 4 * gw, y1)], fill=(71, 85, 105), width=2)
        for i, k in enumerate(keys):
            v = FIN[k] * 100
            bx = x0 + i * gw + 50
            top = y1 - v / 110 * 330
            d.rectangle([bx, top, bx + 130, y1], fill=cols[i])
            d.text((bx + 65, top - 16), f"{v:.1f}%", font=vk.font(20, mono=True), fill=cols[i], anchor="mm")
            for j, line in enumerate(labs[i].split("\n")):
                d.text((bx + 65, y1 + 22 + j * 28), line, font=F_SMALL, fill=MUTED, anchor="mm")
        d.text((180, 170), "第 800 步的测试准确率", font=F_SMALL, fill=MUTED)
    if seg >= 2:
        vk.text_c(d, (640, 660), "后归一化对预热和学习率非常敏感（Xiong et al., 2020）", F_SMALL, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("信号", "每层乘一个因子 → 消失或爆炸"), ("初始化", "He / Xavier：保持方差不变"),
                      ("归一化", "前归一化比后归一化稳定得多")], seg, "下一期：超参数迁移 μP")


RENDER = {"title": sc_title, "signal": sc_signal, "he": sc_he, "norm": sc_norm, "experiment": sc_experiment, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "op-init")
