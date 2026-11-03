"""Optimization 04 — 自适应方法：AdaGrad、RMSProp 与 Adam.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] OP_CACHE=<dir> python op04_adam.py <out_dir>

Real computations (opx.optimizers, JAX): the tfm.py addition Transformer (2 layers, d = 64) trained 600 steps
(batch 256, constant learning rate, no warmup) with SGD (lr 0.1 / 0.3 / 1 / 3), SGD + momentum 0.9
(0.03 / 0.1 / 0.3 / 1) and Adam (1e-3 / 3e-3 / 1e-2 / 3e-2); exact-match accuracy on the 2,000 test problems
every 20 steps; "steps to 99%" = first evaluation with accuracy >= 99%.
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

O = opx.cached(CACHE, "opt", opx.optimizers)
H = {(o, lr): O[o][lr]["hist"] for o in ("sgd", "momentum", "adam") for lr in O[o]}
FIN = {k: h[-1, 2] for k, h in H.items()}
T99 = {k: opx._first(h) for k, h in H.items()}
BEST = {"sgd": 0.3, "momentum": 0.03, "adam": 1e-3}
print({k: (round(FIN[k], 3), T99[k]) for k in H})
assert abs(FIN[("sgd", 0.3)] - 0.945) < 1e-3 and max(FIN[("sgd", l)] for l in O["sgd"]) == FIN[("sgd", 0.3)]
assert all(T99[("sgd", l)] is None for l in O["sgd"]) and T99[("momentum", 0.03)] == 500
assert T99[("adam", 1e-3)] == 340 and T99[("adam", 3e-3)] == 340 and T99[("adam", 1e-2)] is None
assert all(abs(O[o][l]["hist"][-1, 1] - np.log(12)) < 1e-3 for o, l in (("sgd", 1.0), ("sgd", 3.0), ("momentum", 1.0)))
ST = H[("sgd", 0.3)][:, 0]

SCENES = [
    ("title", [
        "前面的方法对所有参数用同一个学习率。但神经网络里，有的参数梯度很大，有的很小，共用一个步长并不合适。这一期看给每个参数各自步长的自适应方法。",
    ]),
    ("adagrad", [
        "二零一一年的自适应梯度方法，记录每个参数历史梯度平方的累计和，用它的平方根去除学习率。梯度一直很大的参数，步子就小一些；梯度很小的参数，步子就大一些。",
        "它的问题是累计和只增不减，学习率会越来越小，最后几乎停下来。均方根传播法改用指数移动平均，只记住最近一段时间的梯度大小，解决了这个问题。",
    ]),
    ("adam", [
        "亚当把动量和均方根传播结合在一起。它用移动平均同时记录梯度本身，也就是一阶矩，和梯度的平方，也就是二阶矩。更新方向用一阶矩，步长用二阶矩的平方根来缩放。",
        "因为两个移动平均都从零开始，前几步会偏小，亚当再除以一个修正因子，这叫偏差修正。默认的参数，一阶矩的系数是零点九，二阶矩是零点九九九。",
    ]),
    ("experiment", [
        "我们在加法模型上比较三种优化器，每种都试四个学习率，训练六百步。",
        f"普通的随机梯度下降，最好的学习率下六百步后准确率是{pct_zh(FIN[('sgd', 0.3)], 1)}，始终没有达到百分之九十九。加上动量，最好的情况在第{int_zh(T99[('momentum', 0.03)])}步达到百分之九十九。",
        f"亚当在第{int_zh(T99[('adam', 1e-3)])}步就达到了百分之九十九。在变换器这样各层梯度大小相差很大的模型上，自适应方法的优势非常明显。",
        "不过学习率依然重要。随机梯度下降学习率取一，亚当学习率取零点零一时，训练都失败了，模型只会输出均匀的猜测，或者停在很差的位置。",
    ]),
    ("adamw", [
        "最后是一个重要的细节，权重衰减。传统做法是在损失里加上权重的平方惩罚，它的梯度会和其他梯度一起，被亚当的二阶矩缩放。结果是，梯度大的参数反而衰减得少。",
        "二零一九年的研究提出把权重衰减从梯度里拿出来，直接在每一步把权重乘以一个略小于一的数，这就是带解耦权重衰减的亚当。今天训练大模型，几乎都用它。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，自适应方法按每个参数历史梯度的大小，调整各自的步长。",
        "第二，亚当结合了动量和二阶矩缩放，在加法模型上比随机梯度下降快得多。",
        "第三，解耦权重衰减的亚当把权重衰减和梯度更新分开，是今天的默认选择。下一期，我们看学习率调度，预热、余弦和预热稳定衰减。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 4, "自适应方法与 Adam", "给每个参数各自的步长", SERIES)


def sc_adagrad(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "AdaGrad（Duchi et al., 2011）→ RMSProp")
    tv.fpaste(c, r"$G_t=\sum_{s\leq t} g_s^2,\qquad \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{G_t}+\epsilon}\,g_t$", 640, 220, 36)
    vk.text_c(d, (640, 300), "梯度大 → 步子小 · 梯度小 → 步子大（逐个参数）", F_SMALL, AMBER)
    if seg >= 1:
        vk.text_c(d, (640, 390), "问题：G 只增不减 → 学习率越来越小", F_SMALL, RED)
        tv.fpaste(c, r"$v_t=\beta_2 v_{t-1}+(1-\beta_2)\,g_t^2$", 640, 470, 34)
        vk.text_c(d, (640, 540), "RMSProp：指数移动平均，只记住最近的梯度大小", F_SMALL, GREEN)


def sc_adam(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "Adam（Kingma & Ba, 2015）")
    tv.fpaste(c, r"$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$", 640, 200, 32)
    tv.fpaste(c, r"$\theta_{t+1}=\theta_t-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$", 640, 300, 36)
    vk.text_c(d, (330, 380), "一阶矩 m：动量（方向）", F_SMALL, CYAN)
    vk.text_c(d, (950, 380), "二阶矩 v：缩放步长", F_SMALL, VIOLET)
    if seg >= 1:
        tv.fpaste(c, r"$\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t}$", 640, 480, 32)
        vk.text_c(d, (640, 560), "偏差修正 · 默认 β₁ = 0.9，β₂ = 0.999", F_SMALL, AMBER)


COLS = {"sgd": RED, "momentum": AMBER, "adam": GREEN}
NAMES = {"sgd": "SGD", "momentum": "SGD + 动量", "adam": "Adam"}


def sc_experiment(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "加法模型 · 600 步 · 每种优化器试 4 个学习率")
    if seg >= 1:
        show = ["sgd", "momentum"] + (["adam"] if seg >= 2 else [])
        series = [(f"{NAMES[o]}（η = {BEST[o]:g}）", H[(o, BEST[o])][:, 2] * 100, COLS[o]) for o in show]
        tv.line_chart(d, (140, 170, 760, 560), ST, series, ylim=(0, 105), xlabel="训练步数", ylabel="测试准确率 %", legend_xy=(820, 520),
                      xticks=[(s, str(s)) for s in (0, 200, 400, 600)])
        rows = [("SGD", f"{FIN[('sgd', 0.3)]:.1%}（未到 99%）", RED), ("SGD + 动量", f"第 {T99[('momentum', 0.03)]} 步", AMBER)]
        if seg >= 2:
            rows.append(("Adam", f"第 {T99[('adam', 1e-3)]} 步", GREEN))
        d.text((820, 200), "达到 99% 的步数", font=F_SMALL, fill=MUTED)
        for i, (a, b, col) in enumerate(rows):
            d.text((820, 245 + i * 45), a, font=F_SMALL, fill=col)
            d.text((1000, 245 + i * 45), b, font=F_SMALL, fill=TEXT)
    if seg >= 3:
        d.text((820, 390), "学习率太大：训练失败", font=F_SMALL, fill=RED)
        d.text((820, 430), "SGD η = 1 · Adam η = 0.01", font=F_SMALL, fill=TEXT)


def sc_adamw(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "AdamW：解耦的权重衰减（Loshchilov & Hutter, 2019）")
    d.text((140, 170), "L2 惩罚：", font=F_BODY, fill=RED)
    tv.fpaste(c, r"$g_t\leftarrow g_t+\lambda\theta_t$", 520, 190, 32)
    d.text((700, 190), "→ 衰减也被 √v 缩放", font=F_BODY, fill=TEXT, anchor="lm")
    vk.text_c(d, (640, 260), "梯度大的参数，反而衰减得少", F_SMALL, RED)
    if seg >= 1:
        d.text((140, 350), "AdamW：", font=F_BODY, fill=GREEN)
        tv.fpaste(c, r"$\theta_{t+1}=\theta_t-\eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\theta_t\right)$", 700, 370, 32)
        vk.text_c(d, (640, 460), "每一步直接把权重缩小一点 · 今天训练大模型的默认选择", F_SMALL, GREEN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("自适应", "按历史梯度大小调整各自步长"), ("Adam", f"加法模型：第 {T99[('adam', 1e-3)]} 步到 99%"),
                      ("AdamW", "权重衰减与梯度更新解耦")], seg, "下一期：学习率调度")


RENDER = {"title": sc_title, "adagrad": sc_adagrad, "adam": sc_adam, "experiment": sc_experiment, "adamw": sc_adamw, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "op-adam")
