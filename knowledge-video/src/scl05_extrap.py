"""Scaling Laws 05 — 从小模型外推：拟合与预测.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] SCL_CACHE=<dir> SHAKESPEARE=<file> python scl05_extrap.py <out_dir>

Real computations (sclx.exp_size): fit on the five smallest models only (widths 16-64), predict the two largest
(widths 96, 128; 2.2x and 4x the largest fitted model): (a) pure power law log L = c + s log N (least squares);
(b) L = E + A N^(-alpha) with E chosen on a grid in [0, 1.5] (step 0.005) and log(L - E) fitted linearly.
Prediction errors against the measured validation losses.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import sclx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Scaling Laws"
CACHE = os.environ.get("SCL_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "sclcache"))

R = sclx.cached(CACHE, "size", sclx.exp_size)
W = list(sclx.WIDTHS)
N = np.array([R[d]["N"] for d in W], float)
L = np.array([R[d]["val"] for d in W])
CP = np.polyfit(np.log(N[:5]), np.log(L[:5]), 1)
PRED_P = np.exp(np.polyval(CP, np.log(N)))
best = None
for E in np.linspace(0, 1.5, 301):
    y = np.log(L[:5] - E)
    A = np.vstack([np.ones(5), np.log(N[:5])]).T
    cf = np.linalg.lstsq(A, y, rcond=None)[0]
    err = ((A @ cf - y) ** 2).sum()
    if best is None or err < best[0]:
        best = (err, E, np.exp(cf[0]), -cf[1])
_, EF, AF, ALF = best
PRED_E = EF + AF * N ** -ALF
ERR_P = PRED_P[5:] - L[5:]
ERR_E = PRED_E[5:] - L[5:]
print(np.round(PRED_P, 3), np.round(PRED_E, 3), L, EF, ALF, ERR_P, ERR_E)
assert round(EF, 2) == 1.39 and round(ALF, 2) == 0.31
assert [round(x, 3) for x in PRED_P[5:]] == [1.561, 1.496] and [round(x, 3) for x in PRED_E[5:]] == [1.608, 1.573]
assert round(ERR_P[1], 2) == -0.10 and round(ERR_E[1], 3) == -0.023

SCENES = [
    ("title", [
        "缩放定律最有价值的用法，是在训练大模型之前，先用一批小模型预测它的效果。这一期我们就这样做：只用小模型的结果拟合，去预测两个更大的模型，再和实际结果比较。",
    ]),
    ("setup", [
        "我们拿上一期的七个模型，只用最小的五个来拟合，把最大的两个藏起来。它们的参数量分别是拟合范围里最大模型的约二点二倍和四倍。",
        "我们试两种公式。第一种是纯幂律，在对数坐标里画一条直线。第二种在幂律前面加一个不可约的常数，也就是第一期讲的那种形式。",
    ]),
    ("predict", [
        f"纯幂律的预测太乐观了。对最大的模型，它预测损失是{dec_zh(PRED_P[-1], 2)}，实际是{dec_zh(L[-1], 2)}，差了约零点一。",
        f"带常数项的公式拟合出的常数约是{dec_zh(EF, 2)}，预测最大的模型是{dec_zh(PRED_E[-1], 2)}，只差约零点零二。常数项让曲线逐渐变平，更符合实际。",
    ]),
    ("pitfalls", [
        "但即使是更好的公式，也略微低估了损失。原因是我们在上一期看到的：大模型开始过拟合、训练也不充分。外推的前提是，大模型和小模型处在同样的条件下，数据足够、训练充分、超参数也调得一样好。",
        "真实的工作对此非常小心。第四代生成式预训练模型的技术报告说，他们用计算量只有千分之一到万分之一的模型，准确预测了最终模型的损失。但这背后，是整套训练流程在所有规模上都保持稳定。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，缩放定律可以用小模型预测大模型。",
        "第二，在我们的实验里，加上不可约常数的公式，误差从约零点一降到约零点零二。",
        "第三，外推要求各个规模的条件一致，数据不够或训练不够都会让预测偏乐观。下一期，我们看超参数怎样随规模变化。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 5, "从小模型外推", "拟合与预测", SERIES)


def sc_setup(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "只用最小的五个模型拟合")
    x0, y0, x1, y1 = 140, 200, 820, 580
    px = lambda n: x0 + (np.log10(n) - 3.7) / (5.7 - 3.7) * (x1 - x0)
    py = lambda l: y1 - (l - 1.45) / (2.1 - 1.45) * (y1 - y0)
    d.rectangle([x0, y0, x1, y1], outline=(30, 41, 59))
    for i in range(7):
        col = GREEN if i < 5 else (71, 85, 105)
        r = 9
        x, y = px(N[i]), py(L[i]) if i < 5 else py(1.5)
        if i < 5:
            d.ellipse([x - r, y - r, x + r, y + r], fill=col)
        else:
            d.text((x, y), "?", font=F_BODY, fill=AMBER, anchor="mm")
    d.text((x1, y1 + 30), "非嵌入参数（对数）", font=F_SMALL, fill=MUTED, anchor="rm")
    if seg >= 1:
        tv.fpaste(c, r"$\mathrm{(a)}\ L=A\,N^{-\alpha}$", 1010, 280, 30)
        tv.fpaste(c, r"$\mathrm{(b)}\ L=E+A\,N^{-\alpha}$", 1010, 350, 30)


def sc_predict(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "预测两个更大的模型")
    x0, y0, x1, y1 = 140, 200, 820, 580
    px = lambda n: x0 + (np.log10(n) - 3.7) / (5.7 - 3.7) * (x1 - x0)
    py = lambda l: y1 - (l - 1.45) / (2.1 - 1.45) * (y1 - y0)
    d.rectangle([x0, y0, x1, y1], outline=(30, 41, 59))
    ns = np.logspace(3.75, 5.65, 60)
    d.line([(px(n), py(np.exp(np.polyval(CP, np.log(n))))) for n in ns], fill=RED, width=3)
    if seg >= 1:
        d.line([(px(n), py(EF + AF * n ** -ALF)) for n in ns], fill=AMBER, width=3)
    for i in range(7):
        x, y = px(N[i]), py(L[i])
        col = GREEN if i < 5 else CYAN
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=col)
    d.text((x1, y1 + 30), "非嵌入参数（对数）", font=F_SMALL, fill=MUTED, anchor="rm")
    f = vk.font(22, mono=True)
    d.text((870, 200), "宽度 128（N = 393k）", font=F_SMALL, fill=MUTED)
    for lab, val, col, y in (("实测", L[-1], CYAN, 250), ("纯幂律", PRED_P[-1], RED, 295)):
        d.text((870, y), lab, font=F_SMALL, fill=col)
        d.text((1010, y), f"{val:.3f}", font=f, fill=col)
    if seg >= 1:
        d.text((870, 340), "带常数项", font=F_SMALL, fill=AMBER)
        d.text((1010, 340), f"{PRED_E[-1]:.3f}", font=f, fill=AMBER)
        d.text((870, 400), f"拟合常数 E ≈ {EF:.2f}", font=F_SMALL, fill=AMBER)


def sc_pitfalls(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "外推的前提：各个规模的条件一致")
    for i, (t, col) in enumerate([("数据足够（不过拟合）", CYAN), ("训练充分", GREEN), ("超参数调得一样好", AMBER)]):
        vk.text_c(d, (640, 210 + i * 70), t, F_BODY, col)
    if seg >= 1:
        vk.text_c(d, (640, 480), "GPT-4 技术报告（2023）：用 1/1,000 – 1/10,000 计算的模型预测了最终损失", F_SMALL, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 4, [("用法", "小模型预测大模型"), ("实验", "加常数项：误差 0.10 → 0.02"),
                      ("前提", "各规模条件一致")], seg, "下一期：超参数")


RENDER = {"title": sc_title, "setup": sc_setup, "predict": sc_predict, "pitfalls": sc_pitfalls, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "scl-extrap")
