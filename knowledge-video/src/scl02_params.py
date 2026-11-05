"""Scaling Laws 02 — 模型越大越好：参数量的缩放.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] SCL_CACHE=<dir> SHAKESPEARE=<file> python scl02_params.py <out_dir>

Real computations (sclx.exp_size, JAX): seven 2-layer character-level Transformers on Tiny Shakespeare with widths
16 / 24 / 32 / 48 / 64 / 96 / 128 (6,144 ... 393,216 non-embedding parameters), each trained 4,000 steps (8.2 M
characters); validation loss. Fit L(N) = E + A N^(-alpha) by least squares in log space over a grid of E (E in
[0, 1.5]); with all seven points the best E is at the boundary 0, i.e. a pure power law with alpha ~ 0.061.
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
TR = np.array([R[d]["hist"][-1]["train"] for d in W])
CF = np.polyfit(np.log(N), np.log(L), 1)
ALPHA = -CF[0]
PER10 = 10 ** (-ALPHA)
print(dict(zip(W, np.round(L, 3))), ALPHA, PER10, np.round(TR, 3))
assert [round(x, 3) for x in L] == [2.054, 1.908, 1.820, 1.727, 1.671, 1.617, 1.596]
assert round(ALPHA, 3) == 0.061 and round(PER10, 3) == 0.870 and N[0] == 6144 and N[-1] == 393216
assert round(TR[-1], 2) == 1.38 and round(L[-1] - TR[-1], 2) == 0.22

SCENES = [
    ("title", [
        "上一期说，损失随规模按幂律下降。这一期我们自己测一测：在莎士比亚文本上训练一系列大小不同的小模型，看损失怎样随参数量变化。",
    ]),
    ("setup", [
        "所有模型都是两层的字符级变换器，只改宽度，从十六到一百二十八，一共七个。按照缩放定律论文的惯例，参数量只算嵌入层以外的部分，从约六千到约三十九万，相差六十多倍。每个模型都训练四千步，读过同样多的文字。",
    ]),
    ("result", [
        f"结果很规整。最小的模型验证损失是{dec_zh(L[0], 2)}，宽度三十二是{dec_zh(L[2], 2)}，六十四是{dec_zh(L[4], 2)}，最大的是{dec_zh(L[-1], 2)}。模型越大，损失越低。",
        f"把横轴和纵轴都取对数，七个点几乎排成一条直线。拟合出的指数约是{dec_zh(ALPHA, 3)}：参数量每增加十倍，损失大约变成原来的百分之{int_zh(round(PER10 * 100))}左右。",
    ]),
    ("bend", [
        f"但仔细看，最右边的两个点开始向上偏离直线，收益在变小。一个原因是这几个大模型开始过拟合了：最大的模型在训练集上的损失只有{dec_zh(TR[-1], 2)}，比验证集低了约零点二二。我们的数据只有一百万个字符，模型大到一定程度，数据就不够了。",
        "另一个原因是训练步数固定。大模型需要更多的训练才能发挥出来，而四千步对它们来说可能还不够。这两个因素，正是后面几期要讨论的：数据量，以及在固定计算下怎样分配模型和数据。",
    ]),
    ("real", [
        "缩放定律论文在大得多的范围里，从几十万到十几亿参数，观察到了同样规整的幂律，参数量的指数约为零点零七六。我们的指数约零点零六，量级是一致的，但小实验里的具体数值不能直接外推到大模型。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，在我们的实验里，参数量增加六十多倍，损失从二点零五降到一点六。",
        "第二，在对数坐标里近似一条直线，指数约零点零六。",
        "第三，最大的几个模型开始偏离直线，因为数据不够、训练不够。下一期，我们看数据量的缩放与过拟合。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 2, "模型越大越好", "参数量的缩放", SERIES)


def sc_setup(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "七个模型：只改宽度（两层，字符级，4,000 步）")
    f = vk.font(24, mono=True)
    d.text((200, 180), "宽度", font=F_SMALL, fill=MUTED)
    d.text((420, 180), "非嵌入参数 N = 24 d²", font=F_SMALL, fill=MUTED)
    for i, w in enumerate(W):
        y = 225 + i * 52
        d.text((200, y), f"{w}", font=f, fill=CYAN)
        bar = np.log10(N[i]) - 3.6
        d.rectangle([420, y + 4, 420 + bar * 330, y + 30], fill=GREEN)
        d.text((430 + bar * 330, y), f"{int(N[i]):,}", font=f, fill=TEXT)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "验证损失 vs 参数量")
    if seg == 0:
        tv.bar_chart(d, (140, 220, 860, 580), [str(w) for w in W], L, [GREEN] * 7, fmt="{:.3f}", vmax=2.3)
        d.text((140, 180), "验证损失（宽度）", font=F_SMALL, fill=MUTED)
    else:
        fit = np.exp(np.polyval(CF, np.log(N)))
        tv.line_chart(d, (140, 200, 820, 580), N, [("实测", L, GREEN), ("幂律拟合", fit, AMBER)], xlog=True, ylog=True,
                      xticks=[(1e4, "1e4"), (1e5, "1e5"), (3.9e5, "3.9e5")], xlabel="非嵌入参数（对数）", legend_xy=(880, 220))
        d.text((140, 165), "验证损失（对数）", font=F_SMALL, fill=MUTED)
        d.text((880, 330), f"指数 α ≈ {ALPHA:.3f}", font=F_BODY, fill=AMBER)
        d.text((880, 390), f"每 10 倍：× {PER10:.2f}", font=F_BODY, fill=TEXT)


def sc_bend(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "大模型开始偏离：训练与验证的差距")
    tv.line_chart(d, (140, 200, 820, 580), list(range(7)), [("验证", L, GREEN), ("训练", TR, CYAN)],
                  xticks=[(i, str(w)) for i, w in enumerate(W)], xlabel="宽度", ylim=(1.3, 2.1), legend_xy=(880, 220))
    d.text((140, 165), "损失", font=F_SMALL, fill=MUTED)
    d.text((880, 330), f"宽度 128：差距 {L[-1] - TR[-1]:.2f}", font=F_BODY, fill=RED)
    if seg >= 1:
        d.text((880, 400), "数据只有约 100 万字符", font=F_SMALL, fill=TEXT)
        d.text((880, 440), "4,000 步对大模型不够", font=F_SMALL, fill=TEXT)


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "和论文对比")
    tv.bar_chart(d, (300, 220, 860, 560), ["Kaplan et al. 2020", "我们的小实验"], [0.076, ALPHA], [CYAN, GREEN], fmt="{:.3f}", vmax=0.09)
    d.text((300, 180), "参数量的指数 α", font=F_SMALL, fill=MUTED)
    d.text((920, 360), "量级一致，", font=F_SMALL, fill=TEXT)
    d.text((920, 400), "但不能直接外推", font=F_SMALL, fill=AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("实验", "参数 × 64，损失 2.05 → 1.60"), ("幂律", "对数坐标近似直线，α ≈ 0.06"),
                      ("偏离", "数据不够、训练不够")], seg, "下一期：数据量与过拟合")


RENDER = {"title": sc_title, "setup": sc_setup, "result": sc_result, "bend": sc_bend, "real": sc_real, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "scl-params")
