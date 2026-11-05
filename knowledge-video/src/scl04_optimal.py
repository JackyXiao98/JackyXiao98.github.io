"""Scaling Laws 04 — 计算最优：Kaplan 与 Chinchilla.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] SCL_CACHE=<dir> SHAKESPEARE=<file> python scl04_optimal.py <out_dir>

Real computations (sclx.exp_isoflop, JAX): three training-compute budgets C = 6 N D = 6e11 / 1.8e12 / 5.4e12 FLOPs.
For each budget every width whose step count C / (6 N x 2,048) lies in 250 ... 18,000 is trained for exactly that
many steps with its own full cosine schedule (Chinchilla's IsoFLOP approach). Validation loss vs N; a parabola in
log N through the points gives the loss-minimising N_opt and D_opt = C / (6 N_opt) per budget; power-law fit
N_opt ~ C^a over the three budgets.
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

R = sclx.cached(CACHE, "isoflop", sclx.exp_isoflop)
CS = sorted({c for c, _ in R})
CURVE = {C: sorted((R[(C, d)]["N"], R[(C, d)]["val"], d) for (c, d) in R if c == C) for C in CS}
NOPT, BESTW = {}, {}
for C in CS:
    n = np.log([x[0] for x in CURVE[C]])
    l = np.array([x[1] for x in CURVE[C]])
    a, b, _ = np.polyfit(n, l, 2)
    NOPT[C] = float(np.exp(-b / (2 * a)))
    BESTW[C] = min(CURVE[C], key=lambda x: x[1])[2]
DOPT = {C: C / (6 * NOPT[C]) for C in CS}
SLOPE = np.polyfit(np.log(CS), np.log([NOPT[C] for C in CS]), 1)[0]
TPP = {C: DOPT[C] / NOPT[C] for C in CS}
print({C: [(x[2], round(x[1], 3)) for x in CURVE[C]] for C in CS}, NOPT, DOPT, SLOPE, TPP, BESTW)
assert BESTW == {6e11: 32, 1.8e12: 48, 5.4e12: 64}
assert round(R[(6e11, 32)]["val"], 2) == 1.91 and round(R[(1.8e12, 48)]["val"], 2) == 1.77 and round(R[(5.4e12, 64)]["val"], 2) == 1.66
assert round(SLOPE, 2) == 0.73 and round(NOPT[6e11] / 1e3) == 17 and round(NOPT[5.4e12] / 1e3) == 86
assert round(DOPT[5.4e12] / DOPT[6e11], 2) == 1.80

SCENES = [
    ("title", [
        "如果计算预算是固定的，应该训练一个大模型、少读一点数据，还是一个小模型、多读一点数据？这是缩放定律最有实际意义的问题。二零二零年和二零二二年的两篇论文给出了很不一样的答案。",
    ]),
    ("isoflop", [
        "训练的计算量约等于六乘参数量乘训练词元数。我们固定三个计算预算，每个预算下训练几个大小不同的模型：模型越大，能读的数据就越少。每个模型都用自己完整的学习率计划训练到预算用完。",
        f"结果在每个预算下都是一条先降后升的曲线。模型太小，学不下；模型太大，数据读得太少。最低点就是这个预算下的最优模型：三个预算的最优宽度分别是三十二、四十八和六十四。",
    ]),
    ("frontier", [
        f"把三个最低点连起来，看最优模型怎样随预算增长。在我们的实验里，计算量每增加三倍，最优参数量增加两倍多，拟合出的指数约是{dec_zh(SLOPE, 2)}；而最优的数据量只增加约三四成。增加的计算大部分给了更大的模型。",
        "这和龙猫论文的结论不同，反而接近更早的结论。一个重要原因是我们的数据只有一百万个字符，最优的训练要把它重复六到十遍。重复的数据价值越来越低，所以多出来的计算更值得花在模型上。数据受限时，计算最优的分配会偏向更大的模型。",
    ]),
    ("history", [
        "二零二零年的缩放定律论文得出的结论是，增加的计算大部分应该用来放大模型，数据只需要增加一点。按照这个思路，当时的大模型参数很多，但训练数据相对较少。",
        "二零二二年的龙猫论文发现，这个结论部分来自一个实验细节：之前的实验让不同长度的训练共用同一个学习率计划，而没有为每个计算量单独设置完整的计划。重新做之后，他们发现参数和数据应该同比例增长，大约每个参数配二十个训练词元。他们用这个比例训练的七百亿参数模型，超过了参数多得多的模型。",
    ]),
    ("today", [
        "但今天的模型往往远远超过这个比例。第三代羊驼的八十亿参数模型训练了约十五万亿个词元，每个参数将近两千个。原因是龙猫比例只考虑训练成本；而模型训练好以后要被调用无数次，小模型推理便宜，所以人们愿意多花训练计算，换一个更小但同样好的模型。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，固定计算时，损失随模型大小先降后升，存在最优的模型大小。",
        "第二，在我们数据受限的实验里，最优模型增长得比数据快，指数约零点七三。",
        "第三，龙猫比例是训练最优，考虑推理成本时，人们会训练更小、读更多数据的模型。下一期，我们看怎样从小模型外推。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 4, "计算最优", "Kaplan 与 Chinchilla", SERIES)


def sc_isoflop(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "固定计算量：C = 6 N D（IsoFLOP）")
    cols = {6e11: VIOLET, 1.8e12: CYAN, 5.4e12: GREEN}
    x0, y0, x1, y1 = 140, 200, 820, 580
    px = lambda n: x0 + (np.log10(n) - 3.6) / (5.8 - 3.6) * (x1 - x0)
    py = lambda l: y1 - (l - 1.6) / (2.3 - 1.6) * (y1 - y0)
    d.rectangle([x0, y0, x1, y1], outline=(30, 41, 59))
    for C in CS:
        pts = [(px(n), py(l)) for n, l, _ in CURVE[C]]
        d.line(pts, fill=cols[C], width=3)
        for (n, l, w), p in zip(CURVE[C], pts):
            best = w == BESTW[C] and seg >= 1
            r = 9 if best else 5
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=AMBER if best else cols[C])
    for n, lab in ((1e4, "1e4"), (1e5, "1e5"), (5e5, "5e5")):
        d.text((px(n), y1 + 18), lab, font=vk.font(18, mono=True), fill=MUTED, anchor="mm")
    d.text((x1, y1 + 44), "非嵌入参数 N（对数）", font=F_SMALL, fill=MUTED, anchor="rm")
    d.text((x0, y0 - 30), "验证损失", font=F_SMALL, fill=MUTED)
    for i, C in enumerate(CS):
        d.text((880, 230 + i * 40), f"C = {C:.1e}", font=vk.font(22, mono=True), fill=cols[C])
    if seg >= 1:
        for i, C in enumerate(CS):
            d.text((880, 380 + i * 40), f"最优宽度 {BESTW[C]}", font=F_SMALL, fill=AMBER)


def sc_frontier(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "最优模型随预算增长")
    tv.line_chart(d, (140, 200, 820, 580), CS, [("N_opt", [NOPT[C] for C in CS], AMBER), ("D_opt / 100", [DOPT[C] / 100 for C in CS], CYAN)],
                  xlog=True, ylog=True, xticks=[(C, f"{C:.1e}") for C in CS], xlabel="计算量 C（对数）", legend_xy=(880, 220))
    f = vk.font(22, mono=True)
    for i, C in enumerate(CS):
        d.text((880, 320 + i * 40), f"N≈{NOPT[C] / 1e3:.0f}k  D≈{DOPT[C] / 1e6:.1f}M", font=f, fill=TEXT)
    d.text((880, 460), f"N_opt ∝ C^{SLOPE:.2f}", font=F_BODY, fill=AMBER)


def sc_history(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "两篇论文")
    d.rounded_rectangle([110, 190, 1170, 290], 12, outline=MUTED, width=2, fill=(12, 19, 34))
    d.text((140, 240), "Kaplan et al.（2020）", font=F_BODY, fill=MUTED, anchor="lm")
    d.text((520, 240), "增加的计算主要用来放大模型（N ∝ C^0.73）", font=F_SMALL, fill=TEXT, anchor="lm")
    if seg >= 1:
        d.rounded_rectangle([110, 320, 1170, 470], 12, outline=GREEN, width=2, fill=(12, 19, 34))
        d.text((140, 360), "Hoffmann et al.（2022）", font=F_BODY, fill=GREEN, anchor="lm")
        d.text((520, 360), "每个计算量用完整的学习率计划重新做", font=F_SMALL, fill=TEXT, anchor="lm")
        d.text((520, 405), "N 与 D 同比例增长：约 20 个词元 / 参数", font=F_SMALL, fill=AMBER, anchor="lm")
        d.text((520, 445), "Chinchilla 70B 超过参数多得多的模型", font=F_SMALL, fill=TEXT, anchor="lm")


def sc_today(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "今天：远超 Chinchilla 比例")
    tv.bar_chart(d, (200, 220, 760, 560), ["Chinchilla 比例", "Llama 3 8B"], [20, 15e12 / 8e9], [GREEN, AMBER], fmt="{:.0f}", vmax=2100)
    d.text((200, 180), "每个参数的训练词元数", font=F_SMALL, fill=MUTED)
    d.text((820, 320), "训练一次，推理无数次", font=F_SMALL, fill=TEXT)
    d.text((820, 360), "→ 训练更小、读更多数据的模型", font=F_SMALL, fill=AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("IsoFLOP", "固定计算，存在最优模型大小"), ("数据受限", "最优模型增长更快（指数 0.73）"),
                      ("推理成本", "实际模型远超 20 词元 / 参数")], seg, "下一期：从小模型外推")


RENDER = {"title": sc_title, "isoflop": sc_isoflop, "frontier": sc_frontier, "history": sc_history, "today": sc_today, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "scl-optimal")
