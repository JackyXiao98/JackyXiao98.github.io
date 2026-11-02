"""Distillation 04 — 自蒸馏与“学生超过老师”.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] KD_CACHE=<dir> TF_CACHE=<dir> python kd04_selfdistill.py <out_dir>

Real computations (JAX):
- kdx.born_again: same-size networks [64, 64, 10] on 300 labelled digits; generation k+1 is trained on
  the hard labels plus the soft labels of generation k (T = 2, alpha = 0.5); 4 generations, 5 seeds;
  compared with label smoothing (0.1).
- kdx.addition_seqkd: the tfm.py addition model after 120 steps (27.7% exact match) labels 7,700
  problems; students (d = 16, 32, 64) trained on those labels plus 300 correct ones.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import kdx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Distillation"
CACHE = os.environ.get("KD_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "kdcache"))
TF_CACHE = os.environ.get("TF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "tfcache"))

B = kdx.cached(CACHE, "ba", kdx.born_again)
GEN = B["gens"].mean(0)
LS = float(B["ls"].mean())
Q = kdx.cached(CACHE, "seq", kdx.addition_seqkd, tf_cache=TF_CACHE)
WT = Q["weak_teacher"]
WEAK = {d: Q["res"][d]["weak"] for d in (16, 32, 64)}
L300 = {d: Q["res"][d]["labels300"] for d in (16, 32, 64)}
print(GEN, LS, WT, Q["q_weak"], WEAK, L300)
assert np.round(GEN * 100, 1).tolist() == [93.2, 93.3, 93.2, 93.4] and abs(LS - 0.947) < 0.001
assert abs(WT - 0.277) < 1e-9 and all(v > WT for v in WEAK.values()) and abs(WEAK[32] - 0.31) < 1e-9 and max(L300.values()) < 0.07
assert abs(Q["q_weak"] - 0.292) < 0.001

SCENES = [
    ("title", [
        "前面的老师都比学生大。这一期看两个有些反直觉的问题。如果老师和学生一样大，蒸馏还有用吗？学生有没有可能超过老师？",
    ]),
    ("ban", [
        "二零一八年的重生网络论文做了这样的实验。先训练一个网络，再用它的软标签，训练一个结构完全相同的新网络，一代一代地传下去。",
        "论文报告，在图像分类上，学生常常超过了老师。这类和自己蒸馏的方法，统称为自蒸馏。",
    ]),
    ("ours", [
        "我们在手写数字上照着做。网络有一个六十四宽的隐藏层，只用三百张带标签的图片，一共四代，每代做五次实验。",
        f"第一代的平均准确率是{pct_zh(GEN[0], 1)}，后面三代分别是{pct_zh(GEN[1], 1)}、{pct_zh(GEN[2], 1)}和{pct_zh(GEN[3], 1)}，几乎没有变化。",
        f"作为对照，直接用标签平滑训练，也就是把硬标签的一小部分概率平均分给其他类别，准确率是{pct_zh(LS, 1)}，反而更高。",
    ]),
    ("why", [
        "所以自蒸馏的效果并不稳定，在我们的小实验里没有出现。研究者对它的解释也有争论。",
        "一种观点认为，自蒸馏的作用主要是一种正则化，类似标签平滑，让模型不要过于自信。另一种观点认为，老师的软标签提供了样本之间的相似信息，这是均匀的标签平滑做不到的。",
    ]),
    ("weak", [
        f"再看学生超过老师的另一种情形，老师本身就会犯错。我们用变换器系列里只训练了一百二十步的加法模型当老师，它的准确率只有{pct_zh(WT, 1)}。",
        f"让它给七千七百道题写答案，其中只有{pct_zh(Q['q_weak'], 1)}是对的。学生用这些答案，再加上三百道正确的题来训练。只用那三百道题，学生的准确率不到百分之七。",
        f"结果三个不同大小的学生，准确率都在百分之三十左右，宽度三十二的学生是{pct_zh(WEAK[32], 1)}，都略微超过了老师。",
        "一种解释是，老师的错误有一部分是随机的，学生在大量数据上学到的是其中一致的规律，随机的错误被平均掉了。但学生终究不会超过老师太多，老师的水平基本决定了学生的上限。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，自蒸馏用和自己一样大的模型做老师，论文中常有提升，但效果并不稳定。",
        "第二，它的作用部分来自正则化，在我们的实验里不如标签平滑。",
        "第三，学生可以略微超过一个会犯错的老师，但上限仍然由老师决定。下一期，我们进入大语言模型的蒸馏，看前向和反向相对熵。感谢大家的观看。",
    ]),
]
assert all(abs(v - 0.30) < 0.02 for v in WEAK.values())


def sc_title(c, u, seg, su):
    tv.title_card(c, 4, "自蒸馏", "学生能超过老师吗？", SERIES)


def sc_ban(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "重生网络（Born-Again Networks, Furlanello et al., 2018）")
    for i in range(4):
        x = 140 + i * 270
        col = [VIOLET, CYAN, CYAN, CYAN][i]
        d.rounded_rectangle([x, 220, x + 200, 320], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 100, 255), f"第 {i + 1} 代", F_BODY, col)
        vk.text_c(d, (x + 100, 295), "同样的结构", F_SMALL, MUTED)
        if i < 3:
            vk.arrow(d, (x + 205, 270), (x + 265, 270), AMBER, 3)
    vk.text_c(d, (640, 370), "软标签 → 下一代", F_SMALL, AMBER)
    if seg >= 1:
        vk.text_c(d, (640, 470), "论文：图像分类上学生常常超过老师 → “自蒸馏”", F_BODY, TEXT)


def sc_ours(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "我们的实验：[64, 64, 10]，300 个标签，5 次平均")
    if seg >= 1:
        labs = ["第 1 代", "第 2 代", "第 3 代", "第 4 代"] + (["标签平滑"] if seg >= 2 else [])
        vals = list(GEN * 100) + ([LS * 100] if seg >= 2 else [])
        tv.bar_chart(d, (150, 200, 150 + 200 * len(vals), 560), labs, vals, [CYAN] * 4 + [GREEN], fmt="{:.1f}%", vmax=105)
        vk.text_c(d, (640, 640), "四代几乎没有变化", F_BODY, MUTED)
    if seg >= 2:
        d.rectangle([380, 620, 900, 660], fill=vk.BG)
        vk.text_c(d, (640, 640), "标签平滑（ε = 0.1）反而更高", F_BODY, GREEN)


def sc_why(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "自蒸馏为什么（有时）有效？")
    vk.text_c(d, (640, 180), "效果不稳定 · 解释有争论", F_BODY, AMBER)
    if seg >= 1:
        for i, (a, b, col) in enumerate([("正则化", "类似标签平滑，不要过于自信（Yuan 2020; Mobahi 2020）", CYAN),
                                         ("样本间相似信息", "软标签因样本而异，均匀的标签平滑做不到", VIOLET)]):
            y = 270 + i * 140
            d.rounded_rectangle([120, y, 1160, y + 100], 14, outline=col, width=2, fill=(12, 19, 34))
            d.text((150, y + 50), a, font=F_BODY, fill=col, anchor="lm")
            d.text((420, y + 50), b, font=F_SMALL, fill=TEXT, anchor="lm")


def sc_weak(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "会犯错的老师：加法模型（训练 120 步）")
    d.text((120, 150), f"老师准确率 {WT:.1%} · 7,700 道题的答案中 {Q['q_weak']:.1%} 正确", font=F_SMALL, fill=VIOLET)
    if seg >= 1:
        d.text((120, 190), f"学生数据 = 老师的答案 + 300 道正确题（只用 300 道：< 7%）", font=F_SMALL, fill=MUTED)
    if seg >= 2:
        labs = ["学生 d=16", "学生 d=32", "学生 d=64"]
        tv.bar_chart(d, (200, 280, 1000, 580), labs, [WEAK[k] * 100 for k in (16, 32, 64)], [CYAN] * 3, fmt="{:.1f}%", vmax=45)
        y = 580 - WT * 100 / 45 * 300
        d.line([(200, y), (1000, y)], fill=VIOLET, width=2)
        d.text((1010, y), f"老师 {WT:.1%}", font=F_SMALL, fill=VIOLET, anchor="lm")
    if seg >= 3:
        vk.text_c(d, (640, 660), "随机的错误被平均掉 · 但上限仍由老师决定", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("自蒸馏", "同样大小的老师，效果不稳定"), ("正则化", "实验中不如标签平滑"),
                      ("超过老师", "略微超过会犯错的老师")], seg, "下一期：前向 KL 与反向 KL")


RENDER = {"title": sc_title, "ban": sc_ban, "ours": sc_ours, "why": sc_why, "weak": sc_weak, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "kd-selfdistill")
