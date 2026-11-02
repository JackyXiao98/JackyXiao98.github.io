"""Distillation 02 — 暗知识：软标签里藏着什么.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] KD_CACHE=<dir> python kd02_dark.py <out_dir>

Real computations (kdx.dark_knowledge, JAX): the teacher of episode 01; its average soft output
(T = 4) on test images of each class; a student [64, 16, 10] trained on a transfer set from which all
117 training images of "3" were removed — once on the teacher's soft labels only (no true labels), once
on hard labels; the never-seen class's logit is then raised until "3" is predicted for 10% of the
(unlabelled) training images; accuracy on the 66 test images of "3", 3 seeds.
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

D = kdx.cached(CACHE, "dark", kdx.dark_knowledge)
SIM = D["sim"]
m = lambda k: float(np.mean(D[k]))
RAW = np.array(D["soft_held_raw"])
print(np.round(SIM[3], 3), np.round(SIM[8], 3), {k: m(k) for k in ("soft_held", "hard_held", "soft_all", "hard_all", "soft_only_full")}, RAW)
assert D["n_removed"] == 117 and abs(SIM[3, 3] - 0.869) < 1e-3 and abs(SIM[3, 2] - 0.038) < 1e-3 and SIM[3].argsort()[-3] == 9
assert abs(SIM[8, 1] - 0.06) < 0.001 and SIM[8].argsort()[-2] == 1
assert RAW.max() < 0.25 and abs(m("soft_held") - 0.939) < 0.001 and abs(m("hard_held") - 0.581) < 0.001
assert abs(m("soft_only_full") - 0.956) < 0.001 and all(np.array(D["hard_held_raw"]) == 0)

SCENES = [
    ("title", [
        "上一期我们看到，软标签能让学生学得更好。这一期深入看软标签里到底藏着什么，辛顿把它叫做暗知识。",
    ]),
    ("similarity", [
        "我们把老师在每一类测试图片上的平均输出画成一张表，温度取四。每一行是真实的数字，每一列是老师给出的概率。",
        f"对角线以外的数字虽然小，却很有规律。对于三，老师平均给三{pct_zh(SIM[3, 3])}，给二{dec_zh(SIM[3, 2] * 100, 1)}个百分点，给九{dec_zh(SIM[3, 9] * 100, 1)}个百分点，给八{dec_zh(SIM[3, 8] * 100, 1)}个百分点，而给四几乎是零。",
        "这些小概率描述了类别之间的相似关系。哪些数字容易混淆，哪些完全不像，这些信息在硬标签里是看不到的。",
    ]),
    ("heldout", [
        f"辛顿的论文里有一个惊人的实验，我们照着做一遍。把训练集里所有的三都拿掉，一共{int_zh(D['n_removed'])}张，学生一次也没见过三。",
        "学生只学习老师在剩下图片上的软标签，完全不用真实标签。三这一类唯一的信息，来自老师在其他数字上给三的那一点点概率。",
        f"直接测试，学生几乎认不出三，三次实验的准确率最高只有{pct_zh(RAW.max())}。原因是三的输出整体偏低，因为在训练中它从来没有当过第一名。",
    ]),
    ("bias", [
        "辛顿的做法是只调高三这一类的偏置。我们让偏置大到，在没有标签的训练图片上，大约十分之一被判为三，这只用到了三大约占十分之一这个先验。",
        f"调整之后，只学过软标签的学生，在三上的准确率达到了{pct_zh(m('soft_held'), 1)}。作为对照，只用硬标签训练的学生做同样的调整，在三上只有{pct_zh(m('hard_held'), 1)}。",
        "学生从来没见过一张三，却从老师对其他数字的犹豫中，学会了三长什么样。这就是暗知识。",
    ]),
    ("nolabel", [
        f"再看一个结果。如果在全部训练图片上，只用老师的软标签，一个真实标签都不用，学生的准确率是{pct_zh(m('soft_only_full'), 1)}，和第一期老师的{pct_zh(D['teacher'], 1)}相差不远。",
        "这说明蒸馏不一定需要标注数据。只要有老师和一批没有标签的输入，就可以把知识传给学生，后面讲大模型蒸馏时，这一点非常重要。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，软标签里的小概率描述了类别之间的相似关系，叫做暗知识。",
        "第二，学生从没见过三，只靠软标签加上一个偏置调整，就能认出九成以上的三。",
        "第三，蒸馏可以只用老师和无标签的数据。下一期，我们看学生能不能模仿老师的中间层，而不只是输出。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 2, "暗知识", "软标签里藏着什么", SERIES)


def sc_similarity(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "老师在每一类上的平均软输出（T = 4）")
    M = SIM * 100
    tv.heat(d, (160, 170, 700, 650), M, 0, 12, labels_x=[str(i) for i in range(10)], labels_y=[str(i) for i in range(10)],
            fmt="{:.0f}", font_size=16)
    d.text((160, 660), "行：真实数字 · 列：老师给出的概率（%，色阶截断在 12%）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        d.rectangle([160, 170 + 3 * 48, 700, 170 + 4 * 48 - 2], outline=AMBER, width=3)
        for i, (k, col) in enumerate([(3, CYAN), (2, AMBER), (9, AMBER), (8, AMBER), (4, MUTED)]):
            d.text((770, 200 + i * 52), f"3 → {k}", font=vk.font(26, mono=True), fill=col)
            d.text((1150, 200 + i * 52), f"{SIM[3, k]:.1%}", font=vk.font(26, mono=True), fill=col, anchor="ra")
    if seg >= 2:
        d.text((770, 480), "哪些数字像，哪些完全不像", font=F_SMALL, fill=TEXT)
        d.text((770, 515), f"（如 8 → 1：{SIM[8, 1]:.0%}）", font=F_SMALL, fill=MUTED)
        d.text((770, 560), "硬标签里看不到", font=F_BODY, fill=RED)


def _digits_row(d, y, crossed):
    for k in range(10):
        x = 200 + k * 90
        col = RED if k == 3 and crossed else CYAN
        d.rounded_rectangle([x, y, x + 70, y + 70], 10, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 35, y + 35), str(k), vk.font(34, mono=True), col)
        if k == 3 and crossed:
            d.line([(x + 5, y + 5), (x + 65, y + 65)], fill=RED, width=4)
            d.line([(x + 65, y + 5), (x + 5, y + 65)], fill=RED, width=4)


def sc_heldout(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "实验：训练集里拿掉所有的 3（Hinton et al., 2015）")
    _digits_row(d, 170, True)
    d.text((200, 260), f"拿掉 {D['n_removed']} 张 3 · 学生一次也没见过 3", font=F_SMALL, fill=RED)
    if seg >= 1:
        vk.text_c(d, (640, 350), "只学老师在其他图片上的软标签，不用真实标签", F_BODY, TEXT)
        vk.text_c(d, (640, 400), "3 的唯一线索：老师给 3 的那一点点概率", F_SMALL, AMBER)
    if seg >= 2:
        for i, v in enumerate(RAW):
            d.text((380 + i * 200, 480), f"实验 {i + 1}", font=F_SMALL, fill=MUTED)
            d.text((380 + i * 200, 515), f"{v:.0%}", font=vk.font(32, mono=True), fill=RED)
        vk.text_c(d, (640, 610), "直接测试：3 的准确率（3 的输出整体偏低）", F_SMALL, MUTED)


def sc_bias(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "只调高 3 的偏置")
    tv.fpaste(c, r"$z_3 \leftarrow z_3 + b$", 330, 190, 34)
    d.text((560, 175), "b：使约 10% 的无标签训练图片被判为 3", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        labs = ["软标签学生", "硬标签学生"]
        tv.bar_chart(d, (260, 280, 1020, 580), labs, [m("soft_held") * 100, m("hard_held") * 100], [GREEN, RED], fmt="{:.1f}%", vmax=110)
        d.text((1040, 400), "3 的准确率", font=F_SMALL, fill=MUTED)
        d.text((1040, 430), "（3 次实验平均）", font=F_SMALL, fill=MUTED)
    if seg >= 2:
        vk.text_c(d, (640, 655), "没见过一张 3，却从老师对其他数字的“犹豫”里学会了 3", F_BODY, AMBER)


def sc_nolabel(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "不用任何真实标签")
    tv.bar_chart(d, (300, 200, 980, 540), ["学生：只用软标签", "老师：用全部标签"], [m("soft_only_full") * 100, D["teacher"] * 100],
                 [GREEN, VIOLET], fmt="{:.1f}%", vmax=110)
    if seg >= 1:
        vk.text_c(d, (640, 630), "老师 + 一批无标签输入 → 就能蒸馏", F_BODY, CYAN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("暗知识", "小概率 = 类别间的相似关系"), ("没见过 3", f"软标签 + 偏置 → {m('soft_held'):.0%} 的 3"),
                      ("无标签", "老师 + 无标签数据即可蒸馏")], seg, "下一期：中间层蒸馏")


RENDER = {"title": sc_title, "similarity": sc_similarity, "heldout": sc_heldout, "bias": sc_bias, "nolabel": sc_nolabel, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "kd-dark")
