"""Distillation 03 — 中间层蒸馏：模仿特征而不只是输出.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] KD_CACHE=<dir> python kd03_features.py <out_dir>

Real computations (kdx.fitnets, JAX): student [64, 32, 32, 10] with only 100 labelled training images;
(a) labels only; (b) labels + a FitNets-style hint loss that maps the student's second hidden layer
(32 units) through a learned linear regressor onto the teacher's first hidden layer (512 units), on all
1,197 unlabelled training images (lambda = 1); (c) the teacher's soft labels (T = 4) on all images, no
true labels; test accuracy, 3 seeds.
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

F = kdx.cached(CACHE, "fit", kdx.fitnets)
LAB, HINT, SOFT = (np.array(F[k]) for k in ("labels", "hint", "soft"))
GAIN = (HINT.mean() - LAB.mean()) * 100
print(LAB, HINT, SOFT, GAIN)
assert (HINT > LAB).all() and abs(LAB.mean() - 0.867) < 0.001 and abs(HINT.mean() - 0.880) < 0.001 and 1.2 < GAIN < 1.4
assert abs(SOFT.mean() - 0.962) < 0.001

SCENES = [
    ("title", [
        "前两期，学生模仿的都是老师的输出。这一期看另一种思路，让学生模仿老师的中间层，也就是老师内部的特征。",
    ]),
    ("idea", [
        "老师的每一层都会把输入变成一组特征。输出只是最后的结论，中间层则记录了老师是怎么一步步看这张图的。",
        "二零一五年的特征网络论文提出，让学生的某个中间层去对齐老师的某个中间层，这个对齐的目标叫做提示。因为两边的宽度通常不同，中间要加一个可以学习的线性映射。",
    ]),
    ("loss", [
        "损失写出来就是，学生的特征经过映射以后，和老师特征之间的平方误差。它不需要任何标签，所以可以在所有没有标签的数据上计算。",
        "后来还有很多变体，比如对齐注意力图，对齐特征之间的关系，在语言模型里对齐隐藏状态和注意力分布。",
    ]),
    ("experiment", [
        "我们做一个实验。学生有两个隐藏层，只有一百张带标签的图片，同时可以看到全部一千一百九十七张没有标签的图片。",
        f"只用一百个标签训练，三次实验的平均准确率是{pct_zh(LAB.mean(), 1)}。加上提示损失，让学生的第二层，去对齐老师第一层的五百一十二个特征，平均是{pct_zh(HINT.mean(), 1)}，三次都有提升，大约{dec_zh(GAIN, 1)}个百分点。",
        f"但如果换成老师在全部图片上的软标签，准确率是{pct_zh(SOFT.mean(), 1)}。在这个小实验里，模仿特征有帮助，但远不如直接模仿输出。",
    ]),
    ("when", [
        "这个结果并不奇怪。中间层的特征有几百维，里面很多信息对分类并不重要，学生容量有限，硬去对齐反而分散了精力。",
        "中间层蒸馏真正发挥作用，往往是在学生很深、很难直接训练的时候，或者学生和老师结构相近、各层可以一一对应的时候，比如把十二层的语言模型压缩成六层。实践中也常常和输出蒸馏一起用。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，中间层蒸馏让学生对齐老师内部的特征，不需要标签。",
        f"第二，在我们的实验里，它带来约{int_zh(round(GAIN))}个百分点的提升，但不如软标签。",
        "第三，它更适合深的学生，或者和老师结构相近的学生。下一期，我们看一个奇怪的现象，学生和老师一样大，甚至超过老师。感谢大家的观看。",
    ]),
]
assert round(GAIN) == 1


def sc_title(c, u, seg, su):
    tv.title_card(c, 3, "中间层蒸馏", "模仿特征而不只是输出", SERIES)


def _net(d, x, widths, col, label, y0=180):
    d.text((x, y0 - 40), label, font=F_SMALL, fill=col)
    pts = []
    for i, w in enumerate(widths):
        h = w
        y = y0 + i * 90
        d.rounded_rectangle([x + 120 - h, y, x + 120 + h, y + 50], 8, outline=col, width=2, fill=(12, 19, 34))
        pts.append((x + 120 + h, y + 25))
    return pts


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "老师的中间层 = 它“怎么看”这张图")
    tp = _net(d, 220, [110, 110, 30], VIOLET, "老师")
    vk.text_c(d, (340, 485), "输出（结论）", F_SMALL, MUTED)
    if seg >= 1:
        sp = _net(d, 760, [40, 40, 30], CYAN, "学生")
        vk.arrow(d, (830, sp[1][1]), (tp[0][0] + 10, tp[0][1]), AMBER, 3)
        vk.text_c(d, (640, 215), "提示（hint）", F_SMALL, AMBER)
        vk.text_c(d, (640, 560), "学生的中间层 → 线性映射 → 对齐老师的中间层（FitNets, Romero et al., 2015）", F_SMALL, TEXT)


def sc_loss(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "提示损失")
    tv.fpaste(c, r"$\mathcal{L}_{\mathrm{hint}}=\|\,W_r\,h^{S}(x)-h^{T}(x)\,\|^2$", 640, 220, 38)
    vk.text_c(d, (640, 300), "W_r：可学习的线性映射（宽度不同）· 不需要标签", F_SMALL, MUTED)
    if seg >= 1:
        for i, t in enumerate(["注意力图（Zagoruyko & Komodakis, 2017）", "特征之间的关系", "语言模型：隐藏状态与注意力分布（TinyBERT, MiniLM）"]):
            vk.text_c(d, (640, 400 + i * 55), t, F_BODY if i < 2 else F_SMALL, CYAN)


def sc_experiment(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "实验：学生 [64, 32, 32, 10]，100 个标签")
    d.text((140, 140), "+ 1,197 张无标签图片 · 3 次实验平均", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        labs = ["只用 100 个标签", "+ 提示（中间层）"] + (["软标签（输出）"] if seg >= 2 else [])
        vals = [LAB.mean() * 100, HINT.mean() * 100] + ([SOFT.mean() * 100] if seg >= 2 else [])
        tv.bar_chart(d, (160, 260, 160 + 280 * len(vals), 600), labs, vals, [RED, AMBER, GREEN], fmt="{:.1f}%", vmax=110)
        d.text((1030, 260), "每次实验", font=F_SMALL, fill=MUTED)
        for i, (a, b) in enumerate(zip(LAB, HINT)):
            d.text((1030, 300 + i * 36), f"{a:.1%} → {b:.1%}", font=vk.font(20, mono=True), fill=AMBER)
    if seg >= 2:
        vk.text_c(d, (640, 670), "模仿特征有帮助，但远不如模仿输出", F_BODY, TEXT)


def sc_when(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "什么时候用中间层蒸馏？")
    vk.text_c(d, (640, 190), "几百维特征 · 很多与任务无关 · 学生容量有限", F_BODY, MUTED)
    if seg >= 1:
        for i, t in enumerate(["学生很深、难以直接训练（FitNets 的原始动机）", "结构相近、各层可一一对应（如 12 层 → 6 层）",
                               "与输出蒸馏一起使用"]):
            d.rounded_rectangle([200, 280 + i * 95, 1080, 350 + i * 95], 12, outline=GREEN, width=2, fill=(12, 19, 34))
            vk.text_c(d, (640, 315 + i * 95), t, F_BODY, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("中间层蒸馏", "对齐老师内部特征，无需标签"), ("实验", f"+{GAIN:.1f} 个百分点，不如软标签"),
                      ("适用", "深的学生 · 结构相近的学生")], seg, "下一期：自蒸馏与“学生超过老师”")


RENDER = {"title": sc_title, "idea": sc_idea, "loss": sc_loss, "experiment": sc_experiment, "when": sc_when, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "kd-features")
