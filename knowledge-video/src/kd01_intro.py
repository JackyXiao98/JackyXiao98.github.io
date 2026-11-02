"""Distillation 01 — 什么是知识蒸馏：让小模型向大模型学习.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] KD_CACHE=<dir> python kd01_intro.py <out_dir>

Real computations (kdx.digits_kd, JAX): an MLP teacher [64, 512, 512, 10] trained on all 1,197
training images of the 8x8 digits dataset; a student [64, 16, 10] that sees only the first 100 images,
trained either on their hard labels or on the teacher's soft labels (T = 4, alpha = 0.1); test accuracy
on 597 held-out images, 5 seeds.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import dpx
import kdx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Distillation"
CACHE = os.environ.get("KD_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "kdcache"))

R = kdx.cached(CACHE, "kd", kdx.digits_kd)
HARD, KD = np.array(R["hard"]), np.array(R["kd"])
TP, SP = R["teacher_params"], R["student_params"]
RATIO = TP / SP
_, _, XTE, YTE = dpx.digits()
IDX = int(np.where(YTE == 3)[0][0])
LOG = R["teacher_logits_te"][IDX]
P1, P4 = kdx.softmax(LOG, 1.0), kdx.softmax(LOG, 4.0)
print(R["teacher"], HARD.mean(), KD.mean(), TP, SP, RATIO, np.round(P1, 4), np.round(P4, 3))
assert TP == 301066 and SP == 1210 and 240 < RATIO < 250 and abs(R["teacher"] - 0.9715) < 1e-3
assert (KD > HARD).all() and abs(HARD.mean() - 0.878) < 0.001 and abs(KD.mean() - 0.915) < 0.001
assert P1[3] > 0.9999 and 0.85 < P4[3] < 0.95 and P4[9] > 0.05 and P4.argsort()[-2] == 9 and P4.argsort()[-3] == 2

SCENES = [
    ("title", [
        "欢迎来到知识蒸馏系列。这一期讲最基本的问题，怎样让一个小模型，向一个大模型学习。",
    ]),
    ("why", [
        "大模型的效果好，但是推理又慢又贵，很难部署到手机或者大规模的线上服务里。直接训练一个小模型，效果往往又差很多。",
        "知识蒸馏的想法是，先训练好一个大的老师模型，再让一个小的学生模型，去模仿老师的输出。学生学的不只是标准答案，还有老师对每个答案有多大把握。",
    ]),
    ("soft", [
        "以识别手写数字为例。一张三的图片，标准答案只说它是三，其他九个数字的概率都是零，这叫硬标签。",
        "老师的输出是十个数字的概率，这叫软标签。不过训练好的老师往往非常自信，直接看它的输出，三的概率几乎是百分之百，其他数字的信息都被压没了。",
        f"所以要引入温度。把老师的输出先除以温度再做归一化，温度越高，分布越平。温度取四的时候，三的概率是{pct_zh(P4[3])}，九是{pct_zh(P4[9])}，二是{pct_zh(P4[2])}。老师在告诉学生，这个三有点像九，也有点像二。",
    ]),
    ("loss", [
        "学生的损失有两部分。一部分是和标准答案的交叉熵，另一部分是在同样的温度下，学生和老师的输出分布之间的相对熵。",
        "后一项要乘上温度的平方，这样调节温度时，两部分梯度的大小比例保持不变。这套方法由辛顿等人在二零一五年提出，蒸馏这个名字也来自这篇论文。",
    ]),
    ("experiment", [
        f"我们在八乘八的手写数字数据集上做实验。老师有{big_zh(TP)}个参数，在全部训练图片上训练，测试准确率{pct_zh(R['teacher'], 1)}。学生只有{big_zh(SP)}个参数，小了大约{int_zh(round(RATIO))}倍。",
        f"假设学生只能看到一百张训练图片。只用硬标签训练，五次实验的平均准确率是{pct_zh(HARD.mean(), 1)}。",
        f"同样的一百张图片，换成老师的软标签，平均准确率升到{pct_zh(KD.mean(), 1)}，五次实验每一次都更好。软标签让每张图片带来了更多的信息。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，知识蒸馏让小的学生模型去模仿大的老师模型的输出。",
        "第二，温度把老师过于自信的输出变平，露出各个类别之间的相似关系。",
        "第三，同样一百张图片，软标签比硬标签多带来了约四个百分点。下一期，我们仔细看软标签里到底藏着什么。感谢大家的观看。",
    ]),
]
assert round((KD.mean() - HARD.mean()) * 100) == 4


def sc_title(c, u, seg, su):
    tv.title_card(c, 1, "什么是知识蒸馏", "让小模型向大模型学习", SERIES)


def sc_why(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "老师 → 学生")
    d.rounded_rectangle([140, 180, 520, 420], 18, outline=VIOLET, width=3, fill=(12, 19, 34))
    vk.text_c(d, (330, 240), "老师（大模型）", F_BODY, VIOLET)
    vk.text_c(d, (330, 300), "效果好", F_SMALL, GREEN)
    vk.text_c(d, (330, 340), "推理慢 · 贵 · 难部署", F_SMALL, RED)
    if seg >= 1:
        vk.arrow(d, (530, 300), (780, 300), AMBER, 4)
        vk.text_c(d, (655, 270), "模仿输出", F_SMALL, AMBER)
        d.rounded_rectangle([790, 230, 1090, 370], 18, outline=CYAN, width=3, fill=(12, 19, 34))
        vk.text_c(d, (940, 280), "学生（小模型）", F_BODY, CYAN)
        vk.text_c(d, (940, 330), "快 · 便宜", F_SMALL, GREEN)
        vk.text_c(d, (640, 520), "学的不只是标准答案，还有老师对每个答案的把握", F_BODY, TEXT)


def _probs(d, box, p, col, title, fmt="{:.2f}"):
    x0, y0, x1, y1 = box
    d.text((x0, y0 - 60), title, font=F_SMALL, fill=col)
    w = (x1 - x0) / 10
    d.line([(x0, y1), (x1, y1)], fill=(71, 85, 105), width=2)
    for k in range(10):
        h = p[k] * (y1 - y0)
        d.rectangle([x0 + k * w + 6, y1 - h, x0 + (k + 1) * w - 6, y1], fill=col)
        d.text((x0 + (k + .5) * w, y1 + 18), str(k), font=vk.font(20, mono=True), fill=MUTED, anchor="mm")
        if p[k] >= 0.015:
            d.text((x0 + (k + .5) * w, y1 - h - 14), fmt.format(p[k]), font=vk.font(16, mono=True), fill=col, anchor="mm")


def sc_soft(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "硬标签 · 软标签 · 温度")
    img = XTE[IDX].reshape(8, 8)
    tv.heat(d, (90, 200, 250, 360), img, 0, 1)
    d.text((90, 375), "测试图片（3）", font=F_SMALL, fill=MUTED)
    _probs(d, (320, 220, 600, 400), np.eye(10)[3], MUTED, "硬标签")
    if seg >= 1:
        _probs(d, (660, 220, 940, 400), P1, CYAN, "老师输出 T = 1")
    if seg >= 2:
        _probs(d, (980, 220, 1240, 400), P4, AMBER, "T = 4")
        tv.fpaste(c, r"$p_i=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$", 640, 520, 32)
        vk.text_c(d, (640, 610), f"T = 4：3 → {P4[3]:.0%} · 9 → {P4[9]:.0%} · 2 → {P4[2]:.0%}", F_BODY, AMBER)


def sc_loss(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "蒸馏损失（Hinton et al., 2015）")
    tv.fpaste(c, r"$\mathcal{L}=\alpha\,\mathrm{CE}(y,\,q^{(1)})+(1-\alpha)\,T^2\,\mathrm{KL}(p^{(T)}\,\|\,q^{(T)})$", 640, 250, 34)
    vk.text_c(d, (400, 340), "标准答案", F_SMALL, MUTED)
    vk.text_c(d, (860, 340), "老师 p 与学生 q 在温度 T 下的分布", F_SMALL, MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 450), "乘 T² ：调节温度时，两项梯度的比例不变", F_BODY, AMBER)


def sc_experiment(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "实验：8×8 手写数字")
    d.text((120, 150), f"老师 [64, 512, 512, 10]：{TP:,} 参数，全部训练图片，准确率 {R['teacher']:.1%}", font=F_SMALL, fill=VIOLET)
    d.text((120, 190), f"学生 [64, 16, 10]：{SP:,} 参数（约小 {RATIO:.0f} 倍）", font=F_SMALL, fill=CYAN)
    if seg >= 1:
        vals = [HARD.mean() * 100] + ([KD.mean() * 100] if seg >= 2 else [])
        labs = ["100 张 · 硬标签"] + (["100 张 · 老师软标签"] if seg >= 2 else [])
        tv.bar_chart(d, (260, 280, 260 + 380 * len(vals), 600), labs, vals, [RED, GREEN], fmt="{:.1f}%", vmax=105)
        y = 600 - R["teacher"] * 100 / 105 * 320
        d.line([(260, y), (1020, y)], fill=VIOLET, width=1)
        d.text((1030, y), f"老师 {R['teacher']:.1%}", font=F_SMALL, fill=VIOLET, anchor="lm")
    if seg >= 2:
        d.text((1030, 420), "5 次实验", font=F_SMALL, fill=MUTED)
        d.text((1030, 450), "每次都更好", font=F_SMALL, fill=GREEN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("老师 → 学生", "小模型模仿大模型的输出"), ("温度", "变平分布，露出类别相似性"),
                      ("实验", f"100 张图：{HARD.mean():.1%} → {KD.mean():.1%}")], seg, "下一期：暗知识——软标签里藏着什么")


RENDER = {"title": sc_title, "why": sc_why, "soft": sc_soft, "loss": sc_loss, "experiment": sc_experiment, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "kd-intro")
