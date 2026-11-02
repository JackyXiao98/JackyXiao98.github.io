"""Distillation 07 — 序列级蒸馏与合成数据.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] KD_CACHE=<dir> TF_CACHE=<dir> python kd07_seqkd.py <out_dir>

Real computations (kdx.addition_seqkd, JAX): the tfm.py addition Transformer from the Transformer
series (100% exact match) is the teacher; it writes answers (greedy) for the 7,700 training problems
that have no label; students of width d = 16, 32, 64 are trained 600 steps on (a) only the 300 labelled
problems, (b) the 300 labelled problems + the teacher's 7,700 answers, (c) the same with a weak
teacher (the 120-step checkpoint, 27.7%); exact match on 2,000 held-out problems.
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

Q = kdx.cached(CACHE, "seq", kdx.addition_seqkd, tf_cache=TF_CACHE)
DS = (16, 32, 64)
R = {k: [Q["res"][d][k] for d in DS] for k in ("labels300", "strong", "weak")}
print(Q["teacher"], Q["q_strong"], R)
assert Q["teacher"] == 1.0 and Q["q_strong"] == 1.0 and max(R["labels300"]) < 0.07 and min(R["labels300"]) > 0.04
assert abs(R["strong"][0] - 0.9445) < 1e-9 and R["strong"][1] == 0.99 and R["strong"][2] == 1.0
assert all(0.29 < v < 0.32 for v in R["weak"])

SCENES = [
    ("title", [
        "上一期讲的是在每个词元上对齐老师的分布。这一期看一种更简单、也更常用的做法，直接让老师写出完整的答案，再让学生去学这些答案。",
    ]),
    ("seq", [
        "二零一六年，金和拉什在机器翻译上提出了序列级蒸馏。老师用束搜索为每个句子生成一个完整的翻译，学生就在这些翻译上做普通的监督训练。",
        "它和逐词元蒸馏的区别是，学生学的不是老师在每一步的概率，而是老师最终写出的那一句话。这样不需要访问老师的概率，只要能调用老师生成文本就行。",
    ]),
    ("synthetic", [
        "今天大模型里大量使用的合成数据，本质上就是序列级蒸馏。用一个强模型写指令和回答，或者写出长长的推理过程，再用来微调小模型。",
        "比如深度求索的推理模型发布时，用它生成的八十万条样本，微调了通义千问和羊驼系列的六个小模型，这些小模型的推理能力都有明显提升。",
    ]),
    ("experiment", [
        "我们在变换器系列的加法任务上实际做一次。训练集有八千道一百以内的加法，假设只有三百道有标准答案。老师是那一系列训练好的加法模型，准确率百分之百。",
        f"只用三百道题训练学生，三个不同大小的学生，准确率都不到百分之七，三百道题远远不够。",
        f"让老师给剩下的七千七百道题写答案，再加上原来的三百道一起训练。宽度十六的学生达到{pct_zh(R['strong'][0], 1)}，宽度三十二的达到{pct_zh(R['strong'][1])}，宽度六十四的达到百分之百。",
    ]),
    ("quality", [
        f"老师的质量决定了这批数据的价值。换成只训练了一百二十步、准确率百分之二十八的弱老师，同样七千七百道题，学生只能达到百分之三十左右。",
        "所以合成数据的关键，是老师足够强，并且对生成的数据做好筛选，比如只保留能通过验证的答案。数据越多不一定越好，错误的答案越多，学生学到的错误也越多。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，序列级蒸馏让老师写出完整答案，学生在上面做普通的监督训练，只需要能调用老师。",
        "第二，大模型的合成数据，本质上就是序列级蒸馏。",
        "第三，老师和数据的质量决定了学生的上限。下一期，我们把蒸馏和剪枝、量化放在一起，看模型压缩的全景。感谢大家的观看。",
    ]),
]
assert round(Q["weak_teacher"] * 100) == 28 and round(R["strong"][1] * 100) == 99


def sc_title(c, u, seg, su):
    tv.title_card(c, 7, "序列级蒸馏与合成数据", "让老师写出完整答案", SERIES)


def sc_seq(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "序列级蒸馏（Kim & Rush, 2016）")
    d.rounded_rectangle([120, 190, 420, 290], 14, outline=VIOLET, width=2, fill=(12, 19, 34))
    vk.text_c(d, (270, 240), "老师：束搜索", F_BODY, VIOLET)
    vk.arrow(d, (425, 240), (500, 240), MUTED, 3)
    d.rounded_rectangle([505, 190, 805, 290], 14, outline=AMBER, width=2, fill=(12, 19, 34))
    vk.text_c(d, (655, 240), "完整的翻译", F_BODY, AMBER)
    vk.arrow(d, (810, 240), (885, 240), MUTED, 3)
    d.rounded_rectangle([890, 190, 1190, 290], 14, outline=CYAN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (1040, 240), "学生：监督训练", F_BODY, CYAN)
    if seg >= 1:
        vk.text_c(d, (640, 400), "逐词元：学老师每一步的概率（需要访问老师的 logits）", F_SMALL, MUTED)
        vk.text_c(d, (640, 450), "序列级：学老师写出的那句话（只需要调用老师生成）", F_BODY, GREEN)


def sc_synthetic(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "合成数据 ≈ 序列级蒸馏")
    for i, t in enumerate(["强模型写指令与回答", "或写出长的推理过程", "→ 微调小模型"]):
        vk.text_c(d, (640, 180 + i * 55), t, F_BODY, [CYAN, CYAN, GREEN][i])
    if seg >= 1:
        d.rounded_rectangle([140, 400, 1140, 560], 14, outline=AMBER, width=2, fill=(12, 19, 34))
        vk.text_c(d, (640, 440), "DeepSeek-R1（2025）：约 80 万条样本", F_BODY, AMBER)
        vk.text_c(d, (640, 495), "微调 Qwen2.5 与 Llama 系列的 6 个小模型（1.5B – 70B）", F_SMALL, TEXT)


def sc_experiment(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "实验：100 以内的加法，只有 300 道有答案")
    d.text((120, 140), "老师：Transformer 系列的加法模型（100%）· 7,700 道无答案的题", font=F_SMALL, fill=VIOLET)
    if seg >= 1:
        series = [("只用 300 道", R["labels300"], RED)]
        if seg >= 2:
            series.append(("+ 老师的 7,700 个答案", R["strong"], GREEN))
        k = len(series)
        gw = 900 / 3
        y1 = 580
        d.line([(160, y1), (1060, y1)], fill=(71, 85, 105), width=2)
        for i, dd in enumerate(DS):
            for j, (_, vals, col) in enumerate(series):
                bx = 160 + i * gw + 50 + j * 100
                top = y1 - vals[i] * 360
                d.rectangle([bx, top, bx + 80, y1], fill=col)
                d.text((bx + 40, top - 16), f"{vals[i]:.1%}", font=vk.font(18, mono=True), fill=col, anchor="mm")
            d.text((160 + i * gw + 50 + k * 50, y1 + 22), f"学生 d={dd}", font=F_SMALL, fill=MUTED, anchor="mm")
        for j, (lab, _, col) in enumerate(series):
            d.text((1080, 260 + j * 34), "● " + lab, font=F_SMALL, fill=col)


def sc_quality(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "老师的质量决定数据的价值")
    labs = [f"d={dd}" for dd in DS]
    tv.bar_chart(d, (140, 200, 640, 520), labs, [v * 100 for v in R["strong"]], [GREEN] * 3, fmt="{:.0f}%", vmax=115)
    tv.bar_chart(d, (700, 200, 1200, 520), labs, [v * 100 for v in R["weak"]], [RED] * 3, fmt="{:.0f}%", vmax=115)
    vk.text_c(d, (390, 170), "强老师（100%）的答案", F_SMALL, GREEN)
    vk.text_c(d, (950, 170), f"弱老师（{Q['weak_teacher']:.0%}）的答案", F_SMALL, RED)
    if seg >= 1:
        vk.text_c(d, (640, 630), "强老师 + 筛选（只保留能通过验证的答案）", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("序列级蒸馏", "学老师写出的完整答案"), ("合成数据", "本质上就是序列级蒸馏"),
                      ("质量", "老师与数据决定学生上限")], seg, "下一期：蒸馏、剪枝与量化")


RENDER = {"title": sc_title, "seq": sc_seq, "synthetic": sc_synthetic, "experiment": sc_experiment, "quality": sc_quality, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "kd-seqkd")
