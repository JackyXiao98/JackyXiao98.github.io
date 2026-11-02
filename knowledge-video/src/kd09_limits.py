"""Distillation 09 — 蒸馏的边界：能力差距、模仿与评估.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] KD_CACHE=<dir> TF_CACHE=<dir> python kd09_limits.py <out_dir>

Real computations (kdx.ood_imitation, JAX): a student (d = 32) trained 600 steps only on the
addition teacher's answers for the 3,988 training problems whose first addend is < 50; exact match
on held-out problems with first addend < 50 (in distribution) and >= 50 (out of distribution); the
teacher itself is 100% on both. Facts about papers are quoted with references in the notes.
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

O = kdx.cached(CACHE, "ood", kdx.ood_imitation, tf_cache=TF_CACHE)
print(O)
assert O["n_train"] == 3988 and abs(O["in_dist"] - 0.994) < 0.001 and abs(O["ood"] - 0.106) < 0.001
assert O["teacher_in"] == 1.0 and O["teacher_ood"] == 1.0

SCENES = [
    ("title", [
        "这是知识蒸馏系列的最后一期。前面我们看到了蒸馏的很多好处，这一期看它的边界，什么时候蒸馏不灵，以及怎样避免被蒸馏的结果骗到。",
    ]),
    ("gap", [
        "第一个边界是能力差距。直觉上，老师越强，学生应该学得越好。但研究发现，当老师比学生大太多时，学生的效果反而可能变差。",
        "原因是学生没有能力去模仿一个太复杂的函数，老师的软标签对它来说太难拟合。一种办法是加一个中等大小的助教模型，先从老师蒸馏到助教，再从助教蒸馏到学生。在大模型里，也常常选一个不那么大的老师，或者用在策略蒸馏，让老师只在学生能到达的地方给出指导。",
    ]),
    ("ood", [
        "第二个边界是模仿只覆盖了见过的范围。我们再用加法模型做一个实验。老师对所有一百以内的加法都能做对。",
        f"学生只学老师在第一个加数小于五十的题目上写的答案，一共{big_zh(O['n_train'])}道。在同样范围的新题上，学生的准确率是{pct_zh(O['in_dist'], 1)}，几乎和老师一样。",
        f"但是第一个加数大于等于五十的题目，学生只有{pct_zh(O['ood'], 1)}，而老师是百分之百。学生学会的是老师在训练分布上的行为，而不是老师背后的能力。",
    ]),
    ("imitation", [
        "大模型里也有同样的现象。二零二三年的一篇论文，用一个强大的商业模型的输出，微调开源的小模型。人类评审一开始觉得这些模仿模型的回答和老师差不多。",
        "但在不同于模仿数据的基准测试上，差距几乎没有缩小。论文的结论是，模仿主要学到了老师的风格，比如语气、格式和自信的口吻，而不是知识和推理能力。",
    ]),
    ("eval", [
        "所以评估蒸馏出来的模型，要特别小心。第一，要在和蒸馏数据分布不同的任务上测试，而不只是在相似的问题上。第二，要检查测试题有没有混进蒸馏数据里。",
        "第三，不要只看回答是否流畅、像不像老师，要看事实是否正确，最好用可以自动验证的任务。另外，很多商业模型的使用条款，也限制用它们的输出训练竞争的模型。",
    ]),
    ("recap", [
        "回顾整个系列。我们从软标签和温度出发，看到了暗知识，比较了模仿特征和模仿输出，又看到自蒸馏和会犯错的老师。",
        "然后进入大模型，比较了前向和反向相对熵，离策略和在策略蒸馏，序列级蒸馏和合成数据，以及蒸馏和剪枝、量化的组合。最后看到了蒸馏的边界。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 9, "蒸馏的边界", "能力差距、模仿与评估", SERIES)


def sc_gap(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "能力差距（Cho & Hariharan, 2019; Mirzadeh et al., 2020）")
    for i, (t, w, col) in enumerate([("巨大的老师", 150, VIOLET), ("学生", 50, CYAN)]):
        x = 300 + i * 600
        d.rounded_rectangle([x - w, 300 - w, x + w, 300 + w], 14, outline=col, width=3, fill=(12, 19, 34))
        vk.text_c(d, (x, 300), t, F_BODY, col)
    if seg == 0:
        vk.arrow(d, (460, 300), (840, 300), RED, 3)
        vk.text_c(d, (650, 270), "太复杂，学不来", F_SMALL, RED)
    if seg >= 1:
        d.rounded_rectangle([590, 220, 710, 380], 12, outline=AMBER, width=2, fill=(12, 19, 34))
        vk.text_c(d, (650, 300), "助教", F_BODY, AMBER)
        vk.arrow(d, (455, 300), (585, 300), MUTED, 3)
        vk.arrow(d, (715, 300), (845, 300), MUTED, 3)
        vk.text_c(d, (640, 540), "或：选不那么大的老师 · 用在策略蒸馏只在学生能到达的地方指导", F_SMALL, TEXT)


def sc_ood(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "实验：学生只学了一半的题")
    d.text((120, 150), "老师：100 以内的加法全部做对", font=F_SMALL, fill=VIOLET)
    if seg >= 1:
        d.text((120, 190), f"学生训练数据：老师在 a < 50 的 {O['n_train']:,} 道题上的答案", font=F_SMALL, fill=CYAN)
        labs = ["a < 50（见过的范围）"] + (["a ≥ 50（没见过）"] if seg >= 2 else [])
        st = [O["in_dist"] * 100] + ([O["ood"] * 100] if seg >= 2 else [])
        tv.bar_chart(d, (200, 280, 200 + 400 * len(st), 580), labs, st, [GREEN, RED][:len(st)], fmt="{:.1f}%", vmax=115)
        d.text((1020, 330), "老师：两个范围", font=F_SMALL, fill=VIOLET)
        d.text((1020, 365), "都是 100%", font=F_SMALL, fill=VIOLET)
    if seg >= 2:
        vk.text_c(d, (640, 660), "学到的是老师在训练分布上的行为，而不是老师的能力", F_BODY, AMBER)


def sc_imitation(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "模仿的假象（Gudibande et al., 2023）")
    vk.text_c(d, (640, 180), "用商业模型的输出微调开源小模型", F_BODY, TEXT)
    d.rounded_rectangle([140, 250, 620, 380], 14, outline=GREEN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (380, 290), "人类评审", F_BODY, GREEN)
    vk.text_c(d, (380, 340), "回答“和老师差不多”", F_SMALL, TEXT)
    if seg >= 1:
        d.rounded_rectangle([660, 250, 1140, 380], 14, outline=RED, width=2, fill=(12, 19, 34))
        vk.text_c(d, (900, 290), "其他基准测试", F_BODY, RED)
        vk.text_c(d, (900, 340), "差距几乎没有缩小", F_SMALL, TEXT)
        vk.text_c(d, (640, 480), "学到了风格（语气、格式、自信）≠ 知识与推理", F_BODY, AMBER)


def sc_eval(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "怎样评估蒸馏出来的模型")
    items = [("分布不同的任务", "不只测和蒸馏数据相似的问题", CYAN, 0), ("检查污染", "测试题是否混进了蒸馏数据", VIOLET, 0),
             ("看正确性，不看像不像", "优先使用可自动验证的任务", GREEN, 1), ("使用条款", "许多商业模型禁止用输出训练竞品", MUTED, 1)]
    for i, (a, b, col, s) in enumerate(items):
        if seg >= s:
            y = 150 + i * 105
            d.rounded_rectangle([120, y, 1160, y + 80], 12, outline=col, width=2, fill=(12, 19, 34))
            d.text((150, y + 40), a, font=F_BODY, fill=col, anchor="lm")
            d.text((560, y + 40), b, font=F_SMALL, fill=TEXT, anchor="lm")


EPS = ["什么是蒸馏", "暗知识", "中间层蒸馏", "自蒸馏", "前向 / 反向 KL", "Off / On-policy", "序列级蒸馏", "压缩全景", "蒸馏的边界"]


def sc_recap(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "系列回顾")
    k = 4 if seg == 0 else 9
    for i, t in enumerate(EPS):
        x, y = 110 + (i % 5) * 215, 180 + (i // 5) * 170
        on = i < k
        d.rounded_rectangle([x, y, x + 195, y + 120], 14, outline=CYAN if on else (30, 41, 59), width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 97, y + 36), f"{i + 1:02d}", vk.font(26, mono=True), CYAN if on else MUTED)
        vk.text_c(d, (x + 97, y + 82), t, F_SMALL, TEXT if on else MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 600), "感谢观看 · Distillation 系列完", F_BODY, VIOLET)


RENDER = {"title": sc_title, "gap": sc_gap, "ood": sc_ood, "imitation": sc_imitation, "eval": sc_eval, "recap": sc_recap}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "kd-limits")
