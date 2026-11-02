"""Distillation 06 — Off-policy 与 On-policy 蒸馏.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] KD_CACHE=<dir> python kd06_onpolicy.py <out_dir>

Real computations (kdx.onpolicy_vs_offpolicy): a 12-token "language" whose teacher depends on the
previous two tokens; a bigram student (sees only the previous token, so it cannot match the teacher).
Off-policy: maximum likelihood on 4,000 teacher sequences of length 30. On-policy: starting from the
off-policy student, 600 steps of the exact per-token reverse-KL gradient evaluated on the student's
own samples (256 sequences per step). Metrics on 4,000 student samples: mean teacher log-probability
of each generated token and the share of tokens the teacher gives < 2% probability; and the student's
NLL on teacher text.
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

O = kdx.cached(CACHE, "onoff", kdx.onpolicy_vs_offpolicy)
OFF, ON, TE = O["off"], O["on"], O["teacher"]
H = O["hist"]
print(OFF, ON, TE, O["off_nll"], O["on_nll"])
assert abs(TE["unlikely"] - 0.029) < 0.001 and abs(OFF["unlikely"] - 0.157) < 0.001 and abs(ON["unlikely"] - 0.141) < 0.001
assert abs(OFF["teacher_logp"] + 2.02) < 0.005 and abs(ON["teacher_logp"] + 1.88) < 0.005 and abs(TE["teacher_logp"] + 1.07) < 0.005
assert O["on_nll"] > O["off_nll"] and abs(O["off_nll"] - 1.44) < 0.005 and abs(O["on_nll"] - 1.52) < 0.005

SCENES = [
    ("title", [
        "这一期讲大模型蒸馏里最重要的一个区分，离策略蒸馏和在策略蒸馏，它们的区别在于，学生是在谁写的文本上学习。",
    ]),
    ("off", [
        "离策略蒸馏，是让学生在老师写的文本上学习。可以只用老师写出的答案做监督微调，也可以在每个位置上对齐老师的整个分布。共同点是，训练用的前缀都来自老师。",
        "问题在于，生成的时候，学生要接着自己写的前缀往下写。学生一旦犯了一个老师不会犯的错，就进入了训练中从没见过的状态，错误会一步步累积，这叫暴露偏差。",
    ]),
    ("on", [
        "在策略蒸馏反过来做。先让学生自己写，然后让老师在学生写的每一个位置上，给出它认为下一个词元的分布，学生再去对齐这个分布，通常用上一期的反向相对熵。",
        "这样学生恰好在自己会遇到的状态上，得到老师的纠正。它和强化学习很像，数据都来自学生自己，只是奖励换成了老师逐个词元的打分，信号密集得多。",
    ]),
    ("toy", [
        "我们用一个小玩具来比较。一个只有十二个词元的语言，老师根据前两个词元决定下一个，学生只能看前一个词元，所以无论怎么学都比不上老师。",
        "先在四千条老师写的序列上做离策略训练，再从这个学生出发，在它自己生成的序列上做六百步在策略训练。评价方法是，让学生自己写，看老师怎么评价学生写的每一个词元。",
    ]),
    ("result", [
        f"老师自己写的文本里，只有{pct_zh(TE['unlikely'], 1)}的词元是它认为不太可能的，也就是概率低于百分之二。离策略的学生写的文本里，这个比例是{pct_zh(OFF['unlikely'], 1)}，在策略训练以后降到{pct_zh(ON['unlikely'], 1)}。",
        f"老师对每个词元的平均对数概率，也从负{dec_zh(-OFF['teacher_logp'], 2)}提高到负{dec_zh(-ON['teacher_logp'], 2)}。在这个玩具里，改进不算大，因为学生的能力上限很低。",
        f"有意思的是代价。在老师写的文本上，在策略学生的负对数似然反而从{dec_zh(O['off_nll'], 2)}升到{dec_zh(O['on_nll'], 2)}。它放弃了覆盖老师的所有写法，换来了自己写出的东西更符合老师的标准。",
    ]),
    ("practice", [
        "在真实的大模型里，在策略蒸馏已经被广泛使用。广义知识蒸馏的论文系统地比较了这两种方式，通义千问三代的小模型，就是在离策略蒸馏之后，再用在策略蒸馏训练出来的。",
        "实践中常见的做法也是先离策略、后在策略。前者便宜，老师的文本可以一次生成、反复使用。后者每一步都要让学生生成、老师打分，计算更贵，但能修正学生自己的错误。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，离策略蒸馏在老师的文本上学习，存在暴露偏差。",
        "第二，在策略蒸馏在学生自己的文本上让老师打分，信号密集，像强化学习。",
        "第三，实践中通常先离策略、后在策略。下一期，我们看序列级蒸馏和合成数据。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 6, "Off-policy 与 On-policy", "在谁写的文本上学习？", SERIES)


def _seq(d, y, words, cols, x0=160):
    for i, (w, col) in enumerate(zip(words, cols)):
        tv.token(d, x0 + i * 120, y, w, col, w=104, font=F_SMALL)


def sc_off(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "离策略（off-policy）：在老师的文本上学习")
    d.text((160, 150), "训练：前缀来自老师", font=F_SMALL, fill=VIOLET)
    _seq(d, 190, ["老师", "写的", "前缀", "→", "?"], [VIOLET] * 3 + [MUTED, CYAN])
    if seg >= 1:
        d.text((160, 330), "生成：前缀来自学生自己", font=F_SMALL, fill=CYAN)
        _seq(d, 370, ["学生", "写错", "一步", "→", "?"], [CYAN, RED, RED, MUTED, RED])
        vk.text_c(d, (640, 520), "进入训练时没见过的状态 → 错误累积（暴露偏差）", F_BODY, RED)


def sc_on(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "在策略（on-policy）：在学生自己的文本上让老师打分")
    steps = [("学生自己写", CYAN), ("老师在每个位置\n给出下一个词元分布", VIOLET), ("学生对齐（反向 KL）", GREEN)]
    for i, (t, col) in enumerate(steps):
        x = 110 + i * 370
        d.rounded_rectangle([x, 180, x + 320, 290], 14, outline=col, width=2, fill=(12, 19, 34))
        for j, line in enumerate(t.split("\n")):
            vk.text_c(d, (x + 160, 225 + j * 34 - (17 if "\n" in t else 0)), line, F_SMALL, col)
        if i < 2:
            vk.arrow(d, (x + 325, 235), (x + 365, 235), MUTED, 3)
    if seg >= 1:
        vk.text_c(d, (640, 400), "像强化学习：数据来自学生自己", F_BODY, TEXT)
        vk.text_c(d, (640, 450), "但“奖励”是老师逐词元的打分 → 信号密集得多", F_BODY, AMBER)


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "玩具：12 个词元的语言")
    d.text((160, 160), "老师：看前 2 个词元", font=F_BODY, fill=VIOLET)
    d.text((160, 210), "学生：只看前 1 个词元（能力不够）", font=F_BODY, fill=CYAN)
    if seg >= 1:
        for i, (a, b) in enumerate([("离策略", "4,000 条老师序列上做最大似然"), ("在策略", "从离策略学生出发，600 步，学生自己的样本")]):
            d.text((160, 320 + i * 60), a, font=F_BODY, fill=[AMBER, GREEN][i])
            d.text((330, 324 + i * 60), b, font=F_SMALL, fill=TEXT)
        vk.text_c(d, (640, 520), "评价：学生自己写 → 老师给每个词元打分", F_BODY, MUTED)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "结果：学生自己写的文本，老师怎么看")
    labs = ["老师自己", "离策略学生", "在策略学生"]
    tv.bar_chart(d, (120, 200, 720, 520), labs, [TE["unlikely"] * 100, OFF["unlikely"] * 100, ON["unlikely"] * 100],
                 [VIOLET, AMBER, GREEN], fmt="{:.1f}%", vmax=20)
    d.text((120, 150), "老师认为“不太可能”（< 2%）的词元比例", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        d.text((790, 180), "老师的平均对数概率", font=F_SMALL, fill=MUTED)
        for i, (lab, v, col) in enumerate([("老师自己", TE["teacher_logp"], VIOLET), ("离策略", OFF["teacher_logp"], AMBER),
                                           ("在策略", ON["teacher_logp"], GREEN)]):
            d.text((790, 220 + i * 40), lab, font=F_SMALL, fill=col)
            d.text((1180, 220 + i * 40), f"{v:.2f}", font=vk.font(24, mono=True), fill=col, anchor="ra")
    if seg >= 2:
        d.text((790, 380), "代价：老师文本上的 NLL", font=F_SMALL, fill=MUTED)
        d.text((790, 420), "离策略", font=F_SMALL, fill=AMBER)
        d.text((1180, 420), f"{O['off_nll']:.2f}", font=vk.font(24, mono=True), fill=AMBER, anchor="ra")
        d.text((790, 460), "在策略", font=F_SMALL, fill=GREEN)
        d.text((1180, 460), f"{O['on_nll']:.2f}", font=vk.font(24, mono=True), fill=RED, anchor="ra")
        vk.text_c(d, (640, 620), "放弃覆盖老师的所有写法 → 换来自己写的东西更合老师的标准", F_BODY, TEXT)


def sc_practice(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "真实系统")
    for i, (a, b) in enumerate([("GKD（Agarwal et al., 2024）", "系统比较 on / off-policy，学生自己的样本更好"),
                                ("Qwen3（2025）", "小模型：off-policy 蒸馏 → on-policy 蒸馏")]):
        d.text((140, 170 + i * 60), a, font=F_BODY, fill=CYAN)
        d.text((560, 175 + i * 60), b, font=F_SMALL, fill=TEXT)
    if seg >= 1:
        for i, (a, b, col) in enumerate([("先离策略", "老师文本一次生成、反复使用 · 便宜", AMBER), ("后在策略", "每步学生生成、老师打分 · 贵 · 能纠错", GREEN)]):
            x = 80 + i * 580
            d.rounded_rectangle([x, 380, x + 540, 500], 14, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 270, 415), a, F_BODY, col)
            vk.text_c(d, (x + 270, 465), b, F_SMALL, TEXT)
        vk.arrow(d, (622, 440), (656, 440), MUTED, 3)


def sc_summary(c, u, seg, su):
    tv.summary(c, 6, [("离策略", "老师的文本 · 暴露偏差"), ("在策略", "学生的文本 + 老师打分"),
                      ("实践", "先离策略，后在策略")], seg, "下一期：序列级蒸馏与合成数据")


RENDER = {"title": sc_title, "off": sc_off, "on": sc_on, "toy": sc_toy, "result": sc_result, "practice": sc_practice, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "kd-onpolicy")
