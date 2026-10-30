"""Differential Privacy 01 — 什么是差分隐私：从差分攻击说起.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python dp01_intro.py <out_dir>

Real computations: a differencing attack on a 10-person salary table (two exact counting queries);
the same two queries answered with Laplace noise (densities, worst-case density ratio = e^eps);
the bound on an attacker's posterior belief, e^eps / (1 + e^eps).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Differential Privacy"

NAMES = ["小李", "小张", "小陈", "小刘", "小赵", "小周", "小吴", "小郑", "小孙", "小王"]
SAL = np.array([12000, 15000, 9000, 22000, 18000, 14000, 11000, 16000, 13000, 25000])
TH = 20000
Q_ALL = int((SAL > TH).sum())
Q_OLD = int((SAL[:-1] > TH).sum())  # "员工中入职满一年的人" — everybody except the newcomer 小王
assert Q_ALL - Q_OLD == 1 and SAL[-1] > TH

# Laplace answers to the counting query on D (with 小王) and D' (without): eps = 1, sensitivity 1
EPS = 1.0
XS = np.linspace(-4, 7, 600)
lap = lambda x, mu: np.exp(-np.abs(x - mu) / (1 / EPS)) * EPS / 2
P1, P0 = lap(XS, Q_ALL), lap(XS, Q_OLD)
RATIO = float((P1 / P0).max())
assert abs(RATIO - np.exp(EPS)) < 1e-9

# posterior bound for a 50/50 prior
EPS_LIST = [0.1, 1.0, 3.0]
POST = [np.exp(e) / (1 + np.exp(e)) for e in EPS_LIST]
assert abs(POST[1] - 0.731) < 1e-3

SCENES = [
    ("title", [
        "欢迎来到差分隐私系列。这一期，我们从一个简单的攻击说起，看看为什么只公布统计结果也会泄露个人隐私，以及差分隐私是怎样定义的。",
    ]),
    ("attack", [
        "假设一家公司有十名员工，人事系统只允许查询统计结果，不能看任何人的工资。",
        f"第一次查询，月薪超过两万的员工有几个，答案是{int_zh(Q_ALL)}。第二次查询，入职满一年的员工里，月薪超过两万的有几个，答案是{int_zh(Q_OLD)}。",
        "公司里只有新来的小王入职不满一年。两个答案一减，就知道小王的月薪超过了两万。这就是差分攻击，两次查询单独看都无害，合在一起却暴露了一个人。",
    ]),
    ("anon", [
        "去掉姓名、只发布统计结果，都不足以保护隐私。历史上，研究者曾经把匿名发布的影评数据，和公开网站上的评分对照，重新识别出了部分用户。",
        "问题在于，攻击者手里可能有我们想不到的背景知识。我们需要一种保证，不管攻击者知道什么，都成立。",
    ]),
    ("noise", [
        f"差分隐私的思路是，在答案里加入精心设计的随机噪声。这里给计数结果加上拉普拉斯噪声，隐私参数艾普西隆取一。",
        "青色是有小王时答案的分布，紫色是没有小王时的分布。两条曲线大部分重叠，看到一个带噪声的答案，攻击者很难判断小王在不在里面。",
        f"在任何一个位置，两条曲线的高度之比，最多是自然常数的一次方，约等于{dec_zh(RATIO)}。",
    ]),
    ("definition", [
        "这就是差分隐私的定义。对于只相差一个人的两个数据集，算法的任何输出出现的概率，最多相差艾普西隆的指数倍。",
        f"直观地说，如果攻击者一开始认为小王有一半的可能在数据集里，那么看到输出以后，他的把握最多变成多少？艾普西隆等于零点一时，最多是{pct_zh(POST[0], 1)}；等于一时，最多是{pct_zh(POST[1], 1)}；等于三时，最多是{pct_zh(POST[2], 1)}。",
        "艾普西隆越小，保护越强，但需要的噪声也越大。这个保证对任何背景知识都成立，因为它只比较算法本身在两个数据集上的行为。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，只发布统计结果并不安全，差分攻击可以从多个查询里推出个人信息。",
        "第二，差分隐私要求有没有某一个人，算法的输出分布几乎不变，差别由艾普西隆控制。",
        "第三，实现方法是加入随机噪声。下一期，我们看两种最基本的加噪方法，随机应答和拉普拉斯机制。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 1, "什么是差分隐私", "从一次差分攻击说起", SERIES)


def sc_attack(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "差分攻击")
    f = vk.font(22, mono=True)
    for i, (nm, s) in enumerate(zip(NAMES, SAL)):
        y = 140 + i * 44
        hidden = seg < 2 or i < 9
        col = AMBER if i == 9 else TEXT
        d.text((140, y), nm, font=F_SMALL, fill=col)
        d.text((240, y), "入职 < 1 年" if i == 9 else "入职 ≥ 1 年", font=F_SMALL, fill=MUTED)
        d.text((420, y), "¥ ••••••" if hidden else f"¥ {s:,}", font=f, fill=MUTED if hidden else RED)
    if seg >= 1:
        rows = [("查询 1：月薪 > 2 万的人数", Q_ALL, CYAN), ("查询 2：入职满一年、月薪 > 2 万的人数", Q_OLD, VIOLET)]
        for i, (q, a, col) in enumerate(rows):
            y = 170 + i * 110
            d.rounded_rectangle([640, y, 1180, y + 80], 12, outline=col, width=2, fill=(12, 19, 34))
            d.text((665, y + 40), q, font=F_SMALL, fill=TEXT, anchor="lm")
            d.text((1150, y + 40), str(a), font=vk.font(34, mono=True), fill=col, anchor="rm")
    if seg >= 2:
        vk.text_c(d, (910, 440), f"{Q_ALL} - {Q_OLD} = 1  →  小王月薪 > 2 万", F_BODY, RED)
        vk.text_c(d, (910, 500), "每次查询都“只看统计”，合起来却暴露了一个人", F_SMALL, MUTED)


def sc_anon(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "匿名化为什么不够")
    for i, (h, col) in enumerate([("匿名影评数据", CYAN), ("公开网站评分", VIOLET)]):
        x = 180 + i * 520
        d.rounded_rectangle([x, 170, x + 400, 400], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 200, 205), h, F_BODY, col)
        rows = [("用户 #4182", "★★★★☆ 2005-03-12"), ("用户 #4182", "★★☆☆☆ 2005-03-15"), ("用户 #4182", "★★★★★ 2005-04-02")] if i == 0 else \
               [("某实名用户", "★★★★☆ 2005-03-12"), ("某实名用户", "★★☆☆☆ 2005-03-15"), ("某实名用户", "★★★★★ 2005-04-02")]
        for j, (a, b) in enumerate(rows):
            d.text((x + 30, 255 + j * 44), a, font=F_SMALL, fill=MUTED)
            d.text((x + 170, 255 + j * 44), b, font=vk.font(18, mono=True), fill=TEXT)
    vk.arrow(d, (590, 285), (690, 285), AMBER, 3)
    vk.text_c(d, (640, 450), "对照评分与日期 → 重新识别（示意；Netflix Prize 数据，Narayanan & Shmatikov, 2008）", F_SMALL, MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 540), "需要一种不依赖“攻击者知道什么”的保证", F_BODY, AMBER)


def sc_noise(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "加入噪声：拉普拉斯机制（ε = 1）")
    tv.fpaste(c, r"$M(D)=\mathrm{count}(D)+\mathrm{Lap}(1/\varepsilon)$", 640, 140, 28)
    if seg >= 1:
        series = [("有小王：count = 2", P1, CYAN), ("没有小王：count = 1", P0, VIOLET)]
        tv.line_chart(d, (160, 200, 1120, 540), XS, series, ylim=(0, 0.55), xlabel="带噪声的查询结果", ylabel="概率密度",
                      legend_xy=(860, 220), xticks=[(x, str(x)) for x in range(-4, 8)])
    if seg >= 2:
        vk.text_c(d, (640, 610), f"任意位置的密度比 ≤ e^ε = {RATIO:.2f}", F_BODY, AMBER)


def sc_definition(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "差分隐私的定义（Dwork et al., 2006）")
    tv.fpaste(c, r"$\Pr[M(D)\in S]\ \leq\ e^{\varepsilon}\,\Pr[M(D')\in S]$", 640, 160, 34)
    vk.text_c(d, (640, 225), "对所有只相差一个人的数据集 D、D′，以及所有输出集合 S", F_SMALL, MUTED)
    if seg >= 1:
        tv.fpaste(c, r"$\Pr[\text{in}\mid \text{output}]\ \leq\ \frac{e^{\varepsilon}}{1+e^{\varepsilon}}$  (prior 50%)", 640, 290, 26)
        tv.bar_chart(d, (330, 350, 950, 560), [f"ε = {e:g}" for e in EPS_LIST], [p * 100 for p in POST], [GREEN, CYAN, AMBER],
                     fmt="{:.1f}%", vmax=110)
        y50 = 560 - 50 / 110 * 210
        d.line([(330, y50), (950, y50)], fill=MUTED, width=1)
        d.text((955, y50), "先验 50%", font=F_SMALL, fill=MUTED, anchor="lm")
    if seg >= 2:
        vk.text_c(d, (640, 630), "ε 越小 → 保护越强 → 噪声越大", F_BODY, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("差分攻击", "多个统计查询可以推出个人信息"), ("定义", "有没有一个人，输出分布几乎不变"),
                      ("实现", "加入随机噪声，强度由 ε 控制")], seg, "下一期：随机应答与拉普拉斯机制")


RENDER = {"title": sc_title, "attack": sc_attack, "anon": sc_anon, "noise": sc_noise, "definition": sc_definition,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "dp-intro")
