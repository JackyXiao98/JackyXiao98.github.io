"""Recursive Self-Improvement 09 — 安全与可控：让自我改进可监督.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python rsi09_safety.py <out_dir>

Real computations: an illustrative oversight model — if a fraction p of accepted changes carries a
hidden problem and each is caught with probability d, the chance that k accepted changes contain no
uncaught problem is (1 - p (1 - d))^k; evaluated for p = 5 %, d = 90 / 99 / 99.9 %, k up to 1,000,
and checked against a Monte Carlo simulation (20,000 runs).
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
SERIES = "Recursive Self-Improvement"

P_BAD = 0.05
DS = [0.9, 0.99, 0.999]
KS = np.arange(0, 1001)
SAFE = {d: (1 - P_BAD * (1 - d)) ** KS for d in DS}
AT100 = {d: float(SAFE[d][100]) for d in DS}
AT1000 = {d: float(SAFE[d][1000]) for d in DS}
rng = np.random.default_rng(0)
MC = {d: float((rng.binomial(100, P_BAD * (1 - d), 20000) == 0).mean()) for d in DS}
print(AT100, AT1000, MC)
assert all(abs(MC[d] - AT100[d]) < 0.01 for d in DS)
assert 0.6 < AT100[0.9] < 0.61 and 0.95 < AT100[0.99] < 0.952 and AT1000[0.9] < 0.01 and 0.95 < AT1000[0.999] < 0.952

SCENES = [
    ("title", [
        "这是递归自我改进系列的最后一期。我们讨论一个越来越重要的问题，怎样让自我改进保持安全和可控。",
    ]),
    ("risk", [
        "前面我们看到，自我改进依赖验证，而验证总有漏洞。在一个循环里，每一次被接受的修改，都可能带进一个没有被发现的问题。",
        "改进越快，单位时间里被接受的修改越多，没被发现的问题也就积累得越快。所以速度本身，就放大了对验证的要求。",
    ]),
    ("math", [
        "我们用一个简单的模型来量化。假设每一百次被接受的修改里，有五次带着隐藏的问题，而检查能发现其中一部分。",
        f"如果检查能发现百分之九十的问题，一百次修改后，完全没有漏网问题的概率只有{pct_zh(AT100[0.9])}；一千次以后，几乎为零。",
        f"检查率提高到百分之九十九，一百次后是{pct_zh(AT100[0.99])}，一千次后又降到{pct_zh(AT1000[0.99])}。要在一千次修改后仍保持{pct_zh(AT1000[0.999])}，检查率需要达到百分之九十九点九。我们用两万次随机模拟核对了这些数字。",
    ]),
    ("practice", [
        "实践中有几层防线。第一，独立且隐藏的评估，被改进的系统看不到评估数据，就很难对它过拟合，上一期的研究正是这样做的。",
        "第二，在沙箱里运行，限制自我修改能接触的资源和权限。第三，持续监测像钻奖励空子这样的行为，而不只看主要指标。第四，关键的修改由人来审核和批准。",
    ]),
    ("oversight", [
        "更长远的挑战是可扩展监督。当被监督的系统比监督者更强时，怎样还能可靠地判断它的工作？",
        "研究者在探索用较弱的模型监督较强的模型、让模型互相辩论、以及把大任务拆成人能检查的小步骤等方法。自我改进越接近闭环，这些方法就越重要。",
    ]),
    ("recap", [
        "回顾整个系列。我们从智能爆炸的增长模型和瓶颈出发，实际运行了自我训练、自我博弈、验证者择优和代码进化，也看到了模型崩溃和奖励黑客。",
        "最后我们看到，二零二六年的研究正在把这些方法用于自动化人工智能研究，而可靠的验证和人的把关，是让这一切安全前进的关键。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 9, "安全与可控", "让自我改进可监督", SERIES)


def sc_risk(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "每一次接受的修改都可能带进隐藏问题")
    rng2 = np.random.default_rng(5)
    for i in range(40):
        x = 140 + i * 25
        bad = rng2.random() < 0.12
        d.rectangle([x, 200, x + 18, 240], fill=RED if bad else CYAN)
    d.text((140, 255), "被接受的修改（红色：带隐藏问题，示意）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 380), "改进越快 → 单位时间接受的修改越多", F_BODY, TEXT)
        vk.text_c(d, (640, 430), "→ 漏网问题积累越快 → 对验证要求越高", F_BODY, AMBER)


def sc_math(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "示意模型：k 次修改后没有漏网问题的概率")
    tv.fpaste(c, r"$\Pr[\mathrm{safe}]=\left(1-p\,(1-d)\right)^{k},\qquad p=5\%$", 640, 140, 30)
    if seg >= 1:
        series = [(f"检查率 d = {d_:.1%}", SAFE[d_] * 100, col) for d_, col in zip(DS, [RED, AMBER, GREEN])][: (1 if seg == 1 else 3)]
        tv.line_chart(d, (160, 200, 840, 570), KS, series, ylim=(0, 100), xlabel="被接受的修改次数 k", ylabel="%",
                      legend_xy=(560, 360), xticks=[(k, str(k)) for k in (0, 250, 500, 750, 1000)])
        d.text((900, 220), "k = 100 / 1000", font=F_SMALL, fill=MUTED)
        for i, (d_, col) in enumerate(zip(DS, [RED, AMBER, GREEN])):
            if seg >= 2 or i == 0:
                d.text((900, 270 + i * 70), f"d = {d_:.1%}", font=F_SMALL, fill=col)
                d.text((900, 300 + i * 70), f"{AT100[d_]:.1%} / {AT1000[d_]:.1%}", font=vk.font(22, mono=True), fill=col)
    if seg >= 2:
        vk.text_c(d, (1010, 520), "蒙特卡洛 2 万次核对", F_SMALL, MUTED)


def sc_practice(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "几层防线")
    items = [("独立、隐藏的评估", "看不到评估数据，难以过拟合", CYAN, 0), ("沙箱", "限制资源与权限", VIOLET, 1),
             ("行为监测", "如钻奖励空子的比例", AMBER, 1), ("人工审核", "关键修改由人批准", GREEN, 1)]
    for i, (a, b, col, s) in enumerate(items):
        if seg >= s:
            y = 160 + i * 100
            d.rounded_rectangle([200, y, 1080, y + 76], 12, outline=col, width=2, fill=(12, 19, 34))
            d.text((230, y + 38), a, font=F_BODY, fill=col, anchor="lm")
            d.text((560, y + 38), b, font=F_SMALL, fill=TEXT, anchor="lm")


def sc_oversight(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "可扩展监督")
    vk.text_c(d, (640, 180), "被监督者比监督者更强时，怎样可靠地判断？", vk.font(32), AMBER)
    if seg >= 1:
        for i, (t, col) in enumerate([("弱模型监督强模型", CYAN), ("模型互相辩论", VIOLET), ("拆成人能检查的小步骤", GREEN)]):
            x = 140 + i * 340
            d.rounded_rectangle([x, 300, x + 320, 380], 12, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 160, 340), t, F_SMALL, col)
        vk.text_c(d, (640, 470), "越接近闭环，越重要", F_BODY, TEXT)


EPS_ = ["智能爆炸", "瓶颈", "自训练 STaR", "自我博弈", "验证者", "失效模式", "改进代码", "2026 进展", "安全与可控"]


def sc_recap(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "系列回顾")
    k = 6 if seg == 0 else 9
    for i, t in enumerate(EPS_):
        x, y = 140 + (i % 5) * 205, 180 + (i // 5) * 160
        on = i < k
        d.rounded_rectangle([x, y, x + 185, y + 120], 14, outline=CYAN if on else (30, 41, 59), width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 92, y + 36), f"{i + 1:02d}", vk.font(26, mono=True), CYAN if on else MUTED)
        vk.text_c(d, (x + 92, y + 82), t, F_SMALL, TEXT if on else MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 560), "感谢观看 · Recursive Self-Improvement 系列完", F_BODY, VIOLET)


RENDER = {"title": sc_title, "risk": sc_risk, "math": sc_math, "practice": sc_practice, "oversight": sc_oversight,
          "recap": sc_recap}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rsi-safety")
