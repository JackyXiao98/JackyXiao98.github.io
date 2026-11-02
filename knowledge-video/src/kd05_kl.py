"""Distillation 05 — 大语言模型的蒸馏：前向 KL 与反向 KL.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] KD_CACHE=<dir> python kd05_kl.py <out_dir>

Real computations (kdx.kl_fits): teacher p = 0.6 N(-2, 0.5^2) + 0.4 N(2, 0.5^2); student q = one
Gaussian N(mu, s^2); exact grid search (mu in [-3, 3], s in [0.2, 3]) minimising KL(p||q) (forward) and
KL(q||p) (reverse) by numerical integration; probability mass in the gap |x| < 0.5.
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

K = kdx.cached(CACHE, "kl", kdx.kl_fits)
(FM, FS), (RM, RS) = K["fwd"], K["rev"]
print(K["fwd"], K["rev"], K["mean"], K["std"], K["gap_mass_p"], K["gap_mass_f"], K["gap_mass_r"])
assert abs(FM + 0.4) < 1e-9 and abs(FS - 2.02) < 1e-9 and abs(K["mean"] + 0.4) < 1e-6 and abs(K["std"] - 2.02) < 0.005
assert (RM, RS) == (-2.0, 0.5) and abs(K["gap_mass_p"] - 0.0013) < 1e-4 and abs(K["gap_mass_f"] - 0.191) < 0.001
SEL = slice(None, None, 10)
X = K["x"][SEL]

SCENES = [
    ("title", [
        "从这一期开始，我们转向大语言模型的蒸馏。第一个问题是，学生应该用哪一种相对熵去接近老师，前向还是反向。",
    ]),
    ("token", [
        "语言模型每一步都会输出下一个词元的概率分布。最直接的蒸馏，就是在每个位置上，让学生的分布接近老师的分布，和第一期的软标签完全一样。",
        "但相对熵不对称。前向相对熵按老师的分布去平均，老师认为可能的地方，学生必须给出概率。反向相对熵按学生的分布去平均，学生给出概率的地方，老师必须也认为可能。",
    ]),
    ("toy", [
        "我们用一个一维的玩具看清两者的区别。老师是两个高斯分布的混合，在负二和正二各有一个峰，中间几乎没有概率。学生只是一个高斯分布，能力不够，无法同时表示两个峰。",
        f"最小化前向相对熵，学生的中心在负零点四，标准差{dec_zh(FS, 2)}，正好等于老师分布的均值和标准差。它张开双臂盖住了两个峰，这叫覆盖模式。",
        f"最小化反向相对熵，学生的中心在负二，标准差零点五，只抓住了较大的那个峰，这叫寻找模式。",
    ]),
    ("gap", [
        f"关键在两个峰之间。老师在这段区间里只有百分之{dec_zh(K['gap_mass_p'] * 100, 2)}的概率，前向拟合的学生却放了{pct_zh(K['gap_mass_f'], 1)}，反向拟合的学生和老师一样，几乎为零。",
        "对语言模型来说，这段区间就是老师认为不该说的话。覆盖模式的学生会以不小的概率生成老师自己都不会写出的内容，比如胡编的事实。寻找模式的学生回答更可靠，代价是多样性下降。",
    ]),
    ("llm", [
        "二零二三年的迷你大模型方法，主张在语言模型蒸馏里使用反向相对熵，在指令跟随任务上，生成的质量比前向的方法更好。也有方法把两者混合，比如广义的詹森香农散度，在两者之间调节。",
        "要注意，反向相对熵按学生的分布求平均，需要在学生自己生成的文本上计算。这就引出了下一期的主题，离策略和在策略蒸馏。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，前向相对熵让学生覆盖老师所有的模式，反向相对熵让学生只抓住主要的模式。",
        f"第二，在玩具里，前向拟合在老师几乎为零的区域放了{pct_zh(K['gap_mass_f'])}的概率。",
        "第三，大模型蒸馏常用反向或混合的相对熵，换取更可靠的生成。下一期，我们看离策略和在策略蒸馏。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 5, "前向 KL 与反向 KL", "大语言模型的蒸馏", SERIES)


def sc_token(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "逐词元蒸馏：每个位置对齐下一个词元的分布")
    words = ["今天", "天气", "很", "?"]
    for i, w in enumerate(words):
        tv.token(d, 150 + i * 130, 160, w, CYAN if i < 3 else AMBER, w=110, font=F_SMALL)
    vk.text_c(d, (900, 182), "老师 p(·) ↔ 学生 q(·)", F_SMALL, AMBER)
    if seg >= 1:
        tv.fpaste(c, r"$\mathrm{KL}(p\|q)=\sum_y p(y)\log\frac{p(y)}{q(y)}$", 360, 330, 32)
        tv.fpaste(c, r"$\mathrm{KL}(q\|p)=\sum_y q(y)\log\frac{q(y)}{p(y)}$", 920, 330, 32)
        vk.text_c(d, (360, 420), "前向：按老师平均", F_BODY, CYAN)
        vk.text_c(d, (920, 420), "反向：按学生平均", F_BODY, VIOLET)
        vk.text_c(d, (360, 470), "老师认为可能 → 学生必须给概率", F_SMALL, TEXT)
        vk.text_c(d, (920, 470), "学生给了概率 → 老师必须认为可能", F_SMALL, TEXT)


def _plot(d, upto_seg):
    series = [("老师：两个峰", K["p"][SEL], MUTED)]
    if upto_seg >= 1:
        series.append((f"前向 KL：N({FM:.1f}, {FS:.2f}²)", K["q_f"][SEL], CYAN))
    if upto_seg >= 2:
        series.append((f"反向 KL：N({RM:.0f}, {RS}²)", K["q_r"][SEL], VIOLET))
    tv.line_chart(d, (140, 160, 1000, 580), X, series, ylim=(0, 0.85), xlabel="x", legend_xy=(160, 175),
                  xticks=[(v, str(v)) for v in (-8, -4, -2, 0, 2, 4, 8)])


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "玩具：老师有两个峰，学生只有一个")
    _plot(d, seg)
    if seg >= 1:
        d.text((1030, 260), "覆盖模式", font=F_BODY, fill=CYAN)
        d.text((1030, 300), "= 均值与标准差", font=F_SMALL, fill=MUTED)
        d.text((1030, 330), f"（{K['mean']:.2f}, {K['std']:.2f}）", font=F_SMALL, fill=MUTED)
    if seg >= 2:
        d.text((1030, 420), "寻找模式", font=F_BODY, fill=VIOLET)
        d.text((1030, 460), "只抓住大的峰", font=F_SMALL, fill=MUTED)


def sc_gap(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "两个峰之间：老师认为“不该说的话”")
    _plot(d, 2)
    px = lambda v: 140 + (v + 8) / 16 * 860
    d.rectangle([px(-0.5), 160, px(0.5), 580], outline=AMBER, width=2)
    for i, (lab, v, col) in enumerate([("老师", K["gap_mass_p"], MUTED), ("前向", K["gap_mass_f"], CYAN), ("反向", K["gap_mass_r"], VIOLET)]):
        d.text((1030, 240 + i * 60), lab, font=F_BODY, fill=col)
        d.text((1240, 240 + i * 60), f"{v:.2%}", font=vk.font(26, mono=True), fill=col, anchor="ra")
    d.text((1030, 180), "|x| < 0.5 的概率", font=F_SMALL, fill=AMBER)
    if seg >= 1:
        vk.text_c(d, (640, 680), "覆盖模式 → 生成老师不会写的内容 · 寻找模式 → 更可靠、但多样性下降", F_SMALL, TEXT)


def sc_llm(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "大语言模型里的选择")
    for i, (a, b, col) in enumerate([("MiniLLM（Gu et al., 2023）", "反向 KL，指令跟随生成质量更好", VIOLET),
                                     ("GKD（Agarwal et al., 2024）", "广义 JSD：在前向与反向之间调节", CYAN)]):
        y = 170 + i * 120
        d.rounded_rectangle([120, y, 1160, y + 90], 14, outline=col, width=2, fill=(12, 19, 34))
        d.text((150, y + 45), a, font=F_BODY, fill=col, anchor="lm")
        d.text((620, y + 45), b, font=F_SMALL, fill=TEXT, anchor="lm")
    if seg >= 1:
        vk.text_c(d, (640, 500), "反向 KL 按学生的分布平均 → 需要学生自己生成的文本", F_BODY, AMBER)
        vk.text_c(d, (640, 560), "→ 下一期：离策略 vs 在策略", F_SMALL, MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("前向 vs 反向", "覆盖所有模式 vs 抓住主要模式"), ("玩具", f"前向在空隙里放了 {K['gap_mass_f']:.0%} 的概率"),
                      ("大模型", "反向或混合 KL，生成更可靠")], seg, "下一期：Off-policy 与 On-policy 蒸馏")


RENDER = {"title": sc_title, "token": sc_token, "toy": sc_toy, "gap": sc_gap, "llm": sc_llm, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "kd-kl")
