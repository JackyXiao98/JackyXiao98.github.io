"""Recursive Self-Improvement 06 — 失效模式：模型崩溃与奖励黑客.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] RSI_CACHE=<dir> python rsi06_failure.py <out_dir>

Real computations: (1) model collapse — a Gaussian refitted, generation after generation, to 50
samples drawn from the previous fit (median over 200 runs), vs. accumulating all data including the
original real samples; (2) Goodhart — true quality q ~ N(0, 1), proxy = q + error (Gaussian or
heavy-tailed Student-t with 2 d.o.f.); pick the best of n by the proxy and record the true quality
(400 repeats per n).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw
from scipy.stats import norm

import rsix
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Recursive Self-Improvement"
CACHE = os.environ.get("RSI_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "rsicache"))

MC = rsix.cached(CACHE, "collapse", rsix.model_collapse)
REP, ACC = MC["replace"], MC["accumulate"]
GENS = np.arange(len(REP))
TAIL = lambda sd: 2 * norm.sf(2 / sd)
GH = rsix.cached(CACHE, "goodhart", rsix.goodhart)
NS, LIGHT, HEAVY = GH["n"], GH["light"], GH["heavy"]
PEAK = int(np.argmax(HEAVY))
print(REP[[0, 50, 100, 200, 300]], ACC[-1], TAIL(REP[50]), LIGHT, HEAVY)
assert 0.3 < REP[50] < 0.4 and REP[200] < 0.02 and ACC[-1] > 0.95 and TAIL(REP[50]) < 1e-6
assert NS[PEAK] == 16 and HEAVY[-1] < 0.1 and LIGHT[-1] > 2.5 and all(a < b for a, b in zip(LIGHT[1:], LIGHT[2:]))

SCENES = [
    ("title", [
        "前几期我们看到了自我改进成功的例子。这一期看它怎样失败，两种最常见的失效模式，模型崩溃和奖励黑客。",
    ]),
    ("collapse", [
        "先看模型崩溃。设想一个模型只用上一代模型生成的数据训练，一代接一代。",
        "我们用最简单的模型来模拟。真实数据是标准正态分布，每一代从上一代拟合的分布里抽五十个样本，重新估计均值和标准差。",
        f"重复两百次取中位数，标准差在第五十代降到{dec_zh(REP[50], 2)}，第二百代只剩{dec_zh(REP[200], 3)}。分布越来越窄，最后缩成一个点。",
    ]),
    ("tails", [
        f"最先消失的是尾部。真实分布里，大约{pct_zh(TAIL(1), 1)}的数据离中心超过两个标准差，到第五十代，这个比例已经小于百万分之一。少见的情况被一代代遗忘了。",
        f"有研究指出，只要保留原始的真实数据，并把新数据累积起来，而不是替换掉旧数据，就能避免崩溃。我们的模拟里，这样做三百代以后，标准差仍然有{dec_zh(ACC[-1], 2)}。",
    ]),
    ("goodhart", [
        "再看奖励黑客。古德哈特定律说，当一个指标变成目标，它就不再是好的指标。",
        "我们模拟按一个不完美的奖励挑选答案。每个答案有真实质量，奖励等于真实质量加上误差，从很多候选里挑奖励最高的一个。",
        f"如果误差是正常的小误差，挑得越多，真实质量越高，从四个里挑是{dec_zh(LIGHT[1], 2)}，从一万多个里挑是{dec_zh(LIGHT[-1], 2)}。",
        f"但如果误差偶尔会特别大，也就是奖励有漏洞，真实质量在从{int_zh(NS[PEAK])}个里挑时最高，只有{dec_zh(HEAVY[PEAK], 2)}，之后反而下降，挑得最狠时只剩{dec_zh(HEAVY[-1], 2)}，被选中的几乎都是钻了奖励空子的答案。",
    ]),
    ("lessons", [
        "这两种失效模式，在递归自我改进里都会被放大。模型崩溃对应用自己的输出训练自己，奖励黑客对应用不完美的评判来选择改进。",
        "对策也很相似：保留真实的数据和信号，使用更可靠的验证，不要对一个指标过度优化，并持续用独立的测试检查真实效果。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，反复用自己的输出训练，分布会变窄，尾部最先消失；保留并累积真实数据可以避免。",
        "第二，按有漏洞的奖励挑选，挑得越狠，真实质量反而越差。",
        "第三，自我改进越强，越需要可靠的验证和独立的检查。下一期，我们看改进自己代码的系统。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 6, "失效模式", "模型崩溃与奖励黑客", SERIES)


def sc_collapse(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "模型崩溃：一代代用自己的输出训练")
    for i in range(5):
        x = 170 + i * 200
        d.rounded_rectangle([x, 150, x + 150, 200], 10, outline=CYAN, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 75, 175), f"第 {i} 代" if i else "真实数据", F_SMALL, CYAN if i else GREEN)
        if i < 4:
            vk.arrow(d, (x + 152, 175), (x + 198, 175), MUTED, 2)
    if seg >= 1:
        vk.text_c(d, (640, 240), "每代：从上一代的正态分布抽 50 个样本 → 重新估计均值和标准差", F_SMALL, TEXT)
    if seg >= 2:
        tv.line_chart(d, (180, 290, 1100, 580), GENS, [("标准差（200 次中位数）", REP, RED)], ylim=(0, 1.05),
                      xlabel="代数", ylabel="标准差", legend_xy=(800, 310), xticks=[(g, str(g)) for g in (0, 50, 100, 200, 300)])


def sc_tails(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "尾部最先消失；累积真实数据可避免")
    xs = np.linspace(-4, 4, 400)
    curves = [("真实", norm.pdf(xs), GREEN), (f"第 10 代（σ={REP[10]:.2f}）", norm.pdf(xs, 0, REP[10]), AMBER),
              (f"第 50 代（σ={REP[50]:.2f}）", norm.pdf(xs, 0, REP[50]), RED)]
    tv.line_chart(d, (130, 160, 640, 500), xs, curves, ylim=(0, 1.2), xlabel="x", ylabel="密度", legend_xy=(150, 215),
                  xticks=[(x, str(x)) for x in (-4, -2, 0, 2, 4)])
    d.text((130, 560), f"|x| > 2 的比例：真实 {TAIL(1):.1%} → 第 50 代 < 1e-6", font=F_SMALL, fill=AMBER)
    if seg >= 1:
        tv.line_chart(d, (720, 160, 1180, 500), GENS, [("替换", REP, RED), ("累积 + 保留真实数据", ACC, GREEN)], ylim=(0, 1.05),
                      xlabel="代数", ylabel="标准差", legend_xy=(860, 300), xticks=[(g, str(g)) for g in (0, 100, 200, 300)])
        vk.text_c(d, (640, 620), "参考：Shumailov et al., 2024；Gerstgrasser et al., 2024", F_SMALL, MUTED)


def sc_goodhart(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "奖励黑客：古德哈特定律")
    vk.text_c(d, (640, 130), "“当一个指标变成目标，它就不再是好的指标”", F_BODY, AMBER)
    if seg >= 1:
        tv.fpaste(c, r"$\mathrm{reward}=q+\mathrm{error},\qquad \hat q=q\left[\arg\max_{i\leq n}\mathrm{reward}_i\right]$", 640, 190, 26)
    if seg >= 2:
        series = [("正常误差（高斯）", LIGHT, GREEN)] + ([("有漏洞的误差（重尾）", HEAVY, RED)] if seg >= 3 else [])
        tv.line_chart(d, (180, 240, 1100, 590), NS, series, xlog=True, ylim=(-0.2, 3), xlabel="候选数 n", ylabel="被选中答案的真实质量",
                      legend_xy=(200, 270), xticks=[(n, str(n)) for n in NS])
    if seg >= 3:
        x = 180 + np.log10(NS[PEAK]) / np.log10(NS[-1]) * 920
        y = 590 - (HEAVY[PEAK] + 0.2) / 3.2 * 350
        d.text((x, y - 30), f"峰值 {HEAVY[PEAK]:.2f}（n = {NS[PEAK]}）", font=F_SMALL, fill=RED, anchor="mm")


def sc_lessons(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "在自我改进中被放大")
    for i, (a, b, col) in enumerate([("模型崩溃", "用自己的输出训练自己", CYAN), ("奖励黑客", "用不完美的评判选择改进", VIOLET)]):
        x = 160 + i * 500
        d.rounded_rectangle([x, 170, x + 460, 270], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 230, 205), a, F_BODY, col)
        vk.text_c(d, (x + 230, 245), b, F_SMALL, TEXT)
    if seg >= 1:
        for i, t in enumerate(["保留真实的数据和信号", "更可靠的验证", "不过度优化单一指标", "独立的测试检查真实效果"]):
            vk.text_c(d, (640, 340 + i * 50), t, F_BODY, GREEN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("模型崩溃", "分布变窄，尾部先消失"), ("奖励黑客", "挑得越狠，真实质量越差"),
                      ("对策", "真实数据 + 可靠验证 + 独立检查")], seg, "下一期：改进自己的代码")


RENDER = {"title": sc_title, "collapse": sc_collapse, "tails": sc_tails, "goodhart": sc_goodhart, "lessons": sc_lessons,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rsi-failure")
