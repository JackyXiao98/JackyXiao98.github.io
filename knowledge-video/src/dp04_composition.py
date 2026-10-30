"""Differential Privacy 04 — 组合定理与隐私预算.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python dp04_composition.py <out_dir>

Real computations: an averaging attack — the same counting query answered many times with Laplace
noise (eps = 1 each), and how often rounding the average recovers the exact count (2,000 repeats);
basic vs advanced composition (Dwork, Rothblum & Vadhan, 2010) for eps0 = 0.01 per query.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import dpx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Differential Privacy"
rng = np.random.default_rng(0)

# 1. averaging attack on the count from episode 01 (true count 2, eps = 1 per answer)
TRUE = 2
KS = [1, 10, 100, 1000]
HIT = {k: float(np.mean([round(TRUE + rng.laplace(0, 1, k).mean()) == TRUE for _ in range(2000)])) for k in KS}
ONE_RUN = TRUE + rng.laplace(0, 1, 1000)
RUN_AVG = np.cumsum(ONE_RUN) / np.arange(1, 1001)
print(HIT)
assert HIT[1] < 0.45 and HIT[100] > 0.99 and HIT[1000] == 1.0

# 2. composition: k queries with eps0 = 0.01 each, delta' = 1e-5
EPS0, DP = 0.01, 1e-5
KGRID = np.unique(np.logspace(0, 4, 60).astype(int))
BASIC = EPS0 * KGRID
ADV = dpx.advanced_composition(EPS0, KGRID, DP)
CROSS = int(KGRID[np.argmax(ADV < BASIC)])
B1K, A1K = EPS0 * 1000, dpx.advanced_composition(EPS0, 1000, DP)
B10K, A10K = EPS0 * 10000, dpx.advanced_composition(EPS0, 10000, DP)
print(CROSS, A1K, A10K)
assert abs(A1K - 1.62) < 0.01 and abs(A10K - 5.80) < 0.01 and 15 < CROSS < 40

SCENES = [
    ("title", [
        "前几期讲的都是只回答一个问题。现实中，同一份数据会被查询很多次。这一期我们看，隐私损失是怎样累积的，以及怎样用隐私预算来管理它。",
    ]),
    ("average", [
        "回到第一期的例子，月薪超过两万的人数，真实答案是二。每次回答都加上艾普西隆等于一的拉普拉斯噪声。",
        f"只问一次，四舍五入以后猜中真实答案的概率只有{pct_zh(HIT[1])}。但如果攻击者把同一个问题问一百次，再取平均，噪声互相抵消，在两千次模拟中猜中的比例达到了{pct_zh(HIT[100])}。",
        "这时差分攻击又成功了。所以每次查询都会消耗隐私，必须把总的隐私损失记下来。",
    ]),
    ("basic", [
        "最简单的规则是基本组合定理，多个算法一起运行，总的艾普西隆就是各自的艾普西隆相加。问一百次，每次艾普西隆等于一，总共就是一百，几乎没有保护。",
        "还有两个有用的性质。第一，如果查询作用在互不相交的人群上，比如年龄分布图里每个人只落在一个区间，总代价只取最大值，而不是相加。第二，对加噪后的结果做任何处理，都不会额外消耗隐私。",
    ]),
    ("advanced", [
        "基本组合定理往往太悲观。高级组合定理指出，只要允许一个很小的德尔塔，总的艾普西隆大约随查询次数的平方根增长，而不是线性增长。",
        f"比如每次查询艾普西隆等于零点零一，查询一千次，基本组合给出十，高级组合只有{dec_zh(A1K)}；查询一万次，基本组合是一百，高级组合是{dec_zh(A10K)}。",
        f"不过次数少时，高级组合反而更大，在这个例子里，超过{int_zh(CROSS)}次以后，它才更有优势。",
    ]),
    ("budget", [
        "实践中，我们给每份数据设定一个总的隐私预算。每回答一个查询，就从预算里扣除相应的艾普西隆，预算用完就停止回答。",
        "训练机器学习模型要迭代成千上万步，每一步都要访问数据，组合定理就变得至关重要。下一期，我们介绍更精确的记账方法，以及随机抽样带来的隐私放大。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，重复查询会让噪声被平均掉，隐私损失会累积。",
        "第二，基本组合把艾普西隆直接相加；高级组合让它近似按平方根增长。",
        "第三，不相交的数据只需付一次代价，后处理不额外消耗隐私，用隐私预算统一管理。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 4, "组合定理与隐私预算", "隐私损失怎样累积", SERIES)


def sc_average(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "平均攻击：同一个问题问很多次（每次 ε = 1）")
    n = 1000 if seg >= 1 else max(2, int(1000 * vk.ease(u)))
    ks = np.arange(1, 1001)
    tv.line_chart(d, (130, 160, 680, 540), ks, [("前 k 次回答的平均", RUN_AVG, CYAN), ("真实答案 2", np.full(1000, 2.0), GREEN)],
                  xlog=True, ylim=(-1, 5), upto=n, xlabel="查询次数 k", ylabel="平均值", legend_xy=(380, 175),
                  xticks=[(1, "1"), (10, "10"), (100, "100"), (1000, "1000")])
    if seg >= 1:
        d.text((760, 170), "四舍五入后猜中真实答案的概率", font=F_SMALL, fill=MUTED)
        tv.bar_chart(d, (760, 220, 1180, 520), [str(k) for k in KS], [HIT[k] * 100 for k in KS], [RED, AMBER, GREEN, GREEN],
                     fmt="{:.0f}%", vmax=115)
        d.text((1190, 540), "k", font=F_SMALL, fill=MUTED)
    if seg >= 2:
        vk.text_c(d, (640, 620), "每次查询都消耗隐私 → 需要记录总的隐私损失", F_BODY, AMBER)


def sc_basic(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "基本组合 · 并行组合 · 后处理")
    tv.fpaste(c, r"$M_1,\dots,M_k\ \text{each}\ \varepsilon_i\text{-DP}\ \Rightarrow\ (M_1,\dots,M_k)\ \text{is}\ \left(\sum_i\varepsilon_i\right)\text{-DP}$", 640, 160, 26)
    if seg >= 1:
        for i, (t, b, col) in enumerate([("并行组合", "不相交的人群：代价取 max，不相加", CYAN),
                                         ("后处理", "对噪声结果做任何计算：不额外消耗", VIOLET)]):
            x = 160 + i * 500
            d.rounded_rectangle([x, 260, x + 460, 420], 14, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 230, 305), t, F_BODY, col)
            vk.text_c(d, (x + 230, 365), b, F_SMALL, TEXT)
        for j, h in enumerate([3, 6, 9, 7, 4, 2]):
            d.rectangle([220 + j * 55, 560 - h * 12, 260 + j * 55, 560], fill=CYAN)
        d.text((220, 570), "年龄分布：每人只在一个区间", font=F_SMALL, fill=MUTED)


def sc_advanced(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "高级组合（ε₀ = 0.01，δ′ = 1e-5）".replace("₀", "0"))
    tv.fpaste(c, r"$\varepsilon_{\mathrm{total}}=\sqrt{2k\ln(1/\delta')}\,\varepsilon_0+k\,\varepsilon_0\,(e^{\varepsilon_0}-1)$", 640, 140, 28)
    n = len(KGRID) if seg >= 1 else max(2, int(len(KGRID) * vk.ease(u * 1.5 if u < .66 else 1)))
    tv.line_chart(d, (150, 200, 700, 570), KGRID, [("基本组合 k·ε0", BASIC, RED), ("高级组合", ADV, CYAN)], xlog=True, ylog=True,
                  upto=n, xlabel="查询次数 k", ylabel="总 ε（对数）", legend_xy=(170, 230),
                  xticks=[(1, "1"), (10, "10"), (100, "100"), (1000, "1K"), (10000, "10K")])
    if seg >= 1:
        rows = [("k = 1000", f"{B1K:.0f}", f"{A1K:.2f}"), ("k = 10000", f"{B10K:.0f}", f"{A10K:.2f}")]
        d.text((920, 230), "基本", font=F_SMALL, fill=RED)
        d.text((1060, 230), "高级", font=F_SMALL, fill=CYAN)
        for i, (k, b, a) in enumerate(rows):
            y = 280 + i * 60
            d.text((890, y + 4), k, font=vk.font(22, mono=True), fill=MUTED, anchor="ra")
            d.text((920, y), b, font=vk.font(26, mono=True), fill=RED)
            d.text((1060, y), a, font=vk.font(26, mono=True), fill=CYAN)
    if seg >= 2:
        vk.text_c(d, (990, 470), f"k 小于约 {CROSS} 时，基本组合更紧", F_SMALL, AMBER)


def sc_budget(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "隐私预算")
    total, spent = 1.0, [0.2, 0.1, 0.3, 0.25, 0.15]
    x0, w = 200, 880
    d.rounded_rectangle([x0, 200, x0 + w, 260], 10, outline=(71, 85, 105), width=2)
    k = len(spent) if seg >= 1 else 1 + int(4 * vk.ease(u))
    x = x0
    for i, e in enumerate(spent[:k]):
        d.rectangle([x + 2, 202, x + e * w - 2, 258], fill=[CYAN, VIOLET, GREEN, AMBER, RED][i])
        vk.text_c(d, (x + e * w / 2, 230), f"{e}", vk.font(20, mono=True), (10, 14, 22))
        x += e * w
    d.text((x0, 170), "每个色块 = 一次查询消耗的 ε", font=F_SMALL, fill=MUTED)
    d.text((x0, 280), f"总预算 ε = {total:g}", font=F_BODY, fill=TEXT)
    if k == len(spent):
        d.text((x0 + w, 280), "预算用完 → 停止回答", font=F_BODY, fill=RED, anchor="ra")
    if seg >= 1:
        vk.text_c(d, (640, 420), "训练模型：上万步迭代，每一步都访问数据", F_BODY, AMBER)
        vk.text_c(d, (640, 480), "→ 需要更精确的记账（下一期）", F_BODY, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("累积", "重复查询会把噪声平均掉"), ("组合定理", "基本：相加；高级：≈ √k 增长"),
                      ("预算", "并行组合、后处理、隐私预算")], seg, "下一期：Rényi 差分隐私与子采样放大")


RENDER = {"title": sc_title, "average": sc_average, "basic": sc_basic, "advanced": sc_advanced, "budget": sc_budget,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "dp-composition")
