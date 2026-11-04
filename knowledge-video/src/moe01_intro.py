"""MoE 01 — 什么是混合专家：参数多、计算少.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MOE_CACHE=<dir> python moe01_intro.py <out_dir>

Real computations (moex.exp_scale, JAX): a mixture of 8 hidden Markov "languages" over 32 tokens (sequence length
32; the model must infer the language from context; floor = true conditional entropy, about 1.557 nats). A 2-layer
Transformer (d = 64) with a dense FFN of hidden 128, a dense FFN of hidden 1,024 (8x the FFN compute), and a MoE
layer of 8 experts of hidden 128 with top-1 routing (Switch auxiliary loss 0.01), each trained 2,000 steps
(batch 32, Adam 3e-3, cosine). Held-out loss on 256 sequences; total and per-token active parameters.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import moex
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Mixture of Experts"
CACHE = os.environ.get("MOE_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "moecache"))

R = moex.cached(CACHE, "scale", moex.exp_scale)
FLOOR = moex.floor_entropy()
D1, D8, M = R["dense128"], R["dense1024"], R["moe8"]
print(FLOOR, {k: (round(v["final"], 4), v["params"], v["active"]) for k, v in R.items()})
assert abs(FLOOR - 1.557) < 2e-3
assert round(D1["final"], 2) == 1.79 and round(D8["final"], 2) == 1.69 and round(M["final"], 2) == 1.69
assert abs(M["final"] - D8["final"]) < 0.005 and D1["params"] == 71680 and M["params"] == 302080 and M["active"] == 72704
assert D8["params"] == 301056
STEPS = [h["step"] for h in M["hist"]]

SCENES = [
    ("title", [
        "今天最大的语言模型，很多都不是把所有参数一起用的。它们有几百个专家，每个词元只交给其中几个来处理。这个系列我们讲混合专家模型。这一期先看它为什么有用：参数多，计算却少。",
    ]),
    ("idea", [
        "变换器的每一层，除了注意力，还有一个前馈网络，它占了模型的大部分参数。混合专家的做法是，把这个前馈网络复制成很多份，每一份叫一个专家。",
        "再加一个很小的路由器，它看每个词元，给每个专家打分，只把这个词元交给得分最高的那一个或几个专家。这样，参数多了很多倍，但每个词元实际用到的计算，只有一个专家那么多。",
    ]),
    ("toy", [
        "我们用一个能完全算清楚的小任务。数据来自八种隐藏的语言，每种语言是一条马尔可夫链，有自己的转移规律。每一段序列来自其中一种，模型要从上下文猜出是哪种语言，再预测下一个词元。",
        f"因为我们知道真实的规律，所以知道最好可能的损失是{dec_zh(FLOOR, 3)}。模型是两层的小变换器，我们只改前馈网络这一部分。",
    ]),
    ("result", [
        f"先看两个普通的稠密模型。前馈网络宽度一百二十八的模型，训练两千步后损失是{dec_zh(D1['final'], 2)}。把宽度加到一千零二十四，计算量变成八倍，损失降到{dec_zh(D8['final'], 2)}。",
        f"再看混合专家。八个宽度一百二十八的专家，每个词元只用一个。它的损失也是{dec_zh(M['final'], 2)}，和八倍计算的稠密模型几乎一样，而每个词元的计算量，和最小的那个模型相同。",
    ]),
    ("cost", [
        f"代价在哪里？总参数。混合专家有约{big_zh(round(M['params'] / 1000))}千个参数，和大的稠密模型差不多，是小模型的四倍多；但每个词元只激活约{big_zh(round(M['active'] / 1000))}千个，和小模型一样。",
        "所以混合专家是用存储换计算，参数要全部放在显存里，但每个词元只算其中一小部分。后面几期我们会看到，这个交换还带来了一系列新问题。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，混合专家把前馈网络换成多个专家，由路由器为每个词元挑选少数几个。",
        "第二，在我们的实验里，八个专家每次用一个，就达到了八倍计算的稠密模型的效果。",
        "第三，它省的是计算，参数总量并没有减少。下一期，我们来看路由器怎样决定每个词元去哪里。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 1, "什么是混合专家", "参数多、计算少", SERIES)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "把前馈网络换成多个专家")
    d.rounded_rectangle([120, 220, 400, 320], 14, outline=MUTED, width=2, fill=(12, 19, 34))
    vk.text_c(d, (260, 270), "前馈网络（FFN）", F_SMALL, TEXT)
    vk.text_c(d, (260, 360), "稠密：每个词元都用全部", F_SMALL, MUTED)
    vk.arrow(d, (430, 270), (500, 270), MUTED, 3)
    for i in range(8):
        x = 540 + i * 82
        on = seg >= 1 and i == 2
        col = GREEN if on else VIOLET
        d.rounded_rectangle([x, 230, x + 66, 310], 10, outline=col, width=3 if on else 2, fill=(12, 19, 34))
        vk.text_c(d, (x + 33, 270), f"E{i + 1}", F_SMALL, col)
    if seg >= 1:
        d.rounded_rectangle([780, 420, 960, 480], 12, outline=AMBER, width=2, fill=(12, 19, 34))
        vk.text_c(d, (870, 450), "路由器", F_SMALL, AMBER)
        tv.token(d, 840, 530, "词元", CYAN, w=60, font=F_SMALL)
        vk.arrow(d, (870, 420), (540 + 2 * 82 + 33, 316), GREEN, 3)
        vk.text_c(d, (640, 640), "参数 × 8，每个词元的计算 × 1", F_BODY, GREEN)


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "小任务：8 种隐藏的语言")
    rng = np.random.default_rng(3)
    x, lang = moex.sample(rng, 4)
    cols = [CYAN, VIOLET, AMBER, GREEN]
    for r in range(4):
        y = 180 + r * 70
        d.text((90, y + 8), f"语言 {lang[r] + 1}", font=F_SMALL, fill=cols[r])
        for t in range(14):
            tv.token(d, 230 + t * 62, y, str(x[r, t]), cols[r], w=52, font=vk.font(20, mono=True))
    vk.text_c(d, (640, 500), "每种语言：一条马尔可夫链（32 个词元，各自的转移规律）", F_SMALL, TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 560), f"最好可能的损失（真实条件熵）：{FLOOR:.3f}", F_BODY, GREEN)
        vk.text_c(d, (640, 620), "两层变换器（d = 64），只改前馈网络", F_SMALL, MUTED)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "相同计算量下，MoE 追上 8 倍计算的稠密模型")
    ser = [("稠密 128", np.minimum([h["loss"] for h in D1["hist"]], 2.6), MUTED),
           ("稠密 1024（8 倍计算）", np.minimum([h["loss"] for h in D8["hist"]], 2.6), CYAN)]
    if seg >= 1:
        ser.append(("MoE 8 × 128，选 1", np.minimum([h["loss"] for h in M["hist"]], 2.6), GREEN))
    upto = int(len(STEPS) * min(1, u * 1.6)) if seg == 0 else None
    tv.line_chart(d, (110, 170, 760, 590), STEPS, ser, ylim=(1.5, 2.7), upto=upto, xlabel="训练步数", xticks=[(0, "0"), (1000, "1000"), (2000, "2000")], legend_xy=(800, 190))
    d.text((110, 140), "验证损失", font=F_SMALL, fill=MUTED)
    y_f = 590 - (FLOOR - 1.5) / 1.2 * 420
    d.line([(110, y_f), (760, y_f)], fill=(71, 85, 105), width=1)
    d.text((765, y_f), f"下限 {FLOOR:.3f}", font=F_SMALL, fill=MUTED, anchor="lm")
    f = vk.font(24, mono=True)
    rows = [("稠密 128", D1, MUTED), ("稠密 1024", D8, CYAN)] + ([("MoE 8×128", M, GREEN)] if seg >= 1 else [])
    for i, (n, r, col) in enumerate(rows):
        d.text((800, 340 + i * 50), n, font=F_SMALL, fill=col)
        d.text((1010, 340 + i * 50), f"{r['final']:.3f}", font=f, fill=TEXT)


def sc_cost(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "总参数 vs 每个词元激活的参数")
    labs = ["稠密 128", "稠密 1024", "MoE 8×128"]
    tv.bar_chart(d, (120, 200, 600, 560), labs, [D1["params"] / 1e3, D8["params"] / 1e3, M["params"] / 1e3],
                 [MUTED, CYAN, GREEN], fmt="{:.0f}k", vmax=340)
    d.text((120, 160), "总参数", font=F_SMALL, fill=MUTED)
    tv.bar_chart(d, (700, 200, 1180, 560), labs, [D1["active"] / 1e3, D8["active"] / 1e3, M["active"] / 1e3],
                 [MUTED, CYAN, GREEN], fmt="{:.0f}k", vmax=340)
    d.text((700, 160), "每个词元激活的参数", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 640), "用存储换计算：参数都要放进显存，每个词元只算一小部分", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("结构", "多个专家 + 路由器选少数几个"), ("实验", "8 选 1 ≈ 8 倍计算的稠密模型"),
                      ("代价", "省计算，不省参数")], seg, "下一期：路由——每个词元去哪个专家")


RENDER = {"title": sc_title, "idea": sc_idea, "toy": sc_toy, "result": sc_result, "cost": sc_cost, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "moe-intro")
