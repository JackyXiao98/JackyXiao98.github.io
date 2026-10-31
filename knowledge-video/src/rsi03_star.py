"""Recursive Self-Improvement 03 — 自举数据：自训练与 STaR.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] RSI_CACHE=<dir> python rsi03_star.py <out_dir>

Real computations (rsix.star, JAX): the tfm.py addition Transformer trained on 300 labelled problems;
each round it answers 7,700 unlabelled training problems, keeps its own answers (all of them, or only
those passing an external check: answer - second addend = first addend), and is retrained from
scratch on the 300 + kept examples; exact-match accuracy on 2,000 held-out problems per round.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import rsix
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Recursive Self-Improvement"
CACHE = os.environ.get("RSI_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "rsicache"))

R = rsix.cached(CACHE, "star", rsix.star)
VER, NOV = R["verify"], R["noverify"]
ACC_V = [h["acc"] for h in VER]
ACC_N = [h["acc"] for h in NOV]
KEPT = [h["kept"] for h in VER]
POOL = R["n_pool"]
print(ACC_V, ACC_N, KEPT)
assert ACC_V[0] < 0.08 and ACC_V[1] > 0.3 and ACC_V[2] > 0.8 and ACC_V[-1] > 0.99
assert max(ACC_N) < 0.07 and NOV[0]["correct"] == KEPT[0] and POOL == 7700
assert all(a < b for a, b in zip(ACC_V, ACC_V[1:]))

SCENES = [
    ("title", [
        "这一期看最直接的一种自我改进，模型自己生成训练数据，再用这些数据训练自己。",
    ]),
    ("idea", [
        "思路很简单。先用少量有标注的数据训练一个模型，然后让它去解大量没有答案的问题。",
        "把它自己做对的解答挑出来，加入训练集，重新训练。新模型更强，能做对更多题，于是又能产生更多训练数据。研究者把这种方法用在推理任务上，叫做自学推理者。",
    ]),
    ("setup", [
        f"我们用变换器系列里的两位数加法模型来实际试一试。只给它三百道带答案的题，另外还有{big_zh(POOL)}道只有题目、没有答案的题。",
        "关键在于怎样判断它自己的答案对不对。我们用一个外部的检验器，用答案减去第二个加数，看是否等于第一个加数。检验器不需要会做加法题，只需要能检查。",
    ]),
    ("rounds", [
        f"第零轮，只用三百道题训练，测试准确率只有{pct_zh(ACC_V[0], 1)}。但它在没有答案的题里，仍然做对了{int_zh(KEPT[0])}道。",
        f"把这些做对的题加进训练集，第一轮准确率升到{pct_zh(ACC_V[1])}，第二轮{pct_zh(ACC_V[2])}，第三轮{pct_zh(ACC_V[3])}。",
        f"到第五轮，准确率达到{pct_zh(ACC_V[5], 1)}，被保留下来的自生成数据有{big_zh(KEPT[5])}道。之后几乎没有新的题可以加入，改进也就停了下来。",
    ]),
    ("noverify", [
        "如果不用检验器，把模型自己的所有答案都当作正确答案呢？",
        f"结果完全不同。六轮下来，准确率一直停在百分之六左右，最后一轮是{pct_zh(ACC_N[-1], 1)}。错误的答案被当成了训练数据，模型只是在重复自己的错误。",
        "所以自我训练能不能起作用，关键不在生成，而在筛选。这正是第五期要讲的验证比生成容易。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，自我训练让模型用自己生成、并通过检验的数据训练自己，能力逐轮提升。",
        "第二，改进会在可用的新数据耗尽时停下来，不会无限增长。",
        "第三，没有可靠的检验，自我训练只会放大错误。下一期，我们看另一种自我改进，和自己下棋的自我博弈。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 3, "自举数据：自训练与 STaR", "用自己生成、通过检验的数据训练自己", SERIES)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "自训练循环")
    nodes = [("少量标注数据", 640, 150, TEXT), ("训练模型", 940, 280, CYAN), ("解无答案的题", 820, 460, VIOLET),
             ("检验 → 保留正确的", 460, 460, GREEN), ("加入训练集", 340, 280, AMBER)]
    k = 2 if seg == 0 else 5
    for i, (t, x, y, col) in enumerate(nodes[:k + 1 if k < 5 else 5]):
        d.rounded_rectangle([x - 130, y - 32, x + 130, y + 32], 12, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x, y), t, F_SMALL, col)
    if seg >= 1:
        order = [(1, 2), (2, 3), (3, 4), (4, 1)]
        for a, b in order:
            (_, x1, y1, _), (_, x2, y2, _) = nodes[a], nodes[b]
            vk.arrow(d, (x1 + (x2 - x1) * 0.3, y1 + (y2 - y1) * 0.3), (x1 + (x2 - x1) * 0.7, y1 + (y2 - y1) * 0.7), MUTED, 2)
        vk.text_c(d, (640, 600), "STaR：Self-Taught Reasoner（Zelikman et al., 2022）", F_SMALL, MUTED)


def sc_setup(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "实验设置")
    rows = [("有答案的题", "300 道", CYAN), ("只有题目的题", f"{POOL:,} 道", VIOLET), ("测试题", "2,000 道（从不参与训练）", MUTED)]
    for i, (a, b, col) in enumerate(rows):
        d.text((200, 170 + i * 60), a, font=F_BODY, fill=col)
        d.text((520, 170 + i * 60), b, font=F_BODY, fill=TEXT)
    if seg >= 1:
        d.rounded_rectangle([200, 380, 1080, 470], 14, outline=GREEN, width=2, fill=(12, 19, 34))
        vk.text_c(d, (640, 410), "检验器：答案 - 第二个加数 = 第一个加数？", F_BODY, GREEN)
        vk.text_c(d, (640, 448), "只需要能检查，不需要会解题", F_SMALL, MUTED)


def sc_rounds(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "逐轮的测试准确率（有检验器）")
    k = 1 if seg == 0 else (4 if seg == 1 else 6)
    tv.bar_chart(d, (150, 180, 760, 540), [f"第 {i} 轮" for i in range(k)], [a * 100 for a in ACC_V[:k]],
                 [CYAN] * k, fmt="{:.1f}%", vmax=115)
    d.text((780, 180), "保留的自生成数据", font=F_SMALL, fill=MUTED)
    for i in range(k):
        y = 220 + i * 50
        d.text((780, y), f"第 {i} 轮", font=F_SMALL, fill=MUTED)
        w = KEPT[i] / POOL * 300
        d.rectangle([870, y + 4, 870 + w, y + 26], fill=GREEN)
        d.text((880 + w, y + 15), f"{KEPT[i]:,}", font=vk.font(18, mono=True), fill=GREEN, anchor="lm")


def sc_noverify(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "不用检验器：所有自生成答案都当作正确")
    if seg >= 1:
        rounds = list(range(6))
        tv.line_chart(d, (200, 170, 1080, 520), rounds, [("有检验器", [a * 100 for a in ACC_V], CYAN),
                                                         ("无检验器", [a * 100 for a in ACC_N], RED)],
                      ylim=(0, 100), xlabel="轮次", ylabel="测试准确率 %", legend_xy=(860, 360),
                      xticks=[(r, str(r)) for r in rounds])
    if seg >= 2:
        vk.text_c(d, (640, 600), "关键不在生成，而在筛选", vk.font(32), AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("自训练", "自生成 + 通过检验 → 逐轮提升"), ("停止", "新数据耗尽，改进停下"),
                      ("检验", "没有检验只会放大错误")], seg, "下一期：自我博弈 —— 从 AlphaGo Zero 说起")


RENDER = {"title": sc_title, "idea": sc_idea, "setup": sc_setup, "rounds": sc_rounds, "noverify": sc_noverify,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rsi-star")
