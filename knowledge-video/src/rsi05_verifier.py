"""Recursive Self-Improvement 05 — 验证比生成容易：用评判者驱动改进.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] RSI_CACHE=<dir> TF_CACHE=<dir> python rsi05_verifier.py <out_dir>

Real computations (rsix.verifier_bestofn): the tfm.py addition Transformer after 120 training steps
answers 1,000 held-out problems 32 times each by sampling; best-of-n accuracy when choosing with a
perfect checker, a weak checker (units digit only) or by majority vote, for n = 1 ... 32.
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
TF_CACHE = os.environ.get("TF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "tfcache"))

R = rsix.cached(CACHE, "ver", rsix.verifier_bestofn, tf_cache=TF_CACHE)
NS = R["n"]
P, W, M = R["perfect"], R["weak"], R["majority"]
print(R)
assert 0.14 < P[0] < 0.17 and P[-1] > 0.85 and 0.48 < W[-1] < 0.54 and M[-1] < 0.3
assert all(a <= b for a, b in zip(P, P[1:])) and W[-1] - W[-2] < 0.03

SCENES = [
    ("title", [
        "前两期的自我改进，都依赖一个可靠的判断，题做对了没有，棋赢了没有。这一期专门讨论这个判断，也就是验证者。",
    ]),
    ("gap", [
        "很多问题有一个共同的特点，检查一个答案对不对，比想出这个答案容易得多。",
        "数独填好了，一眼就能检查；证明写出来，可以一步步核对；代码写好了，可以跑测试。这叫生成和验证之间的差距，它是自我改进的重要杠杆。",
    ]),
    ("bestofn", [
        "最简单的用法是多次采样，择优录取。让模型对同一道题回答很多次，再用验证者挑出一个通过检查的答案。",
        f"我们用一个只训练了一百二十步的加法模型来实验，它每次随机作答，只有{pct_zh(P[0], 1)}的概率答对。",
    ]),
    ("result", [
        f"有一个完美的验证者时，回答三十二次再挑选，准确率升到{pct_zh(P[-1], 1)}。模型本身没有变，变的只是挑选的方法。",
        f"如果验证者不完美，比如只检查个位数对不对，准确率最多只到{pct_zh(W[-1])}左右，再多采样也没有用，因为个位数对、其他位错的答案会混进来。",
        f"如果没有验证者，只能让多个答案投票，结果只有{pct_zh(M[-1], 1)}。验证者的质量，决定了这种改进的上限。",
    ]),
    ("loop", [
        "把择优录取的结果拿去训练模型，模型本身就会变强，这就是第三期的自我训练。验证者越可靠，循环越能持续下去。",
        "反过来，一旦验证者有漏洞，优化的压力就会集中到漏洞上。下一期，我们会看到这种失效模式。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，验证往往比生成容易，这是自我改进的重要杠杆。",
        "第二，多次采样再用验证者择优，可以大幅提高准确率。",
        "第三，验证者不完美时，改进有明确的上限。下一期，我们讨论自我改进的失效模式，模型崩溃和奖励黑客。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 5, "验证比生成容易", "用评判者驱动改进", SERIES)


def sc_gap(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "生成-验证差距")
    vk.text_c(d, (640, 180), "检查答案 ≪ 想出答案", vk.font(36), AMBER)
    if seg >= 1:
        for i, (a, b, col) in enumerate([("数独", "填好后一眼检查", CYAN), ("证明", "逐步核对", VIOLET), ("代码", "跑测试", GREEN)]):
            x = 160 + i * 340
            d.rounded_rectangle([x, 280, x + 300, 400], 14, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 150, 320), a, vk.font(30), col)
            vk.text_c(d, (x + 150, 365), b, F_SMALL, TEXT)


def sc_bestofn(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "多次采样，择优录取（best-of-n）")
    for i in range(8):
        ok = i in (2, 5)
        x = 160 + i * 120
        d.rounded_rectangle([x, 190, x + 100, 250], 10, outline=GREEN if ok else (71, 85, 105), width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 50, 220), "112" if ok else ["102", "113", "", "122", "012", "", "111", "212"][i], vk.font(24, mono=True),
                  GREEN if ok else MUTED)
    d.text((160, 150), "35+77= ：采样 8 次", font=F_SMALL, fill=TEXT)
    vk.arrow(d, (640, 270), (640, 330), AMBER, 3)
    vk.text_c(d, (640, 360), "验证者挑出通过检查的答案", F_BODY, AMBER)
    vk.text_c(d, (640, 330 - 90 + 200), "（示意）", F_SMALL, MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 520), f"实验：训练 120 步的加法模型，单次采样正确率 {P[0]:.1%}", F_BODY, TEXT)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "best-of-n 的准确率（1,000 道测试题）")
    series = [("完美验证者", [p * 100 for p in P], GREEN)]
    if seg >= 1:
        series.append(("只查个位数", [w * 100 for w in W], AMBER))
    if seg >= 2:
        series.append(("多数投票（无验证者）", [m * 100 for m in M], RED))
    tv.line_chart(d, (160, 170, 820, 560), NS, series, xlog=True, ylim=(0, 100), xlabel="采样次数 n", ylabel="准确率 %",
                  legend_xy=(180, 230), xticks=[(n, str(n)) for n in NS])
    vals = [("完美", P[-1], GREEN), ("个位数", W[-1], AMBER), ("投票", M[-1], RED)][: len(series)]
    d.text((880, 190), "n = 32", font=F_SMALL, fill=MUTED)
    for i, (k, v, col) in enumerate(vals):
        d.text((880, 240 + i * 60), k, font=F_BODY, fill=col)
        d.text((1150, 240 + i * 60), f"{v:.1%}", font=vk.font(26, mono=True), fill=col, anchor="ra")


def sc_loop(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "验证者驱动的改进循环")
    for i, (t, col) in enumerate([("采样", CYAN), ("验证择优", GREEN), ("训练", VIOLET)]):
        x = 200 + i * 320
        d.rounded_rectangle([x, 200, x + 240, 280], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 120, 240), t, F_BODY, col)
        if i < 2:
            vk.arrow(d, (x + 245, 240), (x + 315, 240), MUTED, 2)
    vk.text_c(d, (640, 330), "验证者越可靠，循环越能持续", F_BODY, GREEN)
    if seg >= 1:
        vk.text_c(d, (640, 430), "验证者有漏洞 → 优化压力集中到漏洞上", F_BODY, RED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("差距", "验证往往比生成容易"), ("best-of-n", "采样 + 验证择优，大幅提高准确率"),
                      ("上限", "验证者不完美 → 改进有上限")], seg, "下一期：失效模式 —— 模型崩溃与奖励黑客")


RENDER = {"title": sc_title, "gap": sc_gap, "bestofn": sc_bestofn, "result": sc_result, "loop": sc_loop,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rsi-verifier")
