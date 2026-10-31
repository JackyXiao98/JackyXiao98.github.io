"""RL for LLMs 06 — 推理模型：思维链与测试时扩展.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] RL_CACHE=<dir> python rl06_reasoning.py <out_dir>

Real computations (rlx.cot, JAX): the same 2-layer Transformer (d = 64) trained for 1,500 steps on
"sum of n digits mod 10", either answering directly (n = 2, 4, 8, 12) or first writing the running
sums (chain of thought, n = 12); greedy accuracy on 2,000 held-out problems.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import rlx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "RL for LLMs"
CACHE = os.environ.get("RL_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "rlcache"))

NS = [2, 4, 8, 12]
DIRECT = {n: rlx.cached(CACHE, "cot", rlx.cot, n_digits=n, use_cot=False) for n in NS}
COT = rlx.cached(CACHE, "cot", rlx.cot, n_digits=12, use_cot=True)
FIN = {n: DIRECT[n]["hist"][-1]["acc"] for n in NS}
COT_FIN = COT["hist"][-1]["acc"]
COT_250 = COT["hist"][1]["acc"]
EXTRA = COT["extra_tokens"]
print(FIN, COT_FIN, COT_250, EXTRA)
assert FIN[2] == 1.0 and FIN[4] > 0.99 and FIN[8] < 0.13 and FIN[12] < 0.12 and COT_FIN == 1.0 and COT_250 == 1.0 and EXTRA == 13
EX = [3, 8, 5, 1, 9, 7, 2, 6, 4, 0, 8, 5]
RUN = list(np.cumsum(EX) % 10)
assert RUN[-1] == sum(EX) % 10

SCENES = [
    ("title", [
        "近年的推理模型，会在回答之前先写出很长的思考过程。这一期看为什么想得更久能让模型更聪明，以及测试时的计算怎样换来准确率。",
    ]),
    ("why", [
        "一个变换器每生成一个词元，只做固定层数的计算。如果一个问题需要的计算步骤，超过了一次前向计算能完成的量，直接给出答案就很难做对。",
        "思维链让模型先写出中间结果，每写一个词元，就多一次完整的计算。中间结果又会被后面的步骤读到，相当于把一个深的计算，拆成很多步浅的计算。",
    ]),
    ("task", [
        "我们用一个干净的实验来看。任务是求一串数字的和，除以十的余数。",
        f"直接回答，就是看完所有数字后马上给出答案。思维链则是一边读，一边写出到目前为止的累计和，最后再给出答案。对十二个数字，思维链要多写{int_zh(EXTRA)}个词元。",
    ]),
    ("result", [
        "我们用同一个两层的小变换器，各训练一千五百步。",
        f"直接回答时，两个数字的准确率是百分之百，四个数字是{pct_zh(FIN[4], 1)}。但到了八个数字，准确率只有{pct_zh(FIN[8])}，十二个数字是{pct_zh(FIN[12])}，和随便猜十分之一差不多。",
        f"同样是十二个数字，用思维链训练，只用二百五十步，准确率就达到了百分之百。模型的大小没变，变的只是允许它在回答前多算几步。",
    ]),
    ("scaling", [
        "这就是测试时扩展的一种，用更长的推理换取更高的准确率。另一种是并行地多次采样，再投票或者用验证器挑选，这在递归自我改进系列里讲过。",
        "强化学习在这里起了关键作用。用可验证的奖励训练，模型会自己学会写更长、更有用的思考过程，在难题上主动花更多的计算。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，每个词元只做固定量的计算，复杂问题需要把计算拆成多步。",
        "第二，思维链让同一个小模型，从只能随便猜变成完全做对。",
        "第三，更长的推理和更多的采样，都是用测试时的计算换取准确率。下一期，我们看奖励黑客和对齐税。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 6, "推理模型", "思维链与测试时扩展", SERIES)


def sc_why(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "每个词元 = 一次固定深度的计算")
    for i in range(3):
        d.rounded_rectangle([160, 170 + i * 50, 460, 210 + i * 50], 8, outline=CYAN, width=2, fill=(12, 19, 34))
        vk.text_c(d, (310, 190 + i * 50), f"第 {i + 1} 层", F_SMALL, CYAN)
    vk.text_c(d, (310, 350), "一次前向：层数固定", F_SMALL, TEXT)
    if seg >= 1:
        for j in range(5):
            x = 620 + j * 110
            for i in range(3):
                d.rectangle([x, 170 + i * 50, x + 90, 205 + i * 50], outline=VIOLET, width=2)
            vk.text_c(d, (x + 45, 340), f"词元 {j + 1}", F_SMALL, VIOLET)
            if j < 4:
                vk.arrow(d, (x + 92, 260), (x + 108, 260), MUTED, 2)
        vk.text_c(d, (840, 400), "思维链：每多写一个词元，多一次完整计算", F_BODY, GREEN)


def sc_task(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "任务：一串数字之和除以 10 的余数")
    f = vk.font(28, mono=True)
    d.text((140, 170), "直接回答：", font=F_BODY, fill=RED)
    d.text((140, 220), "".join(map(str, EX)) + " = " + str(RUN[-1]), font=f, fill=TEXT)
    if seg >= 1:
        d.text((140, 320), "思维链：", font=F_BODY, fill=GREEN)
        d.text((140, 370), "".join(map(str, EX)) + " = " + "".join(map(str, RUN)) + " | " + str(RUN[-1]), font=f, fill=TEXT)
        d.text((140, 420), "                 ↑ 累计和（多写 13 个词元）", font=F_SMALL, fill=GREEN)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "同一个 2 层小模型，各训练 1500 步")
    if seg >= 1:
        labs = [f"{n} 个数字" for n in NS] + (["12 个 + 思维链"] if seg >= 2 else [])
        vals = [FIN[n] * 100 for n in NS] + ([COT_FIN * 100] if seg >= 2 else [])
        cols = [CYAN, CYAN, RED, RED] + ([GREEN] if seg >= 2 else [])
        tv.bar_chart(d, (160, 190, 1120, 540), labs, vals, cols, fmt="{:.1f}%", vmax=115)
        y10 = 540 - 10 / 115 * 350
        d.line([(160, y10), (1120, y10)], fill=MUTED, width=1)
        d.text((1125, y10), "瞎猜 10%", font=F_SMALL, fill=MUTED, anchor="lm")
    if seg >= 2:
        vk.text_c(d, (640, 610), "思维链：第 250 步就达到 100%", F_BODY, GREEN)


def sc_scaling(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "测试时扩展")
    for i, (a, b, col) in enumerate([("更长的推理（串行）", "思维链 / 推理模型", GREEN), ("更多的采样（并行）", "投票 / 验证器择优", CYAN)]):
        x = 160 + i * 500
        d.rounded_rectangle([x, 180, x + 460, 290], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 230, 215), a, F_BODY, col)
        vk.text_c(d, (x + 230, 260), b, F_SMALL, TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 400), "可验证奖励 + 强化学习 → 模型自己学会写更长、更有用的思考", F_BODY, AMBER)
        vk.text_c(d, (640, 450), "（如 OpenAI o1, 2024；DeepSeek-R1, 2025）", F_SMALL, MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("固定计算", "每个词元只做固定层数的计算"), ("思维链", "同一小模型：瞎猜 → 全对"),
                      ("测试时扩展", "更长的推理 · 更多的采样")], seg, "下一期：奖励黑客与对齐税")


RENDER = {"title": sc_title, "why": sc_why, "task": sc_task, "result": sc_result, "scaling": sc_scaling, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rl-reasoning")
