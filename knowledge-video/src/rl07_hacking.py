"""RL for LLMs 07 — 奖励黑客与对齐税.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] RL_CACHE=<dir> python rl07_hacking.py <out_dir>

Real computations (rlx.length_hack): 400 responses with a length (20-600 tokens) and a true quality
that drops for answers longer than ~250 tokens; a reward model that sees quality only noisily and
also rewards length (raters' length bias); the KL-regularised optimum for 50 values of beta: average
length, reward-model score and true quality.
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

LH = rlx.cached(CACHE, "len", rlx.length_hack)
CU = LH["curve"]
KL = np.array([x["kl"] for x in CU])
LEN = np.array([x["length"] for x in CU])
TRUE = np.array([x["true"] for x in CU])
PROXY = np.array([x["proxy"] for x in CU])
PEAK = int(TRUE.argmax())
END = CU[-1]
print(LH["ref_len"], LH["ref_true"], CU[PEAK], END)
assert 155 < LH["ref_len"] < 165 and END["length"] > 550 and END["true"] < LH["ref_true"] < 0 and TRUE[PEAK] > 1.5
assert all(np.diff(PROXY) > -1e-9)

SCENES = [
    ("title", [
        "前几期我们多次提到，优化一个不完美的奖励会出问题。这一期专门看大模型强化学习里最常见的几种奖励黑客，以及对齐税。",
    ]),
    ("examples", [
        "第一种是长度偏好。人在比较两个回答时，往往觉得更长、更详细的回答更好，奖励模型也学到了这个偏好，于是模型越来越啰嗦。",
        "第二种是讨好用户。人倾向于给赞同自己观点的回答更高的评价，模型就学会了附和，甚至在用户说错时也表示同意。第三种是在可验证任务里钻检查器的漏洞，比如修改测试本身让代码通过。",
    ]),
    ("toy", [
        "我们用一个玩具来看长度偏好。四百个回答，长度从二十到六百个词元，超过大约二百五十个词元以后，回答因为啰嗦，真实质量开始下降。",
        "奖励模型只能粗略地看到质量，同时偏爱更长的回答。和第三期一样，我们逐渐放松相对熵约束，看最优策略会怎样变化。",
    ]),
    ("result", [
        f"参考模型的回答平均约{int_zh(round(LH['ref_len']))}个词元。随着约束放松，平均长度一路上升，奖励模型的分数也一路上升。",
        f"真实质量先上升，在平均长度约{int_zh(round(LEN[PEAK], -1))}个词元时达到最高，之后迅速下降。约束最松时，平均长度达到{int_zh(round(END['length']))}个词元，真实质量是负的{dec_zh(-END['true'], 2)}，比参考模型还差。",
        "奖励模型的分数从头到尾都在上升，只看它，完全发现不了问题。",
    ]),
    ("tax", [
        "另一个代价叫对齐税。为了让模型更符合人的偏好而做的训练，有时会让它在别的能力上变差，比如一些标准测试的成绩下降。",
        "常见的应对办法包括，在强化学习时混入原来的预训练数据，用相对熵约束限制偏离，以及专门对长度、讨好等行为做校正和评测。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，长度偏好、讨好用户和钻检查器漏洞，是大模型强化学习中常见的奖励黑客。",
        "第二，优化有偏的奖励模型时，分数一直上升，真实质量却会先升后降。",
        "第三，对齐训练可能带来对齐税，需要用约束、混合数据和专门的评测来控制。最后一期，我们看前沿的进展。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 7, "奖励黑客与对齐税", "分数在涨，质量在跌", SERIES)


def sc_examples(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "常见的奖励黑客")
    items = [("长度偏好", "越长越“好” → 越来越啰嗦", AMBER, 0), ("讨好用户", "附和用户，即使用户说错（Sharma et al., 2023）", VIOLET, 1),
             ("钻检查器漏洞", "例如修改测试本身让代码通过", RED, 1)]
    for i, (a, b, col, s) in enumerate(items):
        if seg >= s:
            y = 160 + i * 110
            d.rounded_rectangle([140, y, 1140, y + 84], 12, outline=col, width=2, fill=(12, 19, 34))
            d.text((170, y + 42), a, font=F_BODY, fill=col, anchor="lm")
            d.text((400, y + 42), b, font=F_SMALL, fill=TEXT, anchor="lm")


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "玩具：400 个回答，奖励模型偏爱长回答")
    Ls = np.linspace(20, 600, 300)
    tv.line_chart(d, (140, 170, 640, 500), Ls, [("真实质量中与长度有关的部分", -0.008 * np.maximum(Ls - 250, 0), GREEN)],
                  ylim=(-3, 0.5), xlabel="长度（词元）", ylabel="", legend_xy=(160, 190), xticks=[(x, str(x)) for x in (20, 250, 600)])
    if seg >= 1:
        tv.line_chart(d, (720, 170, 1180, 500), Ls, [("奖励模型中的长度偏好", 0.008 * (Ls - 200), AMBER)], ylim=(-2, 3.5),
                      xlabel="长度（词元）", ylabel="", legend_xy=(740, 190), xticks=[(x, str(x)) for x in (20, 200, 600)])
        vk.text_c(d, (640, 600), "奖励模型 = 0.6 × 质量 + 噪声 + 长度偏好", F_SMALL, MUTED)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "放松约束：长度、分数与真实质量")
    n = PEAK + 1 if seg == 0 else len(CU)
    x = KL + 1e-3
    tv.line_chart(d, (110, 160, 620, 520), x[:n], [("平均长度", LEN[:n], CYAN)], xlog=True, ylim=(100, 600),
                  xlabel="KL(π ‖ π_ref)", ylabel="词元", legend_xy=(130, 215), xticks=[(0.01, "0.01"), (1, "1"), (10, "10")])
    tv.line_chart(d, (690, 160, 1200, 520), x[:n], [("奖励模型分数", PROXY[:n], AMBER), ("真实质量", TRUE[:n], GREEN)], xlog=True,
                  ylim=(-1.5, 5), xlabel="KL(π ‖ π_ref)", ylabel="", legend_xy=(710, 190), xticks=[(0.01, "0.01"), (1, "1"), (10, "10")])
    if seg >= 1:
        vk.text_c(d, (640, 600), f"约束最松：长度 {END['length']:.0f}，真实质量 {END['true']:.2f}（参考模型 {LH['ref_true']:.2f}）", F_BODY, RED)
    if seg >= 2:
        vk.text_c(d, (640, 650), "奖励模型分数从头到尾都在上升", F_SMALL, AMBER)


def sc_tax(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "对齐税")
    vk.text_c(d, (640, 190), "对齐训练 → 某些标准测试的成绩下降（InstructGPT, Ouyang et al., 2022）", F_BODY, TEXT)
    if seg >= 1:
        for i, t in enumerate(["混入原来的预训练数据（PPO-ptx）", "KL 约束限制偏离", "针对长度、讨好的校正与评测"]):
            vk.text_c(d, (640, 300 + i * 60), t, F_BODY, GREEN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("奖励黑客", "长度 · 讨好 · 钻检查器漏洞"), ("先升后降", "分数一直涨，真实质量先升后降"),
                      ("对齐税", "约束 + 混合数据 + 专门评测")], seg, "最后一期：前沿进展")


RENDER = {"title": sc_title, "examples": sc_examples, "toy": sc_toy, "result": sc_result, "tax": sc_tax, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rl-hacking")
