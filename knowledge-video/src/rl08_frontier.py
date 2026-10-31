"""RL for LLMs 08 — 前沿进展：从 RLHF 到智能体强化学习.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] RL_CACHE=<dir> python rl08_frontier.py <out_dir>

Real computations (rlx.process_vs_outcome): an 8-step task with 4 actions per step (one correct);
REINFORCE with a running baseline, rewarded either only for solving the whole task (outcome) or for
the fraction of correct steps (process); probability of solving the task, averaged over 200 runs.
Facts about real systems are quoted with references in the notes.
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

PV = rlx.cached(CACHE, "pvo", rlx.process_vs_outcome)
EPS = np.array(PV["episodes"])
OUT, PRO = np.array(PV["outcome"]), np.array(PV["process"])
K, A = 8, 4
P_RANDOM = A ** -K
EXP_HITS = 3000 * P_RANDOM
at = lambda arr, e: float(arr[list(EPS).index(e)])
print(P_RANDOM, EXP_HITS, at(OUT, 3000), at(PRO, 1000), at(PRO, 3000))
assert P_RANDOM == 1 / 65536 and EXP_HITS < 0.05 and at(OUT, 3000) < 0.01 and at(PRO, 3000) > 0.9 and at(PRO, 1000) > 0.6

SCENES = [
    ("title", [
        "这是大模型强化学习系列的最后一期。我们看几个前沿方向，过程奖励、人工智能反馈，以及多轮的智能体强化学习。",
    ]),
    ("credit", [
        "推理和智能体任务往往要很多步才能完成。如果只在最后告诉模型对还是错，它很难知道是哪一步出了问题，这叫信用分配问题。",
        "我们用一个玩具来看。任务有八步，每一步有四个选项，只有一个是对的。只有八步全对，才算完成任务。",
    ]),
    ("pvo", [
        f"随机乱选，八步全对的概率只有六万五千五百三十六分之一。只用结果奖励，在三千次尝试里，平均预计只能碰巧成功{dec_zh(EXP_HITS, 2)}次，几乎永远拿不到正的信号，所以完全学不会。",
        f"改用过程奖励，每一步做对都给一部分奖励。一千次尝试以后，完成整个任务的概率就达到{pct_zh(at(PRO, 1000))}，三千次以后是{pct_zh(at(PRO, 3000))}。",
        "这就是过程奖励模型的思路，给推理的每一步打分。研究发现，在数学推理上，逐步的监督比只看最终答案更有效。不过，给每一步标注好坏的成本也高得多。",
    ]),
    ("rlaif", [
        "第二个方向是用人工智能来提供反馈。人类标注昂贵而且慢，于是让一个模型按照一组明确写出的原则，去评判和修改其他回答，这就是宪法式方法和人工智能反馈的强化学习。",
        "它大大降低了收集偏好数据的成本，但评判模型自身的偏差，也会被带进训练里。",
    ]),
    ("agentic", [
        "第三个方向是智能体强化学习。模型不再只写一段回答，而是在多轮交互中调用工具、运行代码、浏览网页，最后完成一个任务。",
        "这里的奖励更稀疏，步骤更多，环境也更复杂，信用分配和奖励黑客的问题都被放大了。可验证的奖励、过程奖励和可靠的评估，在这里都变得更加重要。",
    ]),
    ("recap", [
        "回顾整个系列。我们从多臂老虎机和策略梯度出发，学习了奖励模型、基于人类反馈的强化学习和相对熵约束，又看到直接偏好优化如何跳过奖励模型。",
        "然后用可验证的奖励和组相对策略优化训练了加法模型，看到了思维链怎样让模型解决一步算不出来的问题，也看到了奖励黑客的风险。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 8, "前沿进展", "从 RLHF 到智能体强化学习", SERIES)


def sc_credit(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "信用分配：是哪一步出了错？")
    for i in range(K):
        x = 120 + i * 130
        d.rounded_rectangle([x, 200, x + 112, 260], 10, outline=CYAN, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 56, 230), f"第 {i + 1} 步", F_SMALL, CYAN)
        if i < K - 1:
            vk.arrow(d, (x + 113, 230), (x + 129, 230), MUTED, 2)
    vk.text_c(d, (640, 300), "只在最后给出：对 / 错", F_BODY, AMBER)
    if seg >= 1:
        vk.text_c(d, (640, 420), f"玩具：{K} 步，每步 {A} 个选项（1 个正确），全对才算完成", F_BODY, TEXT)


def sc_pvo(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "结果奖励 vs 过程奖励（200 次实验平均）")
    series = [("结果奖励（全对才给 1）", OUT * 100, RED)] + ([("过程奖励（每步都给一部分）", PRO * 100, GREEN)] if seg >= 1 else [])
    tv.line_chart(d, (140, 160, 800, 540), EPS, series, ylim=(0, 100), xlabel="尝试次数", ylabel="完成任务的概率 %",
                  legend_xy=(160, 230), xticks=[(e, str(e)) for e in (0, 1000, 2000, 3000)])
    d.text((860, 200), "随机全对的概率", font=F_SMALL, fill=MUTED)
    d.text((860, 230), "1 / 65536", font=vk.font(28, mono=True), fill=RED)
    d.text((860, 280), f"3000 次里预计成功 {EXP_HITS:.2f} 次", font=F_SMALL, fill=RED)
    if seg >= 1:
        d.text((860, 350), "过程奖励", font=F_SMALL, fill=MUTED)
        d.text((860, 380), f"1000 次：{at(PRO, 1000):.0%}", font=F_BODY, fill=GREEN)
        d.text((860, 420), f"3000 次：{at(PRO, 3000):.0%}", font=F_BODY, fill=GREEN)
    if seg >= 2:
        vk.text_c(d, (640, 640), "过程奖励模型（Lightman et al., 2023）：逐步监督优于只看结果；但标注成本更高", F_SMALL, AMBER)


def sc_rlaif(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "AI 反馈与宪法式方法（Bai et al., 2022）")
    for i, (t, col) in enumerate([("一组明确写出的原则", AMBER), ("评判模型", VIOLET), ("评判 / 修改回答 → 偏好数据", CYAN)]):
        x = 90 + i * 380
        d.rounded_rectangle([x, 200, x + 340, 280], 12, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 170, 240), t, F_SMALL, col)
        if i < 2:
            vk.arrow(d, (x + 345, 240), (x + 375, 240), MUTED, 2)
    if seg >= 1:
        vk.text_c(d, (640, 380), "成本大幅降低 · 但评判模型的偏差会被带进训练", F_BODY, TEXT)


def sc_agentic(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "智能体强化学习：多轮 + 工具")
    steps = ["思考", "调用工具", "读结果", "运行代码", "…", "完成任务"]
    for i, t in enumerate(steps):
        x = 130 + i * 175
        d.rounded_rectangle([x, 200, x + 150, 260], 10, outline=VIOLET if i < 5 else GREEN, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 75, 230), t, F_SMALL, TEXT)
    if seg >= 1:
        for i, t in enumerate(["奖励更稀疏、步骤更多、环境更复杂", "信用分配与奖励黑客被放大", "→ 可验证奖励 · 过程奖励 · 可靠评估更重要"]):
            vk.text_c(d, (640, 340 + i * 55), t, F_BODY, [TEXT, RED, GREEN][i])


EPS_ = ["老虎机与策略梯度", "奖励模型", "RLHF 与 KL", "DPO", "可验证奖励 GRPO", "思维链", "奖励黑客", "前沿进展"]


def sc_recap(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "系列回顾")
    k = 4 if seg == 0 else 8
    for i, t in enumerate(EPS_):
        x, y = 170 + (i % 4) * 245, 200 + (i // 4) * 170
        on = i < k
        d.rounded_rectangle([x, y, x + 215, y + 120], 14, outline=CYAN if on else (30, 41, 59), width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 107, y + 36), f"{i + 1:02d}", vk.font(26, mono=True), CYAN if on else MUTED)
        vk.text_c(d, (x + 107, y + 82), t, F_SMALL, TEXT if on else MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 590), "感谢观看 · RL for LLMs 系列完", F_BODY, VIOLET)


RENDER = {"title": sc_title, "credit": sc_credit, "pvo": sc_pvo, "rlaif": sc_rlaif, "agentic": sc_agentic, "recap": sc_recap}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rl-frontier")
