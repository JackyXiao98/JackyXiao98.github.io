"""RL for LLMs 05 — 可验证奖励与 GRPO.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] RL_CACHE=<dir> TF_CACHE=<dir> python rl05_grpo.py <out_dir>

Real computations (rlx.grpo, JAX): the tfm.py addition Transformer after 120 supervised steps
(27.7% exact match) is trained only with reinforcement learning: for 64 training prompts per step
it samples 8 answers each, the reward is 1 if an automatic check confirms the sum, advantages are
normalised within each group of 8, plus a small KL penalty to the starting model; 400 steps; greedy
exact-match accuracy on 2,000 held-out problems every 25 steps.
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
TF_CACHE = os.environ.get("TF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "tfcache"))

G = rlx.cached(CACHE, "grpo", rlx.grpo, tf_cache=TF_CACHE)
H = G["hist"]
STEPS = [h["step"] for h in H]
ACC = [h["acc"] for h in H]
at = lambda s: ACC[STEPS.index(s)]
print(list(zip(STEPS, ACC)))
assert abs(ACC[0] - 0.277) < 0.002 and at(100) > 0.5 and at(400) > 0.97 and G["group"] == 8 and G["batch"] == 64

# a worked example of group-normalised advantages
EX_R = np.array([1, 0, 0, 1, 0, 0, 0, 1], float)
EX_A = (EX_R - EX_R.mean()) / EX_R.std()
assert abs(EX_A[0] - 1.291) < 1e-3 and abs(EX_A[1] + 0.775) < 1e-3

SCENES = [
    ("title", [
        "上一期的直接偏好优化，依赖人工标注的比较。这一期看另一类任务，答案对不对可以自动检查，以及专门为此设计的组相对策略优化。",
    ]),
    ("verifiable", [
        "数学题有标准答案，代码可以跑测试，这类任务的奖励不需要人来给，也不需要奖励模型，一个程序就能判断对错，这叫可验证的奖励。",
        "可验证奖励有两个好处。它不会被模型钻空子，至少比奖励模型难得多；而且可以无限量地自动产生，非常适合大规模的在线强化学习。近年的推理模型，大多是这样训练出来的。",
    ]),
    ("grpo", [
        "组相对策略优化的做法很简单。对同一道题，让模型回答一组，比如八次，每个回答根据对错得到奖励。",
        f"然后用组内的平均奖励作为基线，再除以组内的标准差。比如八次里答对三次，答对的回答优势是正的{dec_zh(EX_A[0], 2)}，答错的是负的{dec_zh(-EX_A[1], 2)}。",
        "最后按优势加权，提高好回答的概率，降低差回答的概率。和近端策略优化相比，它不需要单独训练一个价值网络来估计基线，节省了大量计算。",
    ]),
    ("experiment", [
        f"我们在变换器系列的加法模型上实际训练。起点是只做了一百二十步监督训练的模型，测试准确率只有{pct_zh(ACC[0], 1)}。之后不再给它任何标准答案，只告诉它答对还是答错。",
        f"每一步取六十四道题，每道题采样八次。训练一百步后，测试准确率升到{pct_zh(at(100))}；二百步后是{pct_zh(at(200))}；四百步后达到{pct_zh(at(400), 1)}。",
        "曲线会有波动，因为强化学习的信号比监督学习噪声更大。但只靠对错的反馈，模型就把自己没学会的加法学会了。",
    ]),
    ("conditions", [
        "这个方法有一个前提，模型一开始必须偶尔能答对。如果一组八个回答全错或全对，组内的优势全是零，这道题就提供不了任何学习信号。",
        "所以强化学习通常建立在一个已经不错的预训练模型之上，它更像是把模型已有的能力挖掘和强化出来。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，可验证的奖励由程序自动判断对错，难以被钻空子，可以大规模使用。",
        "第二，组相对策略优化用组内的平均和标准差作为基线，不需要价值网络。",
        f"第三，只靠对错的反馈，加法模型的准确率从{pct_zh(ACC[0])}升到{pct_zh(at(400))}。下一期，我们看推理模型怎样通过想得更久来提高准确率。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 5, "可验证奖励与 GRPO", "只靠对错的反馈", SERIES)


def sc_verifiable(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "可验证的奖励")
    for i, (a, b, col) in enumerate([("数学", "对照标准答案", CYAN), ("代码", "跑单元测试", VIOLET), ("加法", "检查和是否正确", GREEN)]):
        x = 160 + i * 340
        d.rounded_rectangle([x, 170, x + 300, 270], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 150, 205), a, F_BODY, col)
        vk.text_c(d, (x + 150, 245), b, F_SMALL, TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 360), "难以被钻空子 · 可无限量自动产生 → 适合大规模在线强化学习", F_BODY, AMBER)
        vk.text_c(d, (640, 420), "推理模型（如 DeepSeek-R1, 2025）的主要训练方式", F_SMALL, MUTED)


def sc_grpo(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "GRPO（Shao et al., 2024）")
    for i, r in enumerate(EX_R):
        x = 140 + i * 125
        col = GREEN if r else RED
        d.rounded_rectangle([x, 160, x + 105, 220], 10, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 52, 190), "对" if r else "错", F_BODY, col)
        if seg >= 1:
            vk.text_c(d, (x + 52, 245), f"{EX_A[i]:+.2f}", vk.font(22, mono=True), col)
    d.text((140, 120), "同一道题采样 8 次（示例）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        tv.fpaste(c, r"$A_i=\frac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)}$", 400, 350, 32)
    if seg >= 2:
        tv.fpaste(c, r"$\mathcal{L}=-\frac{1}{G}\sum_i A_i\log\pi_\theta(y_i)+\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})$", 640, 450, 28)
        vk.text_c(d, (640, 530), "不需要价值网络（PPO 需要）", F_BODY, GREEN)


def sc_experiment(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "实验：加法模型，只用对错作为奖励")
    n = 1 if seg == 0 else len(STEPS)
    tv.line_chart(d, (140, 160, 800, 560), STEPS, [("测试准确率（贪心）", [a * 100 for a in ACC], CYAN)], ylim=(0, 100), upto=n,
                  xlabel="强化学习步数", ylabel="%", legend_xy=(160, 470), xticks=[(s, str(s)) for s in (0, 100, 200, 300, 400)])
    d.text((860, 190), "每步：64 道题 × 8 次采样", font=F_SMALL, fill=MUTED)
    d.text((860, 220), "起点：监督训练 120 步的模型", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        for i, s in enumerate([0, 100, 200, 400]):
            d.text((860, 280 + i * 60), f"第 {s} 步", font=F_SMALL, fill=TEXT)
            d.text((1180, 280 + i * 60), f"{at(s):.1%}", font=vk.font(26, mono=True), fill=CYAN, anchor="ra")


def sc_conditions(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "前提：模型要偶尔能答对")
    for row, (lab, rs) in enumerate([("全错", [0] * 8), ("全对", [1] * 8)]):
        for i, r in enumerate(rs):
            x = 300 + i * 90
            col = GREEN if r else RED
            d.rounded_rectangle([x, 170 + row * 90, x + 75, 220 + row * 90], 8, outline=col, width=2, fill=(12, 19, 34))
        d.text((200, 185 + row * 90), lab, font=F_BODY, fill=TEXT)
        d.text((1040, 185 + row * 90), "优势全为 0", font=F_SMALL, fill=AMBER)
    if seg >= 1:
        vk.text_c(d, (640, 450), "强化学习 ≈ 把预训练模型已有的能力挖掘、强化出来", F_BODY, CYAN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("可验证奖励", "程序判断对错，难钻空子"), ("GRPO", "组内基线，无需价值网络"),
                      ("实验", f"{ACC[0]:.0%} → {at(400):.0%}，只靠对错反馈")], seg, "下一期：推理模型与测试时扩展")


RENDER = {"title": sc_title, "verifiable": sc_verifiable, "grpo": sc_grpo, "experiment": sc_experiment,
          "conditions": sc_conditions, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rl-grpo")
