"""RL for LLMs 01 — 强化学习基础：从多臂老虎机到策略梯度.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] RL_CACHE=<dir> python rl01_bandit.py <out_dir>

Real computations (rlx.bandit): the 10-armed Gaussian testbed (2,000 random problems, 1,000 pulls
each); epsilon-greedy with epsilon = 0, 0.01, 0.1 and a softmax policy-gradient learner with a
running-average baseline; average reward and share of optimal pulls over the last 100 steps.
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

B = rlx.cached(CACHE, "bandit", rlx.bandit)
KEYS = [("eps0.0", "ε = 0（纯贪心）", RED), ("eps0.01", "ε = 0.01", AMBER), ("eps0.1", "ε = 0.1", CYAN), ("pg", "策略梯度", GREEN)]
LAST = {k: (float(B[k][0][-100:].mean()), float(B[k][1][-100:].mean())) for k, _, _ in KEYS}
BEST = B["best_mean"]
print(LAST, BEST)
assert LAST["eps0.0"][1] < 0.4 and LAST["eps0.1"][1] > 0.75 and LAST["pg"][0] > LAST["eps0.1"][0] and LAST["pg"][0] < BEST

SCENES = [
    ("title", [
        "欢迎来到大模型强化学习系列。这一期从最基本的概念讲起，奖励、策略、探索，以及策略梯度。",
    ]),
    ("rl", [
        "强化学习里有一个智能体，它在某个情境下做出动作，环境给它一个奖励，它的目标是让得到的奖励尽量多。智能体做决定的规则，叫做策略。",
        "放到大语言模型上，情境就是用户的提示，动作就是模型写出的回答，奖励可以来自人的评价、奖励模型，或者自动的检查器。整个系列，就是在讲怎样用奖励来改进模型的策略。",
    ]),
    ("bandit", [
        "最简单的强化学习问题叫多臂老虎机。面前有十台老虎机，每台的平均奖励不同，但你事先不知道。每次只能拉一台，一共拉一千次。",
        "这里有一个根本的矛盾。总拉目前看起来最好的那台，叫利用；去试试别的，叫探索。只利用不探索，可能永远错过真正最好的那台。",
    ]),
    ("eps", [
        "一个简单的办法是艾普西隆贪心，大部分时候选当前估计最好的，以艾普西隆的小概率随机探索。我们在两千个随机生成的问题上做了实验。",
        f"完全不探索时，最后一百次里只有{pct_zh(LAST['eps0.0'][1])}的时候拉到了真正最好的那台，平均奖励是{dec_zh(LAST['eps0.0'][0])}。",
        f"以百分之十的概率探索，拉到最好那台的比例升到{pct_zh(LAST['eps0.1'][1])}，平均奖励是{dec_zh(LAST['eps0.1'][0])}。适度的探索，换来了长期更高的回报。",
    ]),
    ("pg", [
        "另一种思路是直接学习策略本身，也就是选每台老虎机的概率，这叫策略梯度。拿到的奖励高于平均水平，就提高刚才那个动作的概率，低于平均就降低它。",
        f"这个平均水平叫基线，它让更新的方向更稳定。策略梯度在最后一百次里，平均奖励达到{dec_zh(LAST['pg'][0], 2)}，拉到最好那台的比例是{pct_zh(LAST['pg'][1])}，而理论上的最好值是{dec_zh(BEST, 2)}。",
        "大模型的强化学习，用的几乎都是策略梯度的思路，只是动作从选一台老虎机，变成了写出一整段文字。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，强化学习用奖励来改进策略，大模型的回答就是动作。",
        "第二，探索和利用之间需要平衡，完全不探索会错过更好的选择。",
        "第三，策略梯度直接提高好动作的概率，基线让学习更稳定。下一期，我们看奖励从哪里来，怎样从人类的偏好中学习奖励模型。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 1, "强化学习基础", "从多臂老虎机到策略梯度", SERIES)


def sc_rl(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "智能体 · 动作 · 奖励")
    d.rounded_rectangle([160, 200, 460, 300], 14, outline=CYAN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (310, 250), "智能体（策略 π）", F_BODY, CYAN)
    d.rounded_rectangle([820, 200, 1120, 300], 14, outline=VIOLET, width=2, fill=(12, 19, 34))
    vk.text_c(d, (970, 250), "环境", F_BODY, VIOLET)
    vk.arrow(d, (465, 230), (815, 230), MUTED, 3)
    vk.arrow(d, (815, 275), (465, 275), MUTED, 3)
    vk.text_c(d, (640, 210), "动作 a", F_SMALL, TEXT)
    vk.text_c(d, (640, 295), "奖励 r", F_SMALL, AMBER)
    if seg >= 1:
        rows = [("情境", "用户的提示"), ("动作", "模型写出的回答"), ("奖励", "人的评价 / 奖励模型 / 自动检查器")]
        for i, (a, b) in enumerate(rows):
            d.text((320, 380 + i * 50), a, font=F_BODY, fill=AMBER)
            d.text((460, 380 + i * 50), b, font=F_BODY, fill=TEXT)


def sc_bandit(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "十臂老虎机")
    rng = np.random.default_rng(1)
    q = rng.standard_normal(10)
    for i in range(10):
        x = 140 + i * 100
        d.rounded_rectangle([x, 180, x + 80, 300], 10, outline=CYAN, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 40, 220), f"{i + 1}", vk.font(28, mono=True), CYAN)
        vk.text_c(d, (x + 40, 270), "?", vk.font(28), MUTED)
    d.text((140, 320), "每台的平均奖励不同（事先未知）；每次拉一台，共 1000 次", font=F_SMALL, fill=TEXT)
    if seg >= 1:
        for i, (t, sub, col) in enumerate([("利用", "拉目前看起来最好的", GREEN), ("探索", "试试别的", AMBER)]):
            x = 240 + i * 440
            d.rounded_rectangle([x, 420, x + 360, 520], 14, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 180, 452), t, F_BODY, col)
            vk.text_c(d, (x + 180, 492), sub, F_SMALL, TEXT)


def sc_eps(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "ε-贪心（2000 个随机问题的平均）")
    steps = np.arange(1, 1001)
    k = 0 if seg == 0 else 3
    series = [(lab, B[key][1][::10] * 100, col) for key, lab, col in KEYS[:3]][: (1 if seg == 1 else 3) if seg else 0] or \
             [("ε = 0（纯贪心）", B["eps0.0"][1][::10] * 100, RED)]
    tv.line_chart(d, (140, 160, 780, 560), steps[::10], series, ylim=(0, 100), xlabel="拉的次数", ylabel="拉到最优的比例 %",
                  legend_xy=(560, 470), xticks=[(s, str(s)) for s in (1, 250, 500, 750, 1000)])
    d.text((840, 190), "最后 100 次", font=F_SMALL, fill=MUTED)
    for i, (key, lab, col) in enumerate(KEYS[:3]):
        if seg >= 1 and (seg >= 2 or i == 0):
            y = 240 + i * 90
            d.text((840, y), lab, font=F_SMALL, fill=col)
            d.text((840, y + 30), f"最优 {LAST[key][1]:.0%} · 奖励 {LAST[key][0]:.2f}", font=F_BODY, fill=col)


def sc_pg(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "策略梯度（带基线）")
    tv.fpaste(c, r"$\theta\leftarrow\theta+\alpha\,(r-b)\,\nabla_\theta\log\pi_\theta(a)$", 640, 150, 32)
    if seg >= 1:
        vals = [LAST["eps0.0"][0], LAST["eps0.1"][0], LAST["pg"][0], BEST]
        tv.bar_chart(d, (240, 240, 1040, 540), ["ε = 0", "ε = 0.1", "策略梯度", "理论最好"], vals, [RED, CYAN, GREEN, MUTED],
                     fmt="{:.2f}", vmax=BEST * 1.2)
        vk.text_c(d, (640, 590), "最后 100 次的平均奖励", F_SMALL, MUTED)
    if seg >= 2:
        vk.text_c(d, (640, 640), "大模型：动作 = 一整段文字；log π = 每个词元的对数概率之和", F_SMALL, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("强化学习", "用奖励改进策略；回答 = 动作"), ("探索", "完全不探索会错过更好的选择"),
                      ("策略梯度", "提高好动作的概率；基线更稳")], seg, "下一期：奖励模型 —— 从人类偏好中学习")


RENDER = {"title": sc_title, "rl": sc_rl, "bandit": sc_bandit, "eps": sc_eps, "pg": sc_pg, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rl-bandit")
