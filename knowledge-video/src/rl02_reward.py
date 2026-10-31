"""RL for LLMs 02 — 奖励模型：从人类偏好中学习.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] RL_CACHE=<dir> python rl02_reward.py <out_dir>

Real computations (rlx.reward_model): 2,000 simulated responses with 8 features and a hidden true
score; pairwise preferences drawn from a Bradley-Terry model on the true score; a linear
Bradley-Terry reward model (logistic regression) fitted on 30 ... 10,000 comparisons; accuracy on
20,000 held-out comparisons (vs. the best achievable with the true score) and correlation with the
true score on held-out responses.
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

R = rlx.cached(CACHE, "rm", rlx.reward_model)
ORACLE = R["oracle"]
RES = {r["n"]: r for r in R["res"]}
NS = [r["n"] for r in R["res"]]
print(ORACLE, RES)
assert 0.74 < ORACLE < 0.79 and RES[30]["acc"] < 0.72 and abs(RES[10000]["acc"] - ORACLE) < 0.005
assert RES[30]["corr"] < 0.85 and RES[1000]["corr"] > 0.99

SCENES = [
    ("title", [
        "上一期我们说，强化学习需要奖励。可是一段回答写得好不好，往往没有标准答案。这一期看怎样从人类的偏好中学出一个奖励模型。",
    ]),
    ("compare", [
        "让人直接给一段回答打分，比如十分制，不同的人标准差别很大。但让人比较两段回答，说出哪一段更好，就容易而且一致得多。",
        "所以常见的做法是收集大量的成对比较。第一代基于人类反馈训练的对话模型，正是让标注员对同一个提示的多个回答排序。",
    ]),
    ("bt", [
        "怎样把“甲比乙好”变成分数？布拉德利-特里模型假设，每个回答有一个分数，甲被选中的概率，等于两者分数之差经过逻辑函数。",
        "分数差越大，选择越确定；分数相同，就是五五开。训练奖励模型，就是找到一组分数，让观察到的比较结果最有可能出现。",
    ]),
    ("experiment", [
        "我们来模拟一下。两千个回答，每个回答有八个特征和一个隐藏的真实分数，人的选择按照布拉德利-特里模型随机产生，所以人也会犯错。",
        f"即使知道真实分数，也只能预测对{pct_zh(ORACLE, 1)}的比较，这是人类标注噪声决定的上限。",
        f"只用三十次比较训练，奖励模型在新的比较上的准确率是{pct_zh(RES[30]['acc'], 1)}；用一千次，达到{pct_zh(RES[1000]['acc'], 1)}；用一万次，是{pct_zh(RES[10000]['acc'], 1)}，几乎碰到上限。",
        f"更重要的是排序。用一千次比较，奖励模型给出的分数和真实分数的相关系数已经达到{dec_zh(RES[1000]['corr'], 3)}。",
    ]),
    ("limits", [
        "不过，奖励模型只在它见过的那类回答上可靠。真实的奖励模型是一个大网络，比较数据只覆盖了一部分回答。",
        "当策略被优化得越来越偏离这些数据，奖励模型就可能给出离谱的高分。下一期我们会看到，这正是需要约束策略的原因。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，让人比较两个回答，比直接打分更容易、更一致。",
        "第二，布拉德利-特里模型把成对的偏好变成每个回答的分数。",
        "第三，奖励模型的准确率受标注噪声限制，而且只在数据覆盖的范围内可靠。下一期，我们看怎样用奖励模型训练策略。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 2, "奖励模型", "从人类偏好中学习", SERIES)


def sc_compare(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "打分 vs 比较")
    d.rounded_rectangle([140, 170, 560, 330], 14, outline=RED, width=2, fill=(12, 19, 34))
    vk.text_c(d, (350, 210), "直接打分：7 分？8 分？", F_BODY, RED)
    vk.text_c(d, (350, 270), "不同的人标准差别很大", F_SMALL, TEXT)
    d.rounded_rectangle([720, 170, 1140, 330], 14, outline=GREEN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (930, 210), "比较：A 比 B 好", F_BODY, GREEN)
    vk.text_c(d, (930, 270), "更容易、更一致", F_SMALL, TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 430), "InstructGPT（Ouyang et al., 2022）：标注员对同一提示的多个回答排序", F_SMALL, MUTED)


def sc_bt(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "Bradley–Terry 模型")
    tv.fpaste(c, r"$\Pr[A\succ B]=\sigma\big(s(A)-s(B)\big)=\frac{1}{1+e^{-(s(A)-s(B))}}$".replace(r"\big", ""), 640, 170, 32)
    xs = np.linspace(-5, 5, 200)
    tv.line_chart(d, (260, 240, 1020, 500), xs, [("σ(Δs)", 1 / (1 + np.exp(-xs)), CYAN)], ylim=(0, 1), xlabel="分数差 s(A) − s(B)".replace("−", "-"),
                  ylabel="选 A 的概率", xticks=[(x, str(x)) for x in (-4, -2, 0, 2, 4)])
    if seg >= 1:
        tv.fpaste(c, r"$\mathcal{L}=-\sum\log\sigma\big(s(y_w)-s(y_l)\big)$".replace(r"\big", ""), 640, 610, 28, "#fbbf24")


def sc_experiment(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "模拟：2000 个回答，8 个特征，带噪声的人类选择")
    if seg >= 1:
        tv.line_chart(d, (140, 170, 740, 540), NS, [("奖励模型的准确率", [RES[n]["acc"] * 100 for n in NS], CYAN),
                                                    ("上限（知道真实分数）", [ORACLE * 100] * len(NS), MUTED)],
                      xlog=True, ylim=(60, 80), xlabel="训练用的比较次数", ylabel="新比较上的准确率 %", legend_xy=(380, 450),
                      xticks=[(n, str(n)) for n in NS], upto=len(NS) if seg >= 2 else 1)
    if seg >= 3:
        d.text((800, 190), "与真实分数的相关系数", font=F_SMALL, fill=MUTED)
        for i, n in enumerate(NS):
            y = 240 + i * 52
            d.text((800, y), f"{n:>6,} 次", font=F_SMALL, fill=TEXT)
            w = RES[n]["corr"] * 250
            d.rectangle([920, y + 4, 920 + w, y + 26], fill=GREEN)
            d.text((930 + w, y + 15), f"{RES[n]['corr']:.3f}", font=vk.font(18, mono=True), fill=GREEN, anchor="lm")


def sc_limits(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "奖励模型只在数据覆盖的范围内可靠")
    rng = np.random.default_rng(0)
    for x, y in rng.normal([400, 380], [80, 60], (150, 2)):
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=CYAN)
    vk.text_c(d, (400, 580), "比较数据覆盖的回答", F_SMALL, CYAN)
    if seg >= 1:
        vk.arrow(d, (520, 350), (850, 250), AMBER, 3)
        vk.text_c(d, (960, 230), "被优化的策略", F_BODY, AMBER)
        vk.text_c(d, (960, 280), "奖励模型可能给出离谱的高分", F_SMALL, RED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("比较", "比直接打分更容易、更一致"), ("Bradley–Terry", "成对偏好 → 每个回答的分数"),
                      ("局限", "受标注噪声限制；超出数据范围不可靠")], seg, "下一期：RLHF 与 PPO —— 为什么需要 KL 约束")


RENDER = {"title": sc_title, "compare": sc_compare, "bt": sc_bt, "experiment": sc_experiment, "limits": sc_limits,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rl-reward")
