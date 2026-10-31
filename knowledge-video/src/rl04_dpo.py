"""RL for LLMs 04 — DPO：不用奖励模型的偏好优化.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] RL_CACHE=<dir> python rl04_dpo.py <out_dir>

Real computations (rlx.dpo): on the 200-response toy of episode 03, ~4,800 preference pairs drawn
from the reference policy and labelled by Bradley-Terry on the true reward; (a) the RLHF route:
fit a reward model on the pairs, then take the KL-optimal policy (beta = 0.3); (b) DPO trained
directly on the same pairs; true reward, KL to the reference, and the distance between the two
policies.
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

D = rlx.cached(CACHE, "dpo", rlx.dpo)
H = D["hist"]
print({k: v for k, v in D.items() if k != "hist"})
assert abs(D["dpo_true"] - D["rlhf_true"]) < 0.02 and abs(D["dpo_kl"] - D["rlhf_kl"]) < 0.05 and D["agree"] < 0.02
assert D["dpo_true"] > 1.8 and D["ref_true"] < 0.21 and abs(H[1]["true"] - D["dpo_true"]) < 0.01

SCENES = [
    ("title", [
        "上一期的方法要先训练奖励模型，再做强化学习，流程长、不稳定。这一期介绍直接偏好优化，它跳过了奖励模型，直接用比较数据训练策略。",
    ]),
    ("derive", [
        "关键在上一期的最优解。最优策略等于参考模型乘以奖励的指数，反过来写，奖励就等于贝塔乘以最优策略和参考模型的对数概率之比，再加一个常数。",
        "把这个关系代入布拉德利-特里模型，常数刚好在两个回答相减时消掉了。于是偏好的概率，只取决于策略本身的对数概率之比。",
        "训练目标变成了一个简单的分类损失。对每一对比较，提高好回答相对于参考模型的概率，降低差回答相对于参考模型的概率。不需要奖励模型，也不需要在训练中采样。",
    ]),
    ("experiment", [
        f"我们在上一期的两百个回答上验证。从参考模型里抽出约{big_zh(D['n_pairs'])}对回答，按真实好坏，用布拉德利-特里模型随机给出偏好。",
        f"第一条路线，先拟合奖励模型，再取贝塔等于零点三时的最优策略，真实质量是{dec_zh(D['rlhf_true'], 3)}。第二条路线，直接偏好优化，真实质量是{dec_zh(D['dpo_true'], 3)}。参考模型只有{dec_zh(D['ref_true'], 2)}。",
        f"两个策略和参考模型的相对熵也几乎相同，两个概率分布之间的差别，只有{dec_zh(D['agree'], 3)}。在这个设定下，直接偏好优化和完整的强化学习流程，得到的几乎是同一个策略。",
    ]),
    ("tradeoff", [
        "直接偏好优化简单稳定，成本低，因此被广泛使用。但它也有局限，它只从固定的比较数据中学习，不会像在线强化学习那样，在新生成的回答上继续探索和纠正。",
        "后来出现了许多变体，比如在线地反复收集新的比较数据。而在答案可以自动检查的任务上，人们又回到了在线强化学习，这就是下一期的内容。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，直接偏好优化利用最优策略和奖励之间的关系，把强化学习变成一个分类问题。",
        "第二，在同样的比较数据上，它和先训练奖励模型再做强化学习，得到的策略几乎相同。",
        "第三，它简单稳定，但只用离线数据，缺少在线探索。下一期，我们看可验证奖励和组相对策略优化。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 4, "DPO", "不用奖励模型的偏好优化", SERIES)


def sc_derive(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "DPO 的推导（Rafailov et al., 2023）")
    tv.fpaste(c, r"$r(y)=\beta\log\frac{\pi^*(y)}{\pi_{\mathrm{ref}}(y)}+\beta\log Z$", 640, 160, 30)
    if seg >= 1:
        tv.fpaste(c, r"$\Pr[y_w\succ y_l]=\sigma\left(\beta\log\frac{\pi(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\beta\log\frac{\pi(y_l)}{\pi_{\mathrm{ref}}(y_l)}\right)$", 640, 270, 28)
        vk.text_c(d, (640, 340), "常数 β log Z 在相减时消掉", F_SMALL, GREEN)
    if seg >= 2:
        tv.fpaste(c, r"$\mathcal{L}_{\mathrm{DPO}}=-\log\sigma\left(\beta\log\frac{\pi(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\beta\log\frac{\pi(y_l)}{\pi_{\mathrm{ref}}(y_l)}\right)$", 640, 440, 28, "#fbbf24")
        vk.text_c(d, (640, 520), "不需要奖励模型 · 不需要在训练中采样", F_BODY, GREEN)


def sc_experiment(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, f"同一份比较数据（{D['n_pairs']:,} 对），两条路线（β = 0.3）")
    if seg >= 1:
        tv.bar_chart(d, (180, 190, 760, 520), ["参考模型", "奖励模型 + RL", "DPO"], [D["ref_true"], D["rlhf_true"], D["dpo_true"]],
                     [MUTED, VIOLET, CYAN], fmt="{:.3f}", vmax=2.2)
        vk.text_c(d, (470, 590), "真实质量", F_SMALL, MUTED)
    if seg >= 2:
        rows = [("KL：奖励模型 + RL", f"{D['rlhf_kl']:.3f}", VIOLET), ("KL：DPO", f"{D['dpo_kl']:.3f}", CYAN),
                ("两个策略的差别（总变差）", f"{D['agree']:.3f}", GREEN)]
        for i, (a, b, col) in enumerate(rows):
            d.text((830, 220 + i * 90), a, font=F_SMALL, fill=MUTED)
            d.text((830, 252 + i * 90), b, font=vk.font(30, mono=True), fill=col)


def sc_tradeoff(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "优点与局限")
    for i, (a, b, col) in enumerate([("优点", "简单 · 稳定 · 成本低", GREEN), ("局限", "只用固定的离线数据，缺少在线探索与纠正", AMBER)]):
        y = 180 + i * 110
        d.rounded_rectangle([200, y, 1080, y + 80], 12, outline=col, width=2, fill=(12, 19, 34))
        d.text((230, y + 40), a, font=F_BODY, fill=col, anchor="lm")
        d.text((360, y + 40), b, font=F_BODY, fill=TEXT, anchor="lm")
    if seg >= 1:
        vk.text_c(d, (640, 470), "答案可自动检查的任务 → 回到在线强化学习（下一期）", F_BODY, CYAN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("DPO", "强化学习 → 分类损失"), ("等价", "与 RM + RL 得到几乎同一策略"),
                      ("局限", "离线数据，缺少在线探索")], seg, "下一期：可验证奖励与 GRPO")


RENDER = {"title": sc_title, "derive": sc_derive, "experiment": sc_experiment, "tradeoff": sc_tradeoff, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rl-dpo")
