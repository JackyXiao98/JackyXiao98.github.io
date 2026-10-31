"""RL for LLMs 03 — RLHF 与 PPO：为什么需要 KL 约束.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] RL_CACHE=<dir> python rl03_rlhf.py <out_dir>

Real computations (rlx.rlhf_curve): 200 discrete "responses" with a true reward, a reference policy,
and a reward model whose errors are large on the 10% of responses the reference rarely produces;
the KL-regularised optimum pi* ∝ pi_ref exp(r_hat / beta) for 60 values of beta (true reward, reward
model score, KL to the reference, probability mass on the rare responses); a natural-gradient
policy optimisation at beta = 0.3 checked against the closed form.
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

C = rlx.cached(CACHE, "rlhf", rlx.rlhf_curve)
CU = C["curve"]
KL = np.array([x["kl"] for x in CU])
TRUE = np.array([x["true"] for x in CU])
PROXY = np.array([x["proxy"] for x in CU])
RARE = np.array([x["rare"] for x in CU])
PEAK = int(TRUE.argmax())
END = CU[-1]
print(CU[PEAK], END, C["ref_true"], C["pg_check"])
assert C["pg_check"] < 1e-9 and 1.2 < TRUE[PEAK] < 1.3 and 1.5 < KL[PEAK] < 1.8 and END["true"] < 0.25 and END["rare"] > 0.99
assert END["proxy"] > 4.8 and C["ref_true"] < 0.21

SCENES = [
    ("title", [
        "有了奖励模型，就可以用强化学习来优化语言模型了，这就是基于人类反馈的强化学习。这一期看它的核心设计，为什么一定要约束策略不能离原来的模型太远。",
    ]),
    ("pipeline", [
        "完整的流程分三步。先用示范数据做监督微调，得到一个参考模型；再用人类的比较数据训练奖励模型；最后用强化学习优化策略，让奖励模型给出的分数更高。",
        "第三步常用的算法叫近端策略优化。它限制每一次更新的幅度，避免一步走得太远。同时，目标里还加上一项惩罚，让新策略和参考模型之间的相对熵不要太大。",
    ]),
    ("objective", [
        "这个带约束的目标有一个漂亮的最优解。最优策略等于参考模型的概率，乘以奖励除以贝塔的指数，再归一化。",
        "贝塔越大，策略越接近参考模型；贝塔越小，策略越贪婪地追逐奖励。我们用策略梯度实际优化了一次，结果和这个公式完全一致。",
    ]),
    ("toy", [
        "我们构造一个玩具。两百个可能的回答，每个都有真实的好坏，参考模型很少写出其中百分之十的回答，因此奖励模型在这些回答上的误差特别大。",
        f"逐渐减小贝塔，放松约束。一开始，奖励模型的分数和真实质量一起上升。当相对熵到达约{dec_zh(KL[PEAK], 1)}时，真实质量达到最高的{dec_zh(TRUE[PEAK], 2)}，而参考模型只有{dec_zh(C['ref_true'], 2)}。",
        f"继续放松约束，奖励模型的分数还在上升，最后达到{dec_zh(END['proxy'], 2)}，可真实质量却掉回{dec_zh(END['true'], 2)}。这时策略几乎全部的概率，都压在了那些很少见、被奖励模型高估的回答上。",
    ]),
    ("lesson", [
        "这就是奖励模型的过度优化。奖励模型只是人类偏好的近似，越用力优化它，就越容易钻它的空子。",
        "相对熵约束让策略留在奖励模型可靠的区域里。贝塔的大小，就是在追求奖励和保持可靠之间做权衡。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，基于人类反馈的强化学习分三步，监督微调、训练奖励模型、强化学习优化。",
        "第二，加了相对熵约束以后，最优策略是参考模型乘以奖励的指数。",
        "第三，约束太松会过度优化奖励模型，真实质量反而下降。下一期，我们看一种不需要奖励模型的方法，直接偏好优化。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 3, "RLHF 与 PPO", "为什么需要 KL 约束", SERIES)


def sc_pipeline(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "RLHF 的三步")
    steps = [("① 监督微调", "示范数据 → 参考模型 π_ref", CYAN), ("② 奖励模型", "比较数据 → 奖励模型 RM", VIOLET), ("③ 强化学习", "最大化 RM 分数（PPO）", GREEN)]
    for i, (a, b, col) in enumerate(steps):
        x = 120 + i * 360
        d.rounded_rectangle([x, 170, x + 320, 270], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 160, 205), a, F_BODY, col)
        vk.text_c(d, (x + 160, 245), b, F_SMALL, TEXT)
        if i < 2:
            vk.arrow(d, (x + 325, 220), (x + 355, 220), MUTED, 2)
    if seg >= 1:
        tv.fpaste(c, r"$\max_\pi\ \mathbb{E}_{y\sim\pi}[\hat r(y)]\ -\ \beta\,\mathrm{KL}(\pi\,\|\,\pi_{\mathrm{ref}})$", 640, 370, 34)
        vk.text_c(d, (640, 450), "PPO（Schulman et al., 2017）：裁剪更新幅度；InstructGPT 用 PPO + KL 惩罚", F_SMALL, MUTED)


def sc_objective(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "最优解")
    tv.fpaste(c, r"$\pi^*(y)=\frac{1}{Z}\,\pi_{\mathrm{ref}}(y)\,\exp\!\left(\frac{\hat r(y)}{\beta}\right)$", 640, 200, 40)
    if seg >= 1:
        vk.text_c(d, (640, 320), "β 大 → 接近参考模型；β 小 → 贪婪地追逐奖励", F_BODY, TEXT)
        vk.text_c(d, (640, 380), f"策略梯度实际优化（β = {C['pg_beta']}）与公式的最大差异：{C['pg_check']:.0e}", F_SMALL, GREEN)


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "玩具：200 个回答，奖励模型在罕见回答上误差大")
    if seg == 0:
        R = rlx.responses()
        order = np.argsort(-R["pi_ref"])
        for i, j in enumerate(order):
            x = 140 + i * 5
            h = R["pi_ref"][j] / R["pi_ref"].max() * 220
            d.rectangle([x, 480 - h, x + 4, 480], fill=RED if j in R["rare"] else CYAN)
        d.text((140, 500), "参考模型的概率（从高到低）；红色：罕见的 10%", font=F_SMALL, fill=MUTED)
        return
    n = len(CU) if seg >= 2 else PEAK + 1
    tv.line_chart(d, (130, 160, 820, 560), KL[:n] + 1e-3, [("奖励模型的分数", PROXY[:n], AMBER), ("真实质量", TRUE[:n], GREEN)],
                  xlog=True, ylim=(0, 5.2), xlabel="KL(π ‖ π_ref)", ylabel="平均奖励", legend_xy=(150, 215),
                  xticks=[(0.01, "0.01"), (0.1, "0.1"), (1, "1"), (8, "8")])
    d.text((880, 200), "真实质量最高", font=F_SMALL, fill=MUTED)
    d.text((880, 230), f"KL {KL[PEAK]:.2f} → {TRUE[PEAK]:.2f}", font=F_BODY, fill=GREEN)
    d.text((880, 280), f"参考模型：{C['ref_true']:.2f}", font=F_SMALL, fill=MUTED)
    if seg >= 2:
        d.text((880, 340), "约束最松时", font=F_SMALL, fill=MUTED)
        d.text((880, 370), f"奖励模型 {END['proxy']:.2f}", font=F_BODY, fill=AMBER)
        d.text((880, 410), f"真实质量 {END['true']:.2f}", font=F_BODY, fill=RED)
        d.text((880, 460), f"罕见回答的概率：{END['rare']:.0%}", font=F_SMALL, fill=RED)


def sc_lesson(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "奖励模型的过度优化")
    vk.text_c(d, (640, 200), "奖励模型 ≈ 人类偏好；越用力优化，越容易钻空子", F_BODY, AMBER)
    if seg >= 1:
        vk.text_c(d, (640, 320), "KL 约束：留在奖励模型可靠的区域", F_BODY, GREEN)
        vk.text_c(d, (640, 380), "β = 追求奖励 ⟷ 保持可靠 的权衡".replace("⟷", "与"), F_BODY, TEXT)
        vk.text_c(d, (640, 460), "参考：Gao, Schulman & Hilton, 2023", F_SMALL, MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("三步", "监督微调 → 奖励模型 → 强化学习"), ("最优解", "π* ∝ π_ref · exp(RM 分数 / β)"),
                      ("KL 约束", "太松会过度优化，真实质量下降")], seg, "下一期：DPO —— 不用奖励模型的偏好优化")


RENDER = {"title": sc_title, "pipeline": sc_pipeline, "objective": sc_objective, "toy": sc_toy, "lesson": sc_lesson,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rl-rlhf")
