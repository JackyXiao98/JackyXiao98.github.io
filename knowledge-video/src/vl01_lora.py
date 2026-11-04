"""Video LoRA 01 — LoRA 基础：低秩适配为什么够用.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] VL_CACHE=<dir> python vl01_lora.py <out_dir>

Real computations (vlx.rank_sweep, JAX): the toy video DiT of this series (616,800 parameters, trained only on
videos with dark backgrounds) adapted to a new domain - light backgrounds - with LoRA of rank 1 / 2 / 4 / 8 / 16
on every attention and MLP matrix (Adam 1e-3, 600 steps, batch 32) or full fine-tuning (Adam 2e-4, 600 steps);
flow-matching loss on 64 held-out light-background videos (8 fixed noise draws); generated background
brightness; singular-value energy of the full fine-tuning update (24 matrices, top-k share, averaged).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import vlx
import vlviz
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Video LoRA Edit"
CACHE = os.environ.get("VL_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "vlcache"))

R = vlx.cached(CACHE, "rank", vlx.rank_sweep, cache_dir=CACHE)
RK = sorted(R["lora"])
LO = {r: R["lora"][r]["loss"] for r in RK}
PA = R["params"]
EN = R["energy_mean"]
print(R["base_loss"], R["base_loss_dark"], LO, R["full"]["loss"], PA, R["n_params"], EN[[0, 7]])
assert R["n_params"] == 616800 and PA[8] == 67584 and PA[1] == 8448 and abs(R["base_loss"] - 2.757) < 1e-3
assert abs(R["base_loss_dark"] - 0.0838) < 1e-3 and abs(LO[1] - 0.238) < 1e-3 and abs(LO[8] - 0.1847) < 1e-3
assert abs(LO[16] - 0.1808) < 1e-3 and abs(R["full"]["loss"] - 0.1672) < 1e-3 and all(np.diff([LO[r] for r in RK]) < 0)
assert all(R["lora"][r]["gen"]["bg_brightness"] > 0.8 for r in RK) and R["base_gen"]["bg_brightness"] < 0.2
assert abs(EN[0] - 0.271) < 1e-3 and abs(EN[7] - 0.757) < 1e-3
RATIO8 = PA[8] / R["n_params"]

SCENES = [
    ("title", [
        "欢迎来到视频 LoRA 编辑系列。这个系列讲怎样用很少的参数，让一个训练好的视频生成模型学会新东西，并用来编辑视频。第一期先讲 LoRA 本身。",
    ]),
    ("idea", [
        "微调一个大模型，本质上是给每个权重矩阵加上一个改变量。LoRA 的想法是，这个改变量不需要是任意的矩阵，可以写成两个又窄又长的小矩阵相乘。",
        "如果原来的矩阵是一千乘一千，有一百万个参数，秩取八时，两个小矩阵加起来只有一万六千个参数。训练时原来的权重冻结不动，只训练这两个小矩阵，其中一个初始化为零，所以一开始模型和原来完全一样。",
        "二零二一年提出 LoRA 的论文，在一千七百五十亿参数的语言模型上，把需要训练的参数减少了一万倍，显存需求减少到三分之一，效果却和全量微调相当。",
    ]),
    ("toy", [
        f"这个系列我们有一个自己的小视频生成模型，约{big_zh(R['n_params'])}个参数。它只见过深色背景的视频，生成的背景亮度大约是{dec_zh(R['base_gen']['bg_brightness'], 2)}。",
        f"现在要让它学会生成浅色背景的视频。原模型在浅色背景视频上的损失高达{dec_zh(R['base_loss'], 2)}，而在它熟悉的深色背景上只有{dec_zh(R['base_loss_dark'], 2)}。",
    ]),
    ("rank", [
        f"我们用不同的秩训练 LoRA，都训练六百步。秩为一时，只有{big_zh(PA[1])}个参数，损失就降到了{dec_zh(LO[1], 3)}，生成的背景已经变亮。",
        f"秩为八时，损失是{dec_zh(LO[8], 3)}，参数只占整个模型的{pct_zh(RATIO8)}。秩为十六时是{dec_zh(LO[16], 3)}，而全量微调所有参数是{dec_zh(R['full']['loss'], 3)}。秩越大越好，但收益越来越小。",
    ]),
    ("lowrank", [
        "为什么低秩就够用？我们看全量微调时，每个矩阵实际的改变量。把它做奇异值分解，看能量集中在多少个方向上。",
        f"平均来看，最大的一个方向就占了{pct_zh(EN[0])}的能量，前八个方向占了{pct_zh(EN[7])}，而这些矩阵最少也有九十六个方向。适应一个新任务需要的改变，确实集中在少数几个方向上，这正是 LoRA 能成功的原因。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，LoRA 把权重的改变量写成两个小矩阵的乘积，只训练它们，原模型冻结。",
        "第二，在我们的视频模型上，秩为八的 LoRA 用百分之十一的参数，就接近了全量微调的效果。",
        "第三，全量微调的改变量本身就集中在少数方向上，所以低秩够用。下一期，我们看在真实的视频扩散模型里，LoRA 该加在哪里。感谢大家的观看。",
    ]),
]
assert round(RATIO8 * 100) == 11


def sc_title(c, u, seg, su):
    tv.title_card(c, 1, "LoRA 基础", "低秩适配为什么够用", SERIES)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "LoRA（Hu et al., 2021）：W' = W + BA")
    d.rectangle([140, 190, 340, 390], outline=MUTED, width=2, fill=(30, 36, 48))
    vk.text_c(d, (240, 290), "W（冻结）", F_BODY, MUTED)
    vk.text_c(d, (395, 290), "+", F_BODY, TEXT)
    d.rectangle([450, 190, 480, 390], outline=AMBER, width=2, fill=(12, 19, 34))
    vk.text_c(d, (465, 410), "B", F_BODY, AMBER)
    d.rectangle([500, 275, 700, 305], outline=AMBER, width=2, fill=(12, 19, 34))
    vk.text_c(d, (600, 255), "A", F_BODY, AMBER)
    vk.text_c(d, (590, 340), "秩 r", F_SMALL, AMBER)
    if seg >= 1:
        tv.fpaste(c, r"$1000\times1000=10^6\quad\mathrm{vs}\quad r\,(1000+1000)=16000\ \ (r=8)$", 640, 480, 28)
        vk.text_c(d, (640, 545), "只训练 A、B · B 初始化为 0 → 一开始与原模型相同", F_SMALL, TEXT)
    if seg >= 2:
        vk.text_c(d, (640, 630), "GPT-3 175B：可训练参数减少 10,000 倍，显存减少到 1/3，效果相当", F_BODY, GREEN)


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, f"本系列的小视频模型（{R['n_params']:,} 参数）")
    d.text((140, 150), "只见过深色背景 · 8 帧 · 16×16", font=F_SMALL, fill=MUTED)
    for i in range(3):
        vlviz.film(c, R["samples_base"][i], 260, 200 + i * 70, cell=60, label=f"样本 {i + 1}")
    if seg >= 1:
        tv.bar_chart(d, (820, 260, 1200, 560), ["深色背景", "浅色背景"], [R["base_loss_dark"], R["base_loss"]], [GREEN, RED], fmt="{:.2f}", vmax=3.2)
        d.text((820, 210), "原模型的损失", font=F_SMALL, fill=MUTED)


def sc_rank(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "新任务：浅色背景 · 600 步")
    labs = [f"秩 {r}" for r in RK] + (["全量"] if seg >= 1 else [])
    vals = [LO[r] for r in RK] + ([R["full"]["loss"]] if seg >= 1 else [])
    n = 1 if seg == 0 else len(vals)
    tv.bar_chart(d, (120, 220, 120 + 120 * n, 540), labs[:n], vals[:n], [CYAN] * len(RK) + [VIOLET], fmt="{:.3f}", vmax=0.3)
    d.text((120, 170), "浅色背景视频上的损失（原模型 2.76）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        d.text((900, 220), "参数量", font=F_SMALL, fill=MUTED)
        for i, r in enumerate(RK):
            d.text((900, 260 + i * 38), f"秩 {r}", font=F_SMALL, fill=CYAN)
            d.text((1200, 260 + i * 38), f"{PA[r]:,}", font=vk.font(22, mono=True), fill=CYAN, anchor="ra")
        d.text((900, 260 + 5 * 38), "全量", font=F_SMALL, fill=VIOLET)
        d.text((1200, 260 + 5 * 38), f"{R['n_params']:,}", font=vk.font(22, mono=True), fill=VIOLET, anchor="ra")
    vlviz.film(c, R["samples_lora8"][0], 260, 600, cell=48, label="秩 8 生成")


def sc_lowrank(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "全量微调的改变量：能量集中在少数方向")
    if seg >= 1:
        k = np.arange(1, 17)
        tv.line_chart(d, (160, 170, 820, 560), k, [("前 k 个奇异方向的能量占比", EN[:16] * 100, AMBER)], ylim=(0, 100), xlabel="k",
                      ylabel="%", legend_xy=(400, 470), xticks=[(x, str(x)) for x in (1, 4, 8, 12, 16)])
        d.text((880, 230), f"前 1 个：{EN[0]:.0%}", font=F_BODY, fill=AMBER)
        d.text((880, 280), f"前 8 个：{EN[7]:.0%}", font=F_BODY, fill=AMBER)
        d.text((880, 340), "每个矩阵至少 96 个方向", font=F_SMALL, fill=MUTED)
    else:
        tv.fpaste(c, r"$\Delta W=U\Sigma V^{\top}$", 640, 330, 48)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("LoRA", "ΔW = BA，原模型冻结"), ("实验", f"秩 8：{RATIO8:.0%} 参数，接近全量微调"),
                      ("低秩", "改变量集中在少数方向")], seg, "下一期：LoRA 加在视频模型的哪里")


RENDER = {"title": sc_title, "idea": sc_idea, "toy": sc_toy, "rank": sc_rank, "lowrank": sc_lowrank, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vl-lora")
