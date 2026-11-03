"""Interpretability 06 — 稀疏自编码器：把叠加拆开.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MI_CACHE=<dir> python mi06_sae.py <out_dir>

Real computations (mix.sae, JAX): synthetic activations in 8 dimensions made of 20 random unit feature
directions, each active with probability 5% (magnitude U[0, 1]); sparse autoencoders with 40 latents
(ReLU encoder, unit-norm decoder columns) trained 20,000 steps with L1 coefficient 0-1; on 20,000 fresh
samples: relative reconstruction error, average number of active latents (L0), and how many true
features have a decoder column with |cosine| > 0.9. Also the best |cosine| of each feature with the 8
neuron (standard-basis) directions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import mix
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Interpretability"
CACHE = os.environ.get("MI_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "micache"))

R = mix.cached(CACHE, "sae", mix.sae)
LS = sorted(R["res"])
RES = R["res"]
NC = float(R["neuron_cos"].mean())
TL0 = RES[0.0]["true_l0"]
print(NC, TL0, {l: (round(RES[l]["rel_mse"], 4), round(RES[l]["l0"], 2), RES[l]["recovered"]) for l in LS})
assert abs(NC - 0.646) < 1e-3 and abs(TL0 - 1.0) < 0.01 and R["neuron_cos"].max() < 0.9
assert RES[0.0]["recovered"] == 0 and RES[0.0]["l0"] > 13 and RES[0.1]["recovered"] == 13
assert RES[0.3]["recovered"] == 18 and abs(RES[0.3]["rel_mse"] - 0.113) < 0.001 and abs(RES[0.3]["l0"] - 1.95) < 0.01
assert RES[0.5]["recovered"] == 20 and abs(RES[0.5]["rel_mse"] - 0.219) < 0.001 and abs(RES[0.5]["l0"] - 1.30) < 0.01
assert RES[1.0]["recovered"] == 20 and abs(RES[1.0]["rel_mse"] - 0.508) < 0.001 and RES[1.0]["l0"] < 0.6

SCENES = [
    ("title", [
        "上一期我们看到，模型会把很多特征叠加在少数几个维度里，单个神经元因此是多义的。这一期看怎样把叠加拆开，工具是稀疏自编码器。",
    ]),
    ("setup", [
        "我们先造一个已知答案的例子。激活只有八维，里面叠加了二十个随机方向的特征，每个特征只有百分之五的时间出现，平均每次只有一个特征在场。",
        f"如果直接看八个神经元，每个特征和最接近的那个神经元方向，余弦相似度平均只有{dec_zh(NC, 2)}，没有一个神经元单独对应一个特征。",
    ]),
    ("sae", [
        "稀疏自编码器把八维的激活，展开成更多的隐变量，这里是四十个，再从这些隐变量重建原来的激活。",
        "关键是加一个稀疏惩罚，用隐变量绝对值的和，也就是一范数，让每次只有很少几个隐变量是激活的。训练好以后，解码器的每一列就是一个候选的特征方向。",
    ]),
    ("sweep", [
        f"我们调节稀疏惩罚的强度。不加惩罚时，重建几乎完美，但平均有{int_zh(round(RES[0.0]['l0']))}个隐变量同时激活，一个真实特征也没有找到。",
        f"惩罚系数是零点三时，平均约两个隐变量激活，找回了二十个特征中的{int_zh(RES[0.3]['recovered'])}个，重建误差是{pct_zh(RES[0.3]['rel_mse'])}。",
        f"系数是零点五时，二十个特征全部找回，但重建误差升到{pct_zh(RES[0.5]['rel_mse'])}。系数到一时，误差达到{pct_zh(RES[1.0]['rel_mse'])}，因为一范数把激活的大小也一起压小了。",
    ]),
    ("tradeoff", [
        "这就是稀疏自编码器的核心权衡，越稀疏，特征越干净，重建越差。实践中还会遇到一直不激活的死隐变量，以及激活被压小的问题，后来的改进如只保留前几个最大值的方法，就是为了缓解这些问题。",
    ]),
    ("scale", [
        "稀疏自编码器已经被用到了真实的大模型上。二零二四年，研究者在一个商用大模型的中间层上训练了包含几千万个隐变量的稀疏自编码器，找到了对应具体城市、代码错误、欺骗等概念的特征。",
        "同年也有团队公开了在开源模型每一层上训练好的稀疏自编码器，供研究者使用。不过，找到的特征是否就是模型真正使用的特征，仍然需要像修补那样的因果检验。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，稀疏自编码器把激活展开成更多、更稀疏的隐变量，用来拆开叠加。",
        "第二，在已知答案的玩具里，合适的稀疏惩罚能找回全部二十个特征，但要付出重建误差。",
        "第三，它已经扩展到大模型，但特征是否被模型真正使用，仍需因果验证。下一期，我们完整地理解一个算法，模加法与顿悟。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 6, "稀疏自编码器", "把叠加拆开", SERIES)


def sc_setup(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "已知答案的例子：20 个特征叠加在 8 维里")
    d.text((140, 160), "每个特征出现概率 5% · 平均每次 1 个", font=F_SMALL, fill=MUTED)
    F = R["F"]
    tv.heat(d, (140, 210, 620, 620), np.abs(F), 0, 1)
    d.text((140, 630), "行：20 个特征 · 列：8 个神经元（|分量|）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        tv.bar_chart(d, (700, 230, 1200, 560), [""] * 20, list(R["neuron_cos"] * 100), [AMBER] * 20, fmt="", vmax=110)
        d.text((700, 575), "20 个特征", font=F_SMALL, fill=MUTED)
        y = 560 - 90 / 110 * 330
        d.line([(700, y), (1200, y)], fill=GREEN, width=1)
        d.text((1200, y - 18), "0.9", font=F_SMALL, fill=GREEN, anchor="rm")
        vk.text_c(d, (950, 630), f"每个特征与最接近的神经元：|cos| 平均 {NC:.2f}", F_SMALL, AMBER)


def sc_sae(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "稀疏自编码器（SAE）")
    for i, (t, h, col) in enumerate([("激活 8 维", 60, CYAN), ("隐变量 40 个（稀疏）", 180, AMBER), ("重建 8 维", 60, CYAN)]):
        x = 230 + i * 410
        d.rounded_rectangle([x - 60, 330 - h, x + 60, 330 + h], 12, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x, 330 + h + 30), t, F_SMALL, col)
    vk.arrow(d, (295, 330), (575, 330), MUTED, 3)
    vk.arrow(d, (705, 330), (985, 330), MUTED, 3)
    vk.text_c(d, (435, 300), "编码 + ReLU", F_SMALL, TEXT)
    vk.text_c(d, (845, 300), "解码器的列 = 特征方向", F_SMALL, TEXT)
    if seg >= 1:
        tv.fpaste(c, r"$\mathcal{L}=\|x-\hat x\|^2+\lambda\,\|z\|_1$", 640, 640, 32)


def sc_sweep(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "调节稀疏惩罚 λ")
    show = [LS[:1], LS[:5], LS][min(seg, 2)]
    labs = [f"λ={l:g}" for l in show]
    tv.bar_chart(d, (110, 220, 110 + 105 * len(show), 520), labs, [RES[l]["recovered"] for l in show], [GREEN] * len(show), fmt="{:.0f}", vmax=24)
    d.text((110, 160), "找回的特征（共 20 个）", font=F_SMALL, fill=GREEN)
    x0 = 900
    d.text((x0, 160), "λ", font=F_SMALL, fill=MUTED)
    d.text((x0 + 90, 160), "L0", font=F_SMALL, fill=MUTED)
    d.text((x0 + 190, 160), "重建误差", font=F_SMALL, fill=MUTED)
    for i, l in enumerate(show):
        y = 205 + i * 50
        f = vk.font(22, mono=True)
        d.text((x0, y), f"{l:g}", font=f, fill=TEXT)
        d.text((x0 + 90, y), f"{RES[l]['l0']:.1f}", font=f, fill=CYAN)
        d.text((x0 + 190, y), f"{RES[l]['rel_mse']:.1%}", font=f, fill=RED if RES[l]["rel_mse"] > 0.2 else AMBER)
    d.text((x0, 580), f"真实 L0 ≈ {TL0:.1f}", font=F_SMALL, fill=MUTED)


def sc_tradeoff(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "核心权衡：稀疏 vs 重建")
    xs = np.array([RES[l]["l0"] for l in LS])
    ys = np.array([RES[l]["rel_mse"] * 100 for l in LS])
    tv.line_chart(d, (160, 170, 820, 560), xs[::-1], [("重建误差 %", ys[::-1], RED)], xlog=True, ylim=(0, 55), xlabel="L0（平均激活的隐变量数，对数）",
                  legend_xy=(560, 190), xticks=[(0.5, "0.5"), (1, "1"), (2, "2"), (5, "5"), (13, "13")])
    for l in LS:
        if l == 0.01:
            continue
        x = 160 + (np.log10(RES[l]["l0"]) - np.log10(xs.min())) / (np.log10(xs.max()) - np.log10(xs.min())) * 660
        y = 560 - RES[l]["rel_mse"] * 100 / 55 * 390
        d.text((x + 8, y - 28), f"{RES[l]['recovered']}", font=vk.font(20, mono=True), fill=GREEN)
    d.text((870, 200), "绿色数字 = 找回的特征数", font=F_SMALL, fill=GREEN)
    for i, t in enumerate(["死隐变量（从不激活）", "激活被压小（L1 收缩）", "改进：TopK、JumpReLU 等"]):
        d.text((870, 290 + i * 50), t, font=F_SMALL, fill=[MUTED, MUTED, CYAN][i])


def sc_scale(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "扩展到大模型")
    items = [("Scaling Monosemanticity（Anthropic, 2024）", "Claude 3 Sonnet 中间层 · 最多约 3,400 万个隐变量", CYAN),
             ("Gemma Scope（Google DeepMind, 2024）", "Gemma 2 每一层的开源 SAE", VIOLET)]
    for i, (a, b, col) in enumerate(items):
        if i == 0 or seg >= 1:
            y = 170 + i * 130
            d.rounded_rectangle([100, y, 1180, y + 100], 14, outline=col, width=2, fill=(12, 19, 34))
            d.text((130, y + 32), a, font=F_BODY, fill=col, anchor="lm")
            d.text((130, y + 72), b, font=F_SMALL, fill=TEXT, anchor="lm")
    vk.text_c(d, (640, 470), "特征示例：具体城市 · 代码错误 · 欺骗 …", F_SMALL, AMBER)
    if seg >= 1:
        vk.text_c(d, (640, 560), "找到的特征 = 模型真正使用的特征？→ 仍需因果检验", F_BODY, RED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 6, [("SAE", "展开成更多、更稀疏的隐变量"), ("玩具", "λ 合适时找回全部 20 个，代价是误差"),
                      ("大模型", "已扩展，但仍需因果验证")], seg, "下一期：模加法与顿悟")


RENDER = {"title": sc_title, "setup": sc_setup, "sae": sc_sae, "sweep": sc_sweep, "tradeoff": sc_tradeoff, "scale": sc_scale, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "mi-sae")
