"""Transformer 08 — 规模定律与现代改进.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python tf08_scaling.py <out_dir>

Real computations: five sizes (d = 8 … 96) of the tfm.py addition Transformer trained for the same
400 steps; held-out loss and accuracy against parameter count. Facts about real models (Chinchilla,
LLaMA, Mixtral) are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import tfm
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
CACHE = os.environ.get("TF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "tfcache"))

DS = [8, 16, 32, 64, 96]
RUNS = [tfm.cached_train(CACHE, d=d, steps=400, seed=1, log_every=400) for d in DS]
NP = np.array([tfm.n_params(p) for p, _, _ in RUNS])
LOSS = np.array([h[-1][2] for _, h, _ in RUNS])
ACC = np.array([h[-1][3] for _, h, _ in RUNS])
print(NP, np.round(LOSS, 4), ACC)
assert (np.diff(LOSS) < 0).all() and (np.diff(NP) > 0).all()
assert ACC[0] < 0.1 and ACC[-1] == 1.0 and LOSS[-1] < 0.003

SCENES = [
    ("title", [
        "这是变换器系列的最后一期。我们来看两个问题，模型变大会怎样，以及今天的大模型对原始的变换器做了哪些改进。",
    ]),
    ("sweep", [
        f"还是两位数加法，我们训练了五个不同大小的模型，参数量从{int_zh(NP[0] // 100 * 100)}多个到{int_zh(NP[-1] // 10000)}十多万个，训练步数都是四百步。",
        f"模型越大，测试损失越低，从{dec_zh(LOSS[0])}一路降到{dec_zh(LOSS[-1], 3)}。测试准确率从{pct_zh(ACC[0], 1)}上升到百分之百。",
    ]),
    ("laws", [
        "在真正的大模型上，研究者发现了更规律的现象。损失随参数量、数据量和计算量的增长，都按照幂律平稳下降，这就是规模定律。",
        "后来的研究进一步发现，在计算量固定时，参数量和训练数据应该同步增长，大约每个参数对应二十个训练词元，才是最划算的配置。",
    ]),
    ("modern", [
        "今天的大模型，主体结构还是只有解码器的变换器，但是许多细节有了改进。",
        "位置编码普遍换成了第三期讲过的旋转位置编码；层归一化换成了更简单的均方根归一化，只除以均方根，不减均值。",
        "前馈网络里的激活函数，换成了带门控的版本；注意力换成了分组查询注意力，多个查询头共享一组键和值，让第七期讲的键值缓存小了好几倍。",
        "还有混合专家模型，每层有很多个前馈网络，每个词元只激活其中的少数几个。这样总参数量很大，每个词元的计算量却不大。",
    ]),
    ("recap", [
        "回顾一下整个系列。我们从词元和嵌入出发，讲了注意力、位置编码和多头注意力，再把残差、层归一化和前馈网络组装成完整的模块。",
        "然后讲了因果掩码，亲手训练了一个会做加法的小模型，最后讲了键值缓存和规模定律。这些就是理解今天大语言模型的基础。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 8, "规模定律与现代改进", "模型变大会怎样 · 今天的大模型改了什么")


def sc_sweep(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "实验：同样训练 400 步，只改变模型大小")
    n = len(DS) if seg >= 1 else max(1, int(len(DS) * vk.ease(su)) + 1)
    tv.line_chart(d, (150, 160, 700, 560), NP, [("测试损失", LOSS, CYAN)], xlog=True, ylog=True, upto=n, ylim=(np.log10(1e-3), np.log10(4)),
                  xlabel="参数量（对数）", ylabel="测试交叉熵（对数）",
                  xticks=[(p, f"{p / 1000:.1f}K" if p < 10000 else f"{p // 1000}K") for p in NP])
    for i in range(min(n, len(DS))):
        y = 180 + i * 72
        d.text((780, y), f"d = {DS[i]:>2}", font=vk.font(22, mono=True), fill=MUTED)
        d.text((900, y), f"{NP[i]:>7,}", font=vk.font(22, mono=True), fill=TEXT)
        if seg >= 1:
            d.text((1040, y), f"{ACC[i]:.1%}", font=vk.font(22, mono=True), fill=GREEN if ACC[i] > 0.9 else RED)
    d.text((900, 140), "参数量", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        d.text((1040, 140), "准确率", font=F_SMALL, fill=MUTED)


def sc_laws(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "规模定律（Scaling Laws）")
    tv.fpaste(c, r"$L(N)\approx\left(\frac{N_c}{N}\right)^{\alpha_N},\qquad L(D)\approx\left(\frac{D_c}{D}\right)^{\alpha_D}$", 640, 170, 30)
    vk.text_c(d, (640, 240), "N：参数量 · D：训练词元数 · 对数坐标下近似直线（Kaplan et al., 2020）", F_SMALL, MUTED)
    xs = np.logspace(6, 11, 40)
    tv.line_chart(d, (200, 290, 640, 580), xs, [("示意", 3 * (xs / 1e6) ** -0.076, CYAN)], xlog=True, ylog=True, ylim=(np.log10(1.0), np.log10(4.5)),
                  xlabel="参数量", ylabel="损失（示意）", xticks=[(1e6, "1M"), (1e8, "100M"), (1e10, "10B")])
    if seg >= 1:
        d.rounded_rectangle([720, 320, 1140, 540], 14, outline=AMBER, width=2, fill=(12, 19, 34))
        vk.text_c(d, (930, 360), "Chinchilla（Hoffmann et al., 2022）", F_SMALL, AMBER)
        vk.text_c(d, (930, 420), "计算量固定时：", F_BODY, TEXT)
        vk.text_c(d, (930, 470), "训练词元 ≈ 20 × 参数量", F_BODY, AMBER)
        vk.text_c(d, (930, 515), "例：70B 参数 × 1.4T 词元", F_SMALL, MUTED)


ITEMS = [
    ("位置编码", "正弦 / 可学习", "RoPE 旋转位置编码", 1),
    ("归一化", "LayerNorm（Post-LN）", "RMSNorm，Pre-norm", 1),
    ("前馈激活", "ReLU / GELU", "SwiGLU（门控）", 2),
    ("注意力", "多头 MHA", "GQA：多个查询头共享 K、V", 2),
    ("前馈网络", "稠密", "MoE：每词元只激活少数专家", 3),
]


def sc_modern(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "现代大模型的改进（如 LLaMA、Mixtral）")
    d.text((190, 140), "部件", font=F_SMALL, fill=MUTED)
    d.text((400, 140), "原始 Transformer（2017）", font=F_SMALL, fill=MUTED)
    d.text((760, 140), "现代大模型", font=F_SMALL, fill=MUTED)
    for i, (k, old, new, s) in enumerate(ITEMS):
        y = 190 + i * 74
        on = seg >= s
        d.rounded_rectangle([170, y - 8, 1150, y + 52], 12, outline=(30, 41, 59), fill=(12, 19, 34))
        d.text((190, y + 22), k, font=F_BODY, fill=TEXT if on else MUTED, anchor="lm")
        d.text((400, y + 22), old, font=F_SMALL, fill=MUTED, anchor="lm")
        if on:
            d.text((760, y + 22), new, font=F_BODY, fill=CYAN, anchor="lm")
    if seg >= 3:
        vk.text_c(d, (640, 590), "Mixtral 8×7B：每层 8 个专家选 2 个，总参数约 47B，每词元激活约 13B", F_SMALL, AMBER)
    elif seg >= 2:
        vk.text_c(d, (640, 590), "LLaMA-2 70B：64 个查询头共享 8 组 K、V → KV 缓存为原来的 1/8", F_SMALL, AMBER)


EPS = ["词元与嵌入", "注意力机制", "多头与位置编码", "Transformer 模块", "掩码与三种架构", "训练一个小模型", "KV 缓存", "规模定律与改进"]


def sc_recap(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "系列回顾")
    k = 4 if seg == 0 else 8
    for i, t in enumerate(EPS):
        x, y = 170 + (i % 4) * 245, 200 + (i // 4) * 170
        on = i < k
        d.rounded_rectangle([x, y, x + 215, y + 120], 14, outline=CYAN if on else (30, 41, 59), width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 107, y + 36), f"{i + 1:02d}", vk.font(26, mono=True), CYAN if on else MUTED)
        vk.text_c(d, (x + 107, y + 82), t, F_SMALL, TEXT if on else MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 590), "感谢观看 · Transformer 系列完", F_BODY, VIOLET)


RENDER = {"title": sc_title, "sweep": sc_sweep, "laws": sc_laws, "modern": sc_modern, "recap": sc_recap}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "scaling")
