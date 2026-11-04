"""MoE 08 — 并行与推理：专家放在哪里.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python moe08_parallel.py <out_dir>

Computations: total vs activated parameters of public MoE models (figures from their reports / model cards, see
notes); bf16 weight memory (2 bytes per parameter) and 80 GB GPUs needed just to hold the weights; the expected
number of distinct experts touched by a batch of B tokens when each token picks k of E experts uniformly at random,
E * (1 - (1 - k / E)^B).
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Mixture of Experts"

MODELS = [("Mixtral 8x7B", 46.7, 12.9, "8 选 2"), ("Qwen3-235B-A22B", 235, 22, "128 选 8"),
          ("DeepSeek-V3", 671, 37, "256 选 8 + 1 共享"), ("Kimi K2", 1040, 32, "384 选 8 + 1 共享")]
RATIO = {m: a / t for m, t, a, _ in MODELS}
DSV3_GB = 671e9 * 2 / 1e9
GPUS = int(np.ceil(DSV3_GB / 80))
touched = lambda B, E=256, k=8: E * (1 - (1 - k / E) ** B)
TB = {B: touched(B) for B in (1, 8, 32, 128, 512)}
print(RATIO, DSV3_GB, GPUS, TB)
assert abs(RATIO["DeepSeek-V3"] - 0.0551) < 1e-3 and abs(RATIO["Kimi K2"] - 0.0308) < 1e-3 and abs(RATIO["Mixtral 8x7B"] - 0.276) < 1e-3
assert round(DSV3_GB) == 1342 and GPUS == 17 and round(TB[1]) == 8 and round(TB[32]) == 163 and TB[512] > 255

SCENES = [
    ("title", [
        "前面几期我们一直在小模型里看混合专家。这一期回到真实的大模型，看它们有多少参数、每次用多少，以及这么多专家在硬件上怎么放。",
    ]),
    ("models", [
        "先看几个公开的混合专家模型。法国公司的八乘七十亿模型，总参数约四百六十七亿，每个词元用两个专家，激活约一百二十九亿，大约是总数的四分之一多。",
        "后来的模型稀疏得多。深度求索第三代有六千七百一十亿参数，每个词元只激活三百七十亿，约百分之五点五。月之暗面的万亿参数模型，每个词元只激活约三百二十亿，约百分之三。",
    ]),
    ("memory", [
        f"但要注意，计算只用激活的参数，存储却要全部参数。以六千七百一十亿参数为例，用十六位浮点数存储需要约{big_zh(round(DSV3_GB))}吉字节，单是放下权重，就要至少{int_zh(GPUS)}张八十吉字节的显卡。",
        "所以混合专家模型省的是计算，不是显存。它适合有很多显卡、大规模服务的场景，而不适合放在一张卡上。",
    ]),
    ("ep", [
        "这么多专家怎么放？常用的方法是专家并行，把不同的专家放在不同的显卡上。每一层，词元先被路由器分到各自的专家，通过全对全的通信，发到专家所在的显卡上，算完以后再发回来。",
        "这种通信每一层都要做两次，往往成为瓶颈。所以实际系统会限制每个词元最多去几台机器，把通信和计算重叠起来，并且尽量让负载均衡，避免某张卡上的专家成为所有人等待的那一个。",
    ]),
    ("batch", [
        "推理时还有一个有趣的现象。只生成一个词元时，只需要读八个专家的权重，非常省。但服务时会同时处理很多请求。",
        f"假设二百五十六个专家里每个词元随机选八个，一批只有一个词元时，用到八个专家；一批三十二个词元，平均就会用到约{int_zh(round(TB[32]))}个；一批五百一十二个词元，几乎所有专家都会被用到。批量大了以后，每一步都要读几乎全部的权重，混合专家省下的读取量就消失了，只剩下计算上的节省。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，今天的大型混合专家模型，每个词元只激活百分之三到百分之二十几的参数。",
        "第二，它省的是计算，不是显存，需要专家并行和全对全通信。",
        "第三，批量越大，被用到的专家越多，读取权重上的优势越小。最后一期，我们看混合专家的权衡与前沿。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 8, "并行与推理", "专家放在哪里", SERIES)


def sc_models(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "公开 MoE 模型：总参数 vs 每个词元激活的参数")
    n = 1 if seg == 0 else 4
    for j, h in enumerate(["模型", "总参数", "激活", "比例", "路由"]):
        d.text((110 + [0, 380, 560, 740, 900][j], 160), h, font=F_SMALL, fill=MUTED)
    for i, (m, t, a, r) in enumerate(MODELS[:n]):
        y = 210 + i * 80
        col = [CYAN, VIOLET, AMBER, GREEN][i]
        d.rounded_rectangle([90, y - 14, 1190, y + 46], 10, outline=col, width=2, fill=(12, 19, 34))
        d.text((110, y + 16), m, font=F_BODY, fill=col, anchor="lm")
        f = vk.font(24, mono=True)
        d.text((490, y + 16), f"{t:g}B", font=f, fill=TEXT, anchor="mm")
        d.text((650, y + 16), f"{a:g}B", font=f, fill=TEXT, anchor="mm")
        d.text((800, y + 16), f"{a / t:.1%}", font=f, fill=AMBER, anchor="mm")
        d.text((900, y + 16), r, font=F_SMALL, fill=TEXT, anchor="lm")


def sc_memory(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "省计算，不省显存")
    tv.bar_chart(d, (200, 220, 760, 560), ["每个词元计算（激活）", "存储（全部）"], [37, 671], [GREEN, RED], fmt="{:.0f}B", vmax=760)
    d.text((200, 170), "DeepSeek-V3", font=F_SMALL, fill=MUTED)
    d.text((840, 260), f"bf16 权重：{DSV3_GB:,.0f} GB", font=F_BODY, fill=RED)
    d.text((840, 320), f"至少 {GPUS} 张 80 GB 显卡", font=F_BODY, fill=RED)
    if seg >= 1:
        d.text((840, 420), "适合大规模服务，", font=F_SMALL, fill=TEXT)
        d.text((840, 455), "不适合单卡部署", font=F_SMALL, fill=TEXT)


def sc_ep(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "专家并行与全对全通信")
    for g in range(4):
        x = 140 + g * 270
        d.rounded_rectangle([x, 330, x + 220, 450], 14, outline=VIOLET, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 110, 365), f"显卡 {g + 1}", F_SMALL, VIOLET)
        vk.text_c(d, (x + 110, 410), f"专家 {2 * g + 1}–{2 * g + 2}", F_SMALL, TEXT)
    for i in range(8):
        x = 160 + i * 125
        tv.token(d, x, 190, "词", CYAN, w=60, font=F_SMALL)
        tgt = 140 + ((i * 3) % 4) * 270 + 110
        d.line([(x + 30, 236), (tgt, 326)], fill=(71, 85, 105), width=1)
    vk.text_c(d, (640, 290), "全对全（all-to-all）：发过去 · 算完 · 发回来（每层两次）", F_SMALL, AMBER)
    if seg >= 1:
        vk.text_c(d, (640, 520), "限制每个词元去的机器数 · 通信与计算重叠 · 负载均衡", F_BODY, GREEN)


def sc_batch(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "一批词元会用到多少个专家？（256 选 8，随机）")
    tv.fpaste(c, r"$E\left(1-(1-k/E)^{B}\right)$", 960, 200, 34)
    if seg >= 1:
        Bs = sorted(TB)
        tv.bar_chart(d, (140, 220, 820, 560), [f"批量 {b}" for b in Bs], [TB[b] for b in Bs], [GREEN, CYAN, AMBER, RED, RED], fmt="{:.0f}", vmax=290)
        d.line([(140, 560 - 256 / 290 * 340), (820, 560 - 256 / 290 * 340)], fill=MUTED, width=1)
        d.text((825, 560 - 256 / 290 * 340), "全部 256", font=F_SMALL, fill=MUTED, anchor="lm")
        d.text((880, 330), "批量大了：几乎每步", font=F_SMALL, fill=RED)
        d.text((880, 365), "都要读全部权重", font=F_SMALL, fill=RED)
        d.text((880, 420), "只剩计算上的节省", font=F_SMALL, fill=TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("稀疏", "每个词元激活约 3% – 28%"), ("代价", "省计算不省显存，需专家并行"),
                      ("批量", "批量越大，读取优势越小")], seg, "最后一期：MoE 的权衡与前沿")


RENDER = {"title": sc_title, "models": sc_models, "memory": sc_memory, "ep": sc_ep, "batch": sc_batch, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "moe-parallel")
