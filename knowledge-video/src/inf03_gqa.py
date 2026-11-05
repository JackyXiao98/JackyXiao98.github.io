"""LLM Inference 03 — 压缩 KV 缓存：MQA、GQA 与 MLA.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] INF_CACHE=<dir> python inf03_gqa.py <out_dir>

Real computations (infx.exp_models, JAX training + numpy evaluation): four versions of the toy model that differ
only in attention - multi-head (4 KV heads), grouped-query (2 KV heads shared by pairs of query heads),
multi-query (1 KV head), and a multi-head latent attention (a 16-dim latent per token is cached; keys and values
of all 4 heads are reconstructed from it) - each trained 3,000 steps; held-out loss on 256 sequences and cached
floats per token (both layers). Cache sizes of real models from their public configurations (see notes).
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import infx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "LLM Inference"
CACHE = os.environ.get("INF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "infcache"))

M = infx.exp_models(CACHE)
KEYS = ["mha", "gqa", "mqa", "mla"]
NAMES = {"mha": "MHA（4 个 KV 头）", "gqa": "GQA（2 个 KV 头）", "mqa": "MQA（1 个 KV 头）", "mla": "MLA（16 维潜变量）"}
L = {k: M[k]["loss"] for k in KEYS}
CA = {k: M[k]["cache"] for k in KEYS}
print({k: (round(L[k], 4), CA[k], M[k]["params"]) for k in KEYS})
assert [CA[k] for k in KEYS] == [256, 128, 64, 32]
assert round(L["mha"], 3) == 1.660 and round(L["gqa"], 3) == 1.659 and round(L["mqa"], 3) == 1.661 and round(L["mla"], 3) == 1.671
assert max(L["gqa"], L["mqa"]) - L["mha"] < 0.002 and 0.008 < L["mla"] - L["mha"] < 0.015

SCENES = [
    ("title", [
        "上一期我们看到，键值缓存常常比模型本身还大。这一期看三种压缩缓存的方法：多查询注意力、分组查询注意力，和深度求索提出的多头潜在注意力。",
    ]),
    ("heads", [
        "普通的多头注意力，每个头有自己的查询、键和值，缓存要存每个头的键和值。多查询注意力让所有的头共用同一组键和值，只保留各自的查询，缓存一下子缩小到原来的头数分之一。",
        "分组查询注意力是两者之间的折中：把查询头分成几组，每组共用一组键和值。比如三十二个查询头分成八组，缓存就是原来的四分之一。上一期提到的第三代羊驼模型就是这样做的。",
    ]),
    ("mla", [
        "多头潜在注意力走得更远。它不直接存键和值，而是先把每个词元压缩成一个低维的潜在向量，只缓存这个向量；用到的时候，再从它还原出每个头的键和值。",
        "深度求索第二代的论文报告，和他们之前的六百七十亿参数稠密模型相比，键值缓存减少了百分之九十三点三。",
    ]),
    ("toy", [
        f"我们在小模型上比较四种做法，其他部分完全相同。四个键值头的普通注意力，每个词元要缓存{int_zh(CA['mha'])}个数，损失{dec_zh(L['mha'], 3)}。",
        f"两组的分组查询注意力，缓存减半，损失{dec_zh(L['gqa'], 3)}；只有一组的多查询注意力，缓存是四分之一，损失{dec_zh(L['mqa'], 3)}。在这个任务上，共用键值几乎没有代价。",
        f"潜在注意力只缓存{int_zh(CA['mla'])}个数，是原来的八分之一，损失{dec_zh(L['mla'], 3)}，多了约零点零一。压缩得越狠，表达能力的损失开始显现。",
    ]),
    ("tradeoff", [
        "所以压缩缓存是一个取舍：缓存越小，能同时服务的请求越多、能支持的上下文越长，每一步要读的数据也越少；但压缩太狠，模型的质量会下降。今天大多数开源模型都用分组查询注意力，而潜在注意力在深度求索的模型里效果很好。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，多查询和分组查询注意力让多个查询头共用键和值。",
        "第二，潜在注意力只缓存一个低维向量，再还原出键和值。",
        "第三，在我们的实验里，缓存缩到四分之一几乎没有损失，缩到八分之一损失略有增加。下一期，我们看怎样让注意力少读显存。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 3, "压缩 KV 缓存", "MQA、GQA 与 MLA", SERIES)


def heads_row(d, y, nkv, label, col):
    d.text((120, y + 30), label, font=F_SMALL, fill=col, anchor="lm")
    for i in range(8):
        x = 420 + i * 70
        tv.token(d, x, y, "Q", CYAN, w=50, font=vk.font(20, mono=True))
        g = i // (8 // nkv)
        if i % (8 // nkv) == 0:
            gx = 420 + g * (8 // nkv) * 70
            wdt = (8 // nkv) * 70 - 20
            d.rounded_rectangle([gx, y + 60, gx + wdt, y + 100], 8, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (gx + wdt / 2, y + 80), "K V", vk.font(18, mono=True), col)


def sc_heads(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "多个查询头共用键和值")
    heads_row(d, 170, 8, "MHA：每头一组", MUTED)
    if seg == 0:
        heads_row(d, 330, 1, "MQA：全部共用一组", RED)
    else:
        heads_row(d, 330, 1, "MQA：全部共用一组", RED)
        heads_row(d, 490, 2, "GQA：分组共用", GREEN)


def sc_mla(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "多头潜在注意力（MLA）：只缓存一个低维向量")
    steps = [("词元 h", CYAN, 140), ("潜在向量 c（缓存）", AMBER, 470), ("每个头的 K、V", GREEN, 820)]
    for t, col, x in steps:
        d.rounded_rectangle([x, 220, x + 290, 300], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 145, 260), t, F_SMALL, col)
    vk.arrow(d, (435, 260), (465, 260), MUTED, 3)
    vk.arrow(d, (765, 260), (815, 260), MUTED, 3)
    d.text((440, 320), "压缩", font=F_SMALL, fill=MUTED)
    d.text((770, 320), "还原", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 460), "DeepSeek-V2（2024）：KV 缓存比 DeepSeek 67B 减少 93.3%", F_BODY, AMBER)


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "小模型对比（只改注意力，3,000 步）")
    n = [1, 3, 4][seg]
    ks = KEYS[:n]
    cols = [MUTED, GREEN, CYAN, AMBER]
    tv.bar_chart(d, (120, 220, 600, 560), [k.upper() for k in ks], [CA[k] for k in ks], cols[:n], fmt="{:.0f}", vmax=290)
    d.text((120, 180), "每个词元缓存的数（两层合计）", font=F_SMALL, fill=MUTED)
    f = vk.font(24, mono=True)
    d.text((700, 180), "结构", font=F_SMALL, fill=MUTED)
    d.text((1040, 180), "损失", font=F_SMALL, fill=MUTED)
    for i, k in enumerate(ks):
        y = 235 + i * 60
        d.text((700, y), NAMES[k], font=F_SMALL, fill=cols[i])
        d.text((1040, y), f"{L[k]:.3f}", font=f, fill=TEXT)


def sc_tradeoff(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "取舍")
    d.rounded_rectangle([120, 200, 600, 460], 14, outline=GREEN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (360, 240), "缓存越小", F_BODY, GREEN)
    for i, t in enumerate(["同时服务更多请求", "支持更长上下文", "每步读的数据更少"]):
        vk.text_c(d, (360, 310 + i * 45), t, F_SMALL, TEXT)
    d.rounded_rectangle([680, 200, 1160, 460], 14, outline=RED, width=2, fill=(12, 19, 34))
    vk.text_c(d, (920, 240), "压缩太狠", F_BODY, RED)
    vk.text_c(d, (920, 330), "模型质量下降", F_SMALL, TEXT)
    vk.text_c(d, (640, 560), "GQA：多数开源模型 · MLA：DeepSeek-V2 / V3", F_SMALL, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("MQA / GQA", "多个查询头共用键值"), ("MLA", "只缓存低维潜在向量"),
                      ("实验", "缩到 1/4 几乎无损，1/8 略有损失")], seg, "下一期：FlashAttention")


RENDER = {"title": sc_title, "heads": sc_heads, "mla": sc_mla, "toy": sc_toy, "tradeoff": sc_tradeoff, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "inf-gqa")
