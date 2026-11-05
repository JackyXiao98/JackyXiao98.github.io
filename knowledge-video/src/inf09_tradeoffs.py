"""LLM Inference 09 — 推理的权衡与前沿.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] INF_CACHE=<dir> python inf09_tradeoffs.py <out_dir>

Recap of this series' measurements (read from the infx caches of episodes 02-08) and a latency / throughput
illustration for the 8B example of episode 01: per-step time = (weight bytes + batch x context x KV bytes per
token) / bandwidth, with 16 GB of bf16 weights, 128 KB of KV cache per token (Llama-3-8B configuration),
2,048-token contexts and 3.35 TB/s (memory-bound decoding only; compute ignored, an upper-bound sketch).
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
Q = infx.cached(CACHE, "quant", infx.exp_quant, cache_dir=CACHE)
SP = infx.cached(CACHE, "spec", infx.exp_spec, cache_dir=CACHE)
EV = infx.cached(CACHE, "evict", infx.exp_evict, cache_dir=CACHE)
BT = infx.exp_batching()
WB, KVB, CTX, BW = 16e9, 131072, 2048, 3.35e12
BS = [1, 4, 16, 64, 128]
STEP = {b: (WB + b * CTX * KVB) / BW for b in BS}
THR = {b: b / STEP[b] for b in BS}
print({b: (round(STEP[b] * 1e3, 2), round(THR[b])) for b in BS})
assert round(STEP[1] * 1e3, 1) == 4.9 and round(STEP[64] * 1e3, 1) == 9.9 and round(THR[1]) == 206 and round(THR[64]) == 6462
assert round(STEP[128] * 1e3, 1) == 15.0 and round(THR[128]) == 8515
RECAP = [("02", "KV 缓存", "输出一致，注意力计算 ÷ 43"),
         ("03", "GQA / MQA / MLA", f"缓存 ÷ 4 损失 +{M['mqa']['loss'] - M['mha']['loss']:.3f}；÷ 8 +{M['mla']['loss'] - M['mha']['loss']:.3f}"),
         ("04", "FlashAttention", "结果精确，不存 T × T 矩阵"),
         ("05", "4 位按列量化", f"损失 {Q['base']:.3f} → {Q[(4, True)]:.3f}"),
         ("06", "投机解码", f"分布不变，估计快 {SP['gen'][2]['per_call'] / (1 + 2 * SP['draft_params'] / SP['target_params']):.1f} 倍"),
         ("07", "连续批处理 + 分页", f"利用率 {BT['util_static']:.0%} → {BT['util_cont']:.0%}"),
         ("08", "重要性驱逐（缓存 8）", f"损失 {EV['full']:.3f} → {EV[('heavy', 8)]:.3f}")]

SCENES = [
    ("title", [
        "这是推理加速系列的最后一期。我们先回顾这个系列在小模型上看到的结果，再看推理里最核心的权衡：延迟、吞吐和成本，最后看几个前沿方向。",
    ]),
    ("recap", [
        "第二期，键值缓存不改变输出，把注意力的计算减少了四十多倍。第三期，共用键值把缓存缩到四分之一几乎没有损失。第四期，快速注意力分块计算，结果精确。",
        "第五期，四位按列量化只让损失增加约零点零四。第六期，投机解码不改变输出分布，估计能快一倍半。第七期，连续批处理把位置利用率从三成多提高到九成多。第八期，按注意力保留缓存，损失最接近完整缓存。",
    ]),
    ("triangle", [
        f"推理里最核心的权衡是延迟和吞吐。还是那个八十亿参数的例子，假设每个请求有两千个词元的上下文。只服务一个请求时，每一步约{dec_zh(STEP[1] * 1e3, 1)}毫秒，每秒约{int_zh(round(THR[1]))}个词元。",
        f"同时服务六十四个请求，每一步要多读它们的缓存，变成约{dec_zh(STEP[64] * 1e3, 1)}毫秒，每个用户感觉慢了一倍，但整体每秒能生成约{big_zh(round(THR[64]))}个词元，是原来的三十多倍。成本按吞吐算，所以服务商总想把批量做大，而用户想要的是低延迟。",
    ]),
    ("frontier", [
        "前沿上有几个方向。一是把预填充和解码分开，放在不同的机器上：预填充受算力限制，解码受带宽限制，分开以后各自用最合适的硬件和批量。",
        "二是一次预测多个词元，比如在训练时就让模型多学几个预测头，推理时配合投机解码使用。三是推理时投入更多计算换更好的答案，比如让模型多想几步、多采样几次再挑选，这让推理效率变得更加重要。",
    ]),
    ("summary", [
        "这个系列就到这里。",
        "推理加速的核心，是认清瓶颈在哪里：解码受带宽限制，所以我们要少读数据、让一次读取服务更多的词元，同时管好不断增长的键值缓存。",
        "希望这个系列能帮你理解，一个回答从模型里生成出来时，背后发生了什么。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 9, "推理的权衡与前沿", "系列回顾", SERIES)


def sc_recap(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "本系列的小实验")
    n = 3 if seg == 0 else len(RECAP)
    for i, (ep, a, b) in enumerate(RECAP[:n]):
        y = 160 + i * 70
        col = [CYAN, GREEN, VIOLET, AMBER, CYAN, GREEN, AMBER][i]
        d.rounded_rectangle([110, y, 1170, y + 56], 10, outline=col, width=2, fill=(12, 19, 34))
        d.text((135, y + 28), f"第 {ep} 期", font=F_SMALL, fill=MUTED, anchor="lm")
        d.text((260, y + 28), a, font=F_SMALL, fill=col, anchor="lm")
        d.text((600, y + 28), b, font=F_SMALL, fill=TEXT, anchor="lm")


def sc_triangle(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "延迟 vs 吞吐（8B · bf16 · 每请求 2K 上下文 · 只算显存读取）")
    labs = [str(b) for b in BS]
    tv.bar_chart(d, (110, 220, 590, 560), labs, [STEP[b] * 1e3 for b in BS], [CYAN] * len(BS), fmt="{:.1f}", vmax=17)
    d.text((110, 180), "每一步的时间（毫秒）· 越低越好", font=F_SMALL, fill=MUTED)
    d.text((590, 615), "同时服务的请求数", font=F_SMALL, fill=MUTED, anchor="rm")
    if seg >= 1:
        tv.bar_chart(d, (690, 220, 1170, 560), labs, [THR[b] for b in BS], [GREEN] * len(BS), fmt="{:.0f}", vmax=9800)
        d.text((690, 180), "整体吞吐（词元 / 秒）· 越高越好", font=F_SMALL, fill=MUTED)
        vk.text_c(d, (640, 672), "成本按吞吐算 · 用户感受的是延迟", F_BODY, AMBER)


def sc_frontier(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "前沿方向")
    rows = [("预填充与解码分离", "DistServe / Splitwise（2024）：各用最合适的硬件", CYAN)]
    if seg >= 1:
        rows += [("多词元预测", "训练时多个预测头，推理时配合投机解码", GREEN), ("推理时计算", "多想几步、多采样再挑选：效率更重要", AMBER)]
    for i, (a, b, col) in enumerate(rows):
        y = 190 + i * 100
        d.rounded_rectangle([110, y, 1170, y + 72], 12, outline=col, width=2, fill=(12, 19, 34))
        d.text((140, y + 36), a, font=F_BODY, fill=col, anchor="lm")
        d.text((470, y + 36), b, font=F_SMALL, fill=TEXT, anchor="lm")


def sc_summary(c, u, seg, su):
    tv.summary(c, 4, [("瓶颈", "解码受显存带宽限制"), ("方法", "少读数据 · 一次读取服务更多词元"),
                      ("缓存", "管好不断增长的 KV 缓存")], seg + 1)


RENDER = {"title": sc_title, "recap": sc_recap, "triangle": sc_triangle, "frontier": sc_frontier, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "inf-tradeoffs")
