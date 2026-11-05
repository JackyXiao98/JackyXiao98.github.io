"""LLM Inference 07 — 批处理与 PagedAttention：服务很多请求.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python inf07_batching.py <out_dir>

Simulation (infx.exp_batching, numpy, seed 0): 400 requests whose output lengths follow a log-normal distribution
(median about 200 tokens, clipped to 8..2,048), served with 16 batch slots, one token per slot per step.
Static batching: groups of 16 run until their longest request finishes. Continuous batching: a finished request is
replaced at the next step. Slot utilisation = useful tokens / (steps x 16). KV-cache memory: reserving 2,048
positions per request up front vs pages of 16 positions allocated on demand; utilisation = tokens actually stored /
positions reserved. Facts about real systems are quoted with references in the notes.
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

R = infx.exp_batching()
OL = R["out_len"]
print({k: v for k, v in R.items() if k != "out_len"})
assert R["median"] == 192 and round(R["p99"]) == 1270 and round(R["util_static"], 2) == 0.32 and round(R["util_cont"], 2) == 0.93
assert round(R["mem_contig"], 2) == 0.13 and round(R["mem_paged"], 2) == 0.97 and round(R["steps_static"] / R["steps_cont"], 1) == 2.9

SCENES = [
    ("title", [
        "第一期我们说过，批处理能让一次读取权重服务很多请求。但真实的服务里，请求来来去去，每个回答的长度也差别很大。这一期看怎样把一批请求安排好，以及怎样管理它们的键值缓存。",
    ]),
    ("lengths", [
        f"我们模拟四百个请求，回答长度大致服从对数正态分布：一半的回答短于{int_zh(R['median'])}个词元，但也有百分之一超过{int_zh(round(R['p99']))}个。长短差别非常大。",
    ]),
    ("static", [
        "最简单的做法是静态批处理：凑满十六个请求，一起解码，等这一批全部结束，再换下一批。问题是，一批里只要有一个很长的回答，其他已经结束的请求就只能空等，它们占着的位置一直空转。",
        f"在我们的模拟里，静态批处理的位置利用率只有约{pct_zh(R['util_static'])}。",
    ]),
    ("continuous", [
        f"连续批处理换一种思路：不以批为单位，而以每一步解码为单位。只要某个请求结束了，下一步就立刻让排队的新请求补上它的位置。利用率提高到约{pct_zh(R['util_cont'])}，完成全部请求需要的步数减少到原来的约三分之一。",
        "这种做法最早在二零二二年的一篇服务系统论文里提出，叫迭代级调度，现在几乎所有推理框架都在用。",
    ]),
    ("paged", [
        f"再看显存。如果为每个请求预先留出最大长度二零四八个位置的连续缓存，因为大多数回答都很短，真正用上的只有约{pct_zh(R['mem_contig'])}。剩下的显存都浪费了，能同时服务的请求也就少了。",
        f"分页注意力借用了操作系统里虚拟内存的想法：把缓存切成固定大小的页，比如每页十六个位置，按需要一页一页地分配，不要求连续。这样浪费的只有每个请求最后一页没填满的部分，利用率达到约{pct_zh(R['mem_paged'])}。",
        "分页还带来一个好处：多个请求如果有相同的开头，比如同一段系统提示，可以共享同一批页，不用各存一份。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，回答长度差别很大，静态批处理会让结束的请求空等。",
        "第二，连续批处理每一步都补上新请求，在我们的模拟里利用率从三成多提高到九成多。",
        "第三，分页注意力按页分配缓存，几乎不浪费显存，还能共享相同的前缀。下一期，我们看长上下文时怎样管理缓存。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 7, "批处理与 PagedAttention", "服务很多请求", SERIES)


def sc_lengths(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "400 个请求的回答长度（模拟）")
    edges = np.logspace(np.log10(8), np.log10(2048), 25)
    h, _ = np.histogram(OL, edges)
    x0, y0, x1, y1 = 140, 200, 1000, 580
    w = (x1 - x0) / len(h)
    for i, v in enumerate(h):
        top = y1 - v / h.max() * (y1 - y0)
        d.rectangle([x0 + i * w, top, x0 + (i + 1) * w - 3, y1], fill=CYAN)
    d.line([(x0, y1), (x1, y1)], fill=MUTED, width=1)
    for v in (8, 32, 128, 512, 2048):
        x = x0 + (np.log10(v) - np.log10(8)) / (np.log10(2048) - np.log10(8)) * (x1 - x0)
        d.text((x, y1 + 18), str(v), font=vk.font(18, mono=True), fill=MUTED, anchor="mm")
    d.text((x1, y1 + 46), "回答长度（词元，对数坐标）", font=F_SMALL, fill=MUTED, anchor="rm")
    d.text((1030, 300), f"中位数 {R['median']:.0f}", font=F_BODY, fill=GREEN)
    d.text((1030, 360), f"99% 分位 {R['p99']:.0f}", font=F_BODY, fill=RED)


def slots_pic(d, box, cont):
    x0, y0, x1, y1 = box
    rng = np.random.default_rng(3)
    lens = [rng.integers(3, 14) for _ in range(6)] + [rng.integers(3, 14) for _ in range(6)] + [rng.integers(3, 14) for _ in range(6)]
    rows = 6
    T = 30
    cw, ch = (x1 - x0) / T, (y1 - y0) / rows
    cols = [CYAN, VIOLET, AMBER, GREEN, (244, 114, 182), (148, 163, 184)]
    if not cont:
        t0 = 0
        for b in range(3):
            L = lens[b * 6:(b + 1) * 6]
            for r in range(rows):
                for t in range(max(L)):
                    if t0 + t >= T:
                        break
                    used = t < L[r]
                    d.rectangle([x0 + (t0 + t) * cw, y0 + r * ch, x0 + (t0 + t + 1) * cw - 2, y0 + (r + 1) * ch - 3],
                                fill=cols[(b * 2 + r) % 6] if used else None, outline=(51, 65, 85) if not used else None)
            t0 += max(L)
    else:
        for r in range(rows):
            t, k = 0, r
            while t < T and k < len(lens):
                L = lens[k]
                for s in range(L):
                    if t + s >= T:
                        break
                    d.rectangle([x0 + (t + s) * cw, y0 + r * ch, x0 + (t + s + 1) * cw - 2, y0 + (r + 1) * ch - 3], fill=cols[(k // rows + r) % 6])
                t += L
                k += rows


def sc_static(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "静态批处理：等最长的那个")
    slots_pic(d, (140, 200, 1140, 470), False)
    d.text((140, 170), "每一行是一个位置，空框 = 已结束、在空等", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 560), f"模拟（16 个位置）：利用率 {R['util_static']:.0%}", F_BODY, RED)


def sc_continuous(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "连续批处理：结束了就补上新请求")
    slots_pic(d, (140, 200, 1140, 470), True)
    vk.text_c(d, (640, 540), f"利用率 {R['util_static']:.0%} → {R['util_cont']:.0%} · 总步数 {R['steps_static']:,} → {R['steps_cont']:,}", F_BODY, GREEN)
    if seg >= 1:
        vk.text_c(d, (640, 610), "Orca（OSDI 2022）：迭代级调度", F_SMALL, TEXT)


def sc_paged(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "PagedAttention：按页分配 KV 缓存")
    tv.bar_chart(d, (140, 220, 560, 560), ["预留 2048", "分页（每页 16）"], [R["mem_contig"] * 100, R["mem_paged"] * 100] if seg >= 1 else [R["mem_contig"] * 100, 0],
                 [RED, GREEN], fmt="{:.0f}%", vmax=110)
    d.text((140, 180), "缓存利用率（真正用上的比例）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        for r in range(3):
            n = [5, 3, 4][r]
            for k in range(n):
                x = 680 + k * 80
                full = k < n - 1
                d.rectangle([x, 220 + r * 80, x + 70, 280 + r * 80], fill=[CYAN, VIOLET, AMBER][r] if full else None, outline=[CYAN, VIOLET, AMBER][r], width=2)
                if not full:
                    d.rectangle([x, 220 + r * 80, x + 30, 280 + r * 80], fill=[CYAN, VIOLET, AMBER][r])
        d.text((680, 470), "只浪费最后一页没填满的部分", font=F_SMALL, fill=GREEN)
    if seg >= 2:
        d.text((680, 520), "相同前缀（系统提示）可共享页", font=F_SMALL, fill=AMBER)
        d.text((680, 560), "vLLM（SOSP 2023）", font=F_SMALL, fill=MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("长度差别", "静态批处理让结束的请求空等"), ("连续批处理", "利用率 32% → 93%（模拟）"),
                      ("分页", "缓存利用率 13% → 97%，可共享前缀")], seg, "下一期：长上下文与缓存驱逐")


RENDER = {"title": sc_title, "lengths": sc_lengths, "static": sc_static, "continuous": sc_continuous, "paged": sc_paged, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "inf-batching")
