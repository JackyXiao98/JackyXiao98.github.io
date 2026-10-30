"""Transformer 07 — 高效生成：KV 缓存.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python tf07_kvcache.py <out_dir>

Real computations: a random 4-layer decoder (numpy, d = 256) generating 256 tokens greedily, once by
re-running the whole prefix every step and once with a key/value cache; the two outputs are compared
and the number of token positions processed and the wall-clock time are measured on this machine.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import int_zh

AMBER, RED = tv.AMBER, tv.RED

# a random 4-layer Pre-LN decoder; block() optionally appends this step's keys/values to a cache
rng = np.random.default_rng(0)
d, h, Lyr, V = 256, 4, 4, 100
P = [dict(qkv=rng.standard_normal((d, 3 * d)) / np.sqrt(d), o=rng.standard_normal((d, d)) / np.sqrt(d) / 3,
          w1=rng.standard_normal((d, 4 * d)) / np.sqrt(d), w2=rng.standard_normal((4 * d, d)) / np.sqrt(4 * d) / 3)
     for _ in range(Lyr)]
E = rng.standard_normal((V, d))
OUT = rng.standard_normal((d, V)) / np.sqrt(d)
_ang = np.arange(1024)[:, None] / 10000 ** (np.arange(0, d, 2) / d)
PE = np.concatenate([np.sin(_ang), np.cos(_ang)], 1)


def ln(x):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)


def block(x, Lp, cache=None):
    T = x.shape[0]
    q, k, v = np.split(ln(x) @ Lp["qkv"], 3, -1)
    if cache is not None:
        cache["k"].append(k)
        cache["v"].append(v)
        k, v = np.concatenate(cache["k"]), np.concatenate(cache["v"])
    S, dh = k.shape[0], d // h
    sh = lambda t, n: t.reshape(n, h, dh).transpose(1, 0, 2)
    s = sh(q, T) @ sh(k, S).transpose(0, 2, 1) / np.sqrt(dh)
    s = np.where(np.arange(S)[None, None, :] <= np.arange(S - T, S)[None, :, None], s, -1e9)
    a = np.exp(s - s.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    x = x + (a @ sh(v, S)).transpose(1, 0, 2).reshape(T, d) @ Lp["o"]
    return x + np.maximum(0, ln(x) @ Lp["w1"]) @ Lp["w2"]


def naive(prompt, n):
    """Re-run the whole prefix every step. Returns (tokens, token positions processed)."""
    seq, work = list(prompt), 0
    for _ in range(n):
        x = E[seq] + PE[:len(seq)]
        work += len(seq)
        for Lp in P:
            x = block(x, Lp)
        seq.append(int((ln(x[-1]) @ OUT).argmax()))
    return seq, work


def cached(prompt, n):
    """Process the prompt once, then only the newest token, reusing cached keys/values."""
    seq, work = list(prompt), 0
    C = [dict(k=[], v=[]) for _ in P]
    x = E[seq] + PE[:len(seq)]
    for _ in range(n):
        work += x.shape[0]
        for Lp, c in zip(P, C):
            x = block(x, Lp, c)
        seq.append(int((ln(x[-1]) @ OUT).argmax()))
        x = (E[seq[-1]] + PE[len(seq) - 1])[None]
    return seq, work


PROMPT, NGEN = [1, 2, 3, 4], 256


def timed(f):
    """Best of 3 runs (the machine may be busy)."""
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        out = f(PROMPT, NGEN)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return out, best


(SEQ_A, WORK_A), T_A = timed(naive)
(SEQ_B, WORK_B), T_B = timed(cached)
SPEED = T_A / T_B
print(f"same={SEQ_A == SEQ_B} work {WORK_A} vs {WORK_B}  time {T_A:.2f}s vs {T_B:.3f}s  x{SPEED:.0f}")
assert SEQ_A == SEQ_B and 100 < WORK_A / WORK_B < 200
assert WORK_A == sum(len(PROMPT) + i for i in range(NGEN)) and WORK_B == len(PROMPT) + NGEN - 1
assert SPEED > 2 and WORK_A % 1000 and WORK_A % 10000 >= 1000
SPEED_R = int(round(SPEED, -1)) if SPEED >= 20 else int(round(SPEED))

# memory of the cache for an example model shape (labelled as an example on screen)
EX_L, EX_D, EX_T = 32, 4096, 4096
MEM = 2 * EX_L * EX_D * EX_T * 2
assert MEM == 2 * 1024 ** 3

SCENES = [
    ("title", [
        "上一期我们训练了一个会做加法的变换器。这一期讨论生成文字时的一个关键优化，键值缓存。",
    ]),
    ("naive", [
        "语言模型生成文字时，每一步都要把已经写出的所有词元送进模型，算出下一个词元，再把它接在后面。",
        "如果每一步都从头计算，第一步算四个位置，第二步算五个位置，越往后越慢，总的计算量随长度的平方增长。",
    ]),
    ("idea", [
        "但因为有因果掩码，前面位置的计算结果，不会被后面的词元改变。上上期我们验证过，改动最后一个词元，前面所有位置的输出差别严格为零。",
        "所以每一层的键和值，算过一次就可以保存下来。新的一步只需要为最新的一个词元，计算它的查询、键和值，再和缓存里的键和值做注意力。",
    ]),
    ("bench", [
        f"我们用一个四层的随机变换器，从四个词元出发，生成{int_zh(NGEN)}个词元，分别用两种方法实际运行。",
        f"两种方法生成的结果完全相同。不用缓存时，一共处理了{int_zh(WORK_A // 10000)}万{int_zh(WORK_A % 10000 // 1000)}千多个位置，用了缓存只处理了{int_zh(WORK_B)}个，少了一百多倍。在这台电脑上，实际运行时间也快了大约{int_zh(SPEED_R)}倍。",
    ]),
    ("memory", [
        "代价是显存。缓存要为每一层、每个词元，保存一个键向量和一个值向量。",
        "以一个三十二层、四千零九十六维的模型为例，用半精度保存四千零九十六个词元的缓存，就要两个吉字节。上下文越长，缓存越大，这也是后来出现分组查询注意力等技术的原因。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，生成是一个词元接一个词元进行的，从头重算的代价随长度的平方增长。",
        "第二，因果掩码保证前面的键和值不会改变，可以缓存起来反复使用，每一步只算新的词元。",
        "第三，缓存用显存换速度，长上下文时显存会成为瓶颈。",
        "最后一期，我们看看规模定律，以及现代大模型对变换器做了哪些改进。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 7, "高效生成：KV 缓存", "用显存换速度")


def grid(d, x0, y0, rows, cell, hl):
    """Row r shows r + 4 tokens; hl(r, j) -> colour of cell (r, j)."""
    for r in range(rows):
        for j in range(r + 4):
            col = hl(r, j)
            d.rectangle([x0 + j * cell, y0 + r * cell, x0 + (j + 1) * cell - 3, y0 + (r + 1) * cell - 3], fill=col)


def sc_naive(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "朴素生成：每一步从头算")
    rows = 6 if seg >= 1 else 1 + int(5 * vk.ease(u * 2 if u < .5 else 1))
    grid(d, 160, 150, rows, 52, lambda r, j: RED if seg >= 1 else (14, 60, 72))
    for r in range(rows):
        d.text((160 - 14, 150 + r * 52 + 24), f"第 {r + 1} 步", font=F_SMALL, fill=MUTED, anchor="rm")
        d.text((160 + (r + 4) * 52 + 10, 150 + r * 52 + 24), f"{r + 4} 个位置", font=F_SMALL, fill=TEXT, anchor="lm")
    if seg >= 1:
        tv.fpaste(c, r"$\sum_{t=1}^{n}(p+t-1)\ \approx\ \frac{n^2}{2}$", 1000, 290, 34, "#f87171")
        vk.text_c(d, (1000, 400), "计算量随长度平方增长", F_BODY, RED)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "KV 缓存：只算最新的一个词元")
    grid(d, 160, 150, 6, 52, lambda r, j: (CYAN if j == r + 3 else (40, 52, 70)) if seg >= 1 else
         ((14, 60, 72) if j < r + 3 else CYAN))
    if seg >= 1:
        for r in range(6):
            d.text((160 + (r + 4) * 52 + 10, 150 + r * 52 + 24), "1 个新位置", font=F_SMALL, fill=CYAN, anchor="lm")
        tv.fpaste(c, r"$q_t\ \cdot\ [\,k_1,\dots,k_{t-1}\ |\ k_t\,]$", 1000, 260, 30)
        d.text((850, 320), "■ 灰色：缓存里的 K、V（不再计算）", font=F_SMALL, fill=MUTED)
        d.text((850, 360), "■ 青色：本步新算的 q、k、v", font=F_SMALL, fill=CYAN)
    else:
        vk.text_c(d, (1000, 300), "因果掩码：前面位置的结果", F_BODY, TEXT)
        vk.text_c(d, (1000, 350), "不受后面词元影响", F_BODY, TEXT)


def sc_bench(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, f"实测：随机 4 层模型（d = 256）生成 {NGEN} 个词元")
    steps = np.arange(1, NGEN + 1)
    tv.line_chart(d, (130, 150, 650, 520), steps, [("从头重算", len(PROMPT) + steps - 1, RED), ("KV 缓存", np.ones(NGEN), CYAN)],
                  ylim=(0, NGEN + 10), xlabel="生成步数", ylabel="本步处理的位置数", legend_xy=(150, 215),
                  xticks=[(1, "1"), (128, "128"), (256, "256")], upto=None if seg >= 1 else max(2, int(NGEN * vk.ease(su))))
    if seg >= 1:
        rows = [("输出是否相同", "完全相同" if SEQ_A == SEQ_B else "不同", GREEN),
                ("处理的位置总数", f"{WORK_A:,}  vs  {WORK_B:,}", TEXT),
                ("运行时间（本机 numpy）", f"{T_A:.1f} s  vs  {T_B:.2f} s", TEXT),
                ("加速", f"约 {SPEED:.0f} 倍", CYAN)]
        for i, (k, v, col) in enumerate(rows):
            d.text((730, 190 + i * 80), k, font=F_SMALL, fill=MUTED)
            d.text((730, 222 + i * 80), v, font=F_BODY, fill=col)


def sc_memory(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "代价：显存")
    tv.fpaste(c, r"$\mathrm{KV\ memory}=2\times L\times d\times T\times \mathrm{bytes}$", 640, 170, 30)
    if seg >= 1:
        tv.fpaste(c, r"$2\times 32\times 4096\times 4096\times 2\ \mathrm{B}=2\ \mathrm{GiB}$", 640, 260, 30, "#fbbf24")
        vk.text_c(d, (640, 320), f"示例：{EX_L} 层、d = {EX_D}、FP16、{EX_T} 个词元（单条序列）", F_SMALL, MUTED)
        Ts = np.array([1024, 4096, 16384, 32768])
        tv.bar_chart(d, (300, 380, 980, 600), [f"{t // 1024}K" for t in Ts], list(2 * EX_L * EX_D * Ts * 2 / 1024 ** 3),
                     [CYAN, AMBER, VIOLET, RED], fmt="{:.1f} GiB")
        d.text((990, 620), "上下文长度", font=F_SMALL, fill=MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("逐词生成", "从头重算：计算量 ∝ 长度²"), ("KV 缓存", "保存 K、V，每步只算新词元"),
                      ("代价", "显存随上下文长度线性增长")], seg, "最后一期：规模定律与现代改进")


RENDER = {"title": sc_title, "naive": sc_naive, "idea": sc_idea, "bench": sc_bench, "memory": sc_memory,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "kv-cache")
