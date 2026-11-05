"""LLM Inference 02 — KV 缓存：用显存换计算.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] INF_CACHE=<dir> python inf02_kvcache.py <out_dir>

Real computations (infx, numpy): the trained 2-layer toy model (4 heads, multi-head attention) decodes 16 held-out
sequences of 64 tokens token by token with a KV cache (infx.Cache) and, for comparison, by re-running the full
prefix at every step; largest absolute difference between the two sets of logits. Attention multiply-adds with
and without the cache for 64 steps. KV-cache size per token = 2 (K and V) x layers x KV heads x head dim x 2 bytes
for the public configurations of Llama-2-7B (32 layers, 32 KV heads, head dim 128) and Llama-3-8B (32 layers,
8 KV heads, head dim 128).
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

M = infx.cached(CACHE, "models", infx.exp_models)
P = M["mha"]["p"]
xs, _ = infx.sample(np.random.default_rng(21), 16)
diff = 0.0
for x in xs:
    full = infx.forward(P, x[None, :infx.LEN])[0]
    c = infx.Cache(P)
    inc = np.stack([c.step(int(t)) for t in x[:infx.LEN]])
    diff = max(diff, float(np.abs(full - inc).max()))
FL_CACHE = c.flops
FL_NO = infx.attn_flops_nocache(P, infx.LEN)
kvb = lambda layers, kvh, hd: 2 * layers * kvh * hd * 2
L2, L3 = kvb(32, 32, 128), kvb(32, 8, 128)
print(diff, FL_CACHE, FL_NO, FL_NO / FL_CACHE, L2, L3)
assert diff < 1e-4 and round(FL_NO / FL_CACHE) == 43
assert L2 == 524288 and L3 == 131072 and L2 * 4096 == 2 * 2 ** 30 and L3 * 8192 == 2 ** 30

SCENES = [
    ("title", [
        "上一期我们看到，解码时每一步只生成一个新词元。但注意力要看前面所有的词元。如果每一步都把前文从头算一遍，就会浪费大量计算。这一期讲解决这个问题的键值缓存。",
    ]),
    ("redundant", [
        "在因果注意力里，每个位置的键和值只取决于它自己和它前面的词元，后面生成什么都不会改变它们。所以第十个词元的键和值，在第十一步、第十二步、一直到最后，都是同样的值。",
        "既然不会变，就没有必要重算。键值缓存的做法是：每生成一个词元，就把它在每一层的键和值存下来；下一步只为新词元算查询、键和值，再和缓存里所有的键做注意力。",
    ]),
    ("exact", [
        f"我们在小模型上验证。对十六段六十四个词元的序列，一边用缓存逐个解码，一边每步把前文重新算一遍。两种方法的输出完全一致，最大差别只有约十的负六次方，这是浮点数的舍入误差。",
        f"计算量差别很大。只算注意力这部分，六十四步下来，不用缓存要做的乘加次数是用缓存的约{int_zh(round(FL_NO / FL_CACHE))}倍。序列越长，差距越大：不用缓存是平方增长，用缓存是线性增长。",
    ]),
    ("size", [
        "代价是显存。每个词元在每一层都要存一份键和一份值。按公开的配置计算，七十亿参数的第二代羊驼模型，每个词元要存零点五兆字节，四千个词元的上下文就是两吉字节。",
        "第三代羊驼的八十亿参数模型用了分组查询注意力，每个词元一百二十八千字节，八千个词元的上下文是一吉字节。如果同时服务三十二个这样的请求，缓存就有三十二吉字节，比模型本身的十六吉字节还大。",
    ]),
    ("tradeoff", [
        "所以，键值缓存把计算的问题变成了显存的问题。它决定了一张卡能同时服务多少请求、能支持多长的上下文。而且每一步解码，不光要读权重，还要把整段缓存读一遍，上下文越长，这一部分越慢。",
        "接下来的几期，很多方法都围绕着它：怎样把缓存变小，怎样少读缓存，怎样更好地管理缓存。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，前面词元的键和值不会改变，存下来就不必重算。",
        "第二，在我们的实验里，用缓存和不用缓存的输出完全一致，注意力的计算量少了四十多倍。",
        "第三，缓存随上下文和请求数线性增长，常常比模型本身还大。下一期，我们看怎样压缩键值缓存。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 2, "KV 缓存", "用显存换计算", SERIES)


def sc_redundant(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "前面词元的 K、V 不会变")
    n = 7
    for i in range(n):
        x = 130 + i * 140
        new = i == n - 1
        col = AMBER if new else CYAN
        tv.token(d, x, 190, f"x{i + 1}", col, w=80, font=vk.font(22, mono=True))
        d.rounded_rectangle([x, 270, x + 80, 320], 8, outline=GREEN if (seg >= 1 and not new) else col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 40, 295), "K, V", vk.font(20, mono=True), GREEN if (seg >= 1 and not new) else col)
    if seg >= 1:
        d.rounded_rectangle([120, 260, 120 + 140 * (n - 1) - 50, 330], 12, outline=GREEN, width=2)
        d.text((130, 345), "缓存：存下来，不再重算", font=F_SMALL, fill=GREEN)
        d.text((130 + 140 * (n - 1) - 20, 345), "新词元：算 Q、K、V", font=F_SMALL, fill=AMBER)
        tv.fpaste(c, r"$\mathrm{softmax}(q_t\,[K_{1..t}]^\top/\sqrt{d})\,V_{1..t}$", 640, 470, 38)


def sc_exact(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "小模型验证（16 段 × 64 个词元）")
    d.text((140, 180), "用缓存 vs 每步重算：最大差别", font=F_BODY, fill=TEXT)
    d.text((780, 180), f"{diff:.1e}", font=vk.font(30, mono=True), fill=GREEN)
    if seg >= 1:
        T = np.arange(1, infx.LEN + 1)
        per = 4 * P["cfg"]["heads"] * P["cfg"]["hd"] * len(P["layers"])
        ser = [("不用缓存（累计）", np.cumsum(per * T * T) / 1e6, RED), ("用缓存（累计）", np.cumsum(per * T) / 1e6, GREEN)]
        tv.line_chart(d, (140, 290, 760, 620), list(T), ser, xlabel="解码步数", xticks=[(1, "1"), (32, "32"), (64, "64")], legend_xy=(800, 330))
        d.text((140, 255), "注意力乘加次数（百万）", font=F_SMALL, fill=MUTED)
        d.text((800, 450), f"64 步：{FL_NO / FL_CACHE:.0f} 倍", font=F_BODY, fill=AMBER)
        d.text((800, 500), "平方 vs 线性", font=F_SMALL, fill=TEXT)


def sc_size(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "缓存有多大：2 × 层数 × KV 头数 × 头维度 × 2 字节")
    f = vk.font(24, mono=True)
    rows = [("Llama-2-7B（MHA）", "32 × 32 × 128", f"{L2 // 1024} KB", "4K 上下文：2 GB", CYAN)]
    if seg >= 1:
        rows.append(("Llama-3-8B（GQA）", "32 × 8 × 128", f"{L3 // 1024} KB", "8K 上下文：1 GB", GREEN))
    d.text((150, 175), "模型", font=F_SMALL, fill=MUTED)
    d.text((480, 175), "层 × KV 头 × 维度", font=F_SMALL, fill=MUTED)
    d.text((760, 175), "每个词元", font=F_SMALL, fill=MUTED)
    for i, (a, b, cc, e, col) in enumerate(rows):
        y = 225 + i * 90
        d.rounded_rectangle([120, y - 10, 1160, y + 60], 12, outline=col, width=2, fill=(12, 19, 34))
        d.text((150, y + 25), a, font=F_SMALL, fill=col, anchor="lm")
        d.text((480, y + 25), b, font=f, fill=TEXT, anchor="lm")
        d.text((760, y + 25), cc, font=f, fill=TEXT, anchor="lm")
        d.text((920, y + 25), e, font=F_SMALL, fill=AMBER, anchor="lm")
    if seg >= 1:
        tv.bar_chart(d, (380, 440, 900, 640), ["模型权重（bf16）", "32 个请求 × 8K 的缓存"], [16, 32], [MUTED, RED], fmt="{:.0f} GB", vmax=38)


def sc_tradeoff(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "把计算的问题变成显存的问题")
    for i, (t, col) in enumerate([("能同时服务多少请求", CYAN), ("能支持多长的上下文", VIOLET), ("每一步还要读整段缓存", RED)]):
        vk.text_c(d, (640, 210 + i * 70), t, F_BODY, col)
    if seg >= 1:
        for i, t in enumerate(["压缩缓存（第 3 期）", "少读显存（第 4 期）", "管理缓存（第 7、8 期）"]):
            vk.text_c(d, (640, 460 + i * 50), t, F_SMALL, GREEN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("不变", "前面的 K、V 存下来不重算"), ("实验", "输出一致，注意力计算少 40 多倍"),
                      ("代价", "缓存线性增长，常比模型还大")], seg, "下一期：压缩 KV 缓存")


RENDER = {"title": sc_title, "redundant": sc_redundant, "exact": sc_exact, "size": sc_size, "tradeoff": sc_tradeoff, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "inf-kvcache")
