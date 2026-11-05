"""LLM Inference 01 — 推理为什么慢：预填充、解码与内存带宽.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] INF_CACHE=<dir> python inf01_roofline.py <out_dir>

Computations: roofline arithmetic for an 8B-parameter model in bf16 (2 bytes / parameter, 2 FLOPs / parameter / token)
on an H100 SXM (989 TFLOPS dense bf16, 3.35 TB/s HBM3; vendor datasheet): weight read time per decoding step,
arithmetic intensity (FLOPs / byte) as a function of the number of tokens processed together, and the ridge point.
Real measurement (infx.exp_timing, numpy on this container's CPU): time of X @ W for a 2,048 x 2,048 float32 matrix
with 1 / 4 / 16 / 64 / 256 rows of X (mean of 50 runs; numbers vary between machines).
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

TM = infx.cached(CACHE, "timing", infx.exp_timing)
PARAMS, FLOPS, BW = 8e9, 989e12, 3.35e12
WBYTES = PARAMS * 2
T_READ = WBYTES / BW                     # seconds per decoding step (weights only)
T_COMP = 2 * PARAMS / FLOPS              # seconds of compute per token at peak
RIDGE = FLOPS / BW                       # FLOPs per byte
TOKS = 1 / T_READ
GAP = T_READ / T_COMP
R256 = TM[256] / TM[1]
PER = TM[1] / (TM[256] / 256)
print(T_READ, T_COMP, RIDGE, TOKS, GAP, {k: round(v * 1e3, 4) for k, v in TM.items()}, R256, PER)
assert round(T_READ * 1e3, 1) == 4.8 and round(TOKS) == 209 and round(RIDGE) == 295 and round(GAP) == 296
assert abs(T_COMP * 1e6 - 16.2) < 0.1 and 10 < R256 < 60 and PER > 4

SCENES = [
    ("title", [
        "大模型训练好以后，真正花钱的地方往往是推理，也就是一次次地生成文字。这个系列讲怎样让推理更快、更省。第一期先回答一个基本问题：生成一个词元，时间到底花在了哪里。",
    ]),
    ("two", [
        "推理分两个阶段。先是预填充：把用户的整段输入一次性送进模型，所有位置并行计算。然后是解码：每次只生成一个新词元，再把它接到后面，生成下一个。",
        "解码是一步一步串行的，一段几百个词元的回答，就要走几百步。而每一步，都要把模型的全部权重从显存里读一遍。",
    ]),
    ("numbers", [
        f"算一笔账。一个八十亿参数的模型，用十六位浮点数存储，有一百六十亿字节。在一张显存带宽每秒三点三五太字节的显卡上，光把权重读一遍就要约{dec_zh(T_READ * 1e3, 1)}毫秒，也就是每秒最多约{int_zh(round(TOKS))}个词元。",
        f"而这一步真正的计算量，每个参数两次浮点运算，在这张卡的峰值算力下只要约{dec_zh(T_COMP * 1e6, 1)}微秒。读数据的时间是计算时间的约{int_zh(round(GAP))}倍。解码时，显卡大部分时间在等数据，算力几乎是闲着的。",
    ]),
    ("roofline", [
        "这可以用屋顶线模型来看。横轴是算术强度，也就是每读一个字节做多少次运算；纵轴是能达到的算力。强度低的时候，受带宽限制，是一条斜线；强度高到一定程度，才碰到算力的屋顶。",
        f"这张卡的拐点约在每字节{int_zh(round(RIDGE))}次运算。单个请求解码时，每个参数读两个字节、算两次，强度只有一，离拐点很远。预填充一千个词元时，同一份权重被一千个位置共用，强度约一千，就进入了算力受限的区域。",
    ]),
    ("batch", [
        f"所以提高解码效率最直接的办法是批处理：同时为很多请求解码，权重读一次，所有请求共用。我们在这台机器的处理器上实测，一个二零四八乘二零四八的矩阵，乘一个向量和乘二百五十六个向量，时间只差约{int_zh(round(R256))}倍，平均到每个向量快了约{int_zh(round(PER))}倍。",
        "当然，批量越大，每个请求要等的时间也可能越长，这就是推理里延迟和吞吐的权衡。后面几期讲的方法，大多是在想办法少读数据，或者让一次读取服务更多的词元。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，推理分为并行的预填充和串行的解码。",
        "第二，解码的每一步都要读全部权重，速度受显存带宽限制，而不是算力。",
        "第三，批处理让一次读取服务多个请求，提高算术强度。下一期，我们看解码时另一个大头：键值缓存。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 1, "推理为什么慢", "预填充、解码与内存带宽", SERIES)


def sc_two(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "预填充（并行）与解码（串行）")
    d.text((110, 170), "预填充：整段输入一次算完", font=F_BODY, fill=CYAN)
    for i in range(8):
        tv.token(d, 110 + i * 70, 220, f"x{i + 1}", CYAN, w=60, font=vk.font(20, mono=True))
    d.text((110, 330), "解码：每次一个新词元", font=F_BODY, fill=AMBER)
    n = 1 + int(min(5, u * 10)) if seg == 0 else 6
    for i in range(n):
        tv.token(d, 110 + i * 70, 380, f"y{i + 1}", AMBER, w=60, font=vk.font(20, mono=True))
    if seg >= 1:
        vk.text_c(d, (640, 520), "每一步：读一遍全部权重", F_BODY, RED)


def sc_numbers(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "8B 模型 · bf16 · H100（算一笔账）")
    f = vk.font(26, mono=True)
    d.text((140, 180), "权重", font=F_BODY, fill=MUTED)
    d.text((420, 180), "8e9 × 2 B = 16 GB", font=f, fill=TEXT)
    d.text((140, 240), "读一遍", font=F_BODY, fill=MUTED)
    d.text((420, 240), f"16 GB ÷ 3.35 TB/s ≈ {T_READ * 1e3:.1f} ms", font=f, fill=RED)
    d.text((140, 300), "上限", font=F_BODY, fill=MUTED)
    d.text((420, 300), f"≈ {TOKS:.0f} 词元/秒", font=F_BODY, fill=RED)
    if seg >= 1:
        d.text((140, 380), "计算", font=F_BODY, fill=MUTED)
        d.text((420, 380), f"2 × 8e9 FLOP ÷ 989 TFLOPS ≈ {T_COMP * 1e6:.1f} µs", font=f, fill=GREEN)
        tv.bar_chart(d, (420, 450, 900, 640), ["读权重", "计算"], [T_READ * 1e6, T_COMP * 1e6], [RED, GREEN], fmt="{:.0f} µs", vmax=5600)
        d.text((940, 560), f"≈ {GAP:.0f} 倍", font=F_BODY, fill=AMBER)


def sc_roofline(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "屋顶线模型（对数坐标）")
    x0, y0, x1, y1 = 140, 180, 900, 600
    lx = lambda I: x0 + (np.log10(I) - 0) / 4 * (x1 - x0)          # 1 .. 1e4 FLOP/byte
    ly = lambda P: y1 - (np.log10(P) - 12) / 3.2 * (y1 - y0)        # 1e12 .. ~1.6e15 FLOPS
    d.rectangle([x0, y0, x1, y1], outline=(30, 41, 59))
    pts = [(lx(I), ly(min(FLOPS, I * BW))) for I in np.logspace(0, 4, 60)]
    d.line(pts, fill=CYAN, width=4)
    for I, lab in ((1, "1"), (10, "10"), (100, "100"), (1000, "1000"), (10000, "10⁴")):
        d.text((lx(I), y1 + 16), lab, font=vk.font(18, mono=True), fill=MUTED, anchor="mm")
    d.text((x1, y1 + 44), "算术强度（FLOP / 字节）", font=F_SMALL, fill=MUTED, anchor="rm")
    d.text((x0 + 8, y0 + 8), "可达算力", font=F_SMALL, fill=MUTED)
    d.text((lx(1200), ly(FLOPS) - 24), "算力屋顶 989 TFLOPS", font=F_SMALL, fill=CYAN, anchor="mm")
    if seg >= 1:
        rx = lx(RIDGE)
        d.line([(rx, y0), (rx, y1)], fill=MUTED, width=1)
        d.text((rx + 8, y1 - 30), f"拐点 ≈ {RIDGE:.0f}", font=F_SMALL, fill=MUTED)
        for I, lab, col in ((1, "解码（批量 1）", RED), (1000, "预填充 1000 词元", GREEN)):
            px, py = lx(I), ly(min(FLOPS, I * BW))
            d.ellipse([px - 9, py - 9, px + 9, py + 9], fill=col)
            d.text((px + 16, py + 10), lab, font=F_SMALL, fill=col)
        vk.text_c(d, (1080, 300), "左：带宽受限", F_SMALL, RED)
        vk.text_c(d, (1080, 345), "右：算力受限", F_SMALL, GREEN)


def sc_batch(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "批处理：权重读一次，多个请求共用（本机 CPU 实测）")
    Bs = sorted(TM)
    tv.bar_chart(d, (120, 210, 620, 560), [f"{b}" for b in Bs], [TM[b] * 1e3 for b in Bs], [CYAN] * len(Bs), fmt="{:.2f}", vmax=TM[256] * 1e3 * 1.15)
    d.text((120, 170), "一次矩阵乘法的时间（毫秒）", font=F_SMALL, fill=MUTED)
    d.text((620, 585), "同时处理的向量数", font=F_SMALL, fill=MUTED, anchor="rm")
    tv.bar_chart(d, (720, 210, 1180, 560), [f"{b}" for b in Bs], [TM[b] / b * 1e3 for b in Bs], [GREEN] * len(Bs), fmt="{:.3f}", vmax=TM[1] * 1e3 * 1.15)
    d.text((720, 170), "平均每个向量（毫秒）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 640), "批量越大：吞吐越高，但每个请求可能等得更久", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("两个阶段", "并行预填充 + 串行解码"), ("瓶颈", "解码受显存带宽限制"),
                      ("批处理", "一次读取服务多个请求")], seg, "下一期：KV 缓存")


RENDER = {"title": sc_title, "two": sc_two, "numbers": sc_numbers, "roofline": sc_roofline, "batch": sc_batch, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "inf-roofline")
