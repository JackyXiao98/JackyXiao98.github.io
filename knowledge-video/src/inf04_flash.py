"""LLM Inference 04 — FlashAttention：分块与在线 softmax.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python inf04_flash.py <out_dir>

Real computations (infx.attention_naive / attention_tiled, numpy, float64): causal attention for random Q, K, V
(T = 512 positions, head dim 64) computed the ordinary way (full T x T score matrix) and tiled with an online
softmax (blocks of 64 x 64, running max and running sum); largest absolute difference. Sizes: the T x T score matrix
vs Q, K, V and O for T = 8,192, head dim 128, 2 bytes per number, one head; the largest intermediate in the tiled
version (one 64 x 64 block). A worked online-softmax example on 6 numbers in two blocks.
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

rng = np.random.default_rng(0)
Q, K, Vv = (rng.normal(size=(512, 64)) for _ in range(3))
DIFF = float(np.abs(infx.attention_naive(Q, K, Vv) - infx.attention_tiled(Q, K, Vv, 64, 64)).max())
N, Dh = 8192, 128
S_MB = N * N * 2 / 2 ** 20
QKVO_MB = 4 * N * Dh * 2 / 2 ** 20
BLOCK_KB = 64 * 64 * 2 / 2 ** 10
EX = np.array([1.0, 3.0, 2.0, 5.0, 0.5, 4.0])
m1, l1 = EX[:3].max(), np.exp(EX[:3] - EX[:3].max()).sum()
m2 = max(m1, EX[3:].max())
l2 = l1 * np.exp(m1 - m2) + np.exp(EX[3:] - m2).sum()
TRUE_L = np.exp(EX - EX.max()).sum()
print(DIFF, S_MB, QKVO_MB, BLOCK_KB, m1, l1, m2, l2, TRUE_L)
assert DIFF < 1e-12 and S_MB == 128 and QKVO_MB == 8 and BLOCK_KB == 8 and abs(l2 - TRUE_L) < 1e-12

SCENES = [
    ("title", [
        "注意力是变换器里最耗显存读写的部分之一。二零二二年的快速注意力算法，没有改变注意力的结果，却让它快了好几倍、省了大量显存。这一期看它是怎么做到的。",
    ]),
    ("naive", [
        "普通的注意力分三步：先算出所有查询和所有键的分数，得到一个序列长度乘序列长度的大矩阵；再对每一行做柔性最大值；最后乘上值。",
        f"问题在中间这个大矩阵。序列长度八千一百九十二、十六位浮点数时，一个头的分数矩阵就有{int_zh(round(S_MB))}兆字节，而查询、键、值和输出加起来只有{int_zh(round(QKVO_MB))}兆字节。这个大矩阵要写回显存，再读出来做柔性最大值，再写回去，再读出来乘值，读写量非常大。",
    ]),
    ("tiling", [
        "显卡上有一小块非常快的片上存储，但只有几百千字节。快速注意力的想法是：把查询、键和值切成小块，一次只把一小块搬进片上存储，在里面算完分数、柔性最大值和加权求和，只把最终的输出写回显存。",
        f"这样，那个大矩阵从来不需要完整地存在显存里。在我们的实现里，每次只处理六十四乘六十四的一块，约{int_zh(round(BLOCK_KB))}千字节。",
    ]),
    ("online", [
        "难点在于柔性最大值需要整行的最大值和总和，但我们一次只看到一块。解决办法叫在线柔性最大值：一边看，一边维护目前为止的最大值和指数和。",
        f"比如一行有六个分数，分两块。看完第一块，最大值是三；看到第二块，最大值变成五。之前算好的指数和，只要乘上一个修正因子，也就是旧最大值减新最大值的指数，就能和新的一块合起来。最后得到的总和，和一次看完整行算出来的完全一样。",
    ]),
    ("exact", [
        f"我们用五百一十二个位置、六十四维的随机数据验证：分块加在线柔性最大值的结果，和普通注意力的最大差别只有约十的负十六次方，就是双精度的舍入误差。所以它是精确的，不是近似。",
        "论文报告，在第二代生成式预训练模型上，注意力的计算最多快了七点六倍，而且显存占用从序列长度的平方降到了线性。后续的第二、第三版继续针对新显卡优化，已经成为几乎所有训练和推理框架的标准配置。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，普通注意力的瓶颈是那个序列长度平方大小的中间矩阵在显存里反复读写。",
        "第二，快速注意力分块计算，用在线柔性最大值合并各块，结果完全精确。",
        "第三，它减少的是显存读写，而不是计算量。下一期，我们看量化：用更少的位来存权重。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 4, "FlashAttention", "分块与在线 softmax", SERIES)


def sc_naive(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "普通注意力：T × T 的中间矩阵")
    steps = [("S = QKᵀ", CYAN), ("P = softmax(S)", AMBER), ("O = PV", GREEN)]
    for i, (t, col) in enumerate(steps):
        x = 120 + i * 360
        d.rounded_rectangle([x, 190, x + 300, 260], 12, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 150, 225), t, vk.font(26, mono=True), col)
        if i < 2:
            vk.arrow(d, (x + 305, 225), (x + 355, 225), MUTED, 3)
    if seg >= 1:
        tv.bar_chart(d, (300, 340, 860, 620), ["分数矩阵 S（一个头）", "Q、K、V、O 合计"], [S_MB, QKVO_MB], [RED, GREEN], fmt="{:.0f} MB", vmax=150)
        d.text((900, 420), "T = 8192，d = 128", font=F_SMALL, fill=MUTED)
        d.text((900, 460), "写回显存 → 读出 → 写回 → 读出", font=F_SMALL, fill=RED)


def sc_tiling(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "分块：在片上存储里算完一块")
    x0, y0, s = 140, 190, 360
    nb = 6
    cell = s / nb
    k = int(min(nb * (nb + 1) / 2, u * 30))
    cnt = 0
    for i in range(nb):
        for j in range(i + 1):
            col = GREEN if cnt < k else (30, 41, 59)
            d.rectangle([x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell - 3, y0 + (i + 1) * cell - 3], fill=col if cnt < k else None, outline=(51, 65, 85))
            cnt += 1
    d.text((x0, y0 + s + 14), "因果注意力的块（只算下三角）", font=F_SMALL, fill=MUTED)
    d.rounded_rectangle([640, 200, 1140, 300], 14, outline=CYAN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (890, 235), "显存（HBM）：大、慢", F_BODY, CYAN)
    vk.text_c(d, (890, 275), "只读 Q、K、V，只写 O", F_SMALL, TEXT)
    d.rounded_rectangle([660, 360, 1120, 450], 14, outline=AMBER, width=2, fill=(12, 19, 34))
    vk.text_c(d, (890, 390), "片上存储（SRAM）：小、快", F_BODY, AMBER)
    vk.text_c(d, (890, 428), "几百 KB", F_SMALL, TEXT)
    if seg >= 1:
        vk.text_c(d, (890, 520), f"我们的一块：64 × 64 ≈ {BLOCK_KB:.0f} KB", F_SMALL, GREEN)


def sc_online(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "在线 softmax：边看边更新最大值与指数和")
    f = vk.font(26, mono=True)
    for i, v in enumerate(EX):
        x = 200 + i * 110 + (40 if i >= 3 else 0)
        col = CYAN if i < 3 else AMBER
        d.rounded_rectangle([x, 180, x + 90, 240], 10, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 45, 210), f"{v:g}", f, col)
    d.text((200, 255), "第 1 块", font=F_SMALL, fill=CYAN)
    d.text((570, 255), "第 2 块", font=F_SMALL, fill=AMBER)
    if seg >= 1:
        rows = [(f"第 1 块后：m = {m1:g}", f"l = Σ e^(x−{m1:g}) = {l1:.4f}", CYAN),
                (f"第 2 块：m = {m2:g}", f"l = {l1:.4f}·e^({m1:g}−{m2:g}) + Σ e^(x−{m2:g}) = {l2:.4f}", AMBER),
                ("一次看完整行", f"Σ e^(x−{EX.max():g}) = {TRUE_L:.4f}", GREEN)]
        for i, (a, b, col) in enumerate(rows):
            y = 330 + i * 80
            d.text((140, y), a, font=F_SMALL, fill=col)
            d.text((440, y), b, font=vk.font(22, mono=True), fill=TEXT)


def sc_exact(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "精确，不是近似")
    d.text((160, 190), "T = 512，d = 64，块 64 × 64：最大差别", font=F_BODY, fill=TEXT)
    d.text((900, 190), f"{DIFF:.1e}", font=vk.font(30, mono=True), fill=GREEN)
    if seg >= 1:
        for i, (t, col) in enumerate([("FlashAttention（Dao et al., 2022）：GPT-2 注意力最多快 7.6 倍", AMBER),
                                      ("显存：从 O(T²) 降到 O(T)", GREEN),
                                      ("FlashAttention-2（2023）· FlashAttention-3（2024，Hopper）", CYAN)]):
            vk.text_c(d, (640, 330 + i * 70), t, F_SMALL, col)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("瓶颈", "T × T 中间矩阵的反复读写"), ("分块", "在线 softmax 合并，结果精确"),
                      ("本质", "减少显存读写，不减计算")], seg, "下一期：量化")


RENDER = {"title": sc_title, "naive": sc_naive, "tiling": sc_tiling, "online": sc_online, "exact": sc_exact, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "inf-flash")
