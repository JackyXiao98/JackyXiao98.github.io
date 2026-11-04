"""Video LoRA 02 — 把 LoRA 装进视频扩散模型.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] VL_CACHE=<dir> python vl02_where.py <out_dir>

Computations: parameter counts of LoRA adapters for the public Wan2.1-14B transformer configuration
(hidden size 5,120, FFN 13,824, 40 blocks; each block: self-attention q/k/v/o, cross-attention q/k/v/o, a
2-layer FFN) - full weights vs rank-r adapters on all ten matrices; bf16 storage (2 bytes per parameter);
full fine-tuning memory with Adam (bf16 weights + bf16 gradients + fp32 master + 2 fp32 moments = 16 bytes per
parameter). And the toy video DiT of this series (vlx): parameters per LoRA group at rank 8.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import vlx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Video LoRA Edit"

D_, F_, L_ = 5120, 13824, 40
BLOCK = 8 * D_ * D_ + 2 * D_ * F_
FULL = BLOCK * L_
PER_R = L_ * (8 * (D_ + D_) + 2 * (D_ + F_))
R32 = PER_R * 32
print(BLOCK, FULL, PER_R, R32, R32 / FULL)
assert BLOCK == 351272960 and FULL == 14050918400 and PER_R == 4792320 and R32 == 153354240
assert abs(R32 / FULL - 0.0109) < 1e-4
GB = lambda n, b: n * b / 1e9
assert round(GB(FULL, 2)) == 28 and round(GB(R32, 2) * 1000) == 307 and round(GB(FULL, 16)) == 225


def toy_counts():
    import jax
    p = vlx.init(jax.random.PRNGKey(0))
    tot = int(sum(a.size for a in jax.tree_util.tree_leaves(p)))
    g = {k: vlx.lora_params(vlx.lora_init(jax.random.PRNGKey(0), 8, m)) for k, m in vlx.GROUPS.items()}
    return tot, g


TOY, TG = toy_counts()
print(TOY, TG)
assert TG["temporal"] + TG["spatial"] == TG["all"]

SCENES = [
    ("title", [
        "上一期我们知道了罗拉只训练两个小矩阵。这一期看在一个真实的视频扩散模型里，罗拉应该加在哪里，要多少参数，又要多少显存。",
    ]),
    ("block", [
        "今天主流的视频生成模型是扩散变换器。视频先被压缩成潜变量，再切成很多小块，每一块是一个词元。模型由几十个相同的块组成，每个块有自注意力、和文本交互的交叉注意力，以及一个前馈网络。",
        "罗拉通常加在注意力的查询、键、值和输出矩阵上，有时也加在前馈网络上。比如艾尔提艾克斯的官方训练器，推荐的配置就同时包括两种注意力和前馈层。",
    ]),
    ("count", [
        "我们用一个公开的一百四十亿参数视频模型的配置来算一算。它的隐藏维度是五千一百二十，前馈层的中间维度是一万三千八百二十四，一共四十个块。",
        f"每个块有十个大矩阵，四十个块加起来约一百四十亿个参数。如果在这十个矩阵上都加秩为三十二的罗拉，总共只有约{dec_zh(R32 / 1e8, 2)}亿个参数，大约是原模型的百分之一。",
    ]),
    ("memory", [
        "用十六位浮点数存储，整个模型要二十八吉字节，而一个这样的罗拉只要大约三百零七兆字节，很容易分享和切换。",
        "训练时的差别更大。全量微调用亚当，每个参数要存权重、梯度、三十二位的主权重和两个动量，大约十六个字节，一百四十亿参数就是二百二十五吉字节。罗拉只需要为很小一部分参数保存这些状态，冻结的模型可以只占二十八吉字节，再加上激活值，一张八十吉字节的显卡就能训练。",
    ]),
    ("st", [
        "还有一个问题，空间和时间。早期的视频模型把注意力分成两种，空间注意力只看同一帧里的词元，时间注意力只看同一位置在不同帧的词元。这样就可以只在时间层上加罗拉去学运动，只在空间层上加罗拉去学外观。",
        "而今天很多大模型使用三维的全注意力，每个词元同时看所有帧的所有位置，空间和时间就分不开了。本系列的小模型采用分开的注意力，下一期我们会用它来比较，只训练时间层和只训练空间层，分别学到了什么。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，视频扩散变换器里，罗拉通常加在注意力的四个矩阵上，也常加在前馈层上。",
        "第二，一百四十亿参数的模型，秩三十二的罗拉只有约一亿五千万参数，占百分之一，几百兆字节。",
        "第三，空间和时间分开的注意力，可以让罗拉分别学外观和运动。下一期，我们用一个视频训练一个罗拉。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 2, "把 LoRA 装进视频扩散模型", "加在哪里、多少参数、多少显存", SERIES)


def sc_block(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "视频扩散变换器（DiT）的一个块")
    items = [("自注意力", "q · k · v · o", CYAN, True), ("交叉注意力（文本）", "q · k · v · o", VIOLET, True), ("前馈网络", "两个线性层", GREEN, seg >= 1)]
    for i, (a, b, col, lo) in enumerate(items):
        y = 170 + i * 120
        d.rounded_rectangle([260, y, 760, y + 95], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (510, y + 32), a, F_BODY, col)
        vk.text_c(d, (510, y + 70), b, F_SMALL, TEXT)
        if seg >= 1 and lo:
            d.rounded_rectangle([800, y + 20, 980, y + 75], 10, outline=AMBER, width=2, fill=(12, 19, 34))
            vk.text_c(d, (890, y + 47), "+ LoRA", F_SMALL, AMBER)
    d.text((100, 200), "× 几十个块", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 560), "LTX 训练器的推荐配置：两种注意力 + 前馈层", F_SMALL, AMBER)


def sc_count(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "参数量：以 Wan2.1-14B 的公开配置为例")
    d.text((140, 160), "隐藏维度 5,120 · 前馈 13,824 · 40 个块", font=F_BODY, fill=TEXT)
    tv.fpaste(c, r"$\mathrm{block}=8\cdot 5120^2+2\cdot 5120\cdot 13824$", 640, 250, 30)
    if seg >= 1:
        rows = [("全部权重（40 个块）", f"{FULL:,}", MUTED), ("LoRA 每增加 1 个秩", f"{PER_R:,}", TEXT), ("LoRA 秩 32", f"{R32:,} ({R32 / FULL:.1%})", GREEN)]
        for i, (a, b, col) in enumerate(rows):
            d.text((200, 340 + i * 60), a, font=F_BODY, fill=col)
            d.text((1100, 340 + i * 60), b, font=vk.font(26, mono=True), fill=col, anchor="ra")


def sc_memory(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "存储与训练显存（bf16，Adam）")
    vals = [GB(FULL, 2), GB(R32, 2)]
    tv.bar_chart(d, (140, 220, 600, 540), ["整个模型", "LoRA 秩 32"], vals, [MUTED, GREEN], fmt="{:.1f} GB", vmax=34)
    d.text((140, 170), "存储（每参数 2 字节）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        tv.bar_chart(d, (700, 220, 1180, 540), ["全量微调", "LoRA（冻结权重）"], [GB(FULL, 16), GB(FULL, 2)], [RED, GREEN], fmt="{:.0f} GB", vmax=260)
        d.text((700, 170), "训练状态（不含激活值）", font=F_SMALL, fill=MUTED)
        vk.text_c(d, (640, 620), "全量：权重 + 梯度 + fp32 主权重 + 两个动量 ≈ 16 字节 / 参数", F_SMALL, TEXT)


def sc_st(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "空间与时间：分开的注意力 vs 三维全注意力")
    for f in range(4):
        for i in range(3):
            for j in range(3):
                x, y = 140 + f * 120 + j * 30, 220 + i * 30
                d.rectangle([x, y, x + 24, y + 24], outline=(51, 65, 85), width=1)
    d.rectangle([140 + 30, 250, 140 + 54, 274], fill=CYAN)
    for f in range(4):
        d.rectangle([140 + f * 120 + 30, 250, 140 + f * 120 + 54, 274], outline=AMBER, width=3)
    d.text((140, 170), "分开：空间注意力（同一帧）+ 时间注意力（同一位置跨帧）", font=F_SMALL, fill=TEXT)
    d.text((140, 340), "→ 只在时间层加 LoRA 学运动，只在空间层加 LoRA 学外观", font=F_SMALL, fill=GREEN)
    if seg >= 1:
        d.text((140, 430), "三维全注意力（如 Wan）：每个词元看所有帧的所有位置 → 空间时间分不开", font=F_SMALL, fill=AMBER)
        d.text((140, 490), f"本系列小模型：分开的注意力 · {TOY:,} 参数", font=F_SMALL, fill=MUTED)
        d.text((140, 525), f"秩 8：时间层 {TG['temporal']:,} · 空间层 + 前馈 {TG['spatial']:,} 参数", font=F_SMALL, fill=MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("位置", "注意力 q / k / v / o，常加前馈"), ("规模", f"秩 32：{R32 / 1e8:.2f} 亿，约 1%"),
                      ("空间 / 时间", "分开的注意力可分别学外观与运动")], seg, "下一期：一个视频一个 LoRA", label_w=470)


RENDER = {"title": sc_title, "block": sc_block, "count": sc_count, "memory": sc_memory, "st": sc_st, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vl-where")
