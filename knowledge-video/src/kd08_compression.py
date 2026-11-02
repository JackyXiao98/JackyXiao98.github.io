"""Distillation 08 — 蒸馏、剪枝与量化：模型压缩全景.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] KD_CACHE=<dir> python kd08_compression.py <out_dir>

Real computations (kdx.compression, JAX + numpy): the episode-01 teacher [64, 512, 512, 10];
global magnitude pruning of its weights (50-98%) without retraining; symmetric per-output-channel
weight quantisation to 8/4/3/2 bits without retraining; a distilled student [64, 32, 10]
(soft labels T = 4, alpha = 0.1, all 1,197 training images) and its 8/4-bit quantised versions;
test accuracy on 597 images. Sizes: fp32 = 4 bytes per parameter; the int4 student stores weights
at 0.5 bytes and biases in fp32.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import kdx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Distillation"
CACHE = os.environ.get("KD_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "kdcache"))

C = kdx.cached(CACHE, "comp", kdx.compression)
PR, QU = C["prune"], C["quant"]
T_BYTES = C["n_params"] * 4
S_W = 64 * 32 + 32 * 10
S_BYTES = S_W * 0.5 + (C["student_params"] - S_W) * 4
RATIO = T_BYTES / S_BYTES
print(C, T_BYTES, S_BYTES, RATIO)
assert C["n_params"] == 301066 and C["student_params"] == 2410 and S_BYTES == 1352 and round(RATIO, -1) == 890
assert [round(PR[f] * 100, 1) for f in (0.5, 0.8, 0.9, 0.95, 0.98)] == [97.0, 94.0, 82.6, 58.1, 13.6]
assert [round(QU[b] * 100, 1) for b in (8, 4, 3, 2)] == [97.2, 97.2, 96.5, 86.8]
assert round(C["student"] * 100, 1) == 96.3 and round(C["student_int4"] * 100, 1) == 96.8

SCENES = [
    ("title", [
        "蒸馏是压缩模型的一种方法，但不是唯一的一种。这一期把蒸馏、剪枝和量化放在一起比较，看它们各自省掉了什么，又怎样组合使用。",
    ]),
    ("three", [
        "剪枝是把不重要的权重直接置零或者删掉，模型的结构基本不变。量化是用更少的比特来存储每个权重，比如从三十二位浮点数降到八位或四位整数。",
        "蒸馏则是换一个更小的结构，重新训练，让它模仿老师。前两种是在原模型上做减法，蒸馏是重新造一个小模型。",
    ]),
    ("prune", [
        f"我们拿第一期的老师来做实验，它有{big_zh(C['n_params'])}个参数，准确率{pct_zh(C['teacher'], 1)}。先按绝对值大小剪掉最小的权重，不做任何重新训练。",
        f"剪掉一半，准确率还有{pct_zh(PR[0.5], 1)}；剪掉百分之八十，是{pct_zh(PR[0.8], 1)}。剪掉百分之九十，降到{pct_zh(PR[0.9], 1)}；剪掉百分之九十八，只剩{pct_zh(PR[0.98], 1)}，接近随便猜。",
    ]),
    ("quant", [
        f"再看量化。每个输出通道单独选一个缩放系数，同样不做重新训练。八位和四位整数的准确率都和原来一样，是{pct_zh(QU[4], 1)}。",
        f"三位是{pct_zh(QU[3], 1)}，二位降到{pct_zh(QU[2], 1)}。四位就把存储降到了原来的八分之一，几乎没有损失。",
    ]),
    ("distill", [
        f"最后是蒸馏。一个只有{big_zh(C['student_params'])}个参数的学生，用老师的软标签训练，准确率{pct_zh(C['student'], 1)}，参数少了一百多倍。",
        f"蒸馏和量化还可以叠加。把这个学生再量化到四位，准确率是{pct_zh(C['student_int4'], 1)}，没有下降。老师用三十二位浮点数需要大约一点二兆字节，学生只要{big_zh(S_BYTES)}个字节，小了约{int_zh(round(RATIO, -1))}倍。",
    ]),
    ("llm", [
        "在大模型里，这几种方法也常常组合使用。蒸馏伯特把伯特缩小了百分之四十，保留了约百分之九十七的语言理解能力。英伟达的方法先剪掉大模型的层和宽度，再用蒸馏把能力恢复回来，比从头训练省下大量数据。",
        "量化方面，四位权重的大模型已经很常见，可以在一张消费级显卡上运行原本放不下的模型。实践中的顺序常常是，先蒸馏或剪枝得到合适的结构，最后再量化部署。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，剪枝删掉权重，量化减少比特，蒸馏换一个小结构重新训练。",
        "第二，在我们的实验里，不重新训练时，剪枝到九成就明显掉点，而四位量化几乎无损。",
        "第三，蒸馏和量化可以叠加，学生加上四位量化，存储小了约八百九十倍，准确率几乎不变。最后一期，我们看蒸馏的边界。感谢大家的观看。",
    ]),
]
assert round(T_BYTES / 1e6, 1) == 1.2 and round(C["n_params"] / C["student_params"]) > 100


def sc_title(c, u, seg, su):
    tv.title_card(c, 8, "蒸馏、剪枝与量化", "模型压缩全景", SERIES)


def sc_three(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "三种压缩方法")
    items = [("剪枝", "权重置零 / 删除", "结构基本不变", AMBER), ("量化", "32 位浮点 → 8 / 4 位整数", "每个权重更少比特", CYAN),
             ("蒸馏", "更小的结构，重新训练", "模仿老师", GREEN)]
    for i, (a, b, e, col) in enumerate(items):
        if i < 2 or seg >= 1:
            x = 100 + i * 370
            d.rounded_rectangle([x, 190, x + 340, 400], 16, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 170, 240), a, F_BODY, col)
            vk.text_c(d, (x + 170, 305), b, F_SMALL, TEXT)
            vk.text_c(d, (x + 170, 350), e, F_SMALL, MUTED)
    if seg >= 1:
        vk.text_c(d, (455, 470), "在原模型上做减法", F_SMALL, AMBER)
        vk.text_c(d, (1010, 470), "重新造一个小模型", F_SMALL, GREEN)


def sc_prune(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, f"剪枝（按绝对值，不重新训练）· 老师 {C['n_params']:,} 参数")
    fs = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98]
    vals = [C["teacher"] * 100] + [PR[f] * 100 for f in fs[1:]]
    n = 1 if seg == 0 else len(fs)
    labs = ["原模型"] + [f"剪 {f:.0%}" for f in fs[1:]]
    cols = [VIOLET] + [GREEN, GREEN, AMBER, RED, RED]
    tv.bar_chart(d, (120, 200, 120 + 175 * n, 560), labs[:n], vals[:n], cols[:n], fmt="{:.1f}%", vmax=110)


def sc_quant(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "量化（每个输出通道一个缩放系数，不重新训练）")
    bs = [32, 8, 4, 3, 2]
    vals = [C["teacher"] * 100] + [QU[b] * 100 for b in bs[1:]]
    n = 3 if seg == 0 else 5
    cols = [VIOLET, GREEN, GREEN, AMBER, RED]
    tv.bar_chart(d, (140, 200, 140 + 200 * n, 560), [f"{b} 位" for b in bs][:n], vals[:n], cols[:n], fmt="{:.1f}%", vmax=110)
    tv.fpaste(c, r"$\hat w = s\cdot\mathrm{round}(w/s),\ \ s=\max|w|/(2^{b-1}-1)$", 640, 640, 26)


def sc_distill(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "蒸馏，再叠加量化")
    rows = [("老师 fp32", C["n_params"], C["teacher"], T_BYTES, VIOLET), ("学生 fp32", C["student_params"], C["student"], C["student_params"] * 4, CYAN)]
    if seg >= 1:
        rows.append(("学生 int4", C["student_params"], C["student_int4"], S_BYTES, GREEN))
    for j, h in enumerate(["", "参数", "准确率", "存储"]):
        d.text((180 + [0, 260, 500, 720][j], 170), h, font=F_SMALL, fill=MUTED)
    for i, (a, n, acc, by, col) in enumerate(rows):
        y = 230 + i * 80
        d.rounded_rectangle([150, y - 10, 1130, y + 55], 12, outline=col, width=2, fill=(12, 19, 34))
        d.text((180, y + 22), a, font=F_BODY, fill=col, anchor="lm")
        d.text((440, y + 22), f"{n:,}", font=vk.font(26, mono=True), fill=TEXT, anchor="lm")
        d.text((680, y + 22), f"{acc:.1%}", font=vk.font(26, mono=True), fill=TEXT, anchor="lm")
        d.text((900, y + 22), f"{by:,.0f} B", font=vk.font(26, mono=True), fill=TEXT, anchor="lm")
    if seg >= 1:
        vk.text_c(d, (640, 520), f"存储：约 1.2 MB → {S_BYTES:,.0f} 字节（小约 {round(RATIO, -1):.0f} 倍）", F_BODY, GREEN)
        vk.text_c(d, (640, 570), "int4 学生：权重 0.5 字节，偏置仍为 fp32", F_SMALL, MUTED)


def sc_llm(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "大模型中的组合")
    items = [("DistilBERT（Sanh et al., 2019）", "小 40%，保留约 97% 的语言理解能力", CYAN),
             ("Minitron（NVIDIA, 2024）", "剪枝层与宽度 → 蒸馏恢复，省下大量训练数据", VIOLET)]
    if seg >= 1:
        items.append(("GPTQ / AWQ 等", "4 位权重，消费级显卡上运行大模型", GREEN))
    for i, (a, b, col) in enumerate(items):
        y = 160 + i * 110
        d.rounded_rectangle([100, y, 1180, y + 85], 14, outline=col, width=2, fill=(12, 19, 34))
        d.text((130, y + 42), a, font=F_BODY, fill=col, anchor="lm")
        d.text((620, y + 42), b, font=F_SMALL, fill=TEXT, anchor="lm")
    if seg >= 1:
        vk.text_c(d, (640, 560), "常见顺序：蒸馏 / 剪枝 得到结构 → 最后量化部署", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 6, [("三种方法", "剪枝 · 量化 · 蒸馏"), ("实验", "剪 90% 明显掉点；4 位几乎无损"),
                      ("叠加", f"学生 + int4：小约 {round(RATIO, -1):.0f} 倍")], seg, "最后一期：蒸馏的边界")


RENDER = {"title": sc_title, "three": sc_three, "prune": sc_prune, "quant": sc_quant, "distill": sc_distill, "llm": sc_llm, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "kd-compression")
