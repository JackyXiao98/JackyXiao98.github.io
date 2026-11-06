"""Long Context 03 — 长度外推：训练短，测试长.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] LC_CACHE=<dir> SHAKESPEARE=<file> python lc03_extrap.py <out_dir>

Real computations (lcx.exp_extrap, JAX): five 2-layer character-level Transformers on Tiny Shakespeare that differ
only in the position scheme (learned absolute, sinusoidal, RoPE, ALiBi, none), each trained 2,000 steps on windows
of 64 characters, then evaluated on 64 validation windows of 512 characters: mean next-character loss at each
position; summaries over positions 32-63 (inside the training length), 64-79, 80-127 and 128-510.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import lcx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Long Context"
CACHE = os.environ.get("LC_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "lccache"))

R = lcx.cached(CACHE, "extrap", lcx.exp_extrap)
SEG = {k: {n: float(R[k]["curve"][a:b].mean()) for n, (a, b) in dict(inside=(32, 64), near=(64, 80), mid=(80, 128), far=(128, 511)).items()} for k in R}
NAMES = {"learned": "可学习绝对位置", "sin": "正弦绝对位置", "rope": "RoPE", "alibi": "ALiBi", "nope": "无位置编码"}
SHORT = {"learned": "可学习", "sin": "正弦", "rope": "RoPE", "alibi": "ALiBi", "nope": "无"}
smooth = lambda y, w=16: np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")
COLS = {"learned": VIOLET, "sin": AMBER, "rope": CYAN, "alibi": GREEN, "nope": MUTED}
print({k: {n: round(v, 3) for n, v in s.items()} for k, s in SEG.items()})
assert all(1.68 < SEG[k]["inside"] < 1.75 for k in ("learned", "sin", "rope", "alibi"))
assert round(SEG["learned"]["far"], 1) == 3.8 and round(SEG["rope"]["near"], 2) == 1.69 and round(SEG["rope"]["far"], 1) == 3.6
assert round(SEG["alibi"]["far"], 2) == 1.70 and round(SEG["nope"]["inside"], 2) == 2.19

SCENES = [
    ("title", [
        "如果一个模型只在短文本上训练过，拿去处理长文本会怎样？这个问题叫长度外推。这一期我们在小模型上直接测：训练长度六十四，测试长度五百一十二。",
    ]),
    ("setup", [
        "我们训练五个只在位置编码上不同的小模型：可学习的绝对位置、正弦绝对位置、旋转位置编码、下一期要讲的线性偏置，以及完全不用位置编码。每个都只在六十四个字符的片段上训练。然后在五百一十二个字符的片段上，看每个位置的预测损失。",
    ]),
    ("inside", [
        f"在训练长度之内，前四种模型的表现差不多，损失都在一点七左右。不用位置编码的模型明显差一些，约{dec_zh(SEG['nope']['inside'], 2)}。",
    ]),
    ("beyond", [
        f"一旦超过训练长度，差别就出来了。可学习的绝对位置和正弦位置立刻崩溃，损失跳到三点五到四之间，比完全不看上下文还差。",
        f"旋转位置编码多撑了一小段，第六十四到八十个位置损失还是{dec_zh(SEG['rope']['near'], 2)}，但再往后也崩溃了，平均约{dec_zh(SEG['rope']['far'], 1)}。原因就是上一期说的：远处的距离对应着训练中从没见过的旋转角度。",
        f"只有线性偏置的模型几乎不受影响，一直到第五百个位置，损失都保持在{dec_zh(SEG['alibi']['far'], 2)}。",
    ]),
    ("lesson", [
        "所以，训练长度之外的表现，几乎完全由位置编码决定。真实的大模型大多使用旋转位置编码，它在训练长度内效果很好，但不能直接外推。要让它处理更长的文本，需要专门的扩展方法，这是第五期的内容。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，绝对位置编码在训练长度之外立刻崩溃。",
        "第二，旋转位置编码能多撑一点，但很快也会失效。",
        "第三，线性偏置在我们的实验里外推得很好。下一期，我们看它是怎么做到的，又付出了什么代价。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 3, "长度外推", "训练短，测试长", SERIES)


def sc_setup(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "五个模型：只改位置编码（训练 64 个字符 → 测试 512 个）")
    for i, k in enumerate(R):
        d.rounded_rectangle([200, 190 + i * 75, 700, 245 + i * 75], 10, outline=COLS[k], width=2, fill=(12, 19, 34))
        d.text((230, 217 + i * 75), NAMES[k], font=F_BODY, fill=COLS[k], anchor="lm")
    d.rectangle([800, 300, 800 + 64 * 0.7, 340], fill=GREEN)
    d.rectangle([800, 380, 800 + 512 * 0.7, 420], fill=AMBER)
    d.text((800, 270), "训练 64", font=F_SMALL, fill=GREEN)
    d.text((800, 350), "测试 512", font=F_SMALL, fill=AMBER)


def curves(d, keys, box=(120, 190, 860, 590)):
    xs = np.arange(len(R["rope"]["curve"]))
    ser = [(NAMES[k], np.minimum(smooth(R[k]["curve"]), 4.4), COLS[k]) for k in keys]
    tv.line_chart(d, box, xs, ser, ylim=(1.4, 4.5), xticks=[(0, "0"), (64, "64"), (128, "128"), (256, "256"), (511, "511")], xlabel="位置", legend_xy=(900, 210))
    x64 = box[0] + 64 / 511 * (box[2] - box[0])
    d.line([(x64, box[1]), (x64, box[3])], fill=GREEN, width=2)
    d.text((x64 + 6, box[1] + 6), "训练长度", font=F_SMALL, fill=GREEN)
    d.text((box[0], box[1] - 32), "每个位置的损失（16 个位置平滑）", font=F_SMALL, fill=MUTED)


def sc_inside(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "训练长度之内（位置 32–63）")
    ks = list(R)
    tv.bar_chart(d, (160, 220, 900, 560), [SHORT[k] for k in ks], [SEG[k]["inside"] for k in ks], [COLS[k] for k in ks], fmt="{:.2f}", vmax=2.5)
    d.text((160, 180), "平均损失", font=F_SMALL, fill=MUTED)


def sc_beyond(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "超过训练长度")
    keys = [["learned", "sin"], ["learned", "sin", "rope"], ["learned", "sin", "rope", "alibi", "nope"]][seg]
    curves(d, keys)


def sc_lesson(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "位置编码决定外推")
    ks = list(R)
    f = vk.font(22, mono=True)
    d.text((300, 190), "位置 32–63   64–79   80–127   128–510", font=F_SMALL, fill=MUTED)
    for i, k in enumerate(ks):
        y = 240 + i * 55
        d.text((120, y), NAMES[k], font=F_SMALL, fill=COLS[k])
        d.text((330, y), "     ".join(f"{SEG[k][n]:.2f}" for n in ("inside", "near", "mid", "far")), font=f, fill=TEXT)
    vk.text_c(d, (640, 560), "大模型多用 RoPE：训练长度内很好，不能直接外推 → 第 5 期", F_SMALL, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("绝对位置", "超出训练长度立刻崩溃"), ("RoPE", "多撑一点，很快失效"),
                      ("ALiBi", "外推得很好")], seg, "下一期：ALiBi")


RENDER = {"title": sc_title, "setup": sc_setup, "inside": sc_inside, "beyond": sc_beyond, "lesson": sc_lesson, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "lc-extrap")
