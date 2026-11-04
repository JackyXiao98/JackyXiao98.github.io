"""Video LoRA 08 — 指令编辑：把生成模型变成编辑器.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] VL_CACHE=<dir> python vl08_instruct.py <out_dir>

Real computations (vlx.instruct_lora, JAX): one in-context LoRA (rank 16, 135,168 parameters) plus a learned
instruction embedding, trained 2,500 steps (batch 16, the four edits mixed uniformly) on (source, instruction,
target) triplets: recolour (red -> blue, green -> yellow, blue -> red, yellow -> green), change shape (square ->
disk -> cross -> square), play backwards, remove the object. 32 held-out sources per instruction: success =
recolour correct / shape correct (IoU template match) / trajectory equals the reversed source (< 1 px) / no
object pixels left. Compared with the single-task recolour IC-LoRA of episode 07 (1,500 steps).
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import vlx
import vlviz
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Video LoRA Edit"
CACHE = os.environ.get("VL_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "vlcache"))

IN = vlx.cached(CACHE, "inst", vlx.instruct_lora, cache_dir=CACHE)
IC = vlx.cached(CACHE, "ic", vlx.ic_lora, cache_dir=CACHE)
SU = {k: IN["per_task"][k]["success"] for k in range(4)}
print(SU, IN["params"])
assert SU == {0: 0.5, 1: 0.1875, 2: 0.0625, 3: 0.9375} and IN["params"] == 135168 and IC["final"]["success"] == 1.0
NAMES = vlx.INSTR
COLS = [CYAN, VIOLET, AMBER, GREEN]

SCENES = [
    ("title", [
        "上一期的上下文 LoRA，一个 LoRA 只学一种编辑。更理想的是一个模型，你用一句话告诉它怎么改，它就怎么改。这一期看指令式的视频编辑。",
    ]),
    ("idea", [
        "做法是在上下文 LoRA 的基础上，再加一个编辑指令作为条件。训练数据变成三元组，原视频、编辑指令、编辑后的视频。模型要学会读懂指令，再对参考视频做相应的修改。",
        "难点在数据。真实世界里很少有成对的编辑前后的视频，所以这些三元组几乎都是合成的。",
    ]),
    ("ditto", [
        "二零二五年的一项工作，构建了超过一百万条视频编辑三元组的合成数据集，生成数据花费了超过一万二千个显卡日。",
        "它以开源的一百四十亿参数文生视频模型，和支持视频参考条件的全能视频编辑模型为基础，只微调上下文模块里的线性投影层，并用一种课程学习，逐步从依赖参考图像过渡到只靠文字指令。",
    ]),
    ("toy", [
        f"我们在小模型上做一个缩小版。一个 LoRA 同时学四种指令，改颜色、改形状、倒放和移除物体，混在一起训练两千五百步。",
        f"移除物体最容易，成功率{pct_zh(SU[3])}。改颜色只有{pct_zh(SU[0])}，而上一期只学改颜色的 LoRA，一千五百步就做到了百分之百。改形状是{pct_zh(SU[1])}，倒放只有{pct_zh(SU[2])}。",
        "同样大小的 LoRA，要学的编辑越多、越难，需要的数据和训练就越多。改形状要重新画出物体，倒放要读懂整段运动再反过来，都比换颜色难得多。这也是真实系统需要百万级数据的原因。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，指令编辑把上下文 LoRA 和编辑指令结合，用原视频、指令、结果的三元组训练。",
        "第二，成对的编辑数据几乎都要合成，真实系统用到了百万级的三元组。",
        "第三，在我们的实验里，一个 LoRA 同时学四种编辑，难的编辑明显学得更慢。下一期，我们看多个 LoRA 能不能叠加使用。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 8, "指令编辑", "把生成模型变成编辑器", SERIES)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "三元组：原视频 + 指令 → 编辑后的视频")
    steps = [("原视频（参考词元）", CYAN), ("编辑指令", AMBER), ("编辑后的视频", GREEN)]
    for i, (t, col) in enumerate(steps):
        x = 110 + i * 370
        d.rounded_rectangle([x, 210, x + 320, 300], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 160, 255), t, F_SMALL, col)
    vk.text_c(d, (455, 255), "+", F_BODY, TEXT)
    vk.arrow(d, (800, 255), (845, 255), MUTED, 3)
    if seg >= 1:
        vk.text_c(d, (640, 420), "难点：成对的编辑数据几乎都要合成", F_BODY, RED)


def sc_ditto(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "Ditto / Editto（2025 · CVPR 2026）")
    d.rounded_rectangle([140, 170, 1140, 280], 14, outline=AMBER, width=2, fill=(12, 19, 34))
    vk.text_c(d, (640, 205), "Ditto-1M：超过 100 万条（原视频, 指令, 编辑后）三元组", F_BODY, AMBER)
    vk.text_c(d, (640, 250), "合成数据花费超过 12,000 个 GPU 日", F_SMALL, TEXT)
    if seg >= 1:
        for i, t in enumerate(["基础：Wan2.1-T2V-14B + VACE-14B", "只微调上下文模块的线性投影层（LoRA）", "课程学习：从参考图像逐步过渡到纯文字指令"]):
            vk.text_c(d, (640, 340 + i * 55), t, F_SMALL, [CYAN, GREEN, VIOLET][i])


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "一个 LoRA 学四种指令（2,500 步）")
    tv.bar_chart(d, (120, 210, 720, 520), NAMES, [SU[k] * 100 for k in range(4)], COLS, fmt="{:.0f}%", vmax=110)
    d.text((120, 165), "成功率（每种 32 个测试视频）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        d.text((780, 200), "对比：只学改颜色", font=F_SMALL, fill=MUTED)
        d.text((780, 240), f"1,500 步 → {IC['final']['success']:.0%}", font=F_BODY, fill=GREEN)
    k = 1 if seg >= 2 else 0
    s = IN["samples"][k]
    vlviz.film(c, s["src"][0], 860, 360, cell=34, label="原视频")
    vlviz.film(c, s["out"][0], 860, 405, cell=34, label=NAMES[k], color=COLS[k])
    vlviz.film(c, s["tgt"][0], 860, 450, cell=34, label="目标")
    if seg >= 2:
        vk.text_c(d, (640, 620), "要学的编辑越多、越难 → 需要的数据和训练越多", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 4, [("指令编辑", "上下文 LoRA + 编辑指令"), ("数据", "合成的百万级三元组"),
                      ("实验", "难的编辑学得更慢")], seg, "下一期：LoRA 的组合与合并")


RENDER = {"title": sc_title, "idea": sc_idea, "ditto": sc_ditto, "toy": sc_toy, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vl-instruct")
