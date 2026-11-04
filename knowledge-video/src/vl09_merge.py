"""Video LoRA 09 — LoRA 的组合、合并与加速.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] VL_CACHE=<dir> python vl09_merge.py <out_dir>

Real computations (vlx.merge_loras, JAX): two adapters from earlier episodes - the light-background LoRA of
episode 01 (rank 8, all modules) and the diagonal-motion LoRA of episode 03 (rank 8, temporal attention) - used
alone or together (sum of the two updates, each scaled by 0.5 / 1 / 1.5). 48 videos per setting (random shapes,
colours blue / green / yellow, motion = none): mean background brightness (light domain about 0.84, base about
0.19), share moving diagonally, share whose colour follows the label.
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

MG = vlx.cached(CACHE, "merge", vlx.merge_loras, cache_dir=CACHE)
B, BG, MV, BO = MG["base"], MG["bg"], MG["motion"], MG["both"]
print({k: (round(v["bg"], 3), v["diag"]) for k, v in (("base", B), ("bg", BG), ("mv", MV))}, {k: (round(v["bg"], 3), v["diag"]) for k, v in BO.items()})
assert abs(B["bg"] - 0.188) < 1e-3 and abs(BG["bg"] - 0.842) < 1e-3 and abs(MV["diag"] - 0.125) < 1e-3 and abs(B["diag"] - 0.0208) < 1e-3
assert abs(BO[(1.0, 1.0)]["bg"] - 0.742) < 1e-3 and abs(BO[(1.0, 1.0)]["diag"] - 0.0417) < 1e-3
assert abs(BO[(1.5, 1.0)]["bg"] - 0.940) < 1e-3 and abs(BO[(1.5, 1.0)]["diag"] - 0.0208) < 1e-3
assert abs(BO[(1.0, 1.5)]["color_acc"] - 0.729) < 1e-3 and max(v["diag"] for v in BO.values()) <= MV["diag"]

SCENES = [
    ("title", [
        "前面我们训练了好几个罗拉，一个改背景，一个学运动。实际使用时，人们常常想把多个罗拉叠在一起，比如一个风格加一个运动。这一期看叠加时会发生什么，以及怎样用罗拉来加速。",
    ]),
    ("sum", [
        "把两个罗拉叠加，最简单的方法是把它们各自的改变量直接相加，每个再乘上一个缩放系数，控制它的强度。因为每个罗拉都很小，叠加和切换都很方便。",
    ]),
    ("experiment", [
        f"我们拿第一期的浅色背景罗拉和第三期的斜向运动罗拉。单独使用时，背景罗拉把背景亮度从{dec_zh(B['bg'], 2)}提高到{dec_zh(BG['bg'], 2)}；运动罗拉让{pct_zh(MV['diag'], 1)}的视频斜着走，而原模型只有{pct_zh(B['diag'])}。",
        f"两个直接相加，背景亮度降到{dec_zh(BO[(1.0, 1.0)]['bg'], 2)}，斜着走的视频只剩{pct_zh(BO[(1.0, 1.0)]['diag'])}，两个效果都被削弱了，运动几乎消失。",
        f"把背景的系数调到一点五，背景亮度回到{dec_zh(BO[(1.5, 1.0)]['bg'], 2)}，但运动依然没有回来。把运动的系数调到一点五，颜色听描述的比例反而降到{pct_zh(BO[(1.0, 1.5)]['color_acc'])}。调系数只能在两者之间来回取舍。",
    ]),
    ("why", [
        "原因是两个罗拉是分别训练的，各自都假设其他部分不变。叠在一起时，它们改动了同一批权重，互相干扰。我们的运动罗拉本来就很弱，很容易被背景罗拉盖掉。",
        "研究者提出了多种合并方法，比如在合并时让两组改变量尽量不重叠的方向，或者在多个罗拉一起出现的数据上做一点联合训练。在图像领域，就有专门把主体罗拉和风格罗拉合并的方法。",
    ]),
    ("speed", [
        "罗拉还有一个重要用途，加速。视频扩散模型通常要采样几十步，蒸馏可以把它压缩到几步。如果把蒸馏学到的改变量也做成一个罗拉，就能像插件一样加到原模型、甚至加到其他微调过的模型上。",
        "图像领域的潜在一致性模型罗拉就是这样一个通用的加速模块。视频模型社区也有类似的少步罗拉，它们和编辑罗拉叠加使用时，同样会遇到刚才看到的干扰问题。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，多个罗拉可以把改变量相加，用缩放系数控制强度。",
        "第二，在我们的实验里，背景和运动两个罗拉叠加后互相削弱，调系数只能取舍。",
        "第三，蒸馏也可以做成罗拉，作为通用的加速插件。最后一期，我们看怎样评估视频编辑的好坏。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 9, "LoRA 的组合、合并与加速", "风格 + 运动 = ？", SERIES)


def sc_sum(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "叠加：改变量相加")
    tv.fpaste(c, r"$W'=W+\lambda_1\,B_1A_1+\lambda_2\,B_2A_2$", 640, 280, 44)
    vk.text_c(d, (640, 380), "λ：每个 LoRA 的缩放系数（强度）", F_BODY, AMBER)


def sc_experiment(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "浅色背景 LoRA + 斜向运动 LoRA（48 个视频）")
    rows = [("原模型", B, MUTED), ("只用背景 LoRA", BG, CYAN), ("只用运动 LoRA", MV, VIOLET)]
    if seg >= 1:
        rows.append(("两者相加（1, 1）", BO[(1.0, 1.0)], RED))
    if seg >= 2:
        rows += [("背景 1.5 + 运动 1", BO[(1.5, 1.0)], AMBER), ("背景 1 + 运动 1.5", BO[(1.0, 1.5)], AMBER)]
    for j, h in enumerate(["", "背景亮度", "斜着走", "颜色听描述"]):
        d.text((140 + [0, 420, 640, 860][j], 160), h, font=F_SMALL, fill=MUTED)
    for i, (a, m, col) in enumerate(rows):
        y = 205 + i * 52
        d.text((140, y), a, font=F_SMALL, fill=col)
        f = vk.font(22, mono=True)
        d.text((560, y), f"{m['bg']:.2f}", font=f, fill=TEXT)
        d.text((780, y), f"{m['diag']:.1%}", font=f, fill=TEXT)
        d.text((1000, y), f"{m['color_acc']:.0%}", font=f, fill=TEXT)
    if seg >= 1:
        vlviz.film(c, BO[(1.0, 1.0)]["samples"][0], 380, 560, cell=48, label="两者相加", color=RED)


def sc_why(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "为什么会互相干扰？")
    vk.text_c(d, (640, 190), "分别训练 · 都假设其他部分不变 · 改的是同一批权重", F_BODY, TEXT)
    if seg >= 1:
        for i, t in enumerate(["合并时让两组改变量尽量不重叠（如 ZipLoRA，2023，主体 + 风格）", "在多个概念同时出现的数据上做联合训练"]):
            vk.text_c(d, (640, 320 + i * 60), t, F_SMALL, GREEN)


def sc_speed(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "把加速也做成 LoRA")
    for i, (t, col) in enumerate([("几十步采样", RED), ("蒸馏", AMBER), ("几步采样的 LoRA 插件", GREEN)]):
        x = 110 + i * 370
        d.rounded_rectangle([x, 200, x + 320, 290], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 160, 245), t, F_SMALL, col)
        if i < 2:
            vk.arrow(d, (x + 325, 245), (x + 365, 245), MUTED, 3)
    if seg >= 1:
        vk.text_c(d, (640, 400), "图像：LCM-LoRA（Luo et al., 2023）通用加速模块 · 视频社区也有少步 LoRA", F_SMALL, TEXT)
        vk.text_c(d, (640, 460), "与编辑 LoRA 叠加时同样会互相干扰", F_SMALL, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("叠加", "改变量相加，系数控制强度"), ("干扰", "背景 + 运动互相削弱"),
                      ("加速", "蒸馏做成 LoRA 插件")], seg, "最后一期：评估与局限")


RENDER = {"title": sc_title, "sum": sc_sum, "experiment": sc_experiment, "why": sc_why, "speed": sc_speed, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vl-merge")
