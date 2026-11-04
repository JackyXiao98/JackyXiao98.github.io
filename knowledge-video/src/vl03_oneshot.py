"""Video LoRA 03 — 一个视频一个 LoRA：单样本微调与运动定制.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] VL_CACHE=<dir> python vl03_oneshot.py <out_dir>

Real computations (vlx.motion_lora, JAX): one reference clip - a red square moving diagonally (+1, +1 px per
frame), a motion the base model never saw. LoRA rank 8 trained 1,000 steps (Adam 2e-3) on that single clip with
labels (square, red, motion = none), attached to: temporal attention only (18,432 parameters), spatial attention
+ MLP (49,152) or everything (67,584). Then 32 videos are generated with other shapes and colours (blue / green /
yellow) and motion = none: share moving diagonally (mean step > 0.4 px in both x and y), share whose colour
follows the label, share that copied the reference red.
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

MO = vlx.cached(CACHE, "motion", vlx.motion_lora, cache_dir=CACHE)
B, RU, PA = MO["base"], MO["runs"], MO["params"]
print(B, {k: {m: v[m] for m in ("diag", "color_acc", "copy")} for k, v in RU.items()}, PA)
assert abs(B["diag"] - 0.0625) < 1e-4 and B["color_acc"] == 1.0
assert abs(RU["temporal"]["diag"] - 0.1875) < 1e-4 and RU["temporal"]["color_acc"] == 1.0 and RU["temporal"]["copy"] == 0.0
assert abs(RU["spatial"]["diag"] - 0.25) < 1e-4 and abs(RU["spatial"]["copy"] - 0.5625) < 1e-4
assert abs(RU["all"]["diag"] - 0.375) < 1e-4 and RU["all"]["color_acc"] == 0.0 and abs(RU["all"]["copy"] - 0.8125) < 1e-4
assert PA == {"temporal": 18432, "spatial": 49152, "all": 67584}
NAMES = {"temporal": "只加在时间层", "spatial": "空间层 + 前馈", "all": "全部层"}
COLS = {"temporal": GREEN, "spatial": VIOLET, "all": RED}

SCENES = [
    ("title", [
        "很多时候，我们手上只有一个视频，希望模型学会其中的某种东西，比如一种特别的运动，再用到别的物体上。这一期看单个视频的 LoRA 微调。",
    ]),
    ("history", [
        "二零二三年的一项工作，只用一个视频就把图像生成模型微调成视频模型，生成和原视频结构相似的新视频。随后的运动导演方法指出一个问题，在一个视频上微调，模型会把外观和运动一起记住。",
        "它的办法是双路的 LoRA，空间层的 LoRA 学外观，时间层的 LoRA 学运动，用的时候只保留时间层的那一路，就能把运动迁移到新的物体上。",
    ]),
    ("setup", [
        "我们在小模型上做这个实验。参考视频是一个红色方块沿对角线移动，这种斜向运动，模型训练时从来没见过。",
        "我们用这一个视频训练秩为八的 LoRA，分别只加在时间注意力上、只加在空间注意力和前馈层上，或者全部都加。然后让它生成蓝色、绿色、黄色的各种形状，看两件事，物体有没有斜着走，颜色有没有听描述的话。",
    ]),
    ("result", [
        f"不加 LoRA 时，只有{pct_zh(B['diag'])}的视频碰巧斜着走。全部层都加 LoRA，斜着走的比例升到{pct_zh(RU['all']['diag'])}，但颜色全都不听描述了，{pct_zh(RU['all']['copy'])}的视频直接画成了参考视频里的红色。模型在复读原视频。",
        f"只加在空间层和前馈层，斜着走的有{pct_zh(RU['spatial']['diag'])}，仍有{pct_zh(RU['spatial']['copy'])}复读了红色。",
        f"只加在时间层，参数最少，只有{big_zh(PA['temporal'])}个，颜色百分之百听描述，没有一个复读红色，但斜着走的比例只有{pct_zh(RU['temporal']['diag'])}。",
    ]),
    ("tradeoff", [
        "这就是单视频微调的核心矛盾。学得越多，越容易把外观一起记住；限制在时间层，外观保住了，运动却学得不够。我们的小模型很弱，运动只学到一部分，但趋势和运动导演论文的动机完全一致。",
        "实际的方法会做更多设计，比如训练时让空间层的 LoRA 专门吸收外观、推理时丢掉，或者在损失里去掉外观相关的部分。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，单个视频的 LoRA 可以学到新的运动，但很容易连外观一起记住。",
        "第二，在我们的实验里，全部层的 LoRA 有八成复读了参考的颜色；只训练时间层，外观完全可控，但运动学得较弱。",
        "第三，分开空间和时间的 LoRA，是解决这个矛盾的常用思路。下一期，我们看不训练也能编辑的反演方法。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 3, "一个视频一个 LoRA", "单样本微调与运动定制", SERIES)


def sc_history(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "Tune-A-Video（2023）→ MotionDirector（2023）")
    vk.text_c(d, (640, 180), "一个视频微调 → 外观和运动一起被记住", F_BODY, RED)
    if seg >= 1:
        for i, (a, b, col) in enumerate([("空间层 LoRA", "学外观 · 推理时丢掉", VIOLET), ("时间层 LoRA", "学运动 · 推理时保留", GREEN)]):
            x = 160 + i * 500
            d.rounded_rectangle([x, 280, x + 460, 400], 14, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 230, 320), a, F_BODY, col)
            vk.text_c(d, (x + 230, 365), b, F_SMALL, TEXT)
        vk.text_c(d, (640, 480), "双路 LoRA：把运动迁移到新的物体上", F_SMALL, AMBER)


def sc_setup(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "参考视频：模型没见过的斜向运动")
    vlviz.film(c, MO["ref"], 300, 180, cell=72, label="参考")
    if seg >= 1:
        for i, (k, n) in enumerate(NAMES.items()):
            d.text((300 + i * 300, 320), n, font=F_BODY, fill=COLS[k])
            d.text((300 + i * 300, 365), f"{PA[k]:,} 参数（秩 8）", font=F_SMALL, fill=MUTED)
        vk.text_c(d, (640, 470), "再生成蓝 / 绿 / 黄的各种形状：斜着走了吗？颜色听描述吗？", F_BODY, TEXT)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "32 个生成视频")
    keys = [["all"], ["all", "spatial"], ["all", "spatial", "temporal"]][min(seg, 2)]
    rows = [("不加 LoRA", B, MUTED)] + [(NAMES[k], RU[k], COLS[k]) for k in keys]
    for j, h in enumerate(["", "斜着走", "颜色听描述", "复读红色"]):
        d.text((160 + [0, 380, 580, 820][j], 170), h, font=F_SMALL, fill=MUTED)
    for i, (a, m, col) in enumerate(rows):
        y = 220 + i * 62
        d.text((160, y), a, font=F_BODY, fill=col)
        f = vk.font(24, mono=True)
        d.text((540, y), f"{m['diag']:.0%}", font=f, fill=TEXT)
        d.text((740, y), f"{m['color_acc']:.0%}", font=f, fill=TEXT)
        d.text((980, y), f"{m['copy']:.0%}", font=f, fill=RED if m["copy"] > 0.5 else TEXT)
    k = keys[-1]
    vlviz.film(c, RU[k]["samples"][0], 400, 520, cell=44, label=f"{NAMES[k]}：样本", color=COLS[k])


def sc_tradeoff(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "学得越多，越容易记住外观")
    xs = [RU[k]["diag"] * 100 for k in ("temporal", "spatial", "all")]
    ys = [RU[k]["copy"] * 100 for k in ("temporal", "spatial", "all")]
    x0, y0, x1, y1 = 200, 170, 800, 560
    d.rectangle([x0, y0, x1, y1], outline=(30, 41, 59))
    for k, x, y in zip(("temporal", "spatial", "all"), xs, ys):
        px, py = x0 + x / 50 * (x1 - x0), y1 - y / 100 * (y1 - y0)
        d.ellipse([px - 9, py - 9, px + 9, py + 9], fill=COLS[k])
        d.text((px + 14, py), NAMES[k], font=F_SMALL, fill=COLS[k], anchor="lm")
    d.text((x1, y1 + 30), "学到斜向运动的比例 →", font=F_SMALL, fill=MUTED, anchor="rm")
    d.text((x0 + 8, y0 + 8), "复读参考外观的比例 ↑", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        d.text((860, 250), "实际方法：", font=F_BODY, fill=AMBER)
        d.text((860, 300), "空间 LoRA 吸收外观、推理时丢掉", font=F_SMALL, fill=TEXT)
        d.text((860, 340), "损失中去掉外观相关部分", font=F_SMALL, fill=TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("单视频 LoRA", "能学运动，也会记住外观"), ("实验", f"全部层复读 {RU['all']['copy']:.0%} · 时间层 0%"),
                      ("思路", "分开空间与时间的 LoRA")], seg, "下一期：反演与注意力注入")


RENDER = {"title": sc_title, "history": sc_history, "setup": sc_setup, "result": sc_result, "tradeoff": sc_tradeoff, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vl-oneshot")
