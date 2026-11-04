"""Video LoRA 10 — 评估与局限：一致性、保真与编辑强度.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] VL_CACHE=<dir> python vl10_eval.py <out_dir>

Real computations: the metrics of this series gathered for every editing method on toy colour edits -
inversion (episode 04, strength 0.2-1.0), regenerating from noise, first-frame propagation (episode 05), mask
inpainting (episode 06, zig-zag clips), in-context LoRA (episode 07): edit success (object colour = target),
trajectory error vs. the source (px), background PSNR (dB, source background pixels), flicker (RMS change of
background pixels between consecutive frames; 0 for the static sources).
Facts about real benchmarks are quoted with references in the notes.
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
CACHE = os.environ.get("VL_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "vlcache"))

INV = vlx.cached(CACHE, "inv", vlx.inversion_edit, cache_dir=CACHE)
FF = vlx.cached(CACHE, "first", vlx.first_frame_edit, cache_dir=CACHE)
MK = vlx.cached(CACHE, "mask", vlx.mask_lora_edit, cache_dir=CACHE)
IC = vlx.cached(CACHE, "ic", vlx.ic_lora, cache_dir=CACHE)
TE = sorted(INV["sweep"])
METHODS = [("从噪声重新生成", INV["fresh"], RED), ("反演 · 强度 0.4", INV["sweep"][0.4]["edit"], AMBER),
           ("反演 · 强度 0.8", INV["sweep"][0.8]["edit"], AMBER), ("首帧传播", FF["label_none"], VIOLET),
           ("掩码重画", MK["agg"]["none"], CYAN), ("上下文 LoRA", IC["final"], GREEN)]
for n, m, _ in METHODS:
    print(n, {k: round(v, 3) for k, v in m.items()})
FL = {n: m["flicker"] for n, m, _ in METHODS}
assert abs(FL["上下文 LoRA"] - 0.0405) < 1e-3 and abs(FL["从噪声重新生成"] - 0.0847) < 1e-3
assert abs(INV["sweep"][0.4]["edit"]["success"] - 0.125) < 1e-3 and abs(INV["sweep"][0.4]["edit"]["bg_psnr"] - 33.6) < 0.05
assert abs(INV["sweep"][0.2]["edit"]["flicker"] - 0.0062) < 1e-3 and abs(IC["final"]["bg_psnr"] - 28.65) < 0.01
assert min(FL, key=FL.get) == "反演 · 强度 0.4"

SCENES = [
    ("title", [
        "这是视频 LoRA 编辑系列的最后一期。怎样判断一次视频编辑做得好不好？这一期看评估的几个维度，以及这个领域还没有解决的问题。",
    ]),
    ("dims", [
        "一次好的编辑至少要满足三点。第一，改对了，结果符合编辑指令。第二，没改的地方保住了，背景、物体的轨迹、其他内容和原视频一致。第三，视频本身是连贯的，帧与帧之间没有闪烁和跳变。",
        "在我们的小实验里，这三点分别用编辑成功率、轨迹偏差和背景的峰值信噪比，以及背景在相邻帧之间的变化，也就是闪烁来衡量。",
    ]),
    ("table", [
        "把这个系列里所有的编辑方法放在一起比较。从噪声重新生成，改对了，但保真最差，闪烁也最大。反演的强度低时保真好，却改不动；强度高时改对了，保真又下降。",
        f"首帧传播和掩码重画各有长处，但都有一部分没改对。上下文 LoRA 同时做到了全部改对、轨迹偏差只有{dec_zh(IC['final']['traj'], 1)}个像素，闪烁也是改对颜色的方法里最小的。代价是它需要训练，也需要成对的数据。",
    ]),
    ("tradeoff", [
        "把编辑成功率和背景保真画在一张图上，就能看到编辑强度和保真度之间的权衡曲线。反演方法沿着这条曲线移动，而好的方法要做的，是把整条曲线往右上方推。",
    ]),
    ("real", [
        "评估真实的视频编辑要难得多。常用的自动指标包括，用图文模型衡量结果和指令的一致性，用相邻帧的特征相似度衡量时间一致性，用未编辑区域的误差衡量保真度，还有综合性的视频生成基准。",
        "但这些指标经常和人的判断不一致，比如一个几乎没改的视频，保真度和一致性都会很高。所以严肃的比较通常还要请人来打分，并且同时报告改了多少和保住了多少。",
    ]),
    ("limits", [
        "最后说几个局限。大的结构性修改，比如改变物体的形状、动作，依然很难。长视频会出现漂移，越往后修改越容易丢失。成对的编辑数据大多是合成的，会带进合成数据本身的偏差。",
        "还有一个越来越重要的问题，编辑后的视频越来越逼真，需要配合水印和来源标注，防止被用来伪造。",
    ]),
    ("recap", [
        "回顾整个系列。我们从 LoRA 的低秩原理出发，看了它在视频模型里加在哪里，单视频微调中外观和运动的纠缠，反演和首帧传播这些不用训练的方法，以及掩码、上下文 LoRA 和指令编辑。",
        "最后看了多个 LoRA 的叠加和编辑的评估。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 10, "评估与局限", "一致性、保真与编辑强度", SERIES)


def sc_dims(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "好的编辑：三个维度")
    items = [("改对了", "符合编辑指令", GREEN, "编辑成功率"), ("保住了", "背景、轨迹与原视频一致", CYAN, "轨迹偏差 · 背景 PSNR"),
             ("连贯", "帧与帧之间没有闪烁", VIOLET, "背景闪烁")]
    for i, (a, b, col, m) in enumerate(items):
        y = 160 + i * 120
        d.rounded_rectangle([140, y, 1140, y + 95], 14, outline=col, width=2, fill=(12, 19, 34))
        d.text((170, y + 32), a, font=F_BODY, fill=col, anchor="lm")
        d.text((170, y + 70), b, font=F_SMALL, fill=TEXT, anchor="lm")
        if seg >= 1:
            d.text((1110, y + 48), m, font=F_SMALL, fill=AMBER, anchor="rm")


def sc_table(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "本系列所有方法（玩具颜色编辑）")
    for j, h in enumerate(["方法", "改对", "轨迹偏差", "背景 PSNR", "闪烁"]):
        d.text((130 + [0, 400, 560, 750, 940][j], 150), h, font=F_SMALL, fill=MUTED)
    n = 3 if seg == 0 else len(METHODS)
    for i, (a, m, col) in enumerate(METHODS[:n]):
        y = 200 + i * 66
        d.rounded_rectangle([110, y - 12, 1170, y + 42], 10, outline=col, width=2, fill=(12, 19, 34))
        d.text((130, y + 15), a, font=F_SMALL, fill=col, anchor="lm")
        f = vk.font(22, mono=True)
        for j, v in enumerate([f"{m['success']:.0%}", f"{m['traj']:.2f}", f"{m['bg_psnr']:.1f}", f"{m['flicker']:.3f}"]):
            d.text((130 + [400, 560, 750, 940][j], y + 15), v, font=f, fill=TEXT, anchor="lm")
    d.text((130, 610), "掩码重画为之字形视频（掩码外完全保留）；其余为同分布的随机视频", font=vk.font(18), fill=MUTED)


def sc_tradeoff(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "编辑强度 vs 保真度")
    x0, y0, x1, y1 = 180, 170, 820, 580
    d.rectangle([x0, y0, x1, y1], outline=(30, 41, 59))
    px = lambda s: x0 + s * (x1 - x0)
    py = lambda b: y1 - (b - 15) / 25 * (y1 - y0)
    pts = [(INV["sweep"][t]["edit"]["success"], INV["sweep"][t]["edit"]["bg_psnr"]) for t in TE]
    d.line([(px(s), py(b)) for s, b in pts], fill=AMBER, width=3)
    for t, (s, b) in zip(TE, pts):
        d.ellipse([px(s) - 6, py(b) - 6, px(s) + 6, py(b) + 6], fill=AMBER)
        d.text((px(s) + 10, py(b) - 22), f"t={t}", font=vk.font(16, mono=True), fill=AMBER)
    for n, m, col in METHODS:
        if n.startswith("反演"):
            continue
        d.ellipse([px(m["success"]) - 9, py(m["bg_psnr"]) - 9, px(m["success"]) + 9, py(m["bg_psnr"]) + 9], fill=col)
        if n == "首帧传播":
            d.text((px(m["success"]), py(m["bg_psnr"]) + 16), n, font=F_SMALL, fill=col, anchor="mt")
        else:
            d.text((px(m["success"]) - 14, py(m["bg_psnr"])), n, font=F_SMALL, fill=col, anchor="rm")
    d.text((x1, y1 + 30), "编辑成功率 →", font=F_SMALL, fill=MUTED, anchor="rm")
    d.text((x0 - 10, y0), "背景 PSNR ↑", font=F_SMALL, fill=MUTED, anchor="rm")
    vk.arrow(d, (930, 480), (1080, 300), GREEN, 4)
    d.text((880, 520), "好的方法：把曲线往右上推", font=F_SMALL, fill=GREEN)


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "真实视频编辑的评估")
    items = [("文本对齐", "图文模型（如 CLIP）衡量结果与指令一致", CYAN), ("时间一致性", "相邻帧特征相似度", VIOLET),
             ("保真度", "未编辑区域的误差", GREEN), ("综合基准", "如 VBench（Huang et al., 2024）", AMBER)]
    for i, (a, b, col) in enumerate(items):
        y = 150 + i * 80
        d.text((160, y), a, font=F_BODY, fill=col)
        d.text((420, y + 4), b, font=F_SMALL, fill=TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 520), "几乎没改的视频：保真和一致性都很高 → 还需人工评价，并同时报告改了多少、保住多少", F_SMALL, RED)


def sc_limits(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "局限")
    for i, (t, col) in enumerate([("大的结构修改（形状、动作）依然困难", RED), ("长视频漂移：越往后修改越容易丢失", AMBER),
                                  ("成对数据多为合成，带入合成数据的偏差", VIOLET)]):
        vk.text_c(d, (640, 190 + i * 70), t, F_BODY, col)
    if seg >= 1:
        vk.text_c(d, (640, 470), "逼真的编辑需要水印与来源标注，防止伪造", F_BODY, GREEN)


EPS = ["LoRA 基础", "加在哪里", "单视频 LoRA", "反演", "首帧传播", "掩码 LoRA", "上下文 LoRA", "指令编辑", "组合与加速", "评估"]


def sc_recap(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 6, "系列回顾")
    k = 8 if seg == 0 else 10
    for i, t in enumerate(EPS):
        x, y = 110 + (i % 5) * 215, 180 + (i // 5) * 170
        on = i < k
        d.rounded_rectangle([x, y, x + 195, y + 120], 14, outline=CYAN if on else (30, 41, 59), width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 97, y + 36), f"{i + 1:02d}", vk.font(26, mono=True), CYAN if on else MUTED)
        vk.text_c(d, (x + 97, y + 82), t, F_SMALL, TEXT if on else MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 600), "感谢观看 · Video LoRA Edit 系列完", F_BODY, VIOLET)


RENDER = {"title": sc_title, "dims": sc_dims, "table": sc_table, "tradeoff": sc_tradeoff, "real": sc_real, "limits": sc_limits, "recap": sc_recap}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vl-eval")
