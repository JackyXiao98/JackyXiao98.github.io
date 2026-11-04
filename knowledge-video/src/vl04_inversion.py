"""Video LoRA 04 — 视频编辑的基本路线：反演与注意力注入.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] VL_CACHE=<dir> python vl04_inversion.py <out_dir>

Real computations (vlx.inversion_edit, JAX): 48 source clips; the target is a different object colour. The
source is integrated with the deterministic flow ODE (Euler, source labels) from t = 0 up to t_end, then sampled
back to t = 0 with the target-colour label (same number of steps). Metrics: edit success (object colour of the
output = target), trajectory error vs. the source (px, mean centroid distance), background PSNR (pixels that
are background in the source), reconstruction PSNR when sampling back with the source label. Baseline:
generating from pure noise with the target labels.
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

I = vlx.cached(CACHE, "inv", vlx.inversion_edit, cache_dir=CACHE)
TE = sorted(I["sweep"])
E = {t: I["sweep"][t]["edit"] for t in TE}
REC = {t: I["sweep"][t]["recon"] for t in TE}
F = I["fresh"]
print({t: (E[t]["success"], round(E[t]["traj"], 2), round(E[t]["bg_psnr"], 1)) for t in TE}, REC, F)
assert TE == [0.2, 0.4, 0.6, 0.8, 1.0] and abs(E[0.2]["success"] - 0.0208) < 1e-3 and abs(E[0.6]["success"] - 0.4583) < 1e-3
assert E[0.8]["success"] == 1.0 and E[1.0]["success"] == 1.0 and abs(E[0.2]["bg_psnr"] - 37.9) < 0.05 and abs(E[0.8]["bg_psnr"] - 24.5) < 0.05
assert abs(E[0.6]["bg_psnr"] - 30.1) < 0.05 and abs(E[0.8]["traj"] - 1.56) < 0.01 and abs(F["traj"] - 3.43) < 0.01 and abs(F["bg_psnr"] - 17.3) < 0.05
assert abs(REC[1.0] - 25.8) < 0.05 and abs(REC[0.2] - 42.4) < 0.05 and F["success"] == 1.0

SCENES = [
    ("title", [
        "在讲用罗拉编辑视频之前，先看最基本的编辑方法，它们不需要任何训练。这一期看反演，以及在它基础上的注意力注入。",
    ]),
    ("naive", [
        "最直接的想法是，用新的描述重新生成一遍。比如原视频里是红色的方块向右移动，我们想把它改成蓝色。",
        f"但从纯噪声重新生成，虽然颜色全都改对了，物体的位置和轨迹却和原视频完全不同，平均偏差{dec_zh(F['traj'], 1)}个像素，背景也变了，背景的峰值信噪比只有{dec_zh(F['bg_psnr'], 1)}分贝。我们要的是编辑，而不是另生成一个视频。",
    ]),
    ("inversion", [
        "反演的想法是，不从随机噪声出发，而是从原视频本身出发。扩散模型的采样可以看成一个常微分方程，从噪声走到视频；把这个方程倒着走，就能从原视频走回到一个特定的噪声。",
        "然后从这个噪声出发，换上新的描述再走回来。因为起点记住了原视频的信息，结果会尽量保留原来的结构。",
    ]),
    ("strength", [
        "关键的参数是倒着走多远，也就是编辑强度。我们在四十八个视频上，把强度从零点二调到一。",
        f"强度零点二时，背景几乎不变，峰值信噪比{dec_zh(E[0.2]['bg_psnr'], 1)}分贝，但只有{pct_zh(E[0.2]['success'])}的视频改了颜色。强度零点六时，成功率{pct_zh(E[0.6]['success'])}，背景降到{dec_zh(E[0.6]['bg_psnr'], 1)}分贝。",
        f"强度达到零点八，颜色全部改对，但背景降到{dec_zh(E[0.8]['bg_psnr'], 1)}分贝，轨迹偏差{dec_zh(E[0.8]['traj'], 2)}个像素。改得越多，保留得越少，这就是编辑的核心权衡。",
    ]),
    ("attention", [
        "为了缓解这个权衡，研究者在反演的基础上加了注意力注入。二零二三年的一些方法，在重新生成时，把原视频反演过程中记录下的注意力图换进去，让结构和运动跟着原视频走，只改变外观。",
        "另一类方法先编辑几个关键帧，再根据原视频帧与帧之间的对应关系，把修改传播到其他帧。这些方法都不需要训练，但受限于原模型的能力，大幅度的形状变化往往做不好，这也是后来转向罗拉和专门编辑模型的原因。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，从噪声重新生成会丢掉原视频的结构；反演从原视频出发，保留结构。",
        "第二，编辑强度越大，改得越彻底，保留得越少，需要权衡。",
        "第三，注意力注入和特征传播能缓解这个权衡，但大的形状变化依然困难。下一期，我们看先编辑首帧、再传播的方法。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 4, "反演与注意力注入", "不训练也能编辑", SERIES)


def sc_naive(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "从噪声重新生成（目标：改颜色）")
    for i in range(2):
        vlviz.film(c, I["src"][i], 260, 180 + i * 190, cell=60, label="原视频")
        if seg >= 1:
            vlviz.film(c, I["fresh_samples"][i], 260, 250 + i * 190, cell=60, label="重新生成", color=RED)
    if seg >= 1:
        d.text((880, 620), f"轨迹偏差 {F['traj']:.1f} 像素 · 背景 PSNR {F['bg_psnr']:.1f} dB", font=F_SMALL, fill=RED, anchor="mm")


def sc_inversion(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "反演：把采样的 ODE 倒着走")
    pts = [(200, "原视频", CYAN), (640, "反演得到的噪声", AMBER), (1080, "编辑后的视频", GREEN)]
    for x, t, col in pts:
        d.rounded_rectangle([x - 150, 220, x + 150, 300], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x, 260), t, F_SMALL, col)
    vk.arrow(d, (355, 260), (485, 260), MUTED, 3)
    vk.text_c(d, (420, 330), "原描述 · t: 0 → 1", vk.font(18), MUTED)
    if seg >= 1:
        vk.arrow(d, (795, 260), (925, 260), GREEN, 3)
        vk.text_c(d, (860, 330), "新描述 · t: 1 → 0", vk.font(18), GREEN)
        vk.text_c(d, (640, 400), "起点记住了原视频 → 尽量保留结构（DDIM 反演 / SDEdit）", F_BODY, TEXT)


def sc_strength(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "编辑强度：倒着走多远（48 个视频）")
    n = [1, 3, 5][min(seg, 2)]
    tv.line_chart(d, (140, 170, 760, 560), TE, [("编辑成功率 %", [E[t]["success"] * 100 for t in TE], GREEN),
                                                ("背景 PSNR（dB × 2.5）", [E[t]["bg_psnr"] * 2.5 for t in TE], CYAN)],
                  ylim=(0, 105), upto=n, xlabel="编辑强度 t", legend_xy=(170, 470), xticks=[(t, f"{t}") for t in TE])
    d.text((820, 190), "强度", font=F_SMALL, fill=MUTED)
    d.text((930, 190), "成功率", font=F_SMALL, fill=MUTED)
    d.text((1060, 190), "背景 PSNR", font=F_SMALL, fill=MUTED)
    for i, t in enumerate(TE[:n]):
        y = 230 + i * 45
        f = vk.font(22, mono=True)
        d.text((820, y), f"{t}", font=f, fill=TEXT)
        d.text((930, y), f"{E[t]['success']:.0%}", font=f, fill=GREEN)
        d.text((1060, y), f"{E[t]['bg_psnr']:.1f}", font=f, fill=CYAN)
    if seg >= 2:
        vlviz.film(c, I["sweep"][0.8]["samples"][0], 900, 520, cell=36, label="t = 0.8")
        vlviz.film(c, I["src"][0], 900, 570, cell=36, label="原视频")


def sc_attention(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "注意力注入与特征传播")
    items = [("FateZero（Qi et al., 2023）", "重新生成时换入反演时记录的注意力图", CYAN),
             ("TokenFlow（Geyer et al., 2023）", "编辑关键帧，按帧间对应传播特征", VIOLET)]
    for i, (a, b, col) in enumerate(items):
        if i == 0 or seg >= 1:
            y = 170 + i * 130
            d.rounded_rectangle([100, y, 1180, y + 100], 14, outline=col, width=2, fill=(12, 19, 34))
            d.text((130, y + 32), a, font=F_BODY, fill=col, anchor="lm")
            d.text((130, y + 72), b, font=F_SMALL, fill=TEXT, anchor="lm")
    if seg >= 1:
        vk.text_c(d, (640, 500), "不需训练 · 受限于原模型 · 大的形状变化难 → 转向 LoRA 与专门编辑模型", F_SMALL, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("反演", "从原视频出发，保留结构"), ("强度", "改得越多，保留越少"),
                      ("注意力注入", "缓解权衡，形状变化仍难")], seg, "下一期：首帧编辑 + 传播")


RENDER = {"title": sc_title, "naive": sc_naive, "inversion": sc_inversion, "strength": sc_strength, "attention": sc_attention, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vl-inversion")
