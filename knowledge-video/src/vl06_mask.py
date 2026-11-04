"""Video LoRA 06 — 掩码感知的 LoRA：保留哪里、重画哪里.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] VL_CACHE=<dir> python vl06_mask.py <out_dir>

Real computations (vlx.mask_lora_edit, JAX): 12 source clips with a zig-zag motion the base model never saw.
Mask = the 4x4-pixel patches the object ever touches (regenerate); all other patches of every frame are given as
clean tokens (kept exactly), and so is the edited first frame (object recoloured). (a) no LoRA; (b) a per-video
LoRA (rank 8, all modules, 200 steps, Adam 1e-3) trained on the source video with the same mask pattern (the
unmasked patches clean, the first frame clean in half of the samples). Metrics: object colour = target, trajectory
error vs. the source (px).
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

M = vlx.cached(CACHE, "mask", vlx.mask_lora_edit, cache_dir=CACHE)
A = M["agg"]
SM = M["samples"]
NREG = int(SM[0]["mask"].sum())
print(A, NREG)
assert abs(A["none"]["success"] - 0.9167) < 1e-3 and A["lora"]["success"] == 0.0
assert abs(A["none"]["traj"] - 1.49) < 0.01 and abs(A["lora"]["traj"] - 1.45) < 0.01 and len(M["per"]["none"]) == 12


def masked_view(v, pm):
    """Darken the patches that will be regenerated."""
    out = v.copy()
    g = pm.reshape(4, 4)
    for i in range(4):
        for j in range(4):
            if g[i, j]:
                out[:, i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] *= 0.25
    return out


SCENES = [
    ("title", [
        "上一期讲到，首帧编辑很难控制第一帧之后的内容。这一期看一种更精细的控制方式，用掩码告诉模型哪里保留、哪里重画，并配合一个针对原视频训练的 LoRA。",
    ]),
    ("mask", [
        "掩码是一个时空的区域。掩码之外的部分，直接照抄原视频；掩码之内的部分，交给模型重新生成。这样背景这些不想改的地方，就能一个像素都不变。",
        "在我们的小模型里，训练时有一部分样本会随机保留一些区域作为已知条件，所以它天然支持这种做法。我们把物体经过的区域设为掩码，掩码外的所有帧都保留，同时把改好颜色的第一帧也给它。",
    ]),
    ("result", [
        f"十二个视频里，掩码内重新生成的物体，有{pct_zh(A['none']['success'])}改成了新颜色，掩码外的背景完全保留。但这些视频里的物体走的是之字形，这是模型从没见过的运动，重画的物体轨迹和原视频平均差{dec_zh(A['none']['traj'], 2)}个像素。",
    ]),
    ("lora", [
        "怎样让重画的部分也学会原视频的运动？二零二五年的掩码感知 LoRA 编辑，先在原视频上训练一个 LoRA，训练时同样使用掩码，让模型学会照抄掩码外的内容，并在掩码内生成和原视频一致的运动。",
        f"我们照着做，对每个视频训练一个 LoRA。结果出乎意料，轨迹几乎没有改善，偏差是{dec_zh(A['lora']['traj'], 2)}个像素，而颜色的成功率从{pct_zh(A['none']['success'])}掉到了零。",
        "原因是，LoRA 在原视频上学到的不只是运动，还有原来物体的颜色。它把改好的第一帧又改回了原来的样子，这就是单视频微调常见的外观泄漏。",
    ]),
    ("paper", [
        "这正是这项工作要解决的问题。它在训练时让掩码同时承担两种含义，哪里照抄、哪里生成，并且可以额外提供用户编辑好的参考帧，让 LoRA 在掩码内学习参考帧的外观，而不是死记原视频的外观。",
        "论文报告，在和商业模型以及多种首帧编辑方法的比较中，它在编辑的可控性上有优势。我们的小实验说明了，为什么这些额外的设计是必要的。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，时空掩码可以精确地规定保留和重画的区域，掩码外一个像素都不变。",
        "第二，在原视频上训练的 LoRA 容易连外观一起学进去，把编辑撤销掉。",
        "第三，掩码感知的 LoRA 编辑用掩码和参考帧，把运动和外观分开控制。下一期，我们看 LTX 的上下文 LoRA 训练。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 6, "掩码感知的 LoRA", "保留哪里、重画哪里", SERIES)


def sc_mask(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "时空掩码：外面照抄，里面重画")
    s = SM[0]
    vlviz.film(c, s["src"], 300, 190, cell=64, label="原视频")
    vlviz.film(c, masked_view(s["src"], s["mask"]), 300, 280, cell=64, label="掩码（暗 = 重画）", color=AMBER)
    if seg >= 1:
        vlviz.film(c, s["edited"], 300, 370, cell=64, frames=[0], label="改好的第一帧", color=GREEN)
        vk.text_c(d, (640, 520), f"本例：每帧 16 个块中 {NREG} 个重画 · 其余照抄", F_SMALL, MUTED)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "不加 LoRA：掩码内重画（12 个之字形视频）")
    for i in range(2):
        s = SM[i]
        vlviz.film(c, s["src"], 300, 170 + i * 200, cell=56, label="原视频")
        vlviz.film(c, s["none"], 300, 235 + i * 200, cell=56, label="编辑结果", color=GREEN)
    d.text((300, 600), f"颜色成功 {A['none']['success']:.0%} · 掩码外完全保留 · 轨迹偏差 {A['none']['traj']:.2f} 像素", font=F_SMALL, fill=TEXT)


def sc_lora(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "再加一个在原视频上训练的 LoRA")
    if seg >= 1:
        tv.bar_chart(d, (140, 220, 600, 520), ["不加 LoRA", "加 LoRA"], [A["none"]["success"] * 100, A["lora"]["success"] * 100],
                     [GREEN, RED], fmt="{:.0f}%", vmax=110)
        d.text((140, 170), "颜色改对的比例", font=F_SMALL, fill=MUTED)
        tv.bar_chart(d, (700, 220, 1160, 520), ["不加 LoRA", "加 LoRA"], [A["none"]["traj"], A["lora"]["traj"]], [CYAN, CYAN], fmt="{:.2f}", vmax=2)
        d.text((700, 170), "轨迹偏差（像素）", font=F_SMALL, fill=MUTED)
    if seg >= 2:
        vlviz.film(c, SM[0]["lora"], 300, 590, cell=48, label="加 LoRA 后", color=RED)
        vk.text_c(d, (640, 680), "外观泄漏：LoRA 把原来的颜色也学进去了", F_SMALL, RED)


def sc_paper(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "LoRA-Edit（Gao et al., 2025 · ICLR 2026）")
    items = [("掩码 = 指令", "哪里照抄原视频、哪里生成新内容", CYAN), ("参考帧", "用户编辑好的帧 → 掩码内学参考的外观", GREEN),
             ("LoRA", "学原视频的运动，而不是死记外观", AMBER)]
    for i, (a, b, col) in enumerate(items):
        y = 160 + i * 105
        d.rounded_rectangle([140, y, 1140, y + 85], 14, outline=col, width=2, fill=(12, 19, 34))
        d.text((170, y + 42), a, font=F_BODY, fill=col, anchor="lm")
        d.text((420, y + 42), b, font=F_SMALL, fill=TEXT, anchor="lm")
    if seg >= 1:
        vk.text_c(d, (640, 520), "对比 Kling 1.6、VACE、AnyV2V、I2VEdit 等，报告可控性更好", F_SMALL, MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("掩码", "外面照抄，里面重画"), ("外观泄漏", f"单视频 LoRA：成功率 {A['none']['success']:.0%} → 0%"),
                      ("LoRA-Edit", "掩码 + 参考帧分开控制运动与外观")], seg, "下一期：LTX 的上下文 LoRA 训练")


RENDER = {"title": sc_title, "mask": sc_mask, "result": sc_result, "lora": sc_lora, "paper": sc_paper, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vl-mask")
