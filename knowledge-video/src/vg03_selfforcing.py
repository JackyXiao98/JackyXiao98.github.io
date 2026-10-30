"""Video Generation 03 — 自回归视频生成与 Self-Forcing.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] VG_CACHE=<dir> python vg03_selfforcing.py <out_dir>

Real computations (vgx.selfforcing_toy, JAX): a next-frame model for a point moving on a circle,
trained with teacher forcing, then fine-tuned on its own 20-frame rollouts; 200-frame rollouts on
256 held-out clips (radius error and speed error per frame).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import tviz as tv
import vgx
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Video Generation"
CACHE = os.environ.get("VG_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "vgcache"))

R = vgx.cached(CACHE, "sf", vgx.selfforcing_toy)
RAD_TF, SP_TF = vgx.rollout_metrics(R["tf"], R["w"])
RAD_SF, SP_SF = vgx.rollout_metrics(R["sf"], R["w"])
L = len(RAD_TF)
print(R["tf_1step"], RAD_TF[[9, 49, 199]], RAD_SF[[9, 49, 199]], SP_TF[-1], SP_SF[-1])
assert R["tf_1step"] < 1e-4 and RAD_TF[-1] > 30 and RAD_SF[-1] < 0.06 and SP_TF[-1] > 0.25 and SP_SF[-1] < 0.05
assert RAD_TF[9] < 0.1

SCENES = [
    ("title", [
        "前两期的模型一次生成整段视频。这一期看另一种思路，自回归生成，一段接一段地生成视频，以及让它真正好用的关键，自我强迫训练。",
    ]),
    ("ar", [
        "双向的视频扩散模型，所有帧一起去噪，必须等整段视频生成完才能播放，长度也是固定的。",
        "自回归模型则像语言模型一样，根据已经生成的帧，生成下一帧或下一小段。它可以边生成边播放，长度不受限制，还能用第七期变换器系列讲过的键值缓存，重复利用前面的计算。",
    ]),
    ("bias", [
        "问题出在训练方式上。常见的教师强迫训练，给模型看的上下文永远是真实视频的帧，扩散强迫训练则给真实帧加上噪声。",
        "但推理时，模型看到的是自己生成的帧，带着自己的小误差。训练和推理看到的上下文不一样，误差会一帧一帧地累积，这叫暴露偏差。",
    ]),
    ("toy", [
        "我们用一个最小的例子来看。一个点在圆上匀速转动，每一帧是它的位置，模型根据前两帧预测下一帧。",
        "用教师强迫训练以后，单步预测的误差几乎为零。但让它连续生成，小误差把点推离了圆周，模型从没见过这样的输入，误差越滚越大。",
        f"生成到第十帧时，半径的误差还不到零点一；到第二百帧，半径误差已经达到{dec_zh(RAD_TF[-1], 1)}，点早已飞出了圆外。",
    ]),
    ("sf", [
        "自我强迫训练的思路很直接，训练时就让模型像推理时一样，用自己生成的帧作为上下文，连续生成一段，再对这些生成结果计算损失。",
        "因为没有和自己生成的上下文对应的真实答案，论文用的是分布匹配损失，只要求生成的视频整体上像真实视频。在我们的玩具里，用一个教师为模型自己的上下文给出目标。",
        f"效果非常明显。连续生成二百帧，半径误差始终在{dec_zh(RAD_SF[-1])}左右，转动速度的误差从{dec_zh(SP_TF[-1])}降到{dec_zh(SP_SF[-1])}。模型学会了从自己的小错误里恢复。",
    ]),
    ("practice", [
        "在真实的视频模型上，自我强迫训练结合了少步蒸馏和滚动的键值缓存，可以在单张显卡上实时地、流式地生成视频，同时保持很好的画质。",
        "少步蒸馏用到的分布匹配蒸馏，我们会在第五期详细介绍。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，自回归视频生成可以边生成边播放，长度不受限制，还能使用键值缓存。",
        "第二，用真实上下文训练，推理时却用自己的上下文，会产生暴露偏差，误差逐帧累积。",
        "第三，自我强迫训练在训练时就用模型自己的生成结果做上下文，消除了这种不一致。下一期，我们看怎样把几十步的采样压缩到几步。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 3, "自回归视频生成与 Self-Forcing", "训练时就用自己生成的上下文", SERIES)


def sc_ar(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "双向生成 vs 自回归生成")
    d.text((140, 150), "双向：所有帧一起去噪", font=F_BODY, fill=VIOLET)
    for i in range(8):
        d.rectangle([140 + i * 70, 200, 200 + i * 70, 250], fill=VIOLET)
    d.text((140, 265), "生成完才能播放；长度固定", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        d.text((140, 340), "自回归：一段接一段", font=F_BODY, fill=CYAN)
        k = 8 if seg >= 1 else 1
        for i in range(k):
            col = CYAN if i < 6 else (20, 70, 85)
            d.rectangle([140 + i * 70, 390, 200 + i * 70, 440], fill=col)
        vk.arrow(d, (140 + 6 * 70 - 10, 470), (140 + 6 * 70 + 30, 470), AMBER, 2)
        d.text((140, 485), "边生成边播放 · 长度不限 · KV 缓存", font=F_SMALL, fill=MUTED)


def sc_bias(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "暴露偏差（exposure bias）")
    rows = [("Teacher Forcing", "上下文 = 真实帧", GREEN), ("Diffusion Forcing", "上下文 = 加噪的真实帧", GREEN),
            ("推理", "上下文 = 自己生成的帧（带误差）", RED)]
    for i, (a, b, col) in enumerate(rows):
        if seg >= (1 if i == 2 else 0):
            y = 170 + i * 110
            d.rounded_rectangle([170, y, 1110, y + 80], 12, outline=col, width=2, fill=(12, 19, 34))
            d.text((200, y + 40), a, font=F_BODY, fill=col, anchor="lm")
            d.text((560, y + 40), b, font=F_BODY, fill=TEXT, anchor="lm")
    if seg >= 1:
        vk.text_c(d, (640, 560), "训练和推理的上下文不一致 → 误差逐帧累积", F_BODY, AMBER)


def draw_traj(d, cx, cy, sc, ys, col, n):
    d.ellipse([cx - sc, cy - sc, cx + sc, cy + sc], outline=(71, 85, 105), width=1)
    pts = [(cx + sc * x, cy - sc * y) for x, y in ys[:n]]
    lim = 1.9 * sc
    keep = [p for p in pts if abs(p[0] - cx) <= lim and abs(p[1] - cy) <= lim]
    if len(keep) < len(pts):
        d.text((cx, cy + lim + 10), "已飞出画面", font=F_SMALL, fill=col, anchor="ma")
    pts = keep
    if len(pts) > 1:
        d.line(pts, fill=col, width=2)
    for x, y in pts[-1:]:
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=col)


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "玩具实验：圆周运动的“视频”")
    n = 30 if seg == 0 else (int(30 + 170 * vk.ease(su)) if seg == 1 else L)
    draw_traj(d, 330, 380, 110, R["tf"][3], RED, n)
    vk.text_c(d, (330, 150), f"Teacher Forcing（第 {n} 帧）", F_SMALL, RED)
    if seg >= 1:
        frames = np.arange(1, L + 1)
        tv.line_chart(d, (680, 170, 1180, 520), frames, [("半径误差", RAD_TF, RED)], ylog=True, ylim=(-2, 2),
                      upto=n, xlabel="生成的帧数", ylabel="半径误差（对数）", xticks=[(1, "1"), (100, "100"), (200, "200")])
    if seg >= 2:
        vk.text_c(d, (930, 620), f"第 10 帧：{RAD_TF[9]:.3f}  →  第 200 帧：{RAD_TF[-1]:.1f}", F_BODY, RED)


def sc_sf(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "Self-Forcing：用自己的生成结果训练")
    if seg <= 1:
        for i in range(6):
            col = GREEN if i < 2 else CYAN
            d.rectangle([180 + i * 120, 180, 270 + i * 120, 250], fill=col)
            if i >= 2:
                vk.text_c(d, (225 + i * 120, 215), "生成", F_SMALL, (10, 14, 22))
        d.text((180, 270), "训练时连续生成 → 对生成结果计算损失", font=F_BODY, fill=TEXT)
        if seg >= 1:
            d.text((180, 350), "论文：分布匹配损失（DMD，第 5 期）—— 整体像真实视频即可", font=F_SMALL, fill=AMBER)
            d.text((180, 390), "玩具：教师为“模型自己的上下文”给出下一帧的目标", font=F_SMALL, fill=MUTED)
        return
    draw_traj(d, 260, 380, 110, R["tf"][3], RED, L)
    draw_traj(d, 640, 380, 110, R["sf"][3], CYAN, L)
    vk.text_c(d, (260, 150), "Teacher Forcing", F_SMALL, RED)
    vk.text_c(d, (640, 150), "Self-Forcing（同一段，200 帧）", F_SMALL, CYAN)
    rows = [("半径误差", RAD_TF[-1], RAD_SF[-1]), ("速度误差", SP_TF[-1], SP_SF[-1])]
    d.text((900, 220), "第 200 帧", font=F_SMALL, fill=MUTED)
    for i, (k, a, b) in enumerate(rows):
        y = 270 + i * 90
        d.text((900, y), k, font=F_SMALL, fill=MUTED)
        d.text((900, y + 30), f"{a:.2f}", font=vk.font(26, mono=True), fill=RED)
        d.text((1040, y + 30), f"{b:.3f}", font=vk.font(26, mono=True), fill=CYAN)


def sc_practice(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "真实视频模型中的 Self-Forcing（Huang et al., 2025）")
    items = [("自回归 + 少步蒸馏（DMD）", CYAN), ("滚动 KV 缓存", VIOLET), ("单卡实时、流式生成", GREEN)]
    for i, (t, col) in enumerate(items):
        x = 100 + i * 370
        d.rounded_rectangle([x, 220, x + 340, 320], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 170, 270), t, F_SMALL, col)
    vk.text_c(d, (640, 420), "具体速度与画质数字见笔记（论文报告）", F_SMALL, MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 6, [("自回归", "边生成边播放，长度不限"), ("暴露偏差", "训练看真实帧，推理看自己的帧"),
                      ("Self-Forcing", "训练时用自己的生成结果")], seg, "下一期：少步生成 —— 从一致性模型到蒸馏")


RENDER = {"title": sc_title, "ar": sc_ar, "bias": sc_bias, "toy": sc_toy, "sf": sc_sf, "practice": sc_practice,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vg-selfforcing")
