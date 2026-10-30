"""Video Generation 01 — 视频生成概览：从图片到视频.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python vg01_overview.py <out_dir>

Real computations: temporal redundancy of 3,000 toy 8-frame videos (video_toy.py): adjacent-frame
difference vs frame energy, and the number of PCA components for 95% of the variance when frames
are compressed separately vs jointly. Token counts for an example setting are plain arithmetic,
labelled as an example on screen.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import tviz as tv
import video_toy as vt
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Video Generation"

VID = vt.make_videos(3000, 0)
N = len(VID)
DIFF = float(np.sqrt(((VID[:, 1:] - VID[:, :-1]) ** 2).mean(-1)).mean())
RMS = float(np.sqrt((VID ** 2).mean(-1)).mean())


def k95(X):
    X = X - X.mean(0)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    return int(np.searchsorted(np.cumsum(s) / s.sum(), 0.95) + 1)


K_FRAME = k95(VID.reshape(-1, 1024))
K_VIDEO = k95(VID.reshape(N, -1))
print(DIFF, RMS, K_FRAME, K_VIDEO)
assert 0.18 < DIFF / RMS < 0.24 and K_FRAME == 15 and K_VIDEO == 48

# example setting: 5 s, 24 fps (121 frames), 1280 x 720; VAE 4x time, 8x8 space, 16 channels; 2x2 patches
FR, HH, WW = 121, 720, 1280
LT, LH, LW, CH = 1 + (FR - 1) // 4, HH // 8, WW // 8, 16
RATIO = 3 * FR * HH * WW / (CH * LT * LH * LW)
TOK_F = (LH // 2) * (LW // 2)
TOK = TOK_F * LT
TOK_IMG = (1024 // 16) ** 2
PAIRS = (TOK / TOK_IMG) ** 2
assert (LT, LH, LW) == (31, 90, 160) and TOK == 111600 and TOK_IMG == 4096 and 46 < RATIO < 47 and 740 < PAIRS < 745

SCENES = [
    ("title", [
        "欢迎来到视频生成系列。这一期先看清楚，从生成图片到生成视频，到底多了什么，难在哪里。",
    ]),
    ("frames", [
        "视频就是一串连续的图片，也就是帧。生成视频，不仅每一帧都要好看，帧与帧之间还要连贯，物体不能突然变形或者闪烁。",
        f"好消息是，相邻的帧非常相似。我们生成了三千段八帧的小视频，相邻两帧的平均差别，只有画面本身幅度的{pct_zh(DIFF / RMS)}左右。",
    ]),
    ("redundancy", [
        "这种相似性意味着视频可以被大幅压缩。我们用主成分分析来衡量，保留百分之九十五的信息，需要多少个数。",
        f"单独压缩一帧需要{int_zh(K_FRAME)}个数，八帧分开压缩就是{int_zh(8 * K_FRAME)}个；把八帧放在一起压缩，只需要{int_zh(K_VIDEO)}个。时间维度上的冗余，让整段视频比逐帧压缩省了一半以上。",
    ]),
    ("vae", [
        "真实的视频模型利用的正是这一点。它先用一个三维的变分自编码器，把视频在时间上压缩四倍，在空间上长宽各压缩八倍，得到一个小得多的时空潜空间。",
        f"举个例子，一段五秒、每秒二十四帧、分辨率一千二百八十乘七百二十的视频，压缩以后是{int_zh(LT)}个潜空间帧，每帧九十乘一百六十，十六个通道。数据量缩小了大约{int_zh(round(RATIO))}倍，扩散模型就在这个潜空间里生成。",
    ]),
    ("tokens", [
        f"然后把潜空间切成二乘二的小块，每一块是一个词元。每帧有{big_zh(TOK_F)}个词元，整段视频一共约{big_zh(TOK)}个。",
        f"而一张一千零二十四像素见方的图片，同样处理以后只有{big_zh(TOK_IMG)}个词元。注意力要计算每两个词元之间的关系，词元对的数量相差约{int_zh(round(PAIRS))}倍。这就是视频生成又慢又贵的根本原因。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，视频需要每帧好看，还要帧间连贯。",
        "第二，相邻帧高度相似，三维变分自编码器同时在时间和空间上压缩。",
        "第三，即使压缩以后，视频的词元数仍然是图片的几十倍，注意力的计算量是几百倍。下一期，我们看视频扩散变换器怎样处理这么多的时空词元。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 1, "视频生成概览", "从图片到视频：多了什么，难在哪里", SERIES)


def draw_frame(d, x, y, img, s=4, col=CYAN):
    img = img.reshape(32, 32)
    for r in range(0, 32, 1):
        for cc in range(32):
            v = img[r, cc]
            if v > 0.05:
                d.rectangle([x + cc * s, y + r * s, x + cc * s + s - 1, y + r * s + s - 1],
                            fill=tuple(int(v * ch) for ch in col))
    d.rectangle([x, y, x + 32 * s, y + 32 * s], outline=(30, 41, 59))


def sc_frames(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "视频 = 一串帧")
    for f in range(8):
        draw_frame(d, 90 + f * 140, 160, VID[0, f], 4)
        d.text((90 + f * 140 + 64, 300), f"第 {f + 1} 帧", font=F_SMALL, fill=MUTED, anchor="mm")
    if seg >= 1:
        tv.bar_chart(d, (380, 380, 900, 600), ["相邻两帧的差", "画面本身的幅度"], [DIFF, RMS], [AMBER, CYAN], fmt="{:.3f}",
                     vmax=RMS * 1.25)
        vk.text_c(d, (640, 680), f"均方根，3000 段 8 帧小视频（差 / 幅度 ≈ {DIFF / RMS:.0%}）", F_SMALL, MUTED)


def sc_redundancy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "冗余：保留 95% 信息需要多少个数（PCA）")
    if seg >= 1:
        tv.bar_chart(d, (260, 180, 1020, 560), ["单独一帧", "八帧分开压缩", "八帧一起压缩"], [K_FRAME, 8 * K_FRAME, K_VIDEO],
                     [MUTED, RED, GREEN], fmt="{:.0f}", vmax=8 * K_FRAME * 1.2)
        vk.text_c(d, (640, 630), f"一起压缩：{K_VIDEO} vs {8 * K_FRAME} → 时间冗余", F_BODY, AMBER)


def sc_vae(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "3D VAE：时空潜空间")
    for i in range(5):
        d.rectangle([140 + i * 18, 160 + i * 18, 460 + i * 18, 340 + i * 18], outline=CYAN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (380, 450), f"{FR} 帧 × {HH} × {WW} × 3", F_SMALL, TEXT)
    vk.arrow(d, (580, 290), (700, 290), AMBER, 3)
    vk.text_c(d, (640, 250), "编码器", F_SMALL, AMBER)
    for i in range(3):
        d.rectangle([760 + i * 12, 220 + i * 12, 880 + i * 12, 290 + i * 12], outline=VIOLET, width=2, fill=(12, 19, 34))
    vk.text_c(d, (850, 360), "时间 ÷4，长宽 ÷8，16 通道", F_SMALL, VIOLET)
    if seg >= 1:
        vk.text_c(d, (850, 400), f"{LT} × {LH} × {LW} × {CH}", vk.font(24, mono=True), VIOLET)
        vk.text_c(d, (640, 520), f"数据量约缩小 {RATIO:.0f} 倍（示例：5 秒、24 帧/秒、720p）", F_BODY, GREEN)


def sc_tokens(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "切块成词元（2 × 2）")
    rows = [("视频（示例）", f"{LH // 2} × {LW // 2} × {LT} = {TOK:,}", CYAN),
            ("图片 1024 × 1024", f"{1024 // 16} × {1024 // 16} = {TOK_IMG:,}", MUTED)]
    for i, (a, b, col) in enumerate(rows):
        d.text((200, 170 + i * 70), a, font=F_BODY, fill=col)
        d.text((560, 172 + i * 70), b, font=vk.font(28, mono=True), fill=col)
    if seg >= 1:
        tv.fpaste(c, r"$\left(\frac{111600}{4096}\right)^2\approx 742$", 640, 400, 40, "#fbbf24")
        vk.text_c(d, (640, 480), "注意力的“词元对”数量：视频是图片的约 742 倍", F_BODY, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("要求", "每帧好看 + 帧间连贯"), ("压缩", "3D VAE：时间 ÷4，空间 ÷8×8"),
                      ("代价", "词元数几十倍，注意力几百倍")], seg, "下一期：视频扩散 Transformer 与时空注意力")


RENDER = {"title": sc_title, "frames": sc_frames, "redundancy": sc_redundancy, "vae": sc_vae, "tokens": sc_tokens,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vg-overview")
