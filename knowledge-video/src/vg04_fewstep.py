"""Video Generation 04 — 少步生成：从一致性模型到蒸馏.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] VG_CACHE=<dir> python vg04_fewstep.py <out_dir>

Real computations: an exact teacher (16-component Gaussian ring, diff2d.py) sampled with DDIM at
1-50 steps, sliced Wasserstein-2 distance to true samples; a one-step student regressed on 20,000
(noise, 50-step DDIM) pairs (vgx.distill_regression).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import diff2d as D
import tviz as tv
import vgx
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Video Generation"
CACHE = os.environ.get("VG_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "vgcache"))

G = vgx.ring()
Z = np.random.default_rng(9).standard_normal((4000, 2))
Y = G.sample(4000, np.random.default_rng(5))
FLOOR = D.sliced_w2(G.sample(4000, np.random.default_rng(7)), Y)
STEPS = [1, 2, 4, 8, 16, 50]
SAMPLES = {n: D.ddim(G, Z, n) for n in STEPS}
W2 = {n: D.sliced_w2(SAMPLES[n], Y) for n in STEPS}
R1 = float(np.sqrt((SAMPLES[1] ** 2).sum(1)).mean())
REG = vgx.cached(CACHE, "reg", vgx.distill_regression)
XS = vgx.sample_generator(REG["params"], Z)
W2S = D.sliced_w2(XS, Y)
print(FLOOR, W2, R1, W2S)
assert W2[1] > 1.3 and R1 < 0.1 and W2[50] < 0.05 and all(W2[a] > W2[b] for a, b in zip(STEPS, STEPS[1:]))
assert W2S < 0.05 and REG["teacher_evals"] == 1_000_000

SCENES = [
    ("title", [
        "扩散模型生成一段视频，通常要去噪几十步。这一期看为什么步数决定了速度，以及怎样把几十步压缩到几步。",
    ]),
    ("cost", [
        "每一步去噪，都要把所有词元完整地送进一次网络。对一个十万词元的视频模型，一次网络计算就很昂贵，五十步就要做五十次。",
        "所以减少步数，是最直接的加速方法，步数减少几倍，生成就快几倍。",
    ]),
    ("steps", [
        "我们用一个可以精确计算的玩具来看。数据分布是平面上的一个圆环，教师模型是精确的去噪器，用确定性的采样器生成样本。",
        f"五十步时，生成的分布和真实分布几乎一样，距离只有{dec_zh(W2[50], 3)}。四步时距离变成{dec_zh(W2[4], 2)}，圆环变得模糊。",
        f"只用一步时，所有样本都挤到了圆心附近，平均半径只有{dec_zh(R1, 2)}。因为从纯噪声出发，一步去噪给出的是所有可能结果的平均值，而圆环的平均正好是圆心。真实图片上，这就是一张模糊的平均图。",
    ]),
    ("distill", [
        "解决办法是蒸馏，训练一个学生模型，直接学会教师多步采样的结果。渐进蒸馏每一轮把步数减半；一致性模型则让轨迹上任意一点都直接映射到终点。",
        f"最简单的版本，是先让教师用五十步生成两万对噪声和结果，再训练学生一步完成这个映射。在玩具上，学生一步生成的距离是{dec_zh(W2S, 3)}，接近教师的五十步。",
        "不过，光是准备这些训练对，教师就要计算一百万次网络。",
    ]),
    ("limits", [
        "在真实的图片和视频上，这种直接回归还有更大的问题。一步映射非常复杂，学生很难精确拟合，结果往往比教师模糊，而且学生最多只能模仿教师的每一个具体样本。",
        "能不能不要求学生复现教师的每一条轨迹，只要求学生生成的分布和教师一样？这就是下一期的分布匹配蒸馏。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，每一步去噪都是一次完整的网络计算，步数直接决定速度。",
        "第二，直接用很少的步数采样，结果会变模糊，一步时退化成平均值。",
        "第三，蒸馏让学生学会教师的多步结果，但逐样本回归代价高、容易模糊。下一期介绍分布匹配蒸馏。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 4, "少步生成", "从一致性模型到蒸馏", SERIES)


def sc_cost(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "步数 = 网络计算的次数")
    k = 50 if seg >= 1 else max(1, int(50 * vk.ease(u * 1.5 if u < .66 else 1)))
    for i in range(k):
        x, y = 140 + (i % 25) * 40, 180 + (i // 25) * 60
        d.rectangle([x, y, x + 32, y + 44], fill=CYAN)
    d.text((140, 320), f"50 步 = 50 次完整的网络计算（每次处理全部词元）", font=F_BODY, fill=TEXT)
    if seg >= 1:
        for i in range(4):
            d.rectangle([140 + i * 40, 420, 172 + i * 40, 464], fill=GREEN)
        d.text((330, 430), "4 步：快约 12 倍", font=F_BODY, fill=GREEN)


def scatter(d, box, X, col, n=1500):
    x0, y0, x1, y1 = box
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 6.0
    d.rectangle(box, outline=(30, 41, 59))
    for x, y in X[:n]:
        px, py = cx + x * s, cy - y * s
        if x0 < px < x1 and y0 < py < y1:
            d.point((px, py), fill=col)
            d.point((px + 1, py), fill=col)


def sc_steps(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "玩具：精确教师 + DDIM（圆环分布）")
    show = [("真实数据", Y, GREEN, None), ("50 步", SAMPLES[50], CYAN, W2[50])]
    if seg >= 1:
        show.append(("4 步", SAMPLES[4], AMBER, W2[4]))
    if seg >= 2:
        show.append(("1 步", SAMPLES[1], RED, W2[1]))
    for i, (t, X, col, w) in enumerate(show):
        x0 = 80 + i * 290
        scatter(d, (x0, 170, x0 + 260, 430), X, col)
        vk.text_c(d, (x0 + 130, 150), t, F_SMALL, col)
        if w is not None:
            vk.text_c(d, (x0 + 130, 455), f"距离 {w:.3f}", F_SMALL, col)
    if seg >= 1:
        d.text((80, 520), "距离：sliced Wasserstein-2（4000 个样本）；两组真实样本之间约 " + f"{FLOOR:.3f}", font=F_SMALL, fill=MUTED)
    if seg >= 2:
        d.text((80, 570), "一步去噪 = 所有可能结果的平均 → 圆环的平均是圆心", font=F_BODY, fill=RED)


def sc_distill(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "蒸馏：学生学会教师的多步结果")
    rows = [("渐进蒸馏（Salimans & Ho, 2022）", "每轮把步数减半：1024 → … → 4", CYAN),
            ("一致性模型（Song et al., 2023）", "轨迹上任意一点 → 直接映射到终点", VIOLET)]
    for i, (a, b, col) in enumerate(rows):
        d.text((120, 150 + i * 70), a, font=F_BODY, fill=col)
        d.text((640, 154 + i * 70), b, font=F_SMALL, fill=TEXT)
    if seg >= 1:
        scatter(d, (120, 320, 380, 580), XS, VIOLET)
        vk.text_c(d, (250, 600), f"学生 1 步：距离 {W2S:.3f}", F_SMALL, VIOLET)
        scatter(d, (440, 320, 700, 580), SAMPLES[50], CYAN)
        vk.text_c(d, (570, 600), f"教师 50 步：距离 {W2[50]:.3f}", F_SMALL, CYAN)
    if seg >= 2:
        vk.text_c(d, (960, 420), "准备训练对：", F_BODY, TEXT)
        vk.text_c(d, (960, 480), f"20,000 × 50 = {REG['teacher_evals']:,}", vk.font(30, mono=True), AMBER)
        vk.text_c(d, (960, 530), "次教师网络计算", F_BODY, TEXT)


def sc_limits(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "逐样本回归的局限")
    for i, t in enumerate(["一步映射极其复杂 → 学生拟合不准 → 往往更模糊", "学生只能模仿教师的每一个具体样本", "准备训练对的代价高"]):
        d.text((160, 180 + i * 70), "· " + t, font=F_BODY, fill=TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 480), "只要求：学生的分布 = 教师的分布？", vk.font(34), AMBER)
        vk.text_c(d, (640, 540), "→ 下一期：分布匹配蒸馏（DMD）", F_BODY, GREEN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("步数", "每一步 = 一次完整网络计算"), ("少步采样", "变模糊；一步退化为平均"),
                      ("蒸馏", "学会多步结果；回归代价高、易模糊")], seg, "下一期：DMD —— 分布匹配蒸馏")


RENDER = {"title": sc_title, "cost": sc_cost, "steps": sc_steps, "distill": sc_distill, "limits": sc_limits,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vg-fewstep")
