"""Video Generation 05 — DMD：分布匹配蒸馏.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] VG_CACHE=<dir> python vg05_dmd.py <out_dir>

Real computations (vgx.train_dmd, JAX): Distribution Matching Distillation of the exact ring teacher
into a one-step MLP generator with a learned "fake" score network; snapshots during training and
sliced Wasserstein-2 distances against true samples, the 50-step teacher and 1-step DDIM.
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
SNAPS = vgx.cached(CACHE, "dmd", vgx.train_dmd)
SHOW = [0, 1000, 2000, 8000]
XS = {k: vgx.sample_generator(SNAPS[k], Z) for k in SHOW}
W2 = {k: D.sliced_w2(XS[k], Y) for k in SHOW}
T50, T1 = D.sliced_w2(D.ddim(G, Z, 50), Y), D.sliced_w2(D.ddim(G, Z, 1), Y)
print(W2, T50, T1, FLOOR)
assert W2[0] > 1.2 and W2[1000] < 0.3 and W2[8000] < 0.06 and abs(W2[8000] - FLOOR) < 0.01 and T1 > 1.3

SCENES = [
    ("title", [
        "上一期我们看到，逐样本地模仿教师代价高、容易模糊。这一期介绍分布匹配蒸馏，它只要求学生生成的分布和教师一样。",
    ]),
    ("idea", [
        "设想一个一步生成器，输入噪声，直接输出样本。我们希望它生成的分布，和真实数据的分布一致。",
        "分布匹配蒸馏用到两个分数函数。分数就是概率密度对数的梯度，指向数据更密集的方向。第一个是真实分数，由冻结的教师扩散模型给出；第二个是虚假分数，由另一个扩散模型在生成器的输出上实时训练得到。",
        "生成器的更新方向，就是两个分数之差。把样本往真实分布密集的地方推，同时远离生成器自己已经过度集中的地方。这个方向正好是两个分布之间散度的梯度。",
    ]),
    ("toy", [
        "我们在圆环玩具上实际训练了一个一步生成器。教师就是上一期的精确去噪器，虚假分数是一个小网络。",
        f"开始时生成器输出一团点，和真实分布的距离是{dec_zh(W2[0], 2)}。训练一千步以后降到{dec_zh(W2[1000], 2)}，圆环已经出现。",
        f"训练八千步以后，距离只有{dec_zh(W2[8000], 3)}，和两组真实样本之间的差距{dec_zh(FLOOR, 3)}几乎一样。",
    ]),
    ("compare", [
        f"对比一下。教师用一步采样，距离是{dec_zh(T1, 2)}，所有点挤在圆心；教师用五十步采样是{dec_zh(T50, 3)}；分布匹配蒸馏的学生只用一步，就达到了{dec_zh(W2[8000], 3)}。",
        "而且训练时不需要事先准备噪声和结果的配对，学生也不必复现教师的某一条具体轨迹。",
    ]),
    ("video", [
        "后续的改进版本去掉了对配对数据的依赖，加入了对抗损失，并支持多步生成。",
        "在视频上，研究者用分布匹配蒸馏，把需要几十步的双向视频模型，蒸馏成只要几步的自回归生成器。上一期讲的自我强迫训练，用的也正是这种分布匹配损失。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，分布匹配蒸馏只要求学生的分布和教师一致，不需要逐样本模仿。",
        "第二，它的梯度是虚假分数和真实分数之差，虚假分数由另一个扩散模型在线学习。",
        "第三，它把多步模型蒸馏成一步到几步的生成器，是实时视频生成的关键。下一期，我们转向另一个瓶颈，注意力。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 5, "DMD：分布匹配蒸馏", "只要求分布一致，不要求逐样本模仿", SERIES)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "DMD（Yin et al., 2024）")
    d.rounded_rectangle([120, 150, 400, 220], 12, outline=CYAN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (260, 185), "一步生成器 G(z)", F_BODY, CYAN)
    if seg >= 1:
        for i, (t, col, sub) in enumerate([("真实分数 s_real", GREEN, "冻结的教师扩散模型"),
                                           ("虚假分数 s_fake", RED, "在 G 的输出上在线训练")]):
            y = 260 + i * 110
            d.rounded_rectangle([120, y, 520, y + 86], 12, outline=col, width=2, fill=(12, 19, 34))
            d.text((140, y + 26), t, font=F_BODY, fill=col, anchor="lm")
            d.text((140, y + 60), sub, font=F_SMALL, fill=MUTED, anchor="lm")
    if seg >= 2:
        tv.fpaste(c, r"$\nabla_\theta D_{KL}\approx\mathbb{E}\left[\left(s_{\mathrm{fake}}(x_t)-s_{\mathrm{real}}(x_t)\right)\frac{\partial G_\theta(z)}{\partial\theta}\right]$", 880, 300, 26)
        vk.text_c(d, (880, 380), "推向真实分布密集处，远离自己过度集中处", F_SMALL, AMBER)
        vk.text_c(d, (880, 420), "x_t：生成样本加噪后的版本", F_SMALL, MUTED)


def scatter(d, box, X, col, n=1500):
    x0, y0, x1, y1 = box
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 6.0
    d.rectangle(box, outline=(30, 41, 59))
    for x, y in X[:n]:
        px, py = cx + x * s, cy - y * s
        if x0 < px < x1 and y0 < py < y1:
            d.point((px, py), fill=col)
            d.point((px + 1, py), fill=col)


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "玩具：训练一个一步生成器")
    k = 1 if seg == 0 else (2 if seg == 1 else 4)
    for i, s in enumerate(SHOW[:k]):
        x0 = 80 + i * 290
        scatter(d, (x0, 170, x0 + 260, 430), XS[s], CYAN if s == 8000 else VIOLET)
        vk.text_c(d, (x0 + 130, 150), f"第 {s} 步", F_SMALL, TEXT)
        vk.text_c(d, (x0 + 130, 455), f"距离 {W2[s]:.3f}", F_SMALL, CYAN if s == 8000 else VIOLET)
    if seg >= 2:
        vk.text_c(d, (640, 540), f"两组真实样本之间：{FLOOR:.3f}", F_BODY, GREEN)


def sc_compare(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "对比（距离越小越好）")
    tv.bar_chart(d, (200, 170, 1080, 520), ["教师 1 步", "教师 50 步", "DMD 学生 1 步", "真实 vs 真实"], [T1, T50, W2[8000], FLOOR],
                 [RED, CYAN, VIOLET, GREEN], fmt="{:.3f}", vmax=T1 * 1.15)
    if seg >= 1:
        vk.text_c(d, (640, 600), "不需要预先生成配对；不必复现教师的具体轨迹", F_BODY, AMBER)


def sc_video(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "从 DMD 到实时视频")
    rows = [("DMD2（Yin et al., 2024）", "去掉配对回归损失；加入 GAN 损失；支持多步", CYAN, 0),
            ("CausVid（Yin et al., 2025）", "双向视频模型 → 几步的自回归生成器", VIOLET, 1),
            ("Self-Forcing（Huang et al., 2025）", "在自己的生成结果上用分布匹配损失训练", GREEN, 1)]
    for i, (a, b, col, s) in enumerate(rows):
        if seg >= s:
            y = 170 + i * 110
            d.rounded_rectangle([120, y, 1160, y + 80], 12, outline=col, width=2, fill=(12, 19, 34))
            d.text((150, y + 40), a, font=F_BODY, fill=col, anchor="lm")
            d.text((640, y + 40), b, font=F_SMALL, fill=TEXT, anchor="lm")


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("目标", "学生分布 = 教师分布"), ("梯度", "s_fake − s_real，虚假分数在线学习"),
                      ("作用", "多步 → 1~4 步，实时视频的关键")], seg, "下一期：注意力为什么是瓶颈")


RENDER = {"title": sc_title, "idea": sc_idea, "toy": sc_toy, "compare": sc_compare, "video": sc_video,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vg-dmd")
