"""Optimization 02 — 随机梯度与批量大小.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] OP_CACHE=<dir> python op02_sgd.py <out_dir>

Real computations (opx.batch_size, JAX): plain SGD on an MLP [64, 128, 10] with the 1,197 training images
of the 8x8 digits; for each batch size B = 4 ... 1,024 the fewest steps (best learning rate from
0.01 / 0.03 / 0.1 / 0.3 / 1 / 3, full-training-set loss checked after every step, at most 20,000 steps)
until the training loss is <= 0.1; examples processed = steps x B.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import opx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Optimization"
CACHE = os.environ.get("OP_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "opcache"))

BS = opx.cached(CACHE, "bs", opx.batch_size)
BB = sorted(BS)
STEPS = np.array([BS[b][0] for b in BB])
EX = STEPS * np.array(BB)
S = dict(zip(BB, STEPS))
print(dict(zip(BB, STEPS)), dict(zip(BB, EX)))
assert S[4] == 592 and S[8] == 286 and S[32] == 111 and S[64] == 85 and S[1024] == 74
assert S[128] == 83 and S[256] == 85 and S[512] == 77 and EX[-1] > 30 * EX[0]

SCENES = [
    ("title", [
        "上一期的梯度下降，每一步都用全部数据计算梯度。真实的训练里数据太多，做不到。这一期看随机梯度下降，以及一个实际的问题，批量应该取多大。",
    ]),
    ("sgd", [
        "随机梯度下降每一步只随机取一小批样本，用它们的平均梯度代替真实的梯度。这个估计是无偏的，但带有噪声，批量越小，噪声越大。",
        "噪声并不全是坏事，它让每一步都很便宜，还可能帮助模型跳出不好的区域。问题是，增大批量能让训练快多少。",
    ]),
    ("experiment", [
        "我们在手写数字上用随机梯度下降训练一个小网络。对每个批量大小，从一组学习率里挑最好的，记录训练损失降到零点一需要多少步。",
        f"批量是四时需要{int_zh(S[4])}步，八时{int_zh(S[8])}步，三十二时{int_zh(S[32])}步，六十四时{int_zh(S[64])}步。在这一段，批量翻倍，步数大约减半，相当于用更多的并行计算换来了更少的步数。",
        f"但是超过六十四以后，步数几乎不再下降。批量一千零二十四时仍然要{int_zh(S[1024])}步，处理的样本数却是批量四的约{int_zh(round(EX[-1] / EX[0]))}倍，大部分计算都浪费了。",
    ]),
    ("critical", [
        "这个拐点叫临界批量。批量小于它时，梯度的噪声是主要瓶颈，增大批量几乎线性地减少步数；大于它时，瓶颈变成了地形本身的弯曲，再大的批量也走不快。",
        "研究者用梯度噪声的大小来估计临界批量。它还会随训练变化，训练越往后、损失越低，临界批量往往越大，所以大模型训练常常在过程中逐渐增大批量。",
    ]),
    ("linear", [
        "增大批量时，学习率通常也要相应增大。一条经验规则是线性缩放，批量乘以几，学习率就乘以几，配合预热。二零一七年有研究用这个方法，把批量增大到八千多，在一小时内训练完了图像网络的分类模型。",
        "但线性缩放只在临界批量以下成立，超过以后，学习率也会被稳定性的上限卡住。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，随机梯度下降用一小批样本估计梯度，便宜但有噪声。",
        "第二，批量小于临界批量时，翻倍批量约能减半步数；超过以后几乎不再加速。",
        "第三，增大批量时学习率常按线性规则放大，但也有上限。下一期，我们看动量怎样在狭长的山谷里加速。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 2, "随机梯度与批量大小", "批量越大越好吗？", SERIES)


def sc_sgd(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "随机梯度下降（SGD）")
    tv.fpaste(c, r"$g_t=\frac{1}{B}\sum_{i\in\mathcal{B}_t}\nabla \ell_i(\theta_t),\qquad \mathbb{E}[g_t]=\nabla f(\theta_t)$", 640, 230, 36)
    vk.text_c(d, (640, 330), "无偏，但有噪声：批量 B 越小，噪声越大", F_BODY, AMBER)
    if seg >= 1:
        vk.text_c(d, (640, 440), "每一步很便宜 · 可能帮助跳出不好的区域", F_SMALL, GREEN)
        vk.text_c(d, (640, 500), "问题：增大批量，训练能快多少？", F_BODY, TEXT)


def sc_experiment(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "手写数字 · MLP · 训练损失降到 0.1 需要多少步")
    n = [0, 5, 9][min(seg, 2)]
    if n:
        tv.line_chart(d, (140, 170, 760, 560), np.array(BB), [("所需步数（最佳学习率）", STEPS, CYAN)], xlog=True, ylog=True,
                      ylim=(1.7, 2.9), upto=n, xlabel="批量大小（对数）", ylabel="步数（对数）", legend_xy=(400, 190),
                      xticks=[(b, str(b)) for b in (4, 16, 64, 256, 1024)])
        d.text((830, 190), "批量", font=F_SMALL, fill=MUTED)
        d.text((960, 190), "步数", font=F_SMALL, fill=MUTED)
        d.text((1080, 190), "样本数", font=F_SMALL, fill=MUTED)
        for i, (b, s, e) in enumerate(list(zip(BB, STEPS, EX))[:n]):
            y = 228 + i * 40
            f = vk.font(22, mono=True)
            d.text((830, y), f"{b}", font=f, fill=TEXT)
            d.text((960, y), f"{s}", font=f, fill=CYAN)
            d.text((1080, y), f"{e:,}", font=f, fill=RED if e > 20000 else MUTED)
    if seg >= 2:
        vk.text_c(d, (450, 640), "超过 64 以后：步数几乎不降，计算大量浪费", F_BODY, RED)


def sc_critical(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "临界批量（McCandlish et al., 2018）")
    tv.line_chart(d, (140, 170, 760, 560), np.array(BB), [("所需步数", STEPS, CYAN)], xlog=True, ylog=True, ylim=(1.7, 2.9),
                  xlabel="批量大小（对数）", xticks=[(b, str(b)) for b in (4, 16, 64, 256, 1024)])
    px = 140 + (np.log10(64) - np.log10(4)) / (np.log10(1024) - np.log10(4)) * 620
    d.line([(px, 170), (px, 560)], fill=AMBER, width=2)
    d.text((px + 8, 180), "≈ 64", font=F_SMALL, fill=AMBER)
    d.text((820, 230), "小于临界批量：噪声是瓶颈", font=F_SMALL, fill=GREEN)
    d.text((820, 270), "→ 翻倍批量 ≈ 减半步数", font=F_SMALL, fill=TEXT)
    d.text((820, 340), "大于临界批量：地形的弯曲是瓶颈", font=F_SMALL, fill=RED)
    d.text((820, 380), "→ 再大的批量也走不快", font=F_SMALL, fill=TEXT)
    if seg >= 1:
        d.text((820, 470), "梯度噪声尺度估计临界批量", font=F_SMALL, fill=AMBER)
        d.text((820, 510), "损失越低，临界批量往往越大", font=F_SMALL, fill=AMBER)


def sc_linear(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "线性缩放规则（Goyal et al., 2017）")
    vk.text_c(d, (640, 200), "批量 × k  →  学习率 × k（配合预热）", F_BODY, CYAN)
    vk.text_c(d, (640, 270), "批量 8,192 · 一小时训练完 ImageNet 分类模型", F_SMALL, MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 400), "只在临界批量以下成立；之后学习率被稳定性上限卡住", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("SGD", "一小批样本估计梯度"), ("临界批量", "以下翻倍减半步数，以上不再加速"),
                      ("线性缩放", "学习率随批量放大，但有上限")], seg, "下一期：动量")


RENDER = {"title": sc_title, "sgd": sc_sgd, "experiment": sc_experiment, "critical": sc_critical, "linear": sc_linear, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "op-sgd")
