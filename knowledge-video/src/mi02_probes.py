"""Interpretability 02 — 探针：模型内部知道什么.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MI_CACHE=<dir> TF_CACHE=<dir> python mi02_probes.py <out_dir>

Real computations (mix.probes): carry probes on the trained addition model vs the same architecture at
random initialisation (step-0 snapshot of the same training run); shuffled-label control (labels
permuted, probe with weak regularisation) at layer 2, position 6.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import mix
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Interpretability"
CACHE = os.environ.get("MI_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "micache"))
TF_CACHE = os.environ.get("TF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "tfcache"))

PR = mix.cached(CACHE, "probe", mix.probes, tf_cache=TF_CACHE)
A, R = PR["trained"], PR["random"]
RMAX = R[1:].max()
TOK = ["3", "7", "+", "4", "5", "=", "0", "8"]
print(RMAX, PR["shuffled_train"], PR["shuffled_test"])
assert abs(RMAX - 0.872) < 1e-3 and A[1:, 4].min() == 1.0 and abs(PR["shuffled_test"] - 0.53) < 1e-9
assert PR["shuffled_train"] < 0.56 and PR["true_test"] == 1.0 and abs(A[0, 1] - 0.763) < 1e-3

SCENES = [
    ("title", [
        "上一期我们用线性探针，在加法模型里读出了进位。这一期看探针的陷阱，探针读得出来，不等于模型真的在用。",
    ]),
    ("random", [
        "第一个对照实验。我们拿一个结构完全一样、但从来没有训练过的随机网络，用同样的方法去训练探针。",
        f"结果出乎意料，在随机网络里，探针最高也能读出{pct_zh(RMAX, 1)}的进位。随机网络当然不会做加法，但它把输入的数字随机地混在一起，线性分类器已经能从这些混合里猜出大半。",
        "所以探针的准确率，要和随机网络这个基线比较才有意义。训练好的模型达到百分之百，比随机网络高出一截，这一截才是训练带来的。",
    ]),
    ("input", [
        f"第二个陷阱是，信息可能直接来自输入。在嵌入层，只看第一个数的个位数字本身，探针就能达到{pct_zh(A[0, 1])}，因为个位数字越大，越容易进位。这不需要模型做任何计算。",
    ]),
    ("shuffle", [
        "第三个对照是打乱标签。把进位的标签随机打乱，再训练探针。如果探针足够强大，它可能连随机标签也能记住，那么高准确率就说明不了什么。",
        f"在我们的实验里，打乱标签后，探针在训练集上只有{pct_zh(PR['shuffled_train'], 1)}，测试集上{pct_zh(PR['shuffled_test'])}，和瞎猜差不多。这个线性探针没有能力记住随机标签，说明真实标签上的百分之百是可信的。",
    ]),
    ("causal", [
        "即使通过了这些对照，探针也只能说明信息在那里，不能说明模型用了它。模型可能顺便写下了某个信息，却从来不去读它。",
        "要判断模型是否在用，需要做因果干预，改动内部的激活，看输出会不会跟着变。这就是下一期的激活修补。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，随机网络也能被读出不少信息，探针的结果要和随机基线比较。",
        "第二，要排除直接来自输入的信息，并用打乱标签检验探针是否在死记。",
        "第三，读得出来不等于在用，确认用途需要因果干预。下一期，我们看激活修补。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 2, "探针", "模型内部知道什么？", SERIES)


def _grid(d, M, box, title, col_title):
    x0, y0, x1, y1 = box
    tv.heat(d, box, M[::-1] * 100, 50, 100, labels_x=TOK, fmt="{:.0f}", font_size=18)
    for l, lab in enumerate(["第 2 层后", "第 1 层后", "嵌入"]):
        d.text((x0 - 8, y0 + (l + .5) * (y1 - y0) / 3), lab, font=vk.font(18), fill=MUTED, anchor="rm")
    d.text((x0, y0 - 80), title, font=F_BODY, fill=col_title)


def sc_random(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "对照 1：没有训练过的随机网络")
    _grid(d, R, (760, 250, 1210, 450), "随机网络", RED)
    if seg >= 1:
        vk.text_c(d, (985, 520), f"最高 {RMAX:.1%}", F_BODY, RED)
    if seg >= 2:
        _grid(d, A, (150, 250, 600, 450), "训练好的模型", GREEN)
        vk.text_c(d, (375, 520), "最高 100%", F_BODY, GREEN)
        vk.text_c(d, (640, 620), "要和随机基线比：高出的那一截才是训练带来的", F_BODY, AMBER)


def sc_input(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "陷阱 2：信息直接来自输入")
    tv.heat(d, (240, 250, 1040, 330), (A[:1] * 100), 50, 100, labels_x=TOK, fmt="{:.0f}", font_size=22)
    d.text((225, 290), "嵌入", font=F_SMALL, fill=MUTED, anchor="rm")
    d.rectangle([340, 250, 440, 328], outline=AMBER, width=3)
    vk.text_c(d, (640, 420), f"只看个位数字 7 本身：{A[0, 1]:.0%}（个位越大越容易进位）", F_BODY, AMBER)
    vk.text_c(d, (640, 480), "不需要模型做任何计算", F_SMALL, MUTED)


def sc_shuffle(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "对照 3：打乱标签（第 2 层后，位置“0”）")
    vk.text_c(d, (640, 170), "探针够强 → 连随机标签也能记住 → 高准确率说明不了什么", F_SMALL, MUTED)
    if seg >= 1:
        labs = ["真实标签 · 训练", "真实标签 · 测试", "打乱 · 训练", "打乱 · 测试"]
        vals = [PR["true_train"] * 100, PR["true_test"] * 100, PR["shuffled_train"] * 100, PR["shuffled_test"] * 100]
        tv.bar_chart(d, (160, 240, 1120, 560), labs, vals, [GREEN, GREEN, RED, RED], fmt="{:.1f}%", vmax=115)
        y = 560 - PR["base_rate"] * 100 / 115 * 320
        d.line([(160, y), (1120, y)], fill=MUTED, width=1)
        d.text((1125, y), "多数类", font=F_SMALL, fill=MUTED, anchor="lm")


def sc_causal(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "读得出来 ≠ 在用")
    d.rounded_rectangle([140, 190, 600, 330], 14, outline=CYAN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (370, 240), "探针（相关）", F_BODY, CYAN)
    vk.text_c(d, (370, 290), "信息在那里", F_SMALL, TEXT)
    if seg >= 1:
        d.rounded_rectangle([680, 190, 1140, 330], 14, outline=GREEN, width=2, fill=(12, 19, 34))
        vk.text_c(d, (910, 240), "因果干预", F_BODY, GREEN)
        vk.text_c(d, (910, 290), "改动激活 → 输出会不会变？", F_SMALL, TEXT)
        vk.arrow(d, (605, 260), (675, 260), MUTED, 3)
        vk.text_c(d, (640, 430), "下一期：激活修补", F_BODY, VIOLET)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("随机基线", f"随机网络也能读出 {RMAX:.0%}"), ("其他对照", "排除输入 · 打乱标签"),
                      ("因果", "读得出来 ≠ 在用")], seg, "下一期：激活修补")


RENDER = {"title": sc_title, "random": sc_random, "input": sc_input, "shuffle": sc_shuffle, "causal": sc_causal, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "mi-probes")
