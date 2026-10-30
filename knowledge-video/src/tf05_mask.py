"""Transformer 05 — 因果掩码与三种架构.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python tf05_mask.py <out_dir>

Real computations: the causal attention pattern of the trained 2-digit-addition model (tfm.py);
changing the last token and measuring how much the outputs at every position change (exactly zero
before it, thanks to the mask).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import tfm
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import dec_zh, int_zh

AMBER, RED = tv.AMBER, tv.RED
CACHE = os.environ.get("TF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "tfcache"))

PARAMS, _, _ = tfm.cached_train(CACHE, steps=600, log_every=20, snapshots=(0, 60, 120, 200, 300, 600))
_, TE, _, _ = tfm.dataset()
SEQ = TE[0].copy()
TXT = "".join(tfm.VOCAB[i] for i in SEQ)
ALT = SEQ.copy()
ALT[-1] = (ALT[-1] + 5) % 10  # change the last digit
ALT_TXT = "".join(tfm.VOCAB[i] for i in ALT)
LO1, ATT = tfm.forward(PARAMS, SEQ[None], xp=np, return_attn=True)
LO2 = tfm.forward(PARAMS, ALT[None], xp=np)
DIFF = np.abs(np.asarray(LO2)[0] - np.asarray(LO1)[0]).max(-1)
print(TXT, ALT_TXT, np.round(DIFF, 3))
assert (DIFF[:-1] == 0).all() and DIFF[-1] > 1
A = np.asarray(ATT[0])[0, 0]  # layer 1, head 1, (T, T)
assert np.allclose(np.triu(A, 1), 0)
T = len(SEQ)

SCENES = [
    ("title", [
        "这一期讨论两个问题，第一，生成文字的模型，为什么不能偷看后面的词；第二，变换器有哪几种不同的架构。",
    ]),
    ("why", [
        "语言模型是从左往右，一个词元接一个词元地生成的。每一步只能根据已经写出来的内容，预测下一个词元。",
        f"训练时，我们希望一次性地在每个位置都做预测，长度为{int_zh(T)}的序列，可以同时提供{int_zh(T - 1)}个预测下一个词元的练习。但如果每个位置都能看到整句话，答案就直接摆在眼前了。",
    ]),
    ("mask", [
        "解决办法叫因果掩码。在计算注意力分数时，把每个位置之后的分数都设成负无穷。",
        "经过软最大函数以后，这些位置的权重就严格等于零。注意力矩阵变成了一个下三角矩阵，每个位置只能看到自己和前面的位置。",
        "右边是训练好的加法模型第一层的真实注意力，上三角全是零。",
    ]),
    ("test", [
        f"我们来验证一下。把最后一个数字换掉，再把两句话分别送进模型，比较每个位置的输出。",
        f"前面{int_zh(T - 1)}个位置的输出完全没有变化，差别严格等于零；只有最后一个位置变了，最大差别约为{dec_zh(DIFF[-1], 1)}。也就是说，后面的词完全影响不到前面的预测。",
        "这个性质还带来一个好处，生成时前面位置的计算结果可以保存下来反复使用，这就是下下期要讲的键值缓存。",
    ]),
    ("arch", [
        "根据是否使用掩码，变换器发展出三种架构。",
        "第一种只有编码器，没有掩码，每个词元能看到整句话，适合理解类的任务，比如文本分类。",
        "第二种只有解码器，使用因果掩码，适合生成，今天的大语言模型几乎都是这一种。",
        "第三种是编码器加解码器，也就是最初用于机器翻译的变换器。解码器除了看自己生成的内容，还通过交叉注意力读取编码器的输出。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，因果掩码让每个位置只能看到自己和前面的词元。",
        "第二，这样一个序列可以同时训练所有位置，而且不会泄露答案。",
        "第三，编码器、解码器和编码器解码器，是变换器的三种主要架构。",
        "下一期，我们真正从零训练一个小变换器，看它怎样学会两位数加法。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 5, "因果掩码与三种架构", "不偷看后面的词 · 编码器与解码器")


def sc_why(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "从左往右生成")
    toks = list(TXT)
    n = T if seg >= 1 else 1 + int((T - 1) * vk.ease(u * 2 if u < .5 else 1))
    x = 220
    for i, t in enumerate(toks[:n]):
        tv.token(d, x + i * 92, 200, t, CYAN, w=72, h=60, font=vk.font(30, mono=True))
    if seg >= 1:
        for i in range(T - 1):
            xc = x + i * 92 + 36
            vk.arrow(d, (xc, 280), (xc + 92, 380), VIOLET, 2)
            tv.token(d, x + (i + 1) * 92, 390, toks[i + 1], VIOLET, w=72, h=50)
        vk.text_c(d, (640, 500), f"一条长度 {T} 的序列 = {T - 1} 个“预测下一个词元”的练习，一次前向全部算完", F_BODY, TEXT)
        vk.text_c(d, (640, 560), "但每个位置都不能看到它要预测的答案", F_BODY, AMBER)


def sc_mask(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "因果掩码")
    tv.fpaste(c, r"$\mathrm{Attention}=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}+M\right)V,\qquad M_{ij}=\begin{cases}0 & j\leq i\\ -\infty & j>i\end{cases}$".replace(
        r"\begin{cases}0 & j\leq i\\ -\infty & j>i\end{cases}", r"0\ (j\leq i),\ -\infty\ (j>i)"), 640, 140, 26)
    labs = list(TXT)
    f = vk.font(18, mono=True)
    x0, y0, cw = 120, 220, 42
    for i in range(T):
        for j in range(T):
            col = (12, 40, 52) if j <= i else (60, 20, 28)
            d.rectangle([x0 + j * cw, y0 + i * cw, x0 + (j + 1) * cw - 3, y0 + (i + 1) * cw - 3], fill=col)
            if j > i:
                d.text((x0 + (j + .5) * cw, y0 + (i + .5) * cw), "-∞", font=f, fill=RED, anchor="mm")
    for i, lab in enumerate(labs):
        d.text((x0 + (i + .5) * cw, y0 - 14), lab, font=f, fill=MUTED, anchor="mm")
        d.text((x0 - 12, y0 + (i + .5) * cw), lab, font=f, fill=MUTED, anchor="rm")
    vk.text_c(d, (x0 + T * cw / 2, y0 + T * cw + 26), "分数 + 掩码", F_SMALL, MUTED)
    if seg >= 1:
        vk.arrow(d, (520, 410), (620, 410), MUTED, 3)
        M = np.where(np.tril(np.ones((T, T))) > 0, A, np.nan) if seg < 2 else A
        tv.heat(d, (700, y0, 700 + T * cw, y0 + T * cw), M, vmin=0, vmax=1, labels_x=labs, labels_y=labs,
                fmt="{:.1f}" if seg >= 2 else None, font_size=13)
        vk.text_c(d, (700 + T * cw / 2, y0 + T * cw + 26),
                  "训练好的模型 · 第 1 层头 1 的注意力权重" if seg >= 2 else "softmax 之后：上三角严格为 0", F_SMALL, GREEN)


def sc_test(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "验证：改动最后一个词元")
    for r, (txt, col) in enumerate([(TXT, CYAN), (ALT_TXT, VIOLET)]):
        for i, t in enumerate(txt):
            hl = i == T - 1 and r == 1
            tv.token(d, 260 + i * 82, 150 + r * 76, t, AMBER if hl else col, w=64, h=56, font=vk.font(28, mono=True))
    if seg >= 1:
        tv.bar_chart(d, (240, 360, 1000, 580), [str(i) for i in range(T)], list(DIFF), [GREEN] * (T - 1) + [RED],
                     fmt="{:.2f}", vmax=DIFF.max() * 1.25)
        d.text((1010, 600), "位置", font=F_SMALL, fill=MUTED)
        d.text((240, 330), "每个位置输出（logits）的最大变化", font=F_SMALL, fill=MUTED)
        vk.text_c(d, (640, 650), f"位置 0–{T - 2}：变化严格为 0.0 · 位置 {T - 1}：{DIFF[-1]:.2f}", F_BODY, TEXT)


def arch_card(d, x, title, sub, mask, items, col, active):
    w = 340
    d.rounded_rectangle([x, 150, x + w, 560], 16, outline=col if active else (30, 41, 59), width=3, fill=(12, 19, 34))
    vk.text_c(d, (x + w / 2, 190), title, F_BODY, col if active else MUTED)
    vk.text_c(d, (x + w / 2, 230), sub, F_SMALL, MUTED)
    n, cw, x0, y0 = 5, 26, x + w / 2 - 65, 270
    for i in range(n):
        for j in range(n):
            on = mask(i, j)
            d.rectangle([x0 + j * cw, y0 + i * cw, x0 + (j + 1) * cw - 3, y0 + (i + 1) * cw - 3],
                        fill=col if (on and active) else ((50, 60, 80) if on else (20, 26, 38)))
    for k, it in enumerate(items):
        vk.text_c(d, (x + w / 2, 430 + k * 36), it, F_SMALL, TEXT if active else MUTED)


def sc_arch(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "三种架构")
    arch_card(d, 80, "只有编码器", "Encoder-only", lambda i, j: True, ["双向注意力，无掩码", "理解：分类、检索", "BERT"], CYAN, seg >= 1)
    arch_card(d, 470, "只有解码器", "Decoder-only", lambda i, j: j <= i, ["因果掩码", "生成：大语言模型", "GPT、LLaMA"], VIOLET, seg >= 2)
    arch_card(d, 860, "编码器 + 解码器", "Encoder–Decoder", lambda i, j: j <= i or j < 2, ["解码器 + 交叉注意力", "翻译、摘要", "原始 Transformer、T5"], GREEN, seg >= 3)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("因果掩码", "每个位置只看自己和前面"), ("并行训练", "一次前向训练所有位置，不泄露答案"),
                      ("三种架构", "编码器 / 解码器 / 编码器-解码器")], seg, "下一期：从零训练一个会做加法的 Transformer")


RENDER = {"title": sc_title, "why": sc_why, "mask": sc_mask, "test": sc_test, "arch": sc_arch, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "mask")
