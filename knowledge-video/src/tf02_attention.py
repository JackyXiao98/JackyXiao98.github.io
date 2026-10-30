"""Transformer 02 — 注意力机制：查询、键与值.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python tf02_attention.py <out_dir>

Real computations: a worked single-query attention example; softmax saturation without the
1/sqrt(d) scaling (20,000 random draws, d = 64); attention of the trained 2-digit-addition model
(tfm.py, cached in $TF_CACHE or ./out/tfcache).
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
from zhnum import dec_zh, int_zh, pct_zh

CACHE = os.environ.get("TF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "tfcache"))


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# worked example: one query, four keys/values (2-D)
Q1 = np.array([1.0, 0.5])
K4 = np.array([[1.0, 0.2], [0.1, 1.0], [-0.8, 0.3], [0.9, 0.6]])
V4 = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.6, 0.8]])
SC4 = K4 @ Q1 / np.sqrt(2)
W4 = softmax(SC4)
O4 = W4 @ V4
ORD = "一二三四"
TOP, LOW = int(W4.argmax()), int(W4.argmin())
assert LOW == 2  # the key pointing away from the query

# scaling experiment
rng = np.random.default_rng(0)
D, N, TR = 64, 10, 20000
Z = np.einsum("td,tnd->tn", rng.standard_normal((TR, D)), rng.standard_normal((TR, N, D)))
P_RAW, P_SC = softmax(Z), softmax(Z / np.sqrt(D))
STD_RAW = float(Z.std())
MAX_RAW, MAX_SC = float(P_RAW.max(1).mean()), float(P_SC.max(1).mean())


def jac_norm(p):
    J = np.einsum("ti,ij->tij", p, np.eye(p.shape[1])) - p[:, :, None] * p[:, None, :]
    return np.linalg.norm(J, axis=(1, 2))


JR, JS = float(np.median(jac_norm(P_RAW[:5000]))), float(np.median(jac_norm(P_SC[:5000])))
print(f"std {STD_RAW:.2f} max weight {MAX_RAW:.3f} vs {MAX_SC:.3f}  jac median {JR:.3f} vs {JS:.3f}")
assert 7.7 < STD_RAW < 8.3 and 0.84 < MAX_RAW < 0.89 and 0.30 < MAX_SC < 0.34 and 0.17 < JR / JS < 0.25

# trained model attention
PARAMS, HIST, SNAPS = tfm.cached_train(CACHE, steps=600, log_every=20, snapshots=(0, 60, 120, 200, 300, 600))
_, TE, _, _ = tfm.dataset()
_, ATT = tfm.forward(PARAMS, TE[:500, :8], xp=np, return_attn=True)
A0 = ATT[0].mean(0)  # (heads, 8, 8), layer 1 averaged over 500 test problems
UNITS = A0[:, 7, 1] + A0[:, 7, 4]  # attention from position 7 (predicts units digit) to the two units digits
print("units-digit attention per head:", np.round(UNITS, 2))
assert UNITS.min() > 0.8
EX = TE[0]
EX_STR = "".join(tfm.VOCAB[i] for i in EX)

SCENES = [
    ("title", [
        "上一期，我们把文字变成了一串向量。但每个向量只代表它自己，还不知道句子里的其他词。",
        "这一期讲变换器的核心，注意力机制，它让每个词元都能从其他词元那里收集信息。",
    ]),
    ("lookup", [
        "注意力可以看成一次柔性的查字典。",
        "每个词元先生成三个向量，查询、键和值。用自己的查询，去和所有词元的键比较，相似度越高，权重越大，最后按权重，把各个词元的值加起来。",
        "比较用的是点积，再除以向量维度的平方根，然后用软最大函数，把分数变成和为一的权重。",
    ]),
    ("example", [
        "我们算一个具体的例子：一个查询和四个键，都是二维向量。",
        f"四个点积分数经过软最大函数以后，和查询方向最接近的第{ORD[TOP]}个键，得到的权重最大，约为{dec_zh(W4.max())}；和查询方向相反的第三个键，权重只有{dec_zh(W4[2])}。",
        "输出就是四个值向量按这些权重的加权平均。",
    ]),
    ("matrix", [
        "对整个序列，所有词元的注意力可以写成矩阵，一次算完。",
        "分数矩阵的每一行，是一个词元对所有词元的注意力，经过软最大以后，每一行加起来等于一。这种矩阵运算可以高度并行，这是变换器训练快的重要原因。",
    ]),
    ("scale", [
        "为什么要除以维度的平方根呢？如果查询和键的每一维都是标准正态的随机数，点积的标准差大约是维度的平方根。我们在六十四维下做了两万次实验，点积的标准差约为八。",
        f"不缩放时，软最大的输出几乎变成独热的，最大权重平均高达{dec_zh(MAX_RAW)}，梯度的中位数还不到缩放后的四分之一。缩放以后，最大权重平均为{dec_zh(MAX_SC)}，分布柔和得多，也更容易训练。",
    ]),
    ("trained", [
        "最后，看一个真正训练过的变换器。它学会了两位数加法，我们会在第六期详细介绍它。",
        f"当它要写出和的个位时，第一层的四个注意力头，都把{pct_zh(np.floor(UNITS.min() * 100) / 100)}以上的注意力，放在两个加数的个位数字上。这正是做加法时最需要看的位置。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，注意力用查询和键的点积计算权重，再对值做加权平均。",
        "第二，整个序列的注意力可以用矩阵一次算完，适合并行。",
        "第三，除以维度的平方根，让分数保持在合适的范围，避免软最大饱和。",
        "下一期，我们讲多头注意力和位置编码。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 2, "注意力机制", "查询、键与值")


def sc_lookup(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "柔性的查字典")
    toks = ["我", "喜欢", "这", "本", "书"]
    for i, t in enumerate(toks):
        x = 160 + i * 200
        tv.token(d, x, 150, t, TEXT, w=120, font=vk.font(26))
        if seg >= 1:
            for j, (lab, col) in enumerate([("q", CYAN), ("k", VIOLET), ("v", GREEN)]):
                tv.token(d, x + j * 42, 220, lab, col, w=36, h=36, font=vk.font(20, mono=True))
    if seg >= 1:
        vk.text_c(d, (640, 330), "查询 q 与每个键 k 比较 → 权重 → 对值 v 加权求和", F_BODY, TEXT)
    if seg >= 2:
        tv.fpaste(c, r"$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V$", 640, 450, 36, "#22d3ee")


def sc_example(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "一个具体的例子（二维）")
    box = (90, 130, 520, 560)
    d.rectangle(box, outline=(30, 41, 59))
    cx, cy, sc = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 170
    vk.arrow(d, (cx, cy), (cx + Q1[0] * sc, cy - Q1[1] * sc), CYAN, 4)
    d.text((cx + Q1[0] * sc + 8, cy - Q1[1] * sc - 18), "q", font=vk.font(26, mono=True), fill=CYAN)
    for i, kk in enumerate(K4):
        w = W4[i] if seg >= 1 else 0.4
        col = tuple(int(70 + (v - 70) * min(1, w * 2.2)) for v in VIOLET)
        vk.arrow(d, (cx, cy), (cx + kk[0] * sc, cy - kk[1] * sc), col, 3)
        d.text((cx + kk[0] * sc + 6, cy - kk[1] * sc + 4), f"k{i + 1}", font=vk.font(22, mono=True), fill=col)
    rows = [("分数 q·k/√2", SC4, "{:+.2f}", 0), ("权重 softmax", W4, "{:.2f}", 1)]
    for r, (lab, vals, fmt, need) in enumerate(rows):
        if seg >= need:
            y = 180 + r * 110
            d.text((600, y), lab, font=F_SMALL, fill=MUTED)
            for i, v in enumerate(vals):
                tv.token(d, 600 + i * 130, y + 34, fmt.format(v), VIOLET if r == 0 else CYAN, w=110, font=vk.font(22, mono=True))
    if seg >= 2:
        tv.fpaste(c, r"$o=\sum_i w_i\,v_i=(" + f"{O4[0]:.2f},\\ {O4[1]:.2f}" + ")$", 860, 480, 32, "#34d399")


def sc_matrix(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "矩阵形式：一次算完整个序列")
    X = np.random.default_rng(3).standard_normal((6, 16))
    Wq, Wk = np.random.default_rng(4).standard_normal((2, 16, 16)) / 4
    A = softmax((X @ Wq) @ (X @ Wk).T / 4)
    labs = ["我", "喜欢", "这", "本", "书", "。"]
    tv.heat(d, (330, 150, 690, 510), A, vmin=0, vmax=A.max(), labels_x=labs, labels_y=labs, fmt="{:.2f}", font_size=16)
    vk.text_c(d, (510, 545), "注意力权重矩阵（随机权重示意，每行和为 1）", F_SMALL, MUTED)
    if seg >= 1:
        for i in range(6):
            d.text((710, 150 + (i + .5) * 60), f"Σ = {A[i].sum():.2f}", font=vk.font(20, mono=True), fill=GREEN, anchor="lm")
        vk.text_c(d, (1020, 330), "矩阵乘法\n高度并行", F_BODY, CYAN)


def sc_scale(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "为什么除以 √d")
    tv.fpaste(c, r"$q,k\sim\mathcal{N}(0,I_d)\ \Rightarrow\ \mathrm{std}(q\cdot k)=\sqrt{d}$", 640, 170, 30)
    if seg >= 0 and u > 0.25:
        vk.text_c(d, (640, 230), f"实验（d = 64，20000 次）：点积标准差 {STD_RAW:.2f}", F_BODY, TEXT)
    if seg >= 1:
        i = 0
        for k, (name, p, col) in enumerate([("不缩放", P_RAW[i], (248, 113, 113)), ("除以 √d", P_SC[i], CYAN)]):
            bx = 170 + k * 520
            box = (bx, 300, bx + 420, 520)
            d.line([(box[0], box[3]), (box[2], box[3])], fill=(71, 85, 105), width=2)
            for j, v in enumerate(p):
                x = box[0] + 10 + j * 40
                d.rectangle([x, box[3] - v * 200, x + 28, box[3]], fill=col)
            vk.text_c(d, ((box[0] + box[2]) / 2, 545), name, F_BODY, col)
        vk.text_c(d, (380, 590), f"最大权重均值 {MAX_RAW:.2f} · 梯度中位数 {JR:.3f}", F_SMALL, (248, 113, 113))
        vk.text_c(d, (900, 590), f"最大权重均值 {MAX_SC:.2f} · 梯度中位数 {JS:.3f}", F_SMALL, CYAN)


def sc_trained(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "训练好的模型在看哪里")
    labs = list(EX_STR[:8])
    vk.text_c(d, (640, 135), f"任务：两位数加法，例如 {EX_STR[:6]}{EX_STR[6:]}", F_BODY, TEXT)
    if seg >= 1:
        for h in range(4):
            x0 = 80 + h * 300
            M = np.full((8, 8), np.nan)
            M[np.tril_indices(8)] = A0[h][np.tril_indices(8)]
            tv.heat(d, (x0 + 30, 200, x0 + 270, 440), M, vmin=0, vmax=1, labels_x=labs, labels_y=labs, font_size=14)
            d.rectangle([x0 + 30, 200 + 7 * 30, x0 + 270, 200 + 8 * 30], outline=(251, 191, 36), width=2)
            vk.text_c(d, (x0 + 150, 470), f"第 1 层 · 头 {h + 1}", F_SMALL, MUTED)
            vk.text_c(d, (x0 + 150, 500), f"个位数字合计 {UNITS[h]:.0%}", F_SMALL, CYAN)
        vk.text_c(d, (640, 560), "黄框：要写出和的个位时的那一行（对 500 道测试题取平均）", F_SMALL, MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 6, [("注意力", "查询·键 → 权重 → 对值加权平均"), ("矩阵形式", "一次算完整个序列，适合并行"),
                      ("缩放 √d", "避免软最大饱和，梯度更健康")], seg, "下一期：多头注意力与位置编码")


RENDER = {"title": sc_title, "lookup": sc_lookup, "example": sc_example, "matrix": sc_matrix, "scale": sc_scale,
          "trained": sc_trained, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "attention")
