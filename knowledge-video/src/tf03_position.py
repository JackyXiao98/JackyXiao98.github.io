"""Transformer 03 — 多头注意力与位置编码.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python tf03_position.py <out_dir>

Real computations: permutation equivariance of attention without positions; sinusoidal position
encodings (dot product depends only on the offset); RoPE (q_m . k_n depends only on m - n);
attention heads of the trained 2-digit-addition model (tfm.py) looking at different digits.
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
from zhnum import pct_zh

CACHE = os.environ.get("TF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "tfcache"))
rng = np.random.default_rng(0)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# 1. order: attention without positions is permutation-equivariant
EMB = {w: rng.standard_normal(16) for w in ["狗", "咬", "人"]}
WQ, WK, WV = (rng.standard_normal((16, 16)) / 4 for _ in range(3))


def attend(X):
    return softmax((X @ WQ) @ (X @ WK).T / 4) @ (X @ WV)


S1, S2 = ["狗", "咬", "人"], ["人", "咬", "狗"]
O1 = attend(np.array([EMB[w] for w in S1]))
O2 = attend(np.array([EMB[w] for w in S2]))
PERM_DIFF = float(np.abs(O2[[2, 1, 0]] - O1).max())
assert PERM_DIFF < 1e-12


# 2. sinusoidal positions
def sinusoid(n, d=64):
    pos = np.arange(n)[:, None]
    i = np.arange(d // 2)[None]
    ang = pos / 10000 ** (2 * i / d)
    out = np.zeros((n, d))
    out[:, 0::2], out[:, 1::2] = np.sin(ang), np.cos(ang)
    return out


PE = sinusoid(100)
OFFS = [1, 5, 20]
DOTS = {k: np.array([PE[p] @ PE[p + k] for p in range(60)]) for k in OFFS}
PE_SPREAD = max(float(np.ptp(v)) for v in DOTS.values())
assert PE_SPREAD < 1e-10


# 3. RoPE
def rope(x, m, d=64):
    i = np.arange(d // 2)
    th = m / 10000 ** (2 * i / d)
    c, s = np.cos(th), np.sin(th)
    out = np.empty_like(x)
    out[0::2], out[1::2] = x[0::2] * c - x[1::2] * s, x[0::2] * s + x[1::2] * c
    return out


QV, KV = rng.standard_normal(64), rng.standard_normal(64)
MS = [3, 10, 50, 99]
ROPE_VALS = [float(rope(QV, m) @ rope(KV, m - 3)) for m in MS]
assert np.ptp(ROPE_VALS) < 1e-9

# 4. heads of the trained model
PARAMS, _, _ = tfm.cached_train(CACHE, steps=600, log_every=20, snapshots=(0, 60, 120, 200, 300, 600))
_, TE, _, _ = tfm.dataset()
_, ATT = tfm.forward(PARAMS, TE[:500, :8], xp=np, return_attn=True)
A0 = ATT[0].mean(0)
H_B, H_A = A0[0, 5, 3], A0[1, 5, 0]  # predicting the hundreds digit: head 1 -> b's tens, head 2 -> a's tens
print(f"perm diff {PERM_DIFF:.1e}  PE spread {PE_SPREAD:.1e}  rope {np.round(ROPE_VALS, 6)}  heads {H_B:.2f} {H_A:.2f}")
assert A0[0, 5].argmax() == 3 and A0[1, 5].argmax() == 0 and H_B > 0.7 and H_A > 0.6
EX = "".join(tfm.VOCAB[i] for i in TE[0])
D_MODEL, N_HEADS = 512, 8
PARAMS_MHA = 4 * D_MODEL * D_MODEL

SCENES = [
    ("title", [
        "上一期我们认识了注意力。这一期解决两个问题，第一，注意力怎么知道词的顺序；第二，为什么要用多个注意力头。",
    ]),
    ("order", [
        "先看一个有点意外的事实，注意力本身并不知道词的顺序。",
        "我们把狗咬人和人咬狗这两句话，分别送进同一个注意力层。",
        "结果两句话得到的输出向量完全一样，只是排列顺序不同，差别在十的负十二次方以下，只是计算误差。对注意力来说，这两句话是一样的。",
    ]),
    ("sinusoid", [
        "解决办法是给每个位置加上一个位置编码。原始的变换器用的是不同频率的正弦和余弦函数。",
        "图中每一行是一个位置的编码，越往右频率越低。",
        "这种编码有一个漂亮的性质，两个位置编码的点积，只取决于它们相隔多远，而和具体在哪个位置无关。",
    ]),
    ("rope", [
        "现在很多大模型用的是旋转位置编码。",
        "它把查询和键的每两维看成一个平面上的向量，按照位置旋转一个角度。",
        "这样，旋转之后查询与键的点积，只取决于两个位置的差，而不是它们各自的绝对位置。我们实际算了四组相隔三个位置的点积，结果完全相同。",
    ]),
    ("heads", [
        "再说多头注意力。与其用一个大的注意力，不如把向量切成几份，每份做一次独立的注意力，最后再拼起来。",
        "在维度不变时，多头注意力的参数量和单头一样，但不同的头可以关注不同的东西。",
        f"在训练好的加法模型里，当它要写出和的百位时，第一个头主要看第二个加数的十位，约占{pct_zh(H_B)}；第二个头主要看第一个加数的十位，约占{pct_zh(H_A)}。两个头分工合作。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，注意力本身对顺序不敏感，必须加入位置信息。",
        "第二，正弦位置编码和旋转位置编码，都让注意力能感知相对位置。",
        "第三，多头注意力在参数量不变的情况下，让不同的头关注不同的信息。",
        "下一期，我们把注意力和其他部件组装成完整的变换器模块。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 3, "多头注意力与位置编码", "顺序从哪里来 · 为什么要多个头")


def sc_order(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "注意力不知道顺序")
    if seg >= 1:
        for r, (sent, O, col) in enumerate([(S1, O1, CYAN), (S2, O2, VIOLET)]):
            y = 150 + r * 200
            tv.tokens_row(d, 120, y, sent, col, font=vk.font(28))
            tv.heat(d, (420, y - 10, 1020, y + 140), O[:, :12], cmap="cyan" if r == 0 else "violet", labels_y=sent)
        vk.text_c(d, (720, 125), "输出向量（前 12 维）", F_SMALL, MUTED)
    if seg >= 2:
        vk.text_c(d, (640, 560), f"重新排列后完全相同：最大差异 {PERM_DIFF:.1e}", F_BODY, GREEN)


def sc_sinusoid(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "正弦位置编码")
    tv.fpaste(c, r"$PE_{(p,2i)}=\sin\frac{p}{10000^{2i/d}},\qquad PE_{(p,2i+1)}=\cos\frac{p}{10000^{2i/d}}$", 640, 150, 26)
    if seg >= 1:
        tv.heat(d, (120, 210, 620, 560), PE[:40], vmin=-1, vmax=1)
        d.text((120, 570), "位置 0–39（行） × 维度 0–63（列）", font=F_SMALL, fill=MUTED)
    if seg >= 2:
        box = (720, 230, 1180, 520)
        d.rectangle(box, outline=(30, 41, 59))
        for j, (k, col) in enumerate(zip(OFFS, [CYAN, VIOLET, GREEN])):
            v = DOTS[k]
            y = box[3] - (v[0] - 10) / 25 * (box[3] - box[1])
            d.line([(box[0], y), (box[2], y)], fill=col, width=3)
            d.text((box[2] - 8, y - 16), f"相隔 {k}：{v[0]:.2f}", font=F_SMALL, fill=col, anchor="rm")
        vk.text_c(d, (950, 545), f"PE(p)·PE(p+k)，p = 0…59：随 p 的变化 ≈ {PE_SPREAD:.0e}（计算误差）", F_SMALL, MUTED)


def sc_rope(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "旋转位置编码（RoPE）")
    cx, cy, r = 330, 360, 170
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(30, 41, 59), width=2)
    for m, col in [(0, MUTED), (1, CYAN), (2, VIOLET), (3, GREEN)]:
        a = 0.35 + m * 0.5 * (vk.ease(su) if seg == 1 else (1 if seg > 1 else 0))
        vk.arrow(d, (cx, cy), (cx + r * np.cos(a), cy - r * np.sin(a)), col, 3)
        d.text((cx + (r + 22) * np.cos(a), cy - (r + 22) * np.sin(a)), f"位置 {m}", font=F_SMALL, fill=col, anchor="mm")
    tv.fpaste(c, r"$x'_{2i}=x_{2i}\cos m\theta_i-x_{2i+1}\sin m\theta_i$", 880, 190, 24)
    tv.fpaste(c, r"$x'_{2i+1}=x_{2i}\sin m\theta_i+x_{2i+1}\cos m\theta_i$", 880, 240, 24)
    tv.fpaste(c, r"$\theta_i=10000^{-2i/d}$", 880, 285, 22, "#94a3b8")
    if seg >= 2:
        tv.fpaste(c, r"$R_m q\cdot R_n k=q\cdot R_{n-m}\,k$", 880, 340, 30, "#34d399")
        for i, (m, v) in enumerate(zip(MS, ROPE_VALS)):
            d.text((720, 400 + i * 36), f"m = {m:2d},  n = m − 3:   {v:+.6f}", font=vk.font(22, mono=True), fill=TEXT)


def sc_heads(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "多头注意力")
    if seg <= 1:
        tv.fpaste(c, r"$\mathrm{MultiHead}(X)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W^O$", 640, 160, 24)
        tv.fpaste(c, r"$\mathrm{head}_j=\mathrm{Attention}(XW_j^Q,\,XW_j^K,\,XW_j^V)$", 640, 210, 24)
        for j in range(4):
            x = 220 + j * 220
            d.rounded_rectangle([x, 260, x + 180, 340], 12, outline=[CYAN, VIOLET, GREEN, (251, 191, 36)][j], width=3)
            vk.text_c(d, (x + 90, 300), f"头 {j + 1}（d/h 维）", F_SMALL, TEXT)
        if seg >= 1:
            vk.text_c(d, (640, 420), f"参数量与头数无关：4·d² = {PARAMS_MHA:,}（d = {D_MODEL}，任意 h）", F_BODY, GREEN)
        return
    labs = list(EX[:8])
    for h, (col, val, name) in enumerate([(CYAN, H_B, "第二个加数的十位"), (VIOLET, H_A, "第一个加数的十位")]):
        x0 = 170 + h * 520
        M = np.full((8, 8), np.nan)
        M[np.tril_indices(8)] = A0[h][np.tril_indices(8)]
        tv.heat(d, (x0, 150, x0 + 320, 470), M, vmin=0, vmax=1, labels_x=labs, labels_y=labs, cmap="cyan" if h == 0 else "violet")
        d.rectangle([x0, 150 + 5 * 40, x0 + 320, 150 + 6 * 40], outline=(251, 191, 36), width=2)
        vk.text_c(d, (x0 + 160, 500), f"头 {h + 1}：{name} {val:.0%}", F_BODY, col)
    vk.text_c(d, (640, 560), "黄框：要写出和的百位时（500 道测试题平均）", F_SMALL, MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("顺序", "注意力本身对顺序不敏感，需要位置信息"), ("位置编码", "正弦编码 / RoPE：感知相对位置"),
                      ("多头", "参数不变，不同的头关注不同信息")], seg, "下一期：Transformer 模块 —— 残差、层归一化与前馈网络")


RENDER = {"title": sc_title, "order": sc_order, "sinusoid": sc_sinusoid, "rope": sc_rope, "heads": sc_heads, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "position")
