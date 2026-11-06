"""Long Context 02 — 位置编码：绝对、相对与旋转.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python lc02_rope.py <out_dir>

Real computations (lcx.rope_np, numpy): rotary position embedding of a random 16-dim query / key pair; the score
q(m+5) . k(m) for m = 0, 10, 100, 1,000 (identical up to rounding: it depends on the distance only); the 8 rotation
frequencies base^(-2i/d) for base 10,000 and their wavelengths 2 pi / theta_i in positions; the score as a function
of distance for the same pair.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import lcx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Long Context"

r = np.random.default_rng(0)
q, k = r.normal(size=16), r.normal(size=16)
score = lambda mq, mk: float(lcx.rope_np(q[None], np.array([float(mq)]))[0] @ lcx.rope_np(k[None], np.array([float(mk)]))[0])
SAME = [score(m + 5, m) for m in (0, 10, 100, 1000)]
THETA = 10000.0 ** (-np.arange(0, 16, 2) / 16)
WAVE = 2 * np.pi / THETA
DIST = np.arange(0, 200)
SC = np.array([score(dd, 0) for dd in DIST])
print(SAME, np.round(WAVE, 1))
assert max(SAME) - min(SAME) < 1e-9 and round(WAVE[0], 2) == 6.28 and round(WAVE[-1]) == 19869

SCENES = [
    ("title", [
        "注意力本身不知道词元的先后顺序：把一句话打乱，每个词元看到的集合是一样的。所以变换器需要位置编码。位置编码的设计，直接决定了模型能不能处理比训练时更长的文本。这一期看三类位置编码，重点是今天最常用的旋转位置编码。",
    ]),
    ("absolute", [
        "最早的做法是绝对位置编码：给每个位置一个向量，加到词元的表示上。可以用固定的正弦余弦函数，也可以让模型自己学。",
        "问题是，学出来的位置向量只覆盖训练时见过的长度。训练只用了两千个位置，第三千个位置的向量就从来没被训练过。",
    ]),
    ("relative", [
        "人们发现，注意力真正需要的往往是相对位置：这个词离我有多远，而不是它在第几个位置。相对位置编码把距离信息直接加进注意力的分数里。",
    ]),
    ("rope", [
        "旋转位置编码用了一个巧妙的办法：把查询和键的每两个维度看成一个平面上的向量，按位置旋转一个角度，位置越靠后转得越多。不同的维度对用不同的旋转速度。",
        "两个向量做点积时，旋转的角度相减，所以分数只取决于两个位置的差，也就是相对距离。我们验证，同一对查询和键，相隔五个位置，不管放在第零、第十、第一百还是第一千个位置，算出的分数完全相同。",
    ]),
    ("freq", [
        "旋转的速度从快到慢呈指数分布。在我们的十六维例子里，最快的一对维度每六点多个位置转一圈，最慢的要将近两万个位置才转一圈。快的维度分辨相邻的词，慢的维度表示远距离。",
        "这也埋下了一个隐患：如果训练时最长只见过几千个位置，那些最慢的维度连一圈都没有转完。测试时遇到更远的距离，就会出现训练中从未见过的角度。下一期我们会看到它带来的问题。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，注意力本身不分先后，需要位置编码。",
        "第二，学出来的绝对位置只覆盖训练长度。",
        "第三，旋转位置编码让注意力分数只依赖相对距离，但慢的维度在训练长度内转不完一圈。下一期，我们看训练短、测试长时会发生什么。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 2, "位置编码", "绝对、相对与旋转", SERIES)


def sc_absolute(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "绝对位置编码：每个位置一个向量")
    for i in range(8):
        tv.token(d, 140 + i * 130, 220, f"x{i + 1}", CYAN, w=90, font=vk.font(20, mono=True))
        tv.token(d, 140 + i * 130, 300, f"p{i + 1}", VIOLET, w=90, font=vk.font(20, mono=True))
        vk.text_c(d, (185 + i * 130, 282), "+", F_SMALL, MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 450), "学出来的位置向量：只覆盖训练时见过的长度", F_BODY, RED)


def sc_relative(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "相对位置：离我多远")
    tv.fpaste(c, r"$s_{ij}=q_i\cdot k_j+b(i-j)$", 640, 260, 46)


def sc_rope(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "旋转位置编码（RoPE）")
    cx, cy, R = 300, 360, 150
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(51, 65, 85), width=2)
    for m, col in ((0, CYAN), (1, GREEN), (2, AMBER), (3, RED)):
        ang = 0.5 + m * 0.45
        vk.arrow(d, (cx, cy), (cx + R * np.cos(ang), cy - R * np.sin(ang)), col, 3)
        d.text((cx + (R + 25) * np.cos(ang), cy - (R + 25) * np.sin(ang)), f"m={m}", font=vk.font(18, mono=True), fill=col, anchor="mm")
    tv.fpaste(c, r"$\mathrm{RoPE}(x,m)_{(2i,2i+1)}=R(m\,\theta_i)\,x_{(2i,2i+1)}$", 860, 230, 28)
    if seg >= 1:
        tv.fpaste(c, r"$\langle R(m\theta)q,\,R(n\theta)k\rangle=\langle q,\,R((n-m)\theta)k\rangle$", 860, 310, 28)
        f = vk.font(22, mono=True)
        d.text((620, 390), "相隔 5 个位置的分数：", font=F_SMALL, fill=MUTED)
        for i, (m, sv) in enumerate(zip((0, 10, 100, 1000), SAME)):
            d.text((640, 430 + i * 38), f"m = {m:<5}  {sv:.12f}", font=f, fill=GREEN)


def sc_freq(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "旋转速度：从快到慢（d = 16，底数 10,000）")
    tv.bar_chart(d, (120, 220, 620, 560), [str(i) for i in range(8)], np.log10(WAVE), [CYAN] * 8, fmt="{:.1f}", vmax=5)
    d.text((120, 180), "转一圈需要的位置数（log10）", font=F_SMALL, fill=MUTED)
    d.text((620, 622), "维度对", font=F_SMALL, fill=MUTED, anchor="rm")
    d.text((700, 240), f"最快：{WAVE[0]:.1f} 个位置一圈", font=F_SMALL, fill=TEXT)
    d.text((700, 280), f"最慢：{WAVE[-1]:,.0f} 个位置一圈", font=F_SMALL, fill=TEXT)
    if seg >= 1:
        d.text((700, 380), "训练只见过几千个位置：", font=F_SMALL, fill=RED)
        d.text((700, 420), "慢的维度连一圈都没转完", font=F_SMALL, fill=RED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("位置编码", "注意力本身不分先后"), ("绝对位置", "只覆盖训练长度"),
                      ("RoPE", "分数只依赖相对距离")], seg, "下一期：长度外推")


RENDER = {"title": sc_title, "absolute": sc_absolute, "relative": sc_relative, "rope": sc_rope, "freq": sc_freq, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "lc-rope")
