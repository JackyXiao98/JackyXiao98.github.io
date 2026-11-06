"""State Space Models 02 — 线性注意力：把 softmax 拿掉.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] SSM_CACHE=<dir> python ssm02_linattn.py <out_dir>

Real computations: (1) ssmx.linattn_parallel / linattn_recurrent (numpy): causal linear attention with feature map
elu(x) + 1 on random q, k, v (T = 32, d = 8), computed as a masked matrix product and as a recurrence over a d x d
state; largest difference. (2) ssmx.exp_lm (softmax attention) and ssmx.exp_lm_linpos (linear attention), JAX:
2-layer width-64 models with learned position embeddings on Tiny Shakespeare, 2,000 steps, context 128; validation
loss every 200 steps.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import ssmx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "State Space Models"
CACHE = os.environ.get("SSM_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "ssmcache"))

r = np.random.default_rng(0)
q, k, v = r.normal(size=(3, 32, 8))
DIFF = float(np.abs(ssmx.linattn_parallel(q, k, v) - ssmx.linattn_recurrent(q, k, v)).max())
LM = ssmx.cached(CACHE, "lm", ssmx.exp_lm)
LP = ssmx.cached(CACHE, "lm_linpos", ssmx.exp_lm_linpos)
A, L = LM["attn"], LP
print(DIFF, A["final"], L["final"])
assert DIFF < 1e-12 and round(A["final"]["val"], 3) == 1.771 and round(L["final"]["val"], 3) == 1.895

SCENES = [
    ("title", [
        "注意力的平方代价来自柔性最大值：每个查询要和所有的键算完分数，归一化之后才能加权求和。二零二零年的一篇论文提出：如果把柔性最大值换成一个简单的核函数，注意力就可以写成循环网络。这一期讲线性注意力。",
    ]),
    ("trick", [
        "关键是改变计算的顺序。普通注意力先算查询和键的乘积，得到一个长度乘长度的矩阵，再乘以值。如果没有柔性最大值，就可以先把键和值乘起来，得到一个固定大小的矩阵，再用查询去读。",
        "对于因果注意力，这个固定大小的矩阵可以一步一步累加：每读一个词元，把它的键和值的外积加进去。这就是一个循环网络，状态就是这个矩阵。",
    ]),
    ("exact", [
        f"我们验证这两种算法完全等价。用三十二个位置、八维的随机数据，先算整张注意力矩阵再加掩码，和一步一步累加状态，结果最大只差约十的负十六次方。",
        "所以线性注意力训练时可以像注意力一样并行，生成时可以像循环网络一样，每一步的代价固定，不需要键值缓存。",
    ]),
    ("cost", [
        f"代价是什么？我们在莎士比亚文本上训练两个只在注意力上不同的小模型。普通注意力的验证损失是{dec_zh(A['final']['val'], 2)}，线性注意力是{dec_zh(L['final']['val'], 2)}，差了不少。",
        "原因是，柔性最大值能产生非常尖锐的注意力，几乎只看某一个词元；而线性的核函数做不到这么尖锐。所有过去的信息被加在同一个矩阵里，越加越模糊，难以精确地取回某个词元。",
    ]),
    ("later", [
        "后来的工作想了很多办法弥补：给状态加上衰减，让旧信息逐渐淡出；用更好的特征映射；或者像增量规则那样，写入前先减去旧的内容。这些改进让线性注意力和状态空间模型越来越接近，第八期我们会看到，它们其实可以统一起来。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，去掉柔性最大值，就能先乘键和值，注意力变成一个状态为固定矩阵的循环网络。",
        "第二，并行形式和循环形式完全等价，我们的实验验证了这一点。",
        "第三，代价是注意力不够尖锐，在我们的实验里损失明显更高。下一期，我们看状态空间模型。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 2, "线性注意力", "把 softmax 拿掉", SERIES)


def sc_trick(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "改变计算顺序")
    tv.fpaste(c, r"$\mathrm{softmax}(QK^\top)\,V\quad\rightarrow\quad \phi(Q)\left(\phi(K)^\top V\right)$", 640, 210, 38)
    d.text((300, 280), "T × T 矩阵", font=F_SMALL, fill=RED)
    d.text((780, 280), "d × d 矩阵（与长度无关）", font=F_SMALL, fill=GREEN)
    if seg >= 1:
        tv.fpaste(c, r"$S_t=S_{t-1}+\phi(k_t)\,v_t^\top,\qquad y_t=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}$", 640, 420, 36)
        vk.text_c(d, (640, 530), "循环网络：状态 S 是一个固定大小的矩阵", F_BODY, GREEN)


def sc_exact(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "并行形式 = 循环形式")
    d.text((200, 250), "T = 32，d = 8，φ(x) = elu(x) + 1：最大差别", font=F_BODY, fill=TEXT)
    d.text((940, 250), f"{DIFF:.0e}", font=vk.font(30, mono=True), fill=GREEN)
    if seg >= 1:
        vk.text_c(d, (640, 400), "训练：像注意力一样并行 · 生成：像 RNN 一样，每步代价固定", F_BODY, CYAN)


def sc_cost(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "莎士比亚：softmax 注意力 vs 线性注意力（2,000 步）")
    xs = [h["step"] for h in A["hist"]]
    ser = [("softmax 注意力", [h["val"] for h in A["hist"]][1:], CYAN), ("线性注意力", [h["val"] for h in L["hist"]][1:], AMBER)]
    tv.line_chart(d, (140, 200, 800, 580), xs[1:], ser, ylim=(1.6, 2.6), xticks=[(200, "200"), (1000, "1000"), (2000, "2000")], xlabel="训练步数", legend_xy=(860, 220))
    d.text((140, 165), "验证损失", font=F_SMALL, fill=MUTED)
    d.text((860, 330), f"{A['final']['val']:.3f} vs {L['final']['val']:.3f}", font=F_BODY, fill=TEXT)
    if seg >= 1:
        d.text((860, 400), "softmax：可以很尖锐", font=F_SMALL, fill=CYAN)
        d.text((860, 440), "线性核：所有信息叠在一起", font=F_SMALL, fill=AMBER)


def sc_later(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "后来的改进")
    for i, (t, col) in enumerate([("给状态加衰减：旧信息逐渐淡出（RetNet、GLA）", CYAN), ("更好的特征映射", VIOLET),
                                  ("增量规则：写入前减去旧内容（DeltaNet）", GREEN), ("→ 与状态空间模型统一（第 8 期）", AMBER)]):
        vk.text_c(d, (640, 210 + i * 75), t, F_BODY if i < 3 else F_SMALL, col)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("线性注意力", "先乘 K、V，变成循环网络"), ("等价", "并行形式 = 循环形式"),
                      ("代价", "不够尖锐，损失更高")], seg, "下一期：状态空间模型")


RENDER = {"title": sc_title, "trick": sc_trick, "exact": sc_exact, "cost": sc_cost, "later": sc_later, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "ssm-linattn")
