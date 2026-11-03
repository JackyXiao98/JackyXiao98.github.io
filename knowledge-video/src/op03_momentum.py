"""Optimization 03 — 动量：在狭长山谷里加速.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] OP_CACHE=<dir> python op03_momentum.py <out_dir>

Real computations (opx.valley_paths, opx.condition): f = 0.5 (x^2 / 50 + y^2) from (10, 1): gradient descent
with lr = 2 / (L + mu) vs heavy-ball momentum with the optimal (alpha, beta), 60 steps; steps for f to drop by
10^6 on f = 0.5 (x^2 / kappa + y^2) for kappa = 10 ... 10,000: GD (lr = 2 / (L + mu)), heavy ball (optimal),
Nesterov (lr = 1 / L, beta = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)).
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

V = opx.cached(CACHE, "valley", opx.valley_paths)
CO = opx.cached(CACHE, "cond", opx.condition)
KS = sorted(CO)
GD, HB, NAG = ([CO[k][m] for k in KS] for m in ("gd", "hb", "nag"))
HB38 = opx.steps_to(1e-6 * V["f_hb"][0], V["f_hb"])
print(V["f_gd"][[20, 60]], V["f_hb"][[20, 60]], HB38, GD, HB, NAG)
assert HB38 == 38 and V["f_gd"][60] > 0.01 and abs(V["b"] - 0.566) < 1e-3
assert GD == [35, 346, 3454, 34539] and HB == [16, 58, 204, 708] and NAG == [21, 63, 165, 387]
RED_GD = V["f_gd"][0] / V["f_gd"][60]

SCENES = [
    ("title", [
        "上一期讲到，地形越狭长，梯度下降越慢。这一期看一个简单而有效的改进，动量。",
    ]),
    ("valley", [
        "想象一条狭长的山谷，两边的山坡很陡，谷底沿着长轴方向却很平缓。梯度主要指向陡的山坡，所以梯度下降会在两壁之间来回弹跳，沿着谷底却只前进一点点。",
    ]),
    ("idea", [
        "动量的想法来自物理里滚下山的球。每一步的更新，不只看当前的梯度，还加上上一步的运动方向，乘以一个系数贝塔，通常是零点九左右。",
        "来回振荡的分量方向不断改变，相加时会互相抵消；沿着谷底的分量方向一致，会不断累积。于是振荡被压住，沿谷底的速度越来越快。这叫重球法。",
    ]),
    ("paths", [
        f"在一个长宽比为五十比一的山谷里，从同一点出发，各用最好的参数走六十步。梯度下降六十步后，损失只降到了原来的{int_zh(round(RED_GD))}分之一左右。",
        f"加了动量以后，只用{int_zh(HB38)}步，损失就下降了一百万倍。",
    ]),
    ("scaling", [
        "理论上，用最好的参数，梯度下降需要的步数和条件数成正比，而动量需要的步数只和条件数的平方根成正比。",
        f"我们实际计算，条件数是一万时，梯度下降需要{big_zh(GD[3])}步，重球动量只要{int_zh(HB[3])}步，涅斯捷罗夫加速法{int_zh(NAG[3])}步。",
        "涅斯捷罗夫的方法先沿着动量方向看一步，再在那个位置计算梯度，相当于提前修正方向。今天几乎所有的神经网络训练，都在用某种形式的动量。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，动量把过去的运动方向累积起来，抵消振荡、加速前进。",
        "第二，在狭长的山谷里，动量把所需步数从和条件数成正比，降到和它的平方根成正比。",
        "第三，涅斯捷罗夫方法先看一步再算梯度，进一步加速。下一期，我们看给每个参数各自步长的自适应方法，以及亚当。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 3, "动量", "在狭长山谷里加速", SERIES)


def _valley(d, box, paths, n):
    x0, y0, x1, y1 = box
    cx, cy = x0 + (x1 - x0) * 0.08, (y0 + y1) / 2
    sx, sy = (x1 - x0) * 0.88 / 10.5, (y1 - y0) / 2 / 1.15
    for lev in (0.02, 0.1, 0.3, 0.6, 1.0, 1.5):
        rx, ry = np.sqrt(2 * lev * 50) * sx, np.sqrt(2 * lev) * sy
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=(40, 52, 72), width=1)
    for P, col in paths:
        pts = [(cx + p[0] * sx, cy - p[1] * sy) for p in P[: n + 1]]
        d.line(pts, fill=col, width=2)
        for q in pts:
            d.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=col)


def sc_valley(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "狭长的山谷：梯度下降在两壁之间弹跳")
    _valley(d, (100, 180, 1180, 560), [(V["gd"], RED)], int(10 + u * 50))
    vk.text_c(d, (640, 620), "梯度主要指向陡的山坡 · 沿谷底只前进一点点", F_SMALL, MUTED)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "动量：像滚下山的球（重球法，Polyak 1964）")
    tv.fpaste(c, r"$\theta_{t+1}=\theta_t-\eta\nabla f(\theta_t)+\beta\,(\theta_t-\theta_{t-1})$", 640, 230, 38)
    vk.text_c(d, (640, 310), "β ≈ 0.9：保留上一步的运动方向", F_SMALL, AMBER)
    if seg >= 1:
        vk.text_c(d, (640, 420), "振荡分量方向不断改变 → 互相抵消", F_BODY, RED)
        vk.text_c(d, (640, 480), "沿谷底分量方向一致 → 不断累积", F_BODY, GREEN)


def sc_paths(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "长宽比 50 : 1 的山谷，各走 60 步（最佳参数）")
    series = [("梯度下降", np.maximum(V["f_gd"], 1e-12), RED)] + ([("动量", np.maximum(V["f_hb"], 1e-12), GREEN)] if seg >= 1 else [])
    tv.line_chart(d, (140, 170, 800, 580), np.arange(61), series, ylog=True, ylim=(-12, 1), xlabel="步数", ylabel="损失（对数）",
                  legend_xy=(170, 470), xticks=[(0, "0"), (20, "20"), (40, "40"), (60, "60")])
    d.text((860, 230), "梯度下降 60 步：", font=F_SMALL, fill=RED)
    d.text((860, 265), f"损失 {V['f_gd'][0]:.2f} → {V['f_gd'][60]:.4f}", font=F_SMALL, fill=TEXT)
    if seg >= 1:
        d.text((860, 350), f"动量 {HB38} 步：", font=F_SMALL, fill=GREEN)
        d.text((860, 385), "损失下降 100 万倍", font=F_SMALL, fill=TEXT)


def sc_scaling(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "所需步数：∝ κ  vs  ∝ √κ")
    series = [("梯度下降 ∝ κ", GD, RED)]
    if seg >= 1:
        series += [("重球动量 ∝ √κ", HB, GREEN), ("Nesterov", NAG, CYAN)]
    tv.line_chart(d, (140, 170, 760, 560), np.array(KS), series, xlog=True, ylog=True, ylim=(1, 5), xlabel="条件数 κ（对数）",
                  ylabel="步数（对数）", legend_xy=(170, 220), xticks=[(k, f"{k:,}") for k in KS])
    if seg >= 1:
        for i, (lab, vals, col) in enumerate([("GD", GD, RED), ("重球", HB, GREEN), ("Nesterov", NAG, CYAN)]):
            d.text((820, 240 + i * 50), lab, font=F_SMALL, fill=col)
            d.text((1180, 240 + i * 50), f"{vals[3]:,}", font=vk.font(24, mono=True), fill=col, anchor="ra")
        d.text((820, 200), "κ = 10,000", font=F_SMALL, fill=MUTED)
    if seg >= 2:
        d.text((820, 420), "Nesterov：先沿动量看一步", font=F_SMALL, fill=CYAN)
        d.text((820, 455), "再在那里算梯度", font=F_SMALL, fill=CYAN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("动量", "累积运动方向，抵消振荡"), ("加速", "步数从 ∝ κ 降到 ∝ √κ"),
                      ("Nesterov", "先看一步再算梯度")], seg, "下一期：自适应方法与 Adam")


RENDER = {"title": sc_title, "valley": sc_valley, "idea": sc_idea, "paths": sc_paths, "scaling": sc_scaling, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "op-momentum")
