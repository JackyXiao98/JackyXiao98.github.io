"""Knowledge Video 07 — Flow Matching 与 Rectified Flow.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python ep07_flow.py <out_dir>

Exact flow matching on the 8-mode ring (same data as episode 04): straight conditional paths
x_t = (1-t) x0 + t x1, exact marginal velocity E[x1 - x0 | x_t]. Euler sampling is compared with
the diffusion probability-flow ODE (DDIM) at equal step counts; crossings / transport cost of
independent vs flow-induced couplings illustrate rectified flow. Numbers are asserted.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import diff2d as D
import fm
import p2d
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL, F_TITLE

ANG = np.arange(8) * np.pi / 4
G = D.GMM(np.stack([2 * np.cos(ANG), 2 * np.sin(ANG)], 1), 0.06)
X0 = np.random.default_rng(1).standard_normal((20000, 2))
STEPS = [3, 5, 10]
FMS = {n: fm.fm_euler(G, X0, n) for n in STEPS}
DDS = {n: D.ddim(G, X0, n) for n in STEPS}
BAD_FM = {n: D.bad_fraction(G, v) for n, v in FMS.items()}
BAD_DD = {n: D.bad_fraction(G, v) for n, v in DDS.items()}
print("FM", BAD_FM, "DDIM", BAD_DD)
_, TR_FM = fm.fm_euler(G, X0[:150], 100, keep=150)

rng = np.random.default_rng(4)
P0 = rng.standard_normal((400, 2))
P1_IND = G.sample(400, rng)
P1_ODE = fm.fm_euler(G, P0, 400)


def crossings(a, b):
    def ccw(p, q, r):
        return (r[..., 1] - p[..., 1]) * (q[..., 0] - p[..., 0]) > (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    i, j = np.triu_indices(len(a), 1)
    A, B, C, E = a[i], b[i], a[j], b[j]
    return float(((ccw(A, C, E) != ccw(B, C, E)) & (ccw(A, B, C) != ccw(A, B, E))).mean())


CROSS_IND, CROSS_ODE = crossings(P0, P1_IND), crossings(P0, P1_ODE)
COST_IND = float(np.mean(np.sum((P1_IND - P0) ** 2, 1)))
COST_ODE = float(np.mean(np.sum((P1_ODE - P0) ** 2, 1)))
print(f"crossings {CROSS_IND:.4f} -> {CROSS_ODE:.4f}   cost {COST_IND:.2f} -> {COST_ODE:.2f}")

assert 0.09 < BAD_FM[5] < 0.13 and 0.72 < BAD_DD[5] < 0.78
assert BAD_FM[10] < 0.006 and 0.17 < BAD_DD[10] < 0.21
assert 0.22 < CROSS_IND < 0.26 and CROSS_ODE < 0.008
assert 5.5 < COST_IND < 6.5 and 0.8 < COST_ODE < 1.3

# ---------------------------------------------------------------------------
# Narration
# ---------------------------------------------------------------------------
SCENES = [
    ("title", [
        "前几期，我们从加噪和去噪的角度来生成数据。这一期换一种思路，直接学习一个把噪声搬运到数据的速度场，这就是流匹配。",
        "一些最新的文生图模型，比如稳定扩散第三代，用的就是这种方法。",
    ]),
    ("idea", [
        "设想每个噪声点，都沿着一条路径流向一个数据点。",
        "流匹配选择最简单的路径，也就是直线。从噪声到数据，位置随时间线性变化，速度就是数据减去噪声，是一个常数。",
        "训练时，随机取一个噪声和一个数据，在它们之间的直线上取一个点，让网络预测这个速度，损失就是预测速度和真实速度的均方误差。",
    ]),
    ("marginal", [
        "但同一个位置，可能有很多条直线经过，它们的方向各不相同。",
        "网络最终学到的，是经过这个位置的所有直线速度的平均值。",
        "所以生成时，沿着这个平均速度场走出来的轨迹，并不一定是直线。在八个峰的例子里，轨迹到后半程才分叉，弯向各自的峰。",
    ]),
    ("compare", [
        "我们和扩散模型的常微分方程比一比，都用最简单的一阶方法。",
        "走五步时，流匹配的坏样本约为百分之十一，扩散模型约为百分之七十五；走十步时，流匹配降到百分之零点四，扩散模型还有百分之十九。",
        "当然，这只是一个玩具例子，不同的方法和求解器各有优劣，但流匹配训练简单、采样高效，这是它流行起来的原因。",
    ]),
    ("reflow", [
        "能不能让轨迹真正变直呢？这就是整流流的想法。",
        "先用训练好的流，把每个噪声点映射到一个数据点，得到新的配对。",
        "左边是随机配对的连线，大约百分之二十四的连线两两相交；右边是按流配对的连线，相交的只有百分之零点五，搬运代价也从六左右降到了一左右。",
        "在新的配对上再训练一次，轨迹会变得更直。原论文显示，这样甚至可以只用一步，就生成不错的结果。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，流匹配学习一个速度场，训练目标是直线路径上的速度，简单直接。",
        "第二，网络学到的是所有路径的平均速度，所以采样轨迹仍然可能弯曲。",
        "第三，整流流通过重新配对，减少交叉，让轨迹更直，从而可以用更少的步数生成。",
        "下一期，我们回到隐私的话题，看看扩散模型会不会记住训练数据。感谢大家的观看。",
    ]),
]

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
EXT = 3.0


def fpaste(c, tex, cx, cy, size=28, color="#e2e8f0"):
    vk.paste_center(c, vk.formula(tex, size, color), cx, cy)


def lines(d, box, a, b, col, n=200, width=1):
    pa, pb = p2d.to_px(box, a[:n], EXT), p2d.to_px(box, b[:n], EXT)
    for p, q in zip(pa, pb):
        d.line([tuple(p), tuple(q)], fill=col, width=width)


def sc_title(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.text_c(d, (640, 190), "Knowledge Video · 07", vk.font(24, mono=True), MUTED)
    vk.text_c(d, (640, 270), "流匹配", F_TITLE, CYAN)
    vk.text_c(d, (640, 340), "Flow Matching 与 Rectified Flow", F_BODY, TEXT)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "直线路径")
    box = (110, 115, 560, 565)
    p2d.frame(d, box)
    if seg >= 1:
        t = vk.ease(su) if seg == 1 else 1.0
        lines(d, box, P0[:40], P1_IND[:40], (60, 80, 110), n=40)
        p2d.scatter(d, box, P0[:40], VIOLET, r=4, ext=EXT)
        p2d.scatter(d, box, (1 - t) * P0[:40] + t * P1_IND[:40], CYAN, r=4, ext=EXT)
    else:
        p2d.scatter(d, box, P0[:300], VIOLET, r=2, ext=EXT)
        p2d.scatter(d, box, P1_IND[:300], CYAN, r=2, ext=EXT)
    vk.text_c(d, (900, 150), "紫：噪声 x0    青：数据 x1", F_SMALL, TEXT)
    if seg >= 1:
        fpaste(c, r"$x_t=(1-t)\,x_0+t\,x_1,\qquad \frac{\mathrm{d}x_t}{\mathrm{d}t}=x_1-x_0$", 900, 250, 28, "#22d3ee")
    if seg >= 2:
        fpaste(c, r"$\mathcal{L}=\mathbb{E}_{t,x_0,x_1}\,\|v_\theta(x_t,t)-(x_1-x_0)\|^2$", 900, 370, 30)


def sc_marginal(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "学到的是平均速度")
    box = (110, 115, 560, 565)
    p2d.frame(d, box)
    lines(d, box, P0[:120], P1_IND[:120], (45, 60, 85), n=120)
    if seg >= 1:
        p2d.arrows(d, box, lambda p: fm.velocity(G, p, 0.5), n=13, ext=EXT, color=(167, 139, 250))
        fpaste(c, r"$v(x,t)=\mathbb{E}\left[x_1-x_0\mid x_t=x\right]$", 900, 200, 30, "#a78bfa")
        vk.text_c(d, (900, 260), "（箭头：t = 0.5 时的速度场）", F_SMALL, MUTED)
    if seg >= 2:
        box2 = (690, 300, 1110, 600)
        k = int((TR_FM.shape[0] - 1) * vk.ease(min(1, su * 1.3)))
        d.rectangle(box2, outline=(30, 41, 59))
        p2d.trails(d, box2, TR_FM[:, :60], k, (34, 150, 170), ext=EXT)
        p2d.scatter(d, box2, TR_FM[k, :150], CYAN, r=2, ext=EXT)


def sc_compare(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "一阶欧拉法：流匹配 vs 扩散 ODE")
    for r, (name, res, bad, col) in enumerate([("流匹配", FMS, BAD_FM, CYAN), ("扩散 ODE（DDIM）", DDS, BAD_DD, VIOLET)]):
        y0 = 120 + r * 225
        vk.text_c(d, (150, y0 + 90), name, F_SMALL, col)
        for i, n in enumerate(STEPS):
            box = (300 + i * 300, y0, 300 + i * 300 + 180, y0 + 180)
            p2d.frame(d, box)
            if seg >= 1 or (seg == 0 and i == 0):
                p2d.scatter(d, box, res[n][:2500], col, r=0.9, ext=EXT)
                b = bad[n]
                d.text((box[2] + 8, box[3] - 12), f"{b:.1%}" if b < 0.1 else f"{b:.0%}", font=vk.font(20, mono=True), fill=col, anchor="lm")
            if r == 0:
                vk.text_c(d, ((box[0] + box[2]) / 2, y0 - 14), f"{n} 步", F_SMALL, TEXT)
    vk.text_c(d, (640, 580), "数字：坏样本比例（2 万个样本）", F_SMALL, MUTED)


def sc_reflow(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "整流流（Rectified Flow）：重新配对")
    if seg <= 1:
        fpaste(c, r"$(x_0,\ x_1)\ \ \longrightarrow\ \ (x_0,\ \Phi(x_0))$", 640, 230, 36, "#22d3ee")
        vk.text_c(d, (640, 310), "Φ：用训练好的流把噪声 x0 送到的数据点", F_BODY, TEXT)
        return
    for i, (b, col, name, cr, cost) in enumerate([(P1_IND, (167, 139, 250), "随机配对", CROSS_IND, COST_IND),
                                                  (P1_ODE, (34, 211, 238), "按流配对", CROSS_ODE, COST_ODE)]):
        box = (120 + i * 560, 110, 120 + i * 560 + 440, 520)
        p2d.frame(d, box)
        lines(d, box, P0, b, tuple(int(v * 0.65) for v in col), n=200)
        p2d.scatter(d, box, b[:200], col, r=2, ext=EXT)
        vk.text_c(d, ((box[0] + box[2]) / 2, box[3] + 25), f"{name}：相交 {cr:.1%}   搬运代价 {cost:.1f}", F_SMALL, col)
    if seg >= 3:
        vk.text_c(d, (640, 590), "在新配对上再训练（reflow）→ 轨迹更直 → 更少步数", F_SMALL, GREEN)


def sc_summary(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "总结")
    items = [
        ("流匹配", "学习速度场，目标是直线路径上的速度"),
        ("平均速度", "多条路径取平均，采样轨迹仍可能弯曲"),
        ("整流流", "重新配对减少交叉，轨迹更直、步数更少"),
    ]
    for i, (k, v) in enumerate(items):
        if seg >= i + 1:
            y = 140 + i * 105
            d.rounded_rectangle([140, y, 1140, y + 82], 14, outline=(30, 41, 59), fill=(12, 19, 34))
            d.text((180, y + 41), f"0{i + 1}", font=vk.font(30, mono=True), fill=CYAN, anchor="lm")
            d.text((260, y + 41), k, font=F_BODY, fill=TEXT, anchor="lm")
            d.text((450, y + 41), v, font=F_BODY, fill=MUTED, anchor="lm")
    if seg >= 4:
        vk.text_c(d, (640, 520), "下一期：扩散模型与隐私 —— 记忆化和差分隐私训练", F_SMALL, VIOLET)


RENDER = {"title": sc_title, "idea": sc_idea, "marginal": sc_marginal, "compare": sc_compare,
          "reflow": sc_reflow, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "flow-matching")
