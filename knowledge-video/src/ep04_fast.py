"""Knowledge Video 04 — 采样加速：DDIM 与 DPM-Solver.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python ep04_fast.py <out_dir>

2D data: 8 Gaussians on a circle (sigma 0.06). DDPM ancestral sampling, DDIM (eta=0) and
DPM-Solver++(2M) all use the exact denoiser E[x0|x_t]. The "bad sample" share (farther than
4 sigma from every mode) is measured on 20,000 samples per setting; the narration numbers come
from this measurement (printed at start-up).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import diff2d as D
import p2d
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL, F_TITLE

ANG = np.arange(8) * np.pi / 4
G = D.GMM(np.stack([2 * np.cos(ANG), 2 * np.sin(ANG)], 1), 0.06)
N = 20000
XT = np.random.default_rng(1).standard_normal((N, 2))
STEPS = [5, 10, 20, 50]
METHODS = [("原始采样（随机）", lambda n: D.ddpm(G, XT, n, np.random.default_rng(2)), VIOLET),
           ("隐式模型（确定）", lambda n: D.ddim(G, XT, n), CYAN),
           ("求解器（二阶）", lambda n: D.dpm_solver_pp_2m(G, XT, n), GREEN)]
RESULTS = {(m, n): f(n) for m, (_, f, _) in enumerate(METHODS) for n in STEPS}
BAD = {k: D.bad_fraction(G, v) for k, v in RESULTS.items()}
DDPM200 = D.ddpm(G, XT, 200, np.random.default_rng(2))
BAD200 = D.bad_fraction(G, DDPM200)
for m, (name, _, _) in enumerate(METHODS):
    print(name, " ".join(f"{n}:{BAD[(m, n)]:.2%}" for n in STEPS))
print("DDPM 200:", f"{BAD200:.2%}")

# trajectories for the DDIM / DDPM comparison (20 steps)
_, TR_DDPM = D.ddpm(G, XT[:400], 20, np.random.default_rng(5), keep=400)
_, TR_DDIM = D.ddim(G, XT[:400], 20, keep=400)

# ODE accuracy at 20 steps: endpoint error vs a 4000-step reference over 2000 samples
XE = XT[:2000]
REF_END = D.ddim(G, XE, 4000)
ERR_DDIM = np.linalg.norm(D.ddim(G, XE, 20) - REF_END, axis=1)
ERR_DPM = np.linalg.norm(D.dpm_solver_pp_2m(G, XE, 20) - REF_END, axis=1)
MED_DDIM, MED_DPM = np.median(ERR_DDIM), np.median(ERR_DPM)
print(f"20-step endpoint error median: DDIM {MED_DDIM:.4f}  DPM++ {MED_DPM:.4f}  better {np.mean(ERR_DPM < ERR_DDIM):.0%}")
# a representative particle: DDIM error near its median, DPM++ error near its median
TRAVEL = np.linalg.norm(REF_END - XE, axis=1)
J = int(np.argmin(np.abs(ERR_DDIM / MED_DDIM - 1) + np.abs(ERR_DPM / MED_DPM - 1) + 10 * (TRAVEL < 2.2)))
P0 = XE[J:J + 1]
_, REF = D.ddim(G, P0, 2000, keep=1)
_, EUL = D.ddim(G, P0, 20, keep=1)
_, DPM = D.dpm_solver_pp_2m(G, P0, 20, keep=1)

# ---------------------------------------------------------------------------
# Narration
# ---------------------------------------------------------------------------
SCENES = [
    ("title", [
        "上一期最后我们看到，扩散模型的采样，可以看成求解一个常微分方程。",
        "这一期的问题是：采样为什么慢，又该怎样加速。",
        "原始的去噪扩散概率模型，训练时把加噪过程分成一千步，采样时也要一步一步走一千次，每一步都要运行一次神经网络。对大模型来说，这非常慢。",
    ]),
    ("naive", [
        "最直接的想法是跳着走，比如只走五步。",
        "我们用一个八个峰排成一圈的例子来看。左边是走两百步的结果，右边是只走五步的结果。",
        "只走五步时，大部分样本都落在了峰与峰之间。每一步跨得太大，误差迅速累积。",
    ]),
    ("ddim", [
        "第一个加速方法，叫去噪扩散隐式模型。",
        "它的做法是：每一步先用网络预测出干净的数据，再直接跳到更早的时刻，把预测出的噪声按比例放回去。",
        "整个过程不再加入新的随机噪声，所以同一份初始噪声，总会生成同一个结果，轨迹也变得平滑。",
        "它不需要重新训练，可以直接使用训练好的去噪模型。原论文报告，在图像数据上，几十步的隐式模型，质量就能接近一千步的原始采样。",
    ]),
    ("ode", [
        "为什么可以这样大步跳呢？",
        "因为这个隐式模型，正是概率流常微分方程的一种离散化，采样就变成了数值求解常微分方程。",
        "既然如此，就可以借用数值分析的经验：同样的步数，更高阶的方法误差更小。在这个例子里走二十步，二阶求解器终点误差的中位数，只有隐式模型的约五分之一。",
    ]),
    ("dpm", [
        "第二个方法叫扩散概率模型求解器，它是专门为这个常微分方程设计的。",
        "它注意到，方程中有一部分是线性的，可以精确求解，只有和网络预测有关的那一项需要近似。",
        "它还把时间换成对数信噪比，并利用前一步的预测结果，做二阶的修正。",
        "这样每一步仍然只调用一次网络，精度却高得多。有趣的是，它的一阶版本，恰好就是隐式模型。",
    ]),
    ("compare", [
        "我们在八个峰的例子上，比较三种方法。每张图有两万个样本，下面标出落在峰与峰之间的坏样本比例。",
        "只走五步时，三种方法都失败了，坏样本都超过七成。",
        "走十步时，原始采样和隐式模型都还有约百分之十九的坏样本，而求解器已经降到百分之一以下。",
        "走二十步时，三种方法都降到百分之二以下，求解器最低，约为百分之零点三。",
        "走到五十步，三者都在千分之二以下，基本没有差别。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，采样慢，是因为每一步都要运行一次网络，而原始方法需要很多小步。",
        "第二，隐式模型去掉了随机性，可以大步跳跃，同一份噪声得到同一个结果。",
        "第三，把采样看成解常微分方程，就能用更高阶的求解器，用十几到二十步达到很好的效果。",
        "实际的文生图模型，常用二三十步的高阶求解器来采样。",
        "下一期，我们讲条件生成和无分类器引导。感谢大家的观看。",
    ]),
]

# sanity checks: the spoken numbers must match the measurements
assert all(BAD[(m, 5)] > 0.7 for m in range(3))
assert 0.15 < BAD[(0, 10)] < 0.23 and 0.15 < BAD[(1, 10)] < 0.23 and BAD[(2, 10)] < 0.01
assert all(BAD[(m, 20)] < 0.02 for m in range(3)) and BAD[(2, 20)] == min(BAD[(m, 20)] for m in range(3))
assert 0.002 < BAD[(2, 20)] < 0.0045
assert all(BAD[(m, 50)] < 0.002 for m in range(3))
assert 0.15 < MED_DPM / MED_DDIM < 0.3

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
EXT = 3.0


def fpaste(c, tex, cx, cy, size=28, color="#e2e8f0"):
    vk.paste_center(c, vk.formula(tex, size, color), cx, cy)


def panel(c, d, box, pts, col, label=None, label_col=MUTED, r=1.2, n=6000):
    p2d.frame(d, box)
    p2d.scatter(d, box, pts[:n], col, r=r, ext=EXT)
    if label:
        vk.text_c(d, ((box[0] + box[2]) / 2, box[3] + 22), label, F_SMALL, label_col)


def sc_title(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.text_c(d, (640, 170), "Knowledge Video · 04", vk.font(24, mono=True), MUTED)
    vk.text_c(d, (640, 250), "采样加速", F_TITLE, CYAN)
    vk.text_c(d, (640, 320), "DDIM 与 DPM-Solver", F_BODY, TEXT)
    if seg >= 2:
        n = int(1000 * vk.ease(su))
        vk.text_c(d, (640, 420), f"网络调用次数：{n} / 1000", vk.font(30), VIOLET)


def sc_naive(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "直接跳步会怎样？")
    if seg >= 1:
        panel(c, d, (140, 120, 560, 540), DDPM200, VIOLET, f"原始采样 200 步   坏样本 {BAD200:.1%}", r=1.4)
        panel(c, d, (720, 120, 1140, 540), RESULTS[(0, 5)], VIOLET, f"原始采样 5 步   坏样本 {BAD[(0, 5)]:.0%}", r=1.4)
    else:
        vk.text_c(d, (640, 330), "1000 步  →  5 步 ?", F_TITLE, MUTED)


def sc_ddim(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "去噪扩散隐式模型（DDIM）")
    if seg <= 1:
        fpaste(c, r"$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}$", 640, 220, 32)
        if seg >= 1:
            fpaste(c, r"$x_s=\sqrt{\bar\alpha_s}\,\hat x_0+\sqrt{1-\bar\alpha_s}\,\epsilon_\theta(x_t,t)$", 640, 330, 32, "#22d3ee")
            vk.text_c(d, (640, 410), "先预测干净数据，再跳到更早的时刻 s（不加新噪声）", F_SMALL, MUTED)
        return
    k = int(20 * (vk.ease(min(1, su * 1.3)) if seg == 2 else 1))
    for i, (tr, col, name) in enumerate([(TR_DDPM, VIOLET, "原始采样 20 步：随机"), (TR_DDIM, CYAN, "隐式模型 20 步：确定、平滑")]):
        box = (140 + i * 580, 115, 140 + i * 580 + 420, 535)
        p2d.frame(d, box)
        p2d.trails(d, box, tr[:, :40], k, tuple(int(v * 0.55) for v in col), ext=EXT)
        p2d.scatter(d, box, tr[min(k, tr.shape[0] - 1)], col, r=1.8, ext=EXT)
        vk.text_c(d, ((box[0] + box[2]) / 2, box[3] + 22), name, F_SMALL, col)


def sc_ode(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "采样 = 解常微分方程")
    fpaste(c, r"$\frac{\mathrm{d}x}{\mathrm{d}t}=-\frac{1}{2}\beta(t)\left[x+\nabla_x\log p_t(x)\right]$", 330, 200, 30)
    if seg >= 1:
        vk.text_c(d, (330, 280), "隐式模型 = 一阶（类欧拉）离散化", F_SMALL, CYAN)
    if seg >= 2:
        box = (640, 110, 1200, 560)
        p2d.frame(d, box)
        ctr = 0.5 * (REF[:, 0].min(0) + REF[:, 0].max(0))
        ext = 0.65 * np.max(REF[:, 0].max(0) - REF[:, 0].min(0)) + 0.3
        p2d.scatter(d, box, G.mu - ctr, (51, 65, 85), r=8, ext=ext)
        p2d.trails(d, box, REF - ctr, REF.shape[0] - 1, TEXT, ext=ext, width=2)
        p = vk.ease(min(1, su * 1.4))
        for tr, col in [(EUL, CYAN), (DPM, GREEN)]:
            k = int((tr.shape[0] - 1) * p)
            p2d.scatter(d, box, tr[: k + 1, 0] - ctr, col, r=4, ext=ext)
        d.text((box[0] + 14, box[1] + 18), "— 精确轨迹", font=F_SMALL, fill=TEXT)
        d.text((box[0] + 14, box[1] + 50), "● 隐式模型 20 步", font=F_SMALL, fill=CYAN)
        d.text((box[0] + 14, box[1] + 82), "● 二阶求解器 20 步", font=F_SMALL, fill=GREEN)
        if p >= 1:
            # zoom on the endpoint
            end = REF[-1, 0]
            zb = (80, 330, 560, 560)
            p2d.frame(d, zb)
            zext = np.linalg.norm(EUL[-1, 0] - end) * 1.8
            zbox = (zb[0] + 120, zb[1], zb[2] - 120, zb[3])
            for q, col, lab, side in [(end, TEXT, "精确终点", -1), (EUL[-1, 0], CYAN, "隐式模型", 1), (DPM[-1, 0], GREEN, "求解器", 1)]:
                x, y = p2d.to_px(zbox, (q - end)[None], zext)[0]
                d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=col)
                dy = 0 if lab != "求解器" else 22
                d.text((x + side * 12, y + dy), lab, font=F_SMALL, fill=col, anchor="lm" if side > 0 else "rm")
            d.text((zb[0] + 10, zb[1] + 14), "终点局部放大", font=F_SMALL, fill=MUTED, anchor="lm")
            d.text((zb[0] + 10, zb[3] - 40), f"误差中位数（2000 个样本）", font=F_SMALL, fill=MUTED, anchor="lm")
            d.text((zb[0] + 10, zb[3] - 14), f"隐式模型 {MED_DDIM:.3f}    求解器 {MED_DPM:.3f}", font=F_SMALL, fill=TEXT, anchor="lm")
        else:
            vk.text_c(d, (330, 380), "同样的步数，高阶方法误差更小", F_BODY, GREEN)


def sc_dpm(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "扩散概率模型求解器（DPM-Solver++）")
    fpaste(c, r"$\lambda_t=\log\frac{\alpha_t}{\sigma_t},\qquad \alpha_t=\sqrt{\bar\alpha_t},\ \ \sigma_t=\sqrt{1-\bar\alpha_t}$", 640, 165, 28)
    if seg >= 1:
        fpaste(c, r"$x_t=\frac{\sigma_t}{\sigma_s}\,x_s\;-\;\alpha_t\left(e^{-h}-1\right)\hat x_0(x_s),\qquad h=\lambda_t-\lambda_s$", 640, 265, 30, "#22d3ee")
        vk.text_c(d, (330, 320), "线性部分：精确", F_SMALL, CYAN)
        vk.text_c(d, (850, 320), "网络部分：近似", F_SMALL, VIOLET)
    if seg >= 2:
        fpaste(c, r"$\hat x_0\ \rightarrow\ \left(1+\frac{1}{2r}\right)\hat x_0^{(s)}-\frac{1}{2r}\,\hat x_0^{(\mathrm{prev})},\qquad r=\frac{h_{\mathrm{prev}}}{h}$", 640, 395, 28, "#34d399")
        vk.text_c(d, (640, 450), "用上一步的预测做二阶修正（多步法）", F_SMALL, GREEN)
    if seg >= 3:
        vk.text_c(d, (640, 520), "一阶版本  =  隐式模型（DDIM）", F_BODY, TEXT)


def sc_compare(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "坏样本比例对比（2 万个样本）")
    col_x = [290 + i * 235 for i in range(4)]
    for i, n in enumerate(STEPS):
        vk.text_c(d, (col_x[i] + 80, 115), f"{n} 步", F_SMALL, TEXT if seg >= i + 1 else (71, 85, 105))
    for m, (name, _, col) in enumerate(METHODS):
        y0 = 135 + m * 150
        vk.text_c(d, (150, y0 + 60), name, F_SMALL, col)
        for i, n in enumerate(STEPS):
            box = (col_x[i], y0, col_x[i] + 160, y0 + 120)
            if seg >= i + 1:
                p2d.frame(d, box)
                p2d.scatter(d, box, RESULTS[(m, n)][:2500], col, r=0.9, ext=EXT)
                b = BAD[(m, n)]
                txt = f"{b:.0%}" if b >= 0.1 else (f"{b:.1%}" if b >= 0.01 else f"{b:.2%}")
                d.text((box[2] + 6, box[3] - 10), txt, font=vk.font(18, mono=True), fill=col, anchor="lm")
            else:
                p2d.frame(d, box)


def sc_summary(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 6, "总结")
    items = [
        ("为什么慢", "每一步都要运行一次网络，原始方法步数多"),
        ("隐式模型", "去掉随机性，大步跳跃，结果可复现"),
        ("高阶求解器", "把采样当 ODE 求解，十几到二十步即可"),
    ]
    for i, (k, v) in enumerate(items):
        if seg >= i + 1:
            y = 140 + i * 105
            d.rounded_rectangle([140, y, 1140, y + 82], 14, outline=(30, 41, 59), fill=(12, 19, 34))
            d.text((180, y + 41), f"0{i + 1}", font=vk.font(30, mono=True), fill=CYAN, anchor="lm")
            d.text((260, y + 41), k, font=F_BODY, fill=TEXT, anchor="lm")
            d.text((470, y + 41), v, font=F_BODY, fill=MUTED, anchor="lm")
    if seg >= 5:
        vk.text_c(d, (640, 520), "下一期：条件生成与 Classifier-Free Guidance", F_SMALL, VIOLET)


RENDER = {
    "title": sc_title, "naive": sc_naive, "ddim": sc_ddim, "ode": sc_ode,
    "dpm": sc_dpm, "compare": sc_compare, "summary": sc_summary,
}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "fast-sampling")
