"""Knowledge Video 02 — DDPM 详解：从加噪链到噪声预测.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python ep02_ddpm.py <out_dir>

Everything is computed exactly on a 1D two-Gaussian data distribution with the DDPM linear
schedule (T=1000, beta 1e-4..0.02). The reverse sampler is DDPM ancestral sampling where the
network is replaced by the exact optimal noise prediction for this distribution.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL, F_H, F_TITLE

# ---------------------------------------------------------------------------
# Narration
# ---------------------------------------------------------------------------
SCENES = [
    ("title", [
        "上一期，我们直观地认识了扩散模型。",
        "这一期我们深入一步，讲讲最经典的去噪扩散概率模型，看看它是怎样定义、又是怎样训练的。",
        "为了把每个公式都画出来，我们用一个一维的小例子：数据集中在负二和正二附近，像两座小山。",
    ]),
    ("step", [
        "先看前向过程中的一步。",
        "从上一步的样本出发，先把它乘以一个略小于一的系数，再加上一点方差为贝塔的高斯噪声。",
        "贝塔很小，在这里从万分之一逐渐增大到百分之二，所以每一步只加一点点噪声。",
        "为什么要先缩小一点呢？因为这样整体的方差会保持在一附近，不会越加越大。",
    ]),
    ("chain", [
        "把这一步重复一千次，数据的分布会怎样变化呢？",
        "可以看到，两座小山先慢慢变宽、互相靠近，最后合成了一个标准正态分布。",
        "更方便的是，把一千个小步骤连起来，可以直接写出任意时刻的分布。",
        "其中阿尔法拔是所有一减贝塔的连乘，它从一开始，逐渐下降到接近零。",
    ]),
    ("posterior", [
        "生成的时候，我们要反过来，从噪声一步一步走回数据。",
        "关键问题是：已知当前时刻的样本，上一步的样本应该在哪里？",
        "如果知道原始数据，这个问题有精确答案：它是一个高斯分布，均值是原始数据和当前样本的加权组合，离当前样本很近。",
        "可惜生成的时候，我们并不知道原始数据，所以要训练一个神经网络来估计它。",
    ]),
    ("loss", [
        "训练的目标，是让模型给出的反向分布，尽量接近这个真实的后验分布。",
        "数学上，数据负对数似然的变分上界，可以拆成每一步两个高斯分布之间的差异。",
        "当两个高斯分布的方差相同时，这个差异只取决于它们的均值之差。",
        "如果把模型的均值，写成用预测噪声来表示的形式，损失就变成了预测噪声和真实噪声之差，再乘一个权重。",
        "原论文发现，去掉这个权重，训练效果反而更好，于是就得到了上一期见过的简单损失。",
    ]),
    ("sampling", [
        "训练好以后，采样就很简单了。",
        "先从标准正态分布中采一个点，然后从最后一步开始往回走。",
        "每一步，用网络预测的噪声算出均值，再加上一点新的随机噪声。",
        "下面是四百条真实的采样轨迹，每条线是一个样本从噪声走回数据的路径。",
        "可以看到，轨迹在后半程逐渐分成两股，最后落在两座小山上。",
    ]),
    ("compare", [
        "最后，我们把两万个生成样本的直方图，和真实的数据分布放在一起比较。",
        "两者几乎完全重合，说明这个采样过程，确实把噪声变回了数据的分布。",
        "这里的噪声预测同样用的是精确计算的最优解，在真实的模型中，它由神经网络学习得到。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，前向过程每一步缩小一点、再加一点噪声，任意时刻的分布都有闭式表达。",
        "第二，已知原始数据时，反向一步的后验是高斯分布，训练就是去逼近它。",
        "第三，变分上界最终化简为预测噪声的均方误差。",
        "第四，采样时从噪声出发，每一步减去预测的噪声，再加一点随机性。",
        "下一期，我们换一个角度，从分数函数来理解扩散模型。感谢大家的观看。",
    ]),
]

# ---------------------------------------------------------------------------
# Exact 1D diffusion
# ---------------------------------------------------------------------------
T = 1000
BETA = np.concatenate([[0.0], np.linspace(1e-4, 0.02, T)])  # BETA[t], t = 1..T
ALPHA = 1.0 - BETA
ABAR = np.cumprod(ALPHA)  # ABAR[0] = 1
MU = np.array([-2.0, 2.0])
SIG0 = 0.35
PI = np.array([0.5, 0.5])


def density(x, t):
    """q(x_t) for the two-Gaussian data distribution."""
    a = ABAR[t]
    var = a * SIG0 ** 2 + 1 - a
    comps = PI[None] * np.exp(-(x[:, None] - np.sqrt(a) * MU[None]) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    return comps.sum(1)


def x0_hat(x, t):
    """Exact E[x0 | x_t]."""
    a = ABAR[t]
    var = a * SIG0 ** 2 + 1 - a
    diff = x[:, None] - np.sqrt(a) * MU[None]
    logw = np.log(PI)[None] - diff ** 2 / (2 * var)
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(1, keepdims=True)
    return (w * (MU[None] + np.sqrt(a) * SIG0 ** 2 / var * diff)).sum(1)


def posterior(x0, xt, t):
    """Mean and variance of q(x_{t-1} | x_t, x_0)."""
    a, a_prev = ABAR[t], ABAR[t - 1]
    mean = np.sqrt(a_prev) * BETA[t] / (1 - a) * x0 + np.sqrt(ALPHA[t]) * (1 - a_prev) / (1 - a) * xt
    var = (1 - a_prev) / (1 - a) * BETA[t]
    return mean, var


def ddpm_sample(n, seed, keep=0, every=5):
    """Ancestral DDPM sampling with the exact optimal noise prediction."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    traj = [x[:keep].copy()]
    for t in range(T, 0, -1):
        eps = (x - np.sqrt(ABAR[t]) * x0_hat(x, t)) / np.sqrt(1 - ABAR[t])
        mean = (x - BETA[t] / np.sqrt(1 - ABAR[t]) * eps) / np.sqrt(ALPHA[t])
        sigma = np.sqrt((1 - ABAR[t - 1]) / (1 - ABAR[t]) * BETA[t]) if t > 1 else 0.0
        x = mean + sigma * rng.standard_normal(n)
        if keep and (t - 1) % every == 0:
            traj.append(x[:keep].copy())
    return x, np.array(traj)


FINAL, TRAJ = ddpm_sample(20000, seed=3, keep=400)
FINAL = FINAL  # 20000 samples at t = 0
HIST_EDGES = np.linspace(-4, 4, 81)
HIST = np.histogram(FINAL, HIST_EDGES, density=True)[0]

# posterior example
T_POST = 500
X0_EX = 2.0
XT_EX = np.sqrt(ABAR[T_POST]) * X0_EX + np.sqrt(1 - ABAR[T_POST]) * (-0.9)
PM, PV = posterior(X0_EX, XT_EX, T_POST)

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
XS = np.linspace(-4, 4, 400)


def axes(d, box, xlabel=None, ticks=(-4, -2, 0, 2, 4), xr=(-4, 4)):
    x0, y0, x1, y1 = box
    d.line([(x0, y1), (x1, y1)], fill=(71, 85, 105), width=2)
    for tk in ticks:
        px = x0 + (tk - xr[0]) / (xr[1] - xr[0]) * (x1 - x0)
        d.line([(px, y1), (px, y1 + 6)], fill=(71, 85, 105), width=2)
        d.text((px, y1 + 20), f"{tk:g}", font=vk.font(20, mono=True), fill=MUTED, anchor="mm")
    if xlabel:
        d.text((x1, y1 - 14), xlabel, font=F_SMALL, fill=MUTED, anchor="rm")


def curve(d, box, xs, ys, color, ymax, xr=(-4, 4), width=3, fill=None):
    x0, y0, x1, y1 = box
    pts = [(x0 + (x - xr[0]) / (xr[1] - xr[0]) * (x1 - x0), y1 - min(y, ymax) / ymax * (y1 - y0)) for x, y in zip(xs, ys)]
    if fill:
        d.polygon([(pts[0][0], y1)] + pts + [(pts[-1][0], y1)], fill=fill)
    d.line(pts, fill=color, width=width, joint="curve")


def xpix(box, x, xr=(-4, 4)):
    return box[0] + (x - xr[0]) / (xr[1] - xr[0]) * (box[2] - box[0])


def marker(d, box, x, color, label, up=True):
    px, y1 = xpix(box, x), box[3]
    d.line([(px, y1 - 70), (px, y1)], fill=color, width=2)
    d.ellipse([px - 7, y1 - 7, px + 7, y1 + 7], fill=color)
    d.text((px, y1 - 90), label, font=F_SMALL, fill=color, anchor="mm")


def fpaste(c, tex, cx, cy, size=30, color="#e2e8f0"):
    vk.paste_center(c, vk.formula(tex, size, color), cx, cy)


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


def sc_title(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.text_c(d, (640, 150), "Knowledge Video · 02", vk.font(24, mono=True), MUTED)
    vk.text_c(d, (640, 230), "去噪扩散概率模型（DDPM）", F_TITLE, CYAN)
    vk.text_c(d, (640, 300), "从加噪链到噪声预测", F_BODY, TEXT)
    if seg >= 2:
        box = (340, 360, 940, 540)
        p = vk.ease(su * 1.5)
        n = max(2, int(len(XS) * p))
        curve(d, box, XS[:n], density(XS, 0)[:n], CYAN, 0.65, fill=(14, 60, 72))
        axes(d, box, "数据分布")


def sc_step(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "前向过程的一步")
    fpaste(c, r"$q(x_t\mid x_{t-1})=\mathcal{N}\!\left(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right)$", 640, 160, 32)
    if seg >= 1:
        fpaste(c, r"$x_t=\sqrt{1-\beta_t}\;x_{t-1}+\sqrt{\beta_t}\;\epsilon,\qquad \epsilon\sim\mathcal{N}(0,1)$", 640, 240, 30, "#a78bfa")
        vk.text_c(d, (330, 300), "缩小一点", F_SMALL, CYAN)
        vk.text_c(d, (620, 300), "加一点噪声", F_SMALL, VIOLET)
    if seg >= 2:
        box = (180, 340, 620, 540)
        d.rectangle(box, outline=(30, 41, 59))
        ts = np.arange(1, T + 1)
        pts = [(box[0] + (t - 1) / (T - 1) * (box[2] - box[0]), box[3] - BETA[t] / 0.022 * (box[3] - box[1])) for t in ts[::10]]
        k = max(2, int(len(pts) * vk.ease(su * 1.4)))
        d.line(pts[:k], fill=CYAN, width=3)
        img = vk.formula(r"$\beta_t$", 24, "#22d3ee")
        c.paste(img, (box[0] + 8, box[1] + 6), img)
        d.text((box[2], box[3] + 22), "t = 1000", font=vk.font(20, mono=True), fill=MUTED, anchor="rm")
        d.text((box[0], box[3] + 22), "t = 1", font=vk.font(20, mono=True), fill=MUTED, anchor="lm")
    if seg >= 3:
        fpaste(c, r"$\mathrm{Var}(x_t)=(1-\beta_t)\cdot 1+\beta_t=1$", 920, 430, 30, "#34d399")
        vk.text_c(d, (920, 490), "方差保持在 1 附近", F_SMALL, GREEN)


def sc_chain(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "一千步之后：闭式公式")
    if seg == 0:
        t = 0
    elif seg == 1:
        t = int(T * vk.ease(su * 1.15))
    else:
        t = T
    box = (80, 140, 740, 440)
    curve(d, box, XS, density(XS, 0), (51, 65, 85), 0.65, width=2)
    curve(d, box, XS, density(XS, max(t, 0)), CYAN, 0.65, fill=(14, 60, 72))
    axes(d, box, None)
    d.text((box[0] + 10, box[1]), f"q(x_t),  t = {t}", font=vk.font(26, mono=True), fill=CYAN)
    # abar curve
    ab = (830, 170, 1190, 400)
    d.rectangle(ab, outline=(30, 41, 59))
    pts = [(ab[0] + s / T * (ab[2] - ab[0]), ab[3] - ABAR[s] * (ab[3] - ab[1])) for s in range(0, T + 1, 10)]
    d.line(pts, fill=VIOLET, width=3)
    mx, my = ab[0] + t / T * (ab[2] - ab[0]), ab[3] - ABAR[t] * (ab[3] - ab[1])
    d.ellipse([mx - 7, my - 7, mx + 7, my + 7], fill=VIOLET)
    img = vk.formula(r"$\bar\alpha_t$", 24, "#a78bfa")
    c.paste(img, (ab[0] + 8, ab[1] + 6), img)
    d.text((ab[2] - 10, ab[1] + 22), f"= {ABAR[t]:.3f}", font=vk.font(22, mono=True), fill=MUTED, anchor="rm")
    if seg >= 2:
        fpaste(c, r"$q(x_t\mid x_0)=\mathcal{N}\!\left(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)\,I\right)$", 410, 515, 30)
    if seg >= 3:
        fpaste(c, r"$\bar\alpha_t=\prod_{s=1}^{t}(1-\beta_s)$", 1010, 460, 26, "#a78bfa")


def sc_posterior(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "反向的一步：真实后验")
    box = (90, 150, 1190, 330)
    curve(d, box, XS, density(XS, T_POST), (51, 65, 85), 0.6, width=2)
    axes(d, box, f"t = {T_POST} 时的分布")
    marker(d, box, XT_EX, VIOLET, "当前样本 x_t")
    if seg >= 1:
        vk.text_c(d, (xpix(box, XT_EX) - 160, 205), "上一步在哪？", F_BODY, MUTED)
    if seg >= 2:
        known = seg == 2
        marker(d, box, X0_EX, CYAN if known else (71, 85, 105), "原始数据 x_0" if known else "x_0 未知 ?")
        # zoom inset around x_t
        sd = np.sqrt(PV)
        lo, hi = PM - 5 * sd, PM + 5 * sd
        ib = (700, 400, 1180, 540)
        d.rectangle([ib[0] - 10, ib[1] - 40, ib[2] + 10, ib[3] + 40], outline=(30, 41, 59))
        d.text((ib[0], ib[1] - 25), "局部放大：", font=F_SMALL, fill=MUTED, anchor="lm")
        img = vk.formula(r"$q(x_{t-1}\mid x_t,x_0)$", 20, "#34d399")
        c.paste(img, (ib[0] + 115, int(ib[1] - 25 - img.height / 2)), img)
        zs = np.linspace(lo, hi, 200)
        pdf = np.exp(-(zs - PM) ** 2 / (2 * PV))
        curve(d, ib, zs, pdf, GREEN, 1.05, xr=(lo, hi), fill=(16, 64, 52))
        d.line([(ib[0], ib[3]), (ib[2], ib[3])], fill=(71, 85, 105), width=2)
        if lo < XT_EX < hi:
            px = xpix(ib, XT_EX, (lo, hi))
            d.line([(px, ib[1]), (px, ib[3])], fill=VIOLET, width=2)
            d.text((px, ib[3] + 18), "x_t", font=F_SMALL, fill=VIOLET, anchor="mm")
        d.text((xpix(ib, PM, (lo, hi)) + 60, ib[1] + 30), "后验均值", font=F_SMALL, fill=GREEN, anchor="lm")
        fpaste(c, r"$\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t$", 360, 440, 24)
        fpaste(c, r"$\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t$", 360, 515, 24)


def sc_loss(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "训练目标：从变分上界到预测噪声")
    rows = [
        (1, r"$-\log p_\theta(x_0)\ \leq\ \sum_{t}\ D_{\mathrm{KL}}\!\left(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\right)+\cdots$", "#e2e8f0"),
        (2, r"$D_{\mathrm{KL}}=\frac{1}{2\sigma_t^2}\,\|\tilde\mu_t-\mu_\theta(x_t,t)\|^2+C$", "#e2e8f0"),
        (3, r"$\mu_\theta=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\right)\ \Rightarrow\ \frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\,\|\epsilon-\epsilon_\theta\|^2$", "#a78bfa"),
        (4, r"$\mathcal{L}_{\mathrm{simple}}=\mathbb{E}_{x_0,t,\epsilon}\,\|\epsilon-\epsilon_\theta(x_t,t)\|^2$", "#22d3ee"),
    ]
    if seg == 0:
        fpaste(c, r"$p_\theta(x_{t-1}\mid x_t)\ \approx\ q(x_{t-1}\mid x_t,x_0)$", 640, 300, 34)
    y = 150
    for k, tex, col in rows:
        if seg >= k:
            fpaste(c, tex, 640, y, 30 if k != 3 else 26, col)
        y += 105
    if seg >= 4:
        img = vk.formula(rows[3][1], 30, rows[3][2])
        d.rounded_rectangle([640 - img.width / 2 - 16, 465 - img.height / 2 - 10, 640 + img.width / 2 + 16, 465 + img.height / 2 + 10], 12, outline=CYAN, width=2)


def sc_sampling(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "采样算法")
    if seg <= 2:
        d.rounded_rectangle([200, 130, 1080, 470], 16, outline=(30, 41, 59), fill=(12, 19, 34))
        lines = [
            (0, r"$x_T\sim\mathcal{N}(0,I)$", "1."),
            (1, r"$\mathrm{for}\ t=T,\dots,1:\quad z\sim\mathcal{N}(0,I)\ \ (t>1),\ \ z=0\ \ (t=1)$", "2."),
            (2, r"$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\right)+\sigma_t z$", "3."),
        ]
        for k, tex, num in lines:
            if seg >= k:
                y = 190 + k * 95
                d.text((240, y), num, font=vk.font(28, mono=True), fill=CYAN, anchor="lm")
                img = vk.formula(tex, 28)
                c.paste(img, (290, int(y - img.height / 2)), img)
        return
    box = (100, 110, 1080, 560)
    d.rectangle(box, outline=(30, 41, 59))
    steps = TRAJ.shape[0]
    p = vk.ease(su * 1.2) if seg == 3 else 1.0
    k = max(2, int(steps * p))
    xs_px = box[0] + np.arange(k) / (steps - 1) * (box[2] - box[0])
    ymap = lambda v: (box[1] + box[3]) / 2 - v / 4.2 * (box[3] - box[1]) / 2
    for j in range(TRAJ.shape[1]):
        col = (34, 211, 238) if TRAJ[-1, j] > 0 else (167, 139, 250)
        if seg == 3:
            col = (120, 130, 160)
        d.line(list(zip(xs_px, ymap(TRAJ[:k, j]))), fill=col, width=1)
    d.text((box[0], box[3] + 22), "t = 1000（噪声）", font=F_SMALL, fill=MUTED, anchor="lm")
    d.text((box[2], box[3] + 22), "t = 0（数据）", font=F_SMALL, fill=MUTED, anchor="rm")
    for v in (-2, 2):
        d.text((box[2] + 12, ymap(v)), f"{v:+d}", font=vk.font(22, mono=True), fill=MUTED, anchor="lm")


def sc_compare(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 6, "生成样本 vs 真实分布")
    box = (140, 140, 1140, 520)
    p = vk.ease(su * 1.3) if seg == 0 else 1.0
    ymax = 0.65
    for i, h in enumerate(HIST):
        xa, xb = xpix(box, HIST_EDGES[i]) + 1, xpix(box, HIST_EDGES[i + 1]) - 1
        top = box[3] - min(h * p, ymax) / ymax * (box[3] - box[1])
        d.rectangle([xa, top, xb, box[3]], fill=(76, 58, 140))
    if seg >= 1:
        curve(d, box, XS, density(XS, 0), CYAN, ymax, width=4)
        d.text((box[2] - 10, box[1] + 10), "— 真实分布", font=F_SMALL, fill=CYAN, anchor="rm")
    d.text((box[2] - 10, box[1] + 45), "生成样本（2 万个）", font=F_SMALL, fill=VIOLET, anchor="rm")
    d.rectangle([box[2] - 250, box[1] + 36, box[2] - 232, box[1] + 54], fill=(76, 58, 140))
    axes(d, box)


def sc_summary(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 7, "总结")
    items = [
        ("前向过程", "缩小一点、加一点噪声；任意时刻有闭式解"),
        ("真实后验", "已知 x0 时反向一步是高斯分布"),
        ("训练目标", "变分上界化简为预测噪声的均方误差"),
        ("采样过程", "减去预测噪声，再加一点随机性"),
    ]
    for i, (k, v) in enumerate(items):
        if seg >= i + 1:
            y = 130 + i * 95
            d.rounded_rectangle([140, y, 1140, y + 76], 14, outline=(30, 41, 59), fill=(12, 19, 34))
            d.text((180, y + 38), f"0{i + 1}", font=vk.font(30, mono=True), fill=CYAN, anchor="lm")
            d.text((260, y + 38), k, font=F_BODY, fill=TEXT, anchor="lm")
            d.text((450, y + 38), v, font=F_BODY, fill=MUTED, anchor="lm")
    if seg >= 5:
        vk.text_c(d, (640, 530), "下一期：Score 与 SDE —— 分数匹配和朗之万采样", F_SMALL, VIOLET)


RENDER = {
    "title": sc_title, "step": sc_step, "chain": sc_chain, "posterior": sc_posterior,
    "loss": sc_loss, "sampling": sc_sampling, "compare": sc_compare, "summary": sc_summary,
}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "ddpm")
