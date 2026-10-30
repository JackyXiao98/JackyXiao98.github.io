"""Knowledge Video 05 — 条件生成与 Classifier-Free Guidance.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python ep05_cfg.py <out_dir>

Six Gaussians on a circle; class A (cyan) = modes 0,2,4, class B (violet) = modes 1,3,5, with
enough width that neighbouring classes overlap. Conditional and unconditional denoisers are
exact; CFG uses eps = (1+w) eps(x,A) - w eps(x,null) with 50 DDIM steps. All statistics in the
narration are measured (and asserted) below.
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

ANG = np.arange(6) * np.pi / 3
MU = np.stack([2 * np.cos(ANG), 2 * np.sin(ANG)], 1)
SIG = 0.45
A, B, U = D.GMM(MU[0::2], SIG), D.GMM(MU[1::2], SIG), D.GMM(MU, SIG)


def p_A(x):
    return 1 / (1 + np.exp(B.logpdf(x) - A.logpdf(x)))


def cfg(xT, w, n=50, keep=None):
    ts = D.grid(n)
    x, traj = xT.copy(), ([xT[:keep].copy()] if keep else None)
    for t, s in zip(ts[:-1], ts[1:]):
        a, a_s = D.abar(t), D.abar(s)
        e = (1 + w) * A.eps_hat(x, a) - w * U.eps_hat(x, a)
        x0 = (x - np.sqrt(1 - a) * e) / np.sqrt(a)
        x = np.sqrt(a_s) * x0 + np.sqrt(1 - a_s) * e
        if keep:
            traj.append(x[:keep].copy())
    a = D.abar(ts[-1])
    e = (1 + w) * A.eps_hat(x, a) - w * U.eps_hat(x, a)
    x = (x - np.sqrt(1 - a) * e) / np.sqrt(a)
    return (x, np.array(traj + [x[:keep].copy()])) if keep else x


N = 20000
XT = np.random.default_rng(0).standard_normal((N, 2))
TRUE_A = A.sample(N, np.random.default_rng(1))
LOGP_THR = np.quantile(A.logpdf(TRUE_A), 0.01)  # 1% of true samples fall below this


def stats(x):
    k = ((x[:, None] - A.mu[None]) ** 2).sum(-1).argmin(1)
    return {
        "ambig": float(np.mean(p_A(x) < 0.5)),
        "conf": float(p_A(x).mean()),
        "spread": float(np.sqrt(np.mean(((x - A.mu[k]) ** 2).sum(1) / 2))),
        # sideways (tangential) spread around the assigned mode: diversity along the circle
        "tang": float(np.std(((x - A.mu[k]) * np.stack([-A.mu[k][:, 1], A.mu[k][:, 0]], 1) / 2.0).sum(1))),
        "radius": float(np.linalg.norm(x, axis=1).mean()),
        "off": float(np.mean(A.logpdf(x) < LOGP_THR)),
    }


WS = [0, 0.5, 1, 2, 3, 5, 7]
SAMPLES = {w: cfg(XT, w) for w in WS}
ST = {w: stats(SAMPLES[w]) for w in WS}
ST_TRUE = stats(TRUE_A)
for w in WS:
    print(f"w={w}: " + "  ".join(f"{k} {v:.3f}" for k, v in ST[w].items()))
print("true: " + "  ".join(f"{k} {v:.3f}" for k, v in ST_TRUE.items()))
_, TR0 = cfg(XT[:300], 0, keep=300)
_, TR3 = cfg(XT[:300], 3, keep=300)

# claims made in the narration
assert 0.02 < ST[0]["ambig"] < 0.04          # "about 3%" ambiguous at w = 0
assert ST[1]["ambig"] < 0.002                # all clearly in class A at w = 1
assert 0.4 < 1 - ST[1]["tang"] / ST[0]["tang"] < 0.6        # sideways spread roughly halves at w = 1
assert ST[7]["tang"] / ST_TRUE["tang"] < 0.1                # under a tenth of the true value at w = 7
assert all(ST[a]["tang"] > ST[b]["tang"] for a, b in zip(WS[:-1], WS[1:]))  # monotone
assert ST[3]["radius"] > ST[1]["radius"] + 0.1 and ST[7]["radius"] > ST[3]["radius"] + 0.15
SHIFT7 = (ST[7]["radius"] - ST_TRUE["radius"]) / SIG
assert 0.85 < SHIFT7 < 1.2  # "shifted outward by about one standard deviation" at w = 7

# ---------------------------------------------------------------------------
# Narration
# ---------------------------------------------------------------------------
SCENES = [
    ("title", [
        "前几期讲的都是无条件生成，模型随意生成一个样本。",
        "但在实际应用中，我们往往想控制生成的内容，比如给一句文字，生成对应的图片，这就是条件生成。",
        "这一期，我们讲条件生成，以及现在几乎所有文生图模型都在用的无分类器引导。",
    ]),
    ("cond", [
        "最简单的做法，是把条件也输入给去噪网络，让它预测给定条件下的噪声。",
        "我们用一个二维的例子：六个峰排成一圈，青色的三个属于甲类，紫色的三个属于乙类，相邻的峰有一部分重叠。",
        "在甲类的条件下采样，样本服从甲类的分布，但大约有百分之三的样本，落在了两类重叠的模糊区域。",
    ]),
    ("cg", [
        "早期的做法叫分类器引导，就是另外训练一个分类器，用它的梯度，把样本往更像这一类的方向推。",
        "根据贝叶斯公式，条件分数等于无条件分数，加上分类器对数概率的梯度，把后一项乘上一个大于一的系数，就加强了引导。",
        "缺点是，需要额外训练一个能处理带噪输入的分类器。",
    ]),
    ("cfg", [
        "无分类器引导的想法很巧妙，同一个网络在训练时，随机丢掉一部分条件，这样它既能做条件预测，也能做无条件预测。",
        "采样时，把条件预测和无条件预测相减，得到一个指向这一类的方向，再沿着这个方向多走一些。",
        "可以证明，这相当于用一个隐式的分类器做引导，所以不再需要单独的分类器。",
    ]),
    ("sweep", [
        "我们把引导强度从零逐渐调大，看看会发生什么。",
        "强度为零时，就是普通的条件采样。强度为一时，所有样本都清楚地落在甲类区域，但每个峰沿着圆周方向的散布缩小了约一半，多样性下降了。",
        "继续加大到三和七，样本被压成一条条细线，并且越推越远。在强度为七时，横向散布不到真实值的十分之一，平均位置还向外偏移了大约一个标准差，这就像图像中过度饱和、失真的效果。",
    ]),
    ("tradeoff", [
        "这就是引导强度的权衡，强度越大，样本越符合条件，但多样性越低，过大还会失真。",
        "实际的文生图模型，引导强度一般取一个适中的值，比如稳定扩散默认的引导系数是七点五，按这里的写法，相当于强度六点五。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，条件生成就是把条件输入去噪网络，学习条件分布。",
        "第二，分类器引导用分类器的梯度加强条件，但需要额外的分类器。",
        "第三，无分类器引导用条件和无条件预测之差作为引导方向，一个网络就够了。",
        "第四，引导强度在符合条件、多样性和失真之间权衡。",
        "下一期，我们看看稳定扩散是怎样在潜空间里高效生成图片的。感谢大家的观看。",
    ]),
]

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
EXT = 3.6
HEAT_A = p2d.heatmap((0, 0, 380, 380), A.logpdf, ext=EXT, color=(34, 211, 238))
HEAT_B = p2d.heatmap((0, 0, 380, 380), B.logpdf, ext=EXT, color=(167, 139, 250))


def both_heat(box):
    a = np.asarray(p2d.heatmap(box, A.logpdf, ext=EXT, color=(34, 211, 238)), float)
    b = np.asarray(p2d.heatmap(box, B.logpdf, ext=EXT, color=(167, 139, 250)), float)
    bg = np.array(vk.BG, float)
    from PIL import Image
    return Image.fromarray(np.clip(a + b - bg, 0, 255).astype(np.uint8))


BOX = (100, 120, 540, 560)
HEAT_BOTH = both_heat(BOX)
HEAT_A_BIG = p2d.heatmap(BOX, A.logpdf, ext=EXT, color=(34, 211, 238))


def fpaste(c, tex, cx, cy, size=28, color="#e2e8f0"):
    vk.paste_center(c, vk.formula(tex, size, color), cx, cy)


def sc_title(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.text_c(d, (640, 170), "Knowledge Video · 05", vk.font(24, mono=True), MUTED)
    vk.text_c(d, (640, 250), "条件生成与无分类器引导", F_TITLE, CYAN)
    vk.text_c(d, (640, 320), "Classifier-Free Guidance", F_BODY, TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 420), "“一只戴墨镜的猫”  →  图片", F_BODY, VIOLET)


def sc_cond(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "条件生成")
    fpaste(c, r"$\epsilon_\theta(x_t,\ t,\ c)$", 900, 190, 34, "#22d3ee")
    vk.text_c(d, (900, 245), "把条件 c 也输入网络", F_SMALL, MUTED)
    if seg >= 1:
        c.paste(HEAT_BOTH, BOX[:2])
        p2d.frame(d, BOX)
        vk.text_c(d, (900, 320), "青色：甲类      紫色：乙类", F_SMALL, TEXT)
    if seg >= 2:
        x = SAMPLES[0][:3000]
        amb = p_A(x) < 0.5
        p2d.scatter(d, BOX, x[~amb], (220, 250, 255), r=1.2, ext=EXT)
        p2d.scatter(d, BOX, x[amb], (248, 113, 113), r=2.4, ext=EXT)
        vk.text_c(d, (900, 400), f"甲类条件采样：模糊样本 {ST[0]['ambig']:.1%}", F_BODY, TEXT)
        vk.text_c(d, (900, 445), "（红点：更像乙类的样本）", F_SMALL, (248, 113, 113))


def sc_cg(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "分类器引导（Classifier Guidance）")
    fpaste(c, r"$p(x\mid c)\ \propto\ p(x)\,p(c\mid x)$", 640, 170, 30)
    if seg >= 1:
        fpaste(c, r"$\nabla_x\log p(x\mid c)=\nabla_x\log p(x)+\nabla_x\log p(c\mid x)$", 640, 260, 30)
        fpaste(c, r"$\tilde s(x)=\nabla_x\log p(x)+\gamma\,\nabla_x\log p_\phi(c\mid x),\qquad \gamma>1$", 640, 350, 30, "#a78bfa")
    if seg >= 2:
        vk.text_c(d, (640, 450), "需要额外训练一个能处理带噪输入 x_t 的分类器", F_BODY, MUTED)


def sc_cfg(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "无分类器引导（CFG）")
    vk.text_c(d, (640, 160), "训练：以一定概率（如 10%）把条件 c 换成“空条件”", F_BODY, TEXT)
    if seg >= 1:
        fpaste(c, r"$\tilde\epsilon=(1+w)\,\epsilon_\theta(x_t,c)-w\,\epsilon_\theta(x_t,\varnothing)=\epsilon_\theta(x_t,c)+w\left[\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\right]$", 640, 260, 26, "#22d3ee")
        vk.text_c(d, (845, 320), "引导方向", F_SMALL, CYAN)
    if seg >= 2:
        fpaste(c, r"$\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\ \approx\ -\sqrt{1-\bar\alpha_t}\;\nabla_{x_t}\log p(c\mid x_t)$", 640, 410, 28, "#a78bfa")
        vk.text_c(d, (640, 470), "隐式分类器：p(c|x) ∝ p(x|c) / p(x)", F_SMALL, MUTED)


def sc_sweep(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "调节引导强度 w")
    shown = [0, 1, 3, 7]
    nshow = 0 if seg == 0 else (2 if seg == 1 else 4)
    for i, w in enumerate(shown):
        box = (60 + i * 300, 130, 60 + i * 300 + 260, 390)
        img = p2d.heatmap(box, A.logpdf, ext=EXT, color=(34, 211, 238)) if i < nshow else None
        if img:
            c.paste(img, box[:2])
            p2d.scatter(d, box, SAMPLES[w][:2500], (230, 245, 255), r=1.0, ext=EXT)
            st = ST[w]
            vk.text_c(d, ((box[0] + box[2]) / 2, box[3] + 25), f"w = {w}", vk.font(26, mono=True), CYAN)
            vk.text_c(d, ((box[0] + box[2]) / 2, box[3] + 60), f"模糊 {st['ambig']:.1%}  横向散布 {st['tang']:.2f}", F_SMALL, TEXT)
            vk.text_c(d, ((box[0] + box[2]) / 2, box[3] + 90), f"半径 {st['radius']:.2f} / 真实 {ST_TRUE['radius']:.2f}", F_SMALL, MUTED)
        p2d.frame(d, box)


def sc_tradeoff(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "权衡：符合条件 / 多样性 / 失真")
    box = (150, 140, 1130, 470)
    d.rectangle(box, outline=(30, 41, 59))
    xs = lambda w: box[0] + w / 7 * (box[2] - box[0])
    series = [("符合条件（甲类置信度）", [ST[w]["conf"] for w in WS], 1.0, CYAN),
              ("多样性（横向散布 / 真实值）", [ST[w]["tang"] / ST_TRUE["tang"] for w in WS], 1.2, GREEN),
              ("失真（平均位置外移 / 标准差）", [max(0.0, (ST[w]["radius"] - ST_TRUE["radius"]) / SIG) for w in WS], 1.2, (248, 113, 113))]
    for j, (name, ys, ymax, col) in enumerate(series):
        pts = [(xs(w), box[3] - y / ymax * (box[3] - box[1])) for w, y in zip(WS, ys)]
        d.line(pts, fill=col, width=3)
        for p in pts:
            d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=col)
        d.text((box[2] - 330, box[1] + 140 + j * 30), "● " + name, font=F_SMALL, fill=col)
    for w in WS:
        d.text((xs(w), box[3] + 20), f"{w:g}", font=vk.font(20, mono=True), fill=MUTED, anchor="mm")
    vk.text_c(d, (640, 520), "w 越大：越符合条件，多样性越低，失真越大", F_BODY, TEXT)
    d.text((box[2], box[3] + 48), "引导强度 w", font=F_SMALL, fill=MUTED, anchor="rm")
    if seg >= 1:
        vk.text_c(d, (640, 560), "Stable Diffusion 默认 guidance_scale = 7.5  （= 1 + w，即 w = 6.5）", F_SMALL, VIOLET)


def sc_summary(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 6, "总结")
    items = [
        ("条件生成", "把条件输入去噪网络，学习条件分布"),
        ("分类器引导", "用分类器梯度加强条件，需额外分类器"),
        ("无分类器引导", "条件与无条件预测之差作引导方向"),
        ("引导强度", "符合条件 / 多样性 / 失真 之间权衡"),
    ]
    for i, (k, v) in enumerate(items):
        if seg >= i + 1:
            y = 130 + i * 95
            d.rounded_rectangle([140, y, 1140, y + 76], 14, outline=(30, 41, 59), fill=(12, 19, 34))
            d.text((180, y + 38), f"0{i + 1}", font=vk.font(30, mono=True), fill=CYAN, anchor="lm")
            d.text((260, y + 38), k, font=F_BODY, fill=TEXT, anchor="lm")
            d.text((480, y + 38), v, font=F_BODY, fill=MUTED, anchor="lm")
    if seg >= 5:
        vk.text_c(d, (640, 530), "下一期：Latent Diffusion 与 Stable Diffusion", F_SMALL, VIOLET)


RENDER = {"title": sc_title, "cond": sc_cond, "cg": sc_cg, "cfg": sc_cfg, "sweep": sc_sweep,
          "tradeoff": sc_tradeoff, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "cfg")
