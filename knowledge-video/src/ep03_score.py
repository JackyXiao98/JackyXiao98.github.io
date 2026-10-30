"""Knowledge Video 03 — Score 与 SDE：分数匹配和朗之万采样.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python ep03_score.py <out_dir>

2D data: three isotropic Gaussians with weights 0.6 / 0.3 / 0.1. All scores are exact.
Plain Langevin, annealed Langevin (NCSN schedule), the reverse VP SDE and the probability-flow
ODE are simulated for real; the mode proportions shown are measured from the samples.
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

G = D.GMM([[-2.0, -1.2], [2.0, -1.2], [0.0, 2.0]], 0.3, [0.6, 0.3, 0.1])
LABELS = ["左下峰 60%", "右下峰 30%", "上方峰 10%"]


def props(x):
    k = ((x[:, None, :] - G.mu[None]) ** 2).sum(-1).argmin(1)
    return np.bincount(k, minlength=3) / len(x)


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------
N = 6000
rng = np.random.default_rng(0)

# plain Langevin on the data score from a uniform start
x = rng.uniform(-4, 4, (N, 2))
LANG = [x[:900].copy()]
for i in range(2000):
    x = x + 0.01 / 2 * G.score(x) + np.sqrt(0.01) * rng.standard_normal(x.shape)
    if i % 10 == 9:
        LANG.append(x[:900].copy())
LANG = np.array(LANG)
P_LANG = props(x)

# annealed Langevin (NCSN): 10 noise levels from 4 to 0.01, 100 steps each
rng = np.random.default_rng(0)
SIGS = np.geomspace(4, 0.01, 10)
x = rng.uniform(-4, 4, (N, 2))
ANN = [x[:900].copy()]
for sig in SIGS:
    a = 2e-5 * sig ** 2 / SIGS[-1] ** 2
    for i in range(100):
        x = x + a / 2 * G.score(x, extra_var=sig ** 2) + np.sqrt(a) * rng.standard_normal(x.shape)
        if i % 10 == 9:
            ANN.append(x[:900].copy())
ANN = np.array(ANN)
P_ANN = props(x)


def reverse(n, sde, x, keep=800):
    ts = np.linspace(1, D.T_MIN, n + 1)
    traj = [x[:keep].copy()]
    for t, s in zip(ts[:-1], ts[1:]):
        dt, b, sc = t - s, D.beta(t), G.score(x, D.abar(t))
        if sde:
            x = x + (0.5 * b * x + b * sc) * dt + np.sqrt(b * dt) * rng.standard_normal(x.shape)
        else:
            x = x + (0.5 * b * x + 0.5 * b * sc) * dt
        traj.append(x[:keep].copy())
    return x, np.array(traj)


rng = np.random.default_rng(1)
XT = rng.standard_normal((N, 2))
X_SDE, TR_SDE = reverse(500, True, XT)
X_ODE, TR_ODE = reverse(500, False, XT)
P_SDE, P_ODE = props(X_SDE), props(X_ODE)
P_TRUE = G.w

print("proportions langevin", P_LANG.round(3), "annealed", P_ANN.round(3), "sde", P_SDE.round(3), "ode", P_ODE.round(3))


def pct(p):
    return "比".join(str(int(round(v * 100))) for v in p)


# ---------------------------------------------------------------------------
# Narration (numbers are taken from the simulations above)
# ---------------------------------------------------------------------------
NUM = {6: "六", 3: "三", 1: "一"}
SCENES = [
    ("title", [
        "前两期，我们从加噪和预测噪声的角度，理解了扩散模型。",
        "这一期换一个角度：分数函数。你会发现，预测噪声和估计分数，其实是同一件事。",
        "我们用一个二维的例子：平面上有三个高斯峰，权重分别是百分之六十、百分之三十和百分之十。",
    ]),
    ("score", [
        "分数函数，就是对数概率密度对输入的梯度。",
        "它是一个向量场：在每个位置，都指向概率增大最快的方向。",
        "离峰越远，箭头越长；到了峰顶，箭头就缩成零。",
        "一个好处是，分数和密度的归一化常数无关，所以比密度本身更容易学习。",
    ]),
    ("langevin", [
        "有了分数，就可以用朗之万动力学来采样。",
        "每一步，沿着分数的方向走一小步，再加上一点随机噪声。",
        "理论上，只要步长足够小、步数足够多，样本就会服从目标分布。",
    ]),
    ("problem", [
        "但实际中有一个问题。",
        "我们数一数三个峰里的样本比例：真实权重是六比三比一，朗之万采样得到的却接近三等分。",
        "原因是，峰与峰之间是低密度区域，样本很难跨过去，最后落在哪个峰，主要取决于起点离谁近。",
        "而且在低密度区域，训练数据很少，分数本身也很难学准。",
    ]),
    ("annealed", [
        "解决办法是，给数据加上不同强度的噪声。",
        "噪声大的时候，分布被抹平，分数场在整个平面都清晰可靠，样本可以在峰之间自由移动。",
        "然后逐渐减小噪声，一边采样一边退火，最后回到原始的分布。",
        "这就是退火朗之万采样。这一次，三个峰的比例和真实权重基本一致。",
    ]),
    ("dsm", [
        "这和去噪有什么关系呢？",
        "对于加了噪声的数据，它对数密度的梯度，恰好等于负的噪声，再除以噪声的标准差。",
        "所以，训练网络去预测噪声，就等价于训练它估计加噪数据的分数，这叫去噪分数匹配。",
    ]),
    ("sde", [
        "把加噪的步数取到无穷多，前向过程就变成了一个随机微分方程。",
        "这个方程有一个对应的反向随机微分方程，只要知道每个时刻的分数，就能从噪声倒推回数据。",
        "它还有一个确定性的版本，叫概率流常微分方程，在每个时刻，它和反向随机微分方程的分布完全相同。",
        "左边是随机的反向轨迹，右边是确定性的常微分方程轨迹，两者最后得到的比例，都和六比三比一基本一致。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，分数是对数密度的梯度，指向概率增大的方向。",
        "第二，朗之万动力学可以用分数采样，但只用一个噪声水平，很难跨越低密度区域。",
        "第三，多尺度加噪解决了这个问题，而预测噪声，就等价于估计分数。",
        "第四，在连续时间下，扩散模型是一对正反向的随机微分方程，还有一个确定性的常微分方程版本。",
        "下一期，我们利用这个常微分方程，把采样从一千步减少到十几步。感谢大家的观看。",
    ]),
]

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
BOX = (110, 115, 560, 565)
HEAT = p2d.heatmap(BOX, G.logpdf)
HEAT_SMALL = {}


def heat_small(box, sig):
    key = (box, sig)
    if key not in HEAT_SMALL:
        HEAT_SMALL[key] = p2d.heatmap(box, lambda p: G.logpdf(p, extra_var=sig ** 2))
    return HEAT_SMALL[key]


def fpaste(c, tex, cx, cy, size=28, color="#e2e8f0"):
    vk.paste_center(c, vk.formula(tex, size, color), cx, cy)


def mode_labels(d, box):
    for (mx, my), lab in zip(G.mu, LABELS):
        px, py = p2d.to_px(box, np.array([[mx, my]]))[0]
        d.text((px, py + 48), lab, font=F_SMALL, fill=TEXT, anchor="mm")


def sc_title(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.text_c(d, (880, 200), "Knowledge Video · 03", vk.font(24, mono=True), MUTED)
    vk.text_c(d, (880, 270), "Score 与 SDE", F_TITLE, CYAN)
    vk.text_c(d, (880, 340), "分数匹配和朗之万采样", F_BODY, TEXT)
    if seg >= 2:
        c.paste(HEAT, BOX[:2])
        p2d.frame(d, BOX)
        mode_labels(d, BOX)


def sc_score(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "分数函数：对数密度的梯度")
    c.paste(HEAT, BOX[:2])
    p2d.frame(d, BOX)
    if seg >= 1:
        p2d.arrows(d, BOX, G.score)
    fpaste(c, r"$s(x)=\nabla_x \log p(x)$", 900, 200, 36, "#22d3ee")
    if seg >= 3:
        fpaste(c, r"$p(x)=\frac{\tilde p(x)}{Z}\ \Rightarrow\ \nabla_x\log p(x)=\nabla_x\log\tilde p(x)$", 900, 320, 26)
        vk.text_c(d, (900, 390), "与归一化常数 Z 无关", F_SMALL, GREEN)


def sc_langevin(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "朗之万动力学")
    c.paste(HEAT, BOX[:2])
    p2d.frame(d, BOX)
    k = 0 if seg == 0 else int((LANG.shape[0] - 1) * vk.ease(min(1, (u - 0.2) / 0.7)))
    p2d.scatter(d, BOX, LANG[max(k, 0)], (167, 139, 250), r=1.5)
    fpaste(c, r"$x_{k+1}=x_k+\frac{\eta}{2}\,\nabla_x\log p(x_k)+\sqrt{\eta}\,z_k$", 900, 220, 30)
    if seg >= 1:
        vk.text_c(d, (830, 290), "沿分数走一步", F_SMALL, CYAN)
        vk.text_c(d, (1040, 290), "加随机噪声", F_SMALL, VIOLET)
    vk.text_c(d, (900, 360), f"步数 {k * 10}", vk.font(26), MUTED)


def sc_problem(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "问题：比例不对")
    c.paste(HEAT, BOX[:2])
    p2d.frame(d, BOX)
    p2d.scatter(d, BOX, LANG[-1], (167, 139, 250), r=1.5)
    if seg >= 1:
        p2d.bars(d, (640, 160, 1180, 470), [P_TRUE, P_LANG], ["左下", "右下", "上方"],
                 [CYAN, VIOLET], ymax=0.75, legend=["真实权重", "朗之万采样"])
    if seg >= 2:
        vk.text_c(d, (910, 530), "低密度区域难以跨越", F_BODY, MUTED)


def sc_annealed(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "多尺度噪声：退火朗之万")
    if seg <= 1:
        for i, sig in enumerate([2.0, 0.8, 0.3]):
            box = (70 + i * 400, 150, 70 + i * 400 + 340, 490)
            c.paste(heat_small(box, sig), box[:2])
            p2d.frame(d, box)
            if seg >= 1:
                p2d.arrows(d, box, lambda p, s=sig: G.score(p, extra_var=s ** 2), n=11)
            vk.text_c(d, ((box[0] + box[2]) / 2, box[3] + 25), f"噪声 σ = {sig}", F_SMALL, MUTED)
        return
    k = int((ANN.shape[0] - 1) * (vk.ease(su) if seg == 2 else 1.0))
    level = min(len(SIGS) - 1, k // 10)
    sig = SIGS[level]
    c.paste(heat_small(BOX, round(float(sig), 2)) if sig > 0.3 else HEAT, BOX[:2])
    p2d.frame(d, BOX)
    p2d.scatter(d, BOX, ANN[k], (167, 139, 250), r=1.5)
    vk.text_c(d, ((BOX[0] + BOX[2]) / 2, BOX[3] + 25), f"当前噪声 σ = {sig:.2f}", F_SMALL, MUTED)
    if seg >= 3:
        p2d.bars(d, (640, 160, 1180, 470), [P_TRUE, P_ANN], ["左下", "右下", "上方"],
                 [CYAN, GREEN], ymax=0.75, legend=["真实权重", "退火朗之万"])


def sc_dsm(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "去噪分数匹配：预测噪声 = 估计分数")
    fpaste(c, r"$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$", 640, 170, 30)
    if seg >= 1:
        fpaste(c, r"$\nabla_{x_t}\log q(x_t\mid x_0)=-\frac{x_t-\sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t}=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$", 640, 270, 30, "#a78bfa")
    if seg >= 2:
        fpaste(c, r"$s_\theta(x_t,t)=-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}$", 640, 380, 32, "#22d3ee")
        fpaste(c, r"$\mathbb{E}\,\|s_\theta(x_t,t)-\nabla_{x_t}\log q(x_t\mid x_0)\|^2\ \propto\ \mathbb{E}\,\|\epsilon_\theta-\epsilon\|^2$", 640, 470, 28)


def sc_sde(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 6, "连续时间：随机微分方程")
    if seg <= 2:
        fpaste(c, r"$\mathrm{d}x=-\frac{1}{2}\beta(t)\,x\,\mathrm{d}t+\sqrt{\beta(t)}\,\mathrm{d}w$", 640, 170, 30)
        vk.text_c(d, (1100, 170), "前向", F_SMALL, MUTED)
        if seg >= 1:
            fpaste(c, r"$\mathrm{d}x=\left[-\frac{1}{2}\beta(t)\,x-\beta(t)\,\nabla_x\log p_t(x)\right]\mathrm{d}t+\sqrt{\beta(t)}\,\mathrm{d}\bar w$", 640, 290, 30, "#a78bfa")
            vk.text_c(d, (1180, 290), "反向", F_SMALL, VIOLET)
        if seg >= 2:
            fpaste(c, r"$\mathrm{d}x=\left[-\frac{1}{2}\beta(t)\,x-\frac{1}{2}\beta(t)\,\nabla_x\log p_t(x)\right]\mathrm{d}t$", 640, 410, 30, "#22d3ee")
            vk.text_c(d, (1110, 410), "概率流 ODE", F_SMALL, CYAN)
        return
    n = TR_SDE.shape[0] - 1
    k = int(n * vk.ease(min(1.0, su * 1.4)))
    for i, (tr, P, name, col) in enumerate([(TR_SDE, P_SDE, "反向 SDE（随机）", (167, 139, 250)),
                                            (TR_ODE, P_ODE, "概率流 ODE（确定）", (34, 211, 238))]):
        box = (90 + i * 560, 115, 90 + i * 560 + 440, 555)
        p2d.frame(d, box)
        p2d.trails(d, box, tr[: k + 1 : 5, :10], (k // 5), tuple(int(v * 0.6) for v in col), width=2)
        p2d.scatter(d, box, tr[k], col, r=1.8)
        vk.text_c(d, ((box[0] + box[2]) / 2, box[1] - 2 + 465), f"{name}   比例 {pct(P)}" if k == n else name, F_SMALL, col)


def sc_summary(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 7, "总结")
    items = [
        ("分数函数", "对数密度的梯度，指向概率增大的方向"),
        ("朗之万", "用分数采样，单一噪声难跨越低密度区"),
        ("多尺度加噪", "解决跨越问题；预测噪声 = 估计分数"),
        ("SDE / ODE", "正反向随机微分方程 + 确定性 ODE"),
    ]
    for i, (k, v) in enumerate(items):
        if seg >= i + 1:
            y = 130 + i * 95
            d.rounded_rectangle([140, y, 1140, y + 76], 14, outline=(30, 41, 59), fill=(12, 19, 34))
            d.text((180, y + 38), f"0{i + 1}", font=vk.font(30, mono=True), fill=CYAN, anchor="lm")
            d.text((260, y + 38), k, font=F_BODY, fill=TEXT, anchor="lm")
            d.text((470, y + 38), v, font=F_BODY, fill=MUTED, anchor="lm")
    if seg >= 5:
        vk.text_c(d, (640, 530), "下一期：采样加速 —— DDIM 与 DPM-Solver", F_SMALL, VIOLET)


RENDER = {
    "title": sc_title, "score": sc_score, "langevin": sc_langevin, "problem": sc_problem,
    "annealed": sc_annealed, "dsm": sc_dsm, "sde": sc_sde, "summary": sc_summary,
}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "score-sde")
