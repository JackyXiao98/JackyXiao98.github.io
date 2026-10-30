"""Knowledge Video 08 — 扩散模型与隐私：记忆化和差分隐私训练.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python ep08_privacy.py <out_dir>

Training set: 20 points from a ring. The optimal (empirical) denoiser memorizes them; a
smoothed model (each training point replaced by N(x_i, h^2 I)) emulates generalization.
Membership inference = denoising-loss threshold attack at t = 0.1 (64 fixed noise draws),
members = the 20 training points, non-members = 200 fresh ring samples. The (eps, delta)-DP
region TPR <= e^eps FPR + delta (and its symmetric counterpart) is drawn for eps = 1.
All numbers in the narration are computed and asserted below.
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

RED = (248, 113, 113)


def ring(n, rng):
    th = rng.uniform(0, 2 * np.pi, n)
    r = 2 + 0.15 * rng.standard_normal(n)
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)


TRAIN = ring(20, np.random.default_rng(0))
NON = ring(200, np.random.default_rng(9))
TRUE = ring(4000, np.random.default_rng(5))
XT = np.random.default_rng(1).standard_normal((4000, 2))
HS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def model(h):
    return D.GMM(TRAIN, max(h, 1e-3))


def nn_dist(x):
    return np.sqrt(((x[:, None] - TRAIN[None]) ** 2).sum(-1)).min(1)


SAMPLES = {h: D.ddim(model(h), XT, 100) for h in HS}
COPY = {h: float(np.mean(nn_dist(SAMPLES[h]) < 0.05)) for h in HS}
RING_ERR = {h: float(np.mean(np.abs(np.linalg.norm(SAMPLES[h], axis=1) - 2))) for h in HS}
RING_TRUE = float(np.mean(np.abs(np.linalg.norm(TRUE, axis=1) - 2)))


def losses(h, x, t=0.1, M=64):
    g, a = model(h), D.abar(t)
    e = np.random.default_rng(11).standard_normal((M, 1, 2))
    xt = np.sqrt(a) * x[None] + np.sqrt(1 - a) * e
    eh = g.eps_hat(xt.reshape(-1, 2), a).reshape(M, -1, 2)
    return ((eh - e) ** 2).sum(-1).mean(0)


def roc(h):
    lm, ln = losses(h, TRAIN), losses(h, NON)
    thr = np.sort(np.concatenate([lm, ln, [np.inf]]))
    tpr = np.array([np.mean(lm <= s) for s in thr])
    fpr = np.array([np.mean(ln <= s) for s in thr])
    auc = float(np.mean(lm[:, None] < ln[None]) + 0.5 * np.mean(lm[:, None] == ln[None]))
    return np.concatenate([[0], fpr]), np.concatenate([[0], tpr]), auc


ROC = {h: roc(h) for h in HS}
AUC = {h: ROC[h][2] for h in HS}
EPS, DELTA = 1.0, 1e-5


def dp_tpr_max(fpr):
    return np.minimum(1.0, np.minimum(np.exp(EPS) * fpr + DELTA, 1 - np.exp(-EPS) * (1 - DELTA - fpr)))


F0, T0, _ = ROC[0.0]
EXCEED = float(np.max(T0 - dp_tpr_max(F0)))
for h in HS:
    print(f"h={h}: copy {COPY[h]:.3f} ring_err {RING_ERR[h]:.3f} AUC {AUC[h]:.3f}")
print(f"true ring_err {RING_TRUE:.3f}; max TPR above the eps=1 bound (h=0): {EXCEED:.3f}")

assert COPY[0.0] > 0.99
assert 0.1 < COPY[0.1] < 0.2 and COPY[0.2] < 0.06
assert RING_ERR[0.5] > 2.5 * RING_TRUE
assert 0.85 < AUC[0.0] < 0.92 and 0.7 < AUC[0.5] < 0.77
assert EXCEED > 0.2

# ---------------------------------------------------------------------------
# Narration
# ---------------------------------------------------------------------------
SCENES = [
    ("title", [
        "前几期我们关注的是怎样生成得更好。这一期换一个角度，扩散模型会不会记住训练数据，泄露隐私？",
        "我们会依次看到记忆化、成员推断攻击，以及差分隐私训练。",
    ]),
    ("memorize", [
        "先做一个实验，从一个圆环分布中抽取二十个点作为训练集。",
        "如果去噪器完美地拟合了这二十个点，也就是使用训练集上的最优去噪器，生成的样本会怎样呢？",
        "结果所有样本都精确地落在了训练点上，模型变成了一台复印机，这就是记忆化。第一期里四张图片的例子，也是同样的现象。",
    ]),
    ("smooth", [
        "真实的神经网络不会完美拟合，它会在训练点之间平滑地插值，从而带来泛化。",
        "我们用一个平滑参数来模拟这种效果，把每个训练点换成一个小的高斯分布，平滑程度越大，模型越不依赖单个训练点。",
        "随着平滑程度增大，复制训练点的比例从百分之百降到百分之十五，再降到百分之四，样本开始填满整个圆环；但平滑过度，样本就会变得模糊，偏离圆环。",
    ]),
    ("mia", [
        "记忆化会带来隐私风险。最简单的攻击叫成员推断，攻击者想知道，某个样本是否在训练集里。",
        "方法很简单，看模型对这个样本的去噪误差，训练过的样本，误差往往更小。",
        "没有平滑时，攻击的曲线下面积约为零点八八，远高于随机猜测的零点五。平滑可以降低攻击效果，但并不能消除，即使平滑到样本已经明显模糊，曲线下面积仍有零点七以上。",
        "在真实世界中，研究者也确实从稳定扩散等模型中，提取出了和训练图片几乎一样的生成结果。",
    ]),
    ("dp", [
        "差分隐私给出了严格的保证，无论有没有某一条训练数据，模型的输出分布都几乎一样，差别由隐私预算艾普西隆来控制。",
        "训练扩散模型时，常用差分隐私随机梯度下降，对每个样本的梯度先裁剪到固定的范数，再加入高斯噪声，最后累计计算隐私预算。",
        "这个保证直接限制了任何成员推断攻击，攻击的真阳性率，最多是假阳性率乘以艾普西隆的指数，再加上德尔塔。",
        "图中灰色区域，是隐私预算为一时，任何攻击都不能超出的范围；而没有保护的记忆化模型，攻击曲线远远超出了这个范围。",
    ]),
    ("practice", [
        "差分隐私的代价是生成质量下降，尤其是在数据少、模型大的时候。",
        "一个实用的路线是，先在公开数据上预训练，再用差分隐私在私有数据上微调，已有研究用这种方式，生成了质量不错的隐私合成数据。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，完美拟合训练集的扩散模型，会直接复制训练数据。",
        "第二，记忆化让成员推断攻击成为可能，平滑等启发式方法只能缓解，不能保证。",
        "第三，差分隐私随机梯度下降给出可证明的保证，严格限制任何攻击的成功率。",
        "第四，预训练加差分隐私微调，是兼顾质量和隐私的实用路线。",
        "下一期，我们讲扩散变换器和视频生成。感谢大家的观看。",
    ]),
]

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
EXT = 3.0
BOX = (110, 115, 560, 565)


def fpaste(c, tex, cx, cy, size=28, color="#e2e8f0"):
    vk.paste_center(c, vk.formula(tex, size, color), cx, cy)


def ring_guide(d, box):
    th = np.linspace(0, 2 * np.pi, 200)
    pts = p2d.to_px(box, np.stack([2 * np.cos(th), 2 * np.sin(th)], 1), EXT)
    d.line([tuple(p) for p in pts], fill=(40, 55, 80), width=1)


def train_marks(d, box, col=RED, r=6):
    for x, y in p2d.to_px(box, TRAIN, EXT):
        d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=2)


def sc_title(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.text_c(d, (640, 190), "Knowledge Video · 08", vk.font(24, mono=True), MUTED)
    vk.text_c(d, (640, 270), "扩散模型与隐私", F_TITLE, CYAN)
    vk.text_c(d, (640, 340), "记忆化 · 成员推断 · 差分隐私训练", F_BODY, TEXT)


def sc_memorize(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "记忆化：完美拟合 = 复印机")
    p2d.frame(d, BOX)
    ring_guide(d, BOX)
    train_marks(d, BOX)
    if seg >= 2:
        p2d.scatter(d, BOX, SAMPLES[0.0][:2000], CYAN, r=1.5, ext=EXT)
        vk.text_c(d, (900, 330), f"复制训练点的比例：{COPY[0.0]:.0%}", F_BODY, CYAN)
    vk.text_c(d, (900, 200), "红圈：20 个训练点", F_BODY, RED)
    if seg >= 1:
        fpaste(c, r"$\hat x_0(x_t)=\sum_i w_i(x_t)\,x_i^{\mathrm{train}}$", 900, 260, 28)


def sc_smooth(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "平滑模拟泛化")
    shown = [0.0, 0.1, 0.2, 0.5]
    for i, h in enumerate(shown):
        box = (60 + i * 300, 130, 60 + i * 300 + 260, 390)
        p2d.frame(d, box)
        ring_guide(d, box)
        if seg >= 2 or (seg == 1 and i == 0):
            p2d.scatter(d, box, SAMPLES[h][:2000], CYAN, r=1.0, ext=EXT)
            train_marks(d, box, r=4)
            vk.text_c(d, ((box[0] + box[2]) / 2, box[3] + 25), f"平滑 h = {h:g}", vk.font(24), CYAN)
            vk.text_c(d, ((box[0] + box[2]) / 2, box[3] + 60), f"复制 {COPY[h]:.0%}   偏离圆环 {RING_ERR[h]:.2f}", F_SMALL, TEXT)
    vk.text_c(d, (640, 540), f"（真实分布的平均偏离为 {RING_TRUE:.2f}）", F_SMALL, MUTED)


def roc_plot(d, box, hs, show_dp=False, p=1.0):
    x0, y0, x1, y1 = box
    d.rectangle(box, outline=(30, 41, 59))
    X = lambda f: x0 + f * (x1 - x0)
    Y = lambda t: y1 - t * (y1 - y0)
    if show_dp:
        fs = np.linspace(0, 1, 200)
        up = dp_tpr_max(fs)
        d.polygon([(X(f), Y(t)) for f, t in zip(fs, up)] + [(X(1), Y(1)), (X(1), Y(0)), (X(0), Y(0))], fill=(40, 48, 64))
        d.text((X(0.62), Y(0.4)), "ε = 1 时任何攻击\n都在灰色区域内", font=F_SMALL, fill=MUTED)
    d.line([(X(0), Y(0)), (X(1), Y(1))], fill=(71, 85, 105), width=1)
    cols = {0.0: RED, 0.2: (251, 191, 36), 0.5: GREEN}
    for j, h in enumerate(hs):
        f, t, auc = ROC[h]
        k = max(2, int(len(f) * p))
        d.line([(X(a), Y(b)) for a, b in zip(f[:k], t[:k])], fill=cols[h], width=3)
        d.text((x1 + 14, y0 + 20 + j * 34), f"h = {h:g}   AUC {auc:.2f}", font=F_SMALL, fill=cols[h])
    d.text((X(0.5), y1 + 22), "假阳性率", font=F_SMALL, fill=MUTED, anchor="mm")
    d.text((x0 - 14, Y(0.5)), "真阳性率", font=F_SMALL, fill=MUTED, anchor="rm")


def sc_mia(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "成员推断攻击")
    if seg <= 1:
        vk.text_c(d, (640, 200), "这个样本在训练集里吗？", F_TITLE, TEXT)
        if seg >= 1:
            fpaste(c, r"$\mathrm{score}(x)=\mathbb{E}_{\epsilon}\,\|\epsilon-\hat\epsilon(\sqrt{\bar\alpha_t}\,x+\sqrt{1-\bar\alpha_t}\,\epsilon)\|^2$", 640, 330, 30, "#a78bfa")
            vk.text_c(d, (640, 400), "误差越小 → 越可能是训练成员", F_BODY, MUTED)
        return
    roc_plot(d, (260, 120, 720, 520), [0.0, 0.2, 0.5], p=vk.ease(min(1, su * 1.5)) if seg == 2 else 1.0)
    if seg >= 3:
        vk.text_c(d, (980, 380), "真实模型中也观察到\n训练图片被提取", F_BODY, RED)


def sc_dp(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "差分隐私（DP）")
    if seg <= 2:
        fpaste(c, r"$\Pr[M(D)\in S]\ \leq\ e^{\varepsilon}\,\Pr[M(D')\in S]+\delta$", 640, 170, 32, "#22d3ee")
        vk.text_c(d, (640, 225), "D 与 D′ 只相差一条训练数据", F_SMALL, MUTED)
        if seg >= 1:
            fpaste(c, r"$g_i\leftarrow g_i\cdot\min\!\left(1,\frac{C}{\|g_i\|}\right),\qquad \tilde g=\frac{1}{B}\left(\sum_i g_i+\mathcal{N}(0,\sigma^2C^2I)\right)$", 640, 320, 28)
            vk.text_c(d, (640, 380), "DP-SGD：逐样本裁剪 + 高斯噪声 + 隐私核算", F_SMALL, MUTED)
        if seg >= 2:
            fpaste(c, r"$\mathrm{TPR}\ \leq\ e^{\varepsilon}\cdot\mathrm{FPR}+\delta$", 640, 470, 34, "#34d399")
        return
    roc_plot(d, (260, 120, 720, 520), [0.0], show_dp=True)
    vk.text_c(d, (980, 380), f"无保护模型的真阳性率\n最多超出边界 {EXCEED:.2f}", F_BODY, RED)


def sc_practice(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "实践中的权衡")
    vk.text_c(d, (640, 200), "隐私越强（ε 越小） → 噪声越大 → 质量越低", F_BODY, TEXT)
    if seg >= 1:
        for i, (lab, col) in enumerate([("公开数据\n预训练", CYAN), ("私有数据\nDP 微调", GREEN), ("隐私\n合成数据", VIOLET)]):
            x = 300 + i * 340
            d.rounded_rectangle([x - 120, 320, x + 120, 440], 16, outline=col, width=3, fill=(12, 19, 34))
            vk.text_c(d, (x, 380), lab, F_BODY, col)
            if i < 2:
                vk.arrow(d, (x + 130, 380), (x + 210, 380), MUTED, 3)


def sc_summary(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 6, "总结")
    items = [
        ("记忆化", "完美拟合训练集 → 直接复制训练数据"),
        ("成员推断", "启发式方法只能缓解，不能保证"),
        ("DP-SGD", "可证明地限制任何攻击的成功率"),
        ("实用路线", "公开数据预训练 + 私有数据 DP 微调"),
    ]
    for i, (k, v) in enumerate(items):
        if seg >= i + 1:
            y = 130 + i * 95
            d.rounded_rectangle([140, y, 1140, y + 76], 14, outline=(30, 41, 59), fill=(12, 19, 34))
            d.text((180, y + 38), f"0{i + 1}", font=vk.font(30, mono=True), fill=CYAN, anchor="lm")
            d.text((260, y + 38), k, font=F_BODY, fill=TEXT, anchor="lm")
            d.text((450, y + 38), v, font=F_BODY, fill=MUTED, anchor="lm")
    if seg >= 5:
        vk.text_c(d, (640, 530), "下一期：Diffusion Transformer 与视频生成", F_SMALL, VIOLET)


RENDER = {"title": sc_title, "memorize": sc_memorize, "smooth": sc_smooth, "mia": sc_mia,
          "dp": sc_dp, "practice": sc_practice, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "privacy")
