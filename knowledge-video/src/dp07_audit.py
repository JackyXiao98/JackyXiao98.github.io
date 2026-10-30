"""Differential Privacy 07 — 成员推断攻击与隐私审计.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python dp07_audit.py <out_dir>

Real computations: 200 audit canaries (held-out digits with a random wrong label, each inserted into
the training set with probability 1/2); the loss-threshold membership attack against an MLP trained
30 and 200 epochs without privacy and 30 epochs with DP-SGD (sigma = 2); AUC, TPR/FPR at loss < 1
and the resulting empirical lower bound on eps.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import dpx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Differential Privacy"
CACHE = os.environ.get("DP_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "dpcache"))
NC, DELTA, TH = 200, 1e-5, 1.0

XC, YC, INC = dpx.canaries(NC)
N_IN = int(INC.sum())
MODELS = {
    "plain30": dpx.cached(CACHE, dpx.train_mlp, dp=False, epochs=30, n_canary=NC),
    "plain200": dpx.cached(CACHE, dpx.train_mlp, dp=False, epochs=200, n_canary=NC),
    "dp": dpx.cached(CACHE, dpx.train_mlp, dp=True, sigma=2.0, epochs=30, n_canary=NC),
}
R = {}
for k, m in MODELS.items():
    l = dpx.losses(m["params"], XC, YC)
    tpr, fpr = float((l[INC] < TH).mean()), float((l[~INC] < TH).mean())
    R[k] = dict(loss=l, auc=dpx.auc(-l[INC], -l[~INC]), tpr=tpr, fpr=fpr, acc=m["hist"][-1][2])
EPS_DP = dpx.dpsgd_eps(MODELS["dp"]["q"], 2.0, MODELS["dp"]["steps"], DELTA)
LB = float(np.log((R["plain200"]["tpr"] - DELTA) / R["plain200"]["fpr"]))
print({k: {a: round(b, 4) for a, b in v.items() if a != "loss"} for k, v in R.items()}, "eps_dp", EPS_DP, "lb", LB, N_IN)
assert 0.65 < R["plain30"]["auc"] < 0.8 and R["plain200"]["auc"] > 0.97 and R["dp"]["auc"] < 0.56
assert R["plain200"]["tpr"] > 0.9 and R["plain200"]["fpr"] < 0.02 and 4 < LB < 5 and 3 < EPS_DP < 4
assert R["plain200"]["acc"] > 0.93

SCENES = [
    ("title", [
        "上一期我们用差分隐私训练了模型。这一期换个角度，站在攻击者的立场，检验一个模型到底记住了多少训练数据。",
    ]),
    ("attack", [
        "最简单的攻击叫成员推断。模型对训练过的样本通常更有把握，损失更低。",
        "所以攻击者拿到一条数据，算一下模型在它上面的损失，如果低于某个阈值，就猜它在训练集里。",
    ]),
    ("canary", [
        f"为了测量最坏的情况，我们做一个审计实验。准备{int_zh(NC)}张手写数字图片，故意给每张图配上一个错误的标签，这些叫金丝雀样本。",
        f"每一张金丝雀都抛硬币决定放不放进训练集，最后放进了{int_zh(N_IN)}张。攻击者的任务，是猜出哪些被放进去了。",
    ]),
    ("plain", [
        f"先看不加隐私保护的模型。训练三十轮时，攻击的曲线下面积是{dec_zh(R['plain30']['auc'])}，随机猜测是零点五。",
        f"训练两百轮以后，曲线下面积达到{dec_zh(R['plain200']['auc'])}。以损失小于一为阈值，放进去的金丝雀有{pct_zh(R['plain200']['tpr'])}被识别出来，没放进去的只有{pct_zh(R['plain200']['fpr'])}被误判。",
        f"值得注意的是，这个模型的测试准确率是{pct_zh(R['plain200']['acc'], 1)}，看起来很正常，但它把那些标签错误的样本都背了下来。",
    ]),
    ("dp", [
        f"再看用差分隐私训练的模型，艾普西隆约为{dec_zh(EPS_DP, 1)}。两组金丝雀的损失分布几乎重合，攻击的曲线下面积只有{dec_zh(R['dp']['auc'])}，和随机猜测差不多。",
        "模型学到了数字的一般规律，却记不住个别错误的标签，这正是差分隐私想要的效果。",
    ]),
    ("audit", [
        "攻击的结果还可以反过来，估计隐私损失的下界。差分隐私要求，真阳性率不能超过假阳性率的艾普西隆指数倍。",
        f"对训练两百轮的模型，这两个比例之比，意味着它的艾普西隆至少约为{dec_zh(LB, 1)}，而它没有任何理论保证。这里只有约一百个样本，只是一个粗略的点估计。",
        "审计只能给出下界，证明给出的是上界。两者越接近，说明攻击越强、分析越紧。实际中的攻击，往往远弱于理论上的最坏情况。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，成员推断攻击利用模型在训练样本上更低的损失，判断数据是否被用过。",
        "第二，少见和异常的样本最容易被记住，测试准确率高并不代表没有泄露。",
        "第三，差分隐私训练让攻击接近随机猜测，审计可以给出隐私损失的经验下界。下一期是本系列最后一期，我们看差分隐私在现实中的应用。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 7, "成员推断攻击与隐私审计", "模型到底记住了什么", SERIES)


def sc_attack(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "损失阈值攻击（Yeom et al., 2018）")
    tv.fpaste(c, r"$\text{guess member}\ \Leftrightarrow\ \ell(f_\theta(x),y)<\tau$", 640, 170, 32)
    if seg >= 1:
        for i, (t, col) in enumerate([("数据 (x, y)", TEXT), ("模型的损失 ℓ", CYAN), ("ℓ < τ → 在训练集里", AMBER)]):
            x = 200 + i * 320
            d.rounded_rectangle([x, 300, x + 260, 380], 12, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 130, 340), t, F_SMALL, col)
            if i < 2:
                vk.arrow(d, (x + 265, 340), (x + 315, 340), MUTED, 3)


def sc_canary(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, f"审计实验：{NC} 个金丝雀样本（错误标签）")
    for i in range(12):
        x, y = 150 + (i % 6) * 110, 150 + (i // 6) * 150
        img = XC[i].reshape(8, 8)
        for r in range(8):
            for cc in range(8):
                v = int(img[r, cc] * 220)
                d.rectangle([x + 4 + cc * 9, y + r * 9, x + 4 + cc * 9 + 8, y + r * 9 + 8], fill=(v // 3, v, v))
        d.text((x + 40, y + 92), f"标签 {YC[i]}", font=F_SMALL, fill=RED, anchor="mm")
        if seg >= 1:
            d.text((x + 40, y + 118), "放入" if INC[i] else "不放", font=F_SMALL, fill=GREEN if INC[i] else MUTED, anchor="mm")
    if seg >= 1:
        vk.text_c(d, (1000, 260), f"抛硬币：{N_IN} 个放入训练集", F_BODY, GREEN)
        vk.text_c(d, (1000, 320), f"{NC - N_IN} 个不放", F_BODY, MUTED)
        vk.text_c(d, (1000, 400), "攻击者：猜哪些被放进去了", F_BODY, AMBER)


def loss_hist(d, box, l, title, col_in=RED, col_out=CYAN):
    x0, y0, x1, y1 = box
    bins = np.linspace(-6, 3, 28)
    li, lo = np.log10(l[INC] + 1e-6), np.log10(l[~INC] + 1e-6)
    hi_, _ = np.histogram(li, bins)
    ho, _ = np.histogram(lo, bins)
    m = max(hi_.max(), ho.max())
    w = (x1 - x0) / (len(bins) - 1)
    for j in range(len(bins) - 1):
        xa = x0 + j * w
        d.rectangle([xa, y1 - hi_[j] / m * (y1 - y0), xa + w / 2 - 1, y1], fill=col_in)
        d.rectangle([xa + w / 2, y1 - ho[j] / m * (y1 - y0), xa + w - 2, y1], fill=col_out)
    xt = x0 + (0 - bins[0]) / (bins[-1] - bins[0]) * (x1 - x0)
    d.line([(xt, y0), (xt, y1)], fill=AMBER, width=2)
    d.text((xt + 4, y0), "ℓ = 1", font=vk.font(18), fill=AMBER)
    for v in (-6, -3, 0, 3):
        d.text((x0 + (v + 6) / 9 * (x1 - x0), y1 + 10), f"1e{v}", font=vk.font(16, mono=True), fill=MUTED, anchor="ma")
    vk.text_c(d, ((x0 + x1) / 2, y0 - 22), title, F_SMALL, TEXT)


def legend(d, x, y):
    d.text((x, y), "■ 放入的金丝雀", font=F_SMALL, fill=RED)
    d.text((x + 190, y), "■ 未放入的", font=F_SMALL, fill=CYAN)


def sc_plain(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "不加隐私的模型")
    loss_hist(d, (120, 190, 580, 480), R["plain30"]["loss"], f"30 轮：AUC = {R['plain30']['auc']:.2f}")
    if seg >= 1:
        p = R["plain200"]
        loss_hist(d, (700, 190, 1160, 480), p["loss"], f"200 轮：AUC = {p['auc']:.2f}")
        vk.text_c(d, (930, 540), f"ℓ < 1：TPR {p['tpr']:.0%}，FPR {p['fpr']:.0%}", F_BODY, AMBER)
    legend(d, 120, 540)
    if seg >= 2:
        vk.text_c(d, (640, 620), f"测试准确率 {R['plain200']['acc']:.1%} —— 看起来正常，却背下了错误标签", F_BODY, RED)


def sc_dp(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, f"DP-SGD 训练的模型（σ = 2，ε ≈ {EPS_DP:.1f}）")
    loss_hist(d, (120, 190, 580, 480), R["plain200"]["loss"], f"不加隐私（200 轮）：AUC = {R['plain200']['auc']:.2f}")
    loss_hist(d, (700, 190, 1160, 480), R["dp"]["loss"], f"DP-SGD：AUC = {R['dp']['auc']:.2f}")
    legend(d, 120, 540)
    if seg >= 1:
        vk.text_c(d, (930, 550), f"测试准确率 {R['dp']['acc']:.1%}", F_BODY, TEXT)
        vk.text_c(d, (640, 620), "学到一般规律，记不住个别错误标签", F_BODY, GREEN)


def sc_audit(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "隐私审计：从攻击结果估计 ε 的下界")
    tv.fpaste(c, r"$\mathrm{TPR}\leq e^{\varepsilon}\,\mathrm{FPR}+\delta\ \Rightarrow\ \varepsilon\geq\ln\frac{\mathrm{TPR}-\delta}{\mathrm{FPR}}$", 640, 160, 30)
    if seg >= 1:
        p = R["plain200"]
        tv.fpaste(c, r"$\ln\frac{%.3f}{%.4f}\approx %.1f$" % (p["tpr"], p["fpr"], LB), 640, 260, 30, "#fbbf24")
        vk.text_c(d, (640, 320), f"不加隐私、200 轮：ε 至少约 {LB:.1f}（约 100 个样本的粗略点估计）", F_SMALL, MUTED)
    if seg >= 2:
        x0, x1 = 200, 1080
        d.line([(x0, 460), (x1, 460)], fill=(71, 85, 105), width=3)
        d.rectangle([x0, 450, x0 + 260, 470], fill=AMBER)
        d.rectangle([x0 + 560, 450, x1, 470], fill=CYAN)
        vk.text_c(d, (x0 + 130, 500), "审计：下界（攻击）", F_SMALL, AMBER)
        vk.text_c(d, (x0 + 720, 500), "证明：上界（记账）", F_SMALL, CYAN)
        vk.text_c(d, (640, 420), "真实的 ε", F_SMALL, TEXT)
        vk.text_c(d, (640, 580), "两者越接近：攻击越强、分析越紧", F_BODY, GREEN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 6, [("成员推断", "训练样本的损失更低"), ("记忆", "少见、异常的样本最容易被记住"),
                      ("审计", "DP 让攻击接近随机；攻击给出 ε 下界")], seg, "下一期：从人口普查到大模型")


RENDER = {"title": sc_title, "attack": sc_attack, "canary": sc_canary, "plain": sc_plain, "dp": sc_dp, "audit": sc_audit,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "dp-audit")
