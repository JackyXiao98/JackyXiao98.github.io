"""Differential Privacy 06 — DP-SGD：用差分隐私训练神经网络.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python dp06_dpsgd.py <out_dir>

Real computations: a 64-128-10 MLP on sklearn's 8x8 digits (1,200 train / 597 test) trained with
plain SGD and with DP-SGD at four noise levels (dpx.train_mlp, cached in DP_CACHE); eps from the
RDP accountant; per-example gradient norms and the size of the added noise.
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
DELTA, CLIP, BATCH = 1e-5, 1.0, 64

PLAIN = dpx.cached(CACHE, dpx.train_mlp, dp=False, epochs=30)
SIGMAS = [0.6, 1.0, 2.0, 4.0]
RUNS = [dpx.cached(CACHE, dpx.train_mlp, dp=True, sigma=s, epochs=30) for s in SIGMAS]
EPSS = [dpx.dpsgd_eps(r["q"], s, r["steps"], DELTA) for r, s in zip(RUNS, SIGMAS)]
ACC_PLAIN = PLAIN["hist"][-1][2]
ACCS = [r["hist"][-1][2] for r in RUNS]
print("plain", ACC_PLAIN, [(s, round(e, 2), round(a, 4)) for s, e, a in zip(SIGMAS, EPSS, ACCS)])
assert ACC_PLAIN > 0.95 and all(np.diff(EPSS) < 0) and all(np.diff(ACCS) < 0)
assert ACCS[2] > 0.9 and 3 < EPSS[2] < 4.5 and ACCS[3] < 0.85 and EPSS[3] < 2

# per-example gradient norms at initialisation and the size of the noise
Xtr, ytr, Xte, yte = dpx.digits()
N_TRAIN = len(Xtr)


def grad_norms(params):
    import jax
    g = jax.vmap(jax.grad(lambda p, x, y: dpx.per_example_loss(p, x[None], y[None])[0]), in_axes=(None, 0, 0))(
        params, Xtr, ytr)
    return np.sqrt(sum((np.asarray(a).reshape(len(Xtr), -1) ** 2).sum(1) for a in jax.tree_util.tree_leaves(g)))


import jax  # noqa: E402

P0 = jax.tree_util.tree_map(np.asarray, dpx._init(jax.random.PRNGKey(0)))
NORMS0 = grad_norms(P0)
FRAC_CLIP = float((NORMS0 > CLIP).mean())
N_PARAMS = sum(a.size for a in jax.tree_util.tree_leaves(P0))
NOISE_NORM = 1.0 * CLIP * np.sqrt(N_PARAMS) / BATCH  # sigma = 1
print("median norm", np.median(NORMS0), "clipped", FRAC_CLIP, "params", N_PARAMS, "noise", NOISE_NORM)
assert N_PARAMS == 9610 and FRAC_CLIP > 0.5 and 1.4 < NOISE_NORM < 1.6

SCENES = [
    ("title", [
        "前面几期我们准备好了所有工具。这一期把它们组合起来，用差分隐私的随机梯度下降，真正训练一个神经网络。",
    ]),
    ("algo", [
        "普通的训练，每一步把一批样本的梯度平均，然后更新参数。问题在于，一个特别的样本，可能产生一个很大的梯度，让模型记住它。",
        "差分隐私的随机梯度下降做了两处修改。第一，单独计算每个样本的梯度，把长度超过阈值的梯度缩短到阈值，这叫逐样本裁剪，保证每个人的影响有上限。",
        "第二，把裁剪后的梯度加起来，再加上高斯噪声，噪声的标准差等于噪声系数乘以裁剪阈值。最后用第五期的方法，计算整个训练过程的艾普西隆。",
    ]),
    ("clip", [
        f"我们在手写数字数据集上训练一个小网络，一千二百个训练样本，大约一万个参数。",
        f"刚开始训练时，每个样本梯度长度的中位数约为{dec_zh(np.median(NORMS0), 1)}，裁剪阈值取一，{'所有样本的梯度都' if FRAC_CLIP == 1 else pct_zh(FRAC_CLIP) + '的样本梯度'}被缩短了。",
        f"噪声系数取一时，每一步加入的噪声向量长度约为{dec_zh(NOISE_NORM)}，比平均梯度本身还要长。但噪声在不同步之间互相独立，多步累积以后，正确的方向依然会显现出来。",
    ]),
    ("results", [
        f"我们用四种不同的噪声系数各训练三十轮。不加隐私保护时，测试准确率是{pct_zh(ACC_PLAIN, 1)}。",
        f"艾普西隆约为{dec_zh(EPSS[1], 1)}时，准确率是{pct_zh(ACCS[1], 1)}；艾普西隆约为{dec_zh(EPSS[2], 1)}时，是{pct_zh(ACCS[2], 1)}；艾普西隆降到{dec_zh(EPSS[3], 1)}时，准确率下降到{pct_zh(ACCS[3], 1)}。",
        "隐私越强，准确率越低，这就是差分隐私训练的代价。实践中，增加数据量、使用更大的批次和预训练模型，都可以缩小这个差距。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，差分隐私的随机梯度下降，对每个样本的梯度单独裁剪，限制每个人的影响。",
        "第二，在梯度和上加入高斯噪声，再用瑞尼差分隐私记账，得到整个训练的艾普西隆。",
        "第三，隐私和准确率之间存在权衡。下一期，我们站在攻击者的角度，检验模型到底记住了什么。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 6, "DP-SGD", "用差分隐私训练神经网络", SERIES)


def sc_algo(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "DP-SGD（Abadi et al., 2016）")
    steps = [("① 随机抽一批样本（概率 q）", TEXT, 0), ("② 逐样本梯度  g_i", TEXT, 0),
             ("③ 裁剪：g_i ← g_i · min(1, C / ‖g_i‖)", AMBER, 1),
             ("④ 求和 + 噪声  N(0, σ²C² I)", RED, 2), ("⑤ 除以批大小，更新参数", TEXT, 2), ("⑥ RDP 记账 → (ε, δ)", GREEN, 2)]
    for i, (t, col, s) in enumerate(steps):
        on = seg >= s
        d.rounded_rectangle([140, 140 + i * 72, 600, 196 + i * 72], 10, outline=col if on else (30, 41, 59), width=2,
                            fill=(12, 19, 34))
        d.text((165, 168 + i * 72), t, font=F_SMALL, fill=col if on else MUTED, anchor="lm")
    if seg >= 2:
        tv.fpaste(c, r"$\tilde g=\frac{1}{B}\left(\sum_{i}\mathrm{clip}_C(g_i)+\mathcal{N}(0,\sigma^2C^2I)\right)$", 960, 330, 26)


def sc_clip(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, f"手写数字：{N_TRAIN:,} 个训练样本，{N_PARAMS:,} 个参数")
    if seg >= 1:
        h, e = np.histogram(NORMS0, bins=30, range=(0, NORMS0.max()))
        x0, x1, y0, y1 = 140, 640, 180, 520
        for j, v in enumerate(h):
            xa = x0 + j * (x1 - x0) / len(h)
            col = AMBER if e[j] >= CLIP else CYAN
            d.rectangle([xa, y1 - v / h.max() * (y1 - y0), xa + (x1 - x0) / len(h) - 2, y1], fill=col)
        xc = x0 + CLIP / e[-1] * (x1 - x0)
        d.line([(xc, y0 - 10), (xc, y1)], fill=RED, width=2)
        d.text((xc + 6, y0 - 10), "C = 1", font=F_SMALL, fill=RED)
        d.text((x0, y1 + 12), "0", font=vk.font(18, mono=True), fill=MUTED)
        d.text((x1, y1 + 12), f"{e[-1]:.1f}", font=vk.font(18, mono=True), fill=MUTED, anchor="ra")
        vk.text_c(d, ((x0 + x1) / 2, y1 + 50), f"初始化时的逐样本梯度长度（{FRAC_CLIP:.0%} 被裁剪）", F_SMALL, MUTED)
    if seg >= 2:
        tv.fpaste(c, r"$\|\mathcal{N}(0,\sigma^2C^2I)\|/B\approx\sigma C\sqrt{p}/B$", 950, 260, 26)
        tv.fpaste(c, r"$=1\times1\times\sqrt{9610}/64\approx1.53$", 950, 330, 26, "#fbbf24")
        vk.text_c(d, (950, 420), "噪声比平均梯度还长", F_BODY, RED)
        vk.text_c(d, (950, 470), "但各步独立，多步累积后方向正确", F_SMALL, TEXT)


def sc_results(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "测试准确率 vs ε（30 轮，δ = 1e-5）")
    labels = ["不加隐私"] + [f"σ = {s}\nε = {e:.1f}" for s, e in zip(SIGMAS, EPSS)]
    vals = [ACC_PLAIN] + ACCS
    k = 1 if seg == 0 else 5
    x0, x1, y0, y1 = 180, 1100, 160, 520
    gw = (x1 - x0) / 5
    d.line([(x0, y1), (x1, y1)], fill=(71, 85, 105), width=2)
    for i in range(k):
        v = vals[i]
        bx = x0 + i * gw + gw * 0.2
        top = y1 - (v - 0.5) / 0.5 * (y1 - y0)
        col = MUTED if i == 0 else [VIOLET, CYAN, GREEN, RED][i - 1]
        d.rectangle([bx, top, bx + gw * 0.6, y1], fill=col)
        d.text((bx + gw * 0.3, top - 16), f"{v:.1%}", font=vk.font(22, mono=True), fill=col, anchor="mm")
        for j, line in enumerate(labels[i].split("\n")):
            d.text((bx + gw * 0.3, y1 + 22 + j * 26), line, font=vk.font(20), fill=MUTED, anchor="mm")
    d.text((x0 - 10, y1), "50%", font=vk.font(18, mono=True), fill=MUTED, anchor="rm")
    if seg >= 2:
        vk.text_c(d, (640, 650), "更多数据 · 更大批次 · 预训练模型 → 缩小差距", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("逐样本裁剪", "每个人对梯度的影响有上限"), ("加噪 + 记账", "高斯噪声，RDP 计算总 ε"),
                      ("权衡", "ε 越小，准确率越低")], seg, "下一期：成员推断攻击与隐私审计")


RENDER = {"title": sc_title, "algo": sc_algo, "clip": sc_clip, "results": sc_results, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "dp-sgd")
