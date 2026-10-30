"""Transformer 06 — 从零训练一个小 Transformer：两位数加法.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python tf06_train.py <out_dir>

Real computations: the 2-layer decoder-only Transformer of tfm.py trained (JAX) on 80% of all
2-digit additions; train/test loss and test exact-match accuracy over training; greedy generations
of held-out problems at several checkpoints.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import tfm
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
CACHE = os.environ.get("TF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "tfcache"))

PARAMS, HIST, SNAPS = tfm.cached_train(CACHE, steps=600, log_every=20, snapshots=(0, 60, 120, 200, 300, 600))
TR, TE, TRP, TEP = tfm.dataset()
NP = tfm.n_params(PARAMS)
H = np.array(HIST)  # step, train_loss, test_loss, acc
STEPS, TRL, TEL, ACC = H[:, 0], H[:, 1], H[:, 2], H[:, 3]
acc_at = lambda s: float(ACC[STEPS == s][0])
FIRST100 = int(STEPS[np.argmax(ACC >= 1.0)])
assert ACC[-1] == 1.0 and FIRST100 <= 300
assert acc_at(60) < 0.05 and 0.2 < acc_at(120) < 0.35 and acc_at(200) > 0.99

PROBS = [(35, 77), (89, 25), (16, 34), (4, 85), (47, 53)]
assert all(p in TEP for p in PROBS)  # held-out problems
SHOW = [0, 60, 120, 200]
PROMPTS = np.array([tfm.encode(a, b)[:tfm.ANS] for a, b in PROBS])
GEN = {s: ["".join(tfm.VOCAB[i] for i in row[tfm.ANS:]) for row in tfm.generate(SNAPS[s], PROMPTS)] for s in SHOW}
TRUTH = [f"{a + b:03d}" for a, b in PROBS]
N_OK = {s: sum(g == t for g, t in zip(GEN[s], TRUTH)) for s in SHOW}
print(NP, FIRST100, {s: acc_at(s) for s in SHOW}, GEN, N_OK)
assert N_OK[0] == 0 and N_OK[200] == 5 and N_OK[120] < 5
assert all(g[0] in "01" for g in GEN[60])
GAP = float(np.abs(TEL - TRL)[STEPS >= 200].max())
assert GAP < 0.05
assert NP == 100736 and len(TR) == 8000 and len(TE) == 2000

SCENES = [
    ("title", [
        "前面几期我们讲了变换器的各个部件，这一期，我们真正从零训练一个小变换器，看它怎样学会两位数加法。",
    ]),
    ("setup", [
        f"任务是这样的，输入三十五加七十七等于，让模型逐位写出三位数的答案一百一十二。两位数加法一共有一万道题，我们随机拿出{int_zh(len(TE))}道作为测试题，训练时从来不给模型看。",
        f"模型是只有解码器的变换器，两层，每层四个注意力头，向量维度六十四，一共大约十万个参数。损失函数是交叉熵，也就是让模型给正确的下一个数字尽量高的概率。",
    ]),
    ("samples", [
        "我们挑五道测试题，看模型在训练过程中的答案。",
        "第零步，模型还没有训练，输出的是一串乱码，有的甚至不是数字。",
        f"第六十步，它已经知道答案是三位数，开头不是零就是一，五道题对了{int_zh(N_OK[60])}道。第一百二十步，对了{int_zh(N_OK[120])}道。",
        "到了第二百步，五道题全部正确。",
    ]),
    ("curve", [
        f"再看全部{int_zh(len(TE))}道测试题的准确率。前六十步几乎为零，到第一百二十步是{pct_zh(acc_at(120))}，然后迅速上升，第二百步达到{pct_zh(acc_at(200), 1)}，第{int_zh(FIRST100)}步达到百分之百。",
        "准确率不是平滑上升的，而是在很短的时间里突然跳上去，这种现象在小模型学习算法类任务时很常见。",
    ]),
    ("generalize", [
        "最重要的是，这些测试题模型从来没有见过。",
        "图中青色是训练集的损失，紫色是测试集的损失，两条曲线几乎重合，说明模型并不是在背答案，而是学会了加法的规律。上一期我们也看到，它的注意力头确实在看对应的数位。",
    ]),
    ("summary", [
        "总结一下。",
        f"第一，一个大约十万参数的两层变换器，用六百步就学会了两位数加法。",
        "第二，训练的目标只是预测下一个词元，能力却是在训练中突然出现的。",
        "第三，测试损失和训练损失一致，模型学到的是规律，而不是记忆。",
        "下一期，我们看模型怎样高效地生成文字，也就是键值缓存。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 6, "从零训练一个 Transformer", "两位数加法 · 10 万参数 · 600 步")


def sc_setup(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "任务与模型")
    tv.tokens_row(d, 240, 150, list("35+77="), CYAN, font=vk.font(30, mono=True))
    x = tv.tokens_row(d, 240 + 6 * 60 + 30, 150, list("112"), VIOLET, font=vk.font(30, mono=True))
    d.text((240, 215), "输入（6 个词元）", font=F_SMALL, fill=CYAN)
    d.text((630, 215), "逐位生成答案", font=F_SMALL, fill=VIOLET)
    W = 600
    d.rectangle([240, 280, 240 + W * 0.8, 320], fill=(14, 60, 72))
    d.rectangle([240 + W * 0.8, 280, 240 + W, 320], fill=(60, 40, 90))
    vk.text_c(d, (240 + W * 0.4, 300), f"训练 {len(TR):,} 道", F_SMALL, TEXT)
    vk.text_c(d, (240 + W * 0.9, 300), f"测试 {len(TE):,}", F_SMALL, TEXT)
    d.text((240 + W + 20, 290), "（00+00 … 99+99，共 10,000 道）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        rows = [("架构", "Decoder-only，Pre-LN"), ("层数 / 头数", "2 层 · 4 头"), ("维度", "d = 64，FFN 256"),
                ("参数量", f"{NP:,}"), ("训练", "AdamW，batch 256，600 步"), ("损失", "答案 3 位的交叉熵")]
        for i, (k, v) in enumerate(rows):
            y = 370 + i * 46
            d.text((160, y), k, font=F_SMALL, fill=MUTED)
            d.text((340, y), v, font=F_BODY, fill=TEXT)
        tv.fpaste(c, r"$\mathcal{L}=-\frac{1}{3}\sum_{t=6}^{8}\log p_\theta(x_t\mid x_{<t})$", 1010, 480, 26)


def sc_samples(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "训练过程中的答案（测试题，贪心解码）")
    cols = [0] + [s for s, k in zip(SHOW[1:], [2, 2, 3]) if seg >= k]
    fm = vk.font(28, mono=True)
    d.text((130, 140), "题目", font=F_SMALL, fill=MUTED)
    d.text((330, 140), "正确答案", font=F_SMALL, fill=MUTED)
    for j, s in enumerate(SHOW):
        if s in cols and seg >= 1:
            d.text((560 + j * 170, 140), f"第 {s} 步", font=F_SMALL, fill=MUTED)
    for i, (a, b) in enumerate(PROBS):
        y = 200 + i * 70
        d.text((130, y), f"{a:02d}+{b:02d}=", font=fm, fill=TEXT)
        d.text((350, y), TRUTH[i], font=fm, fill=GREEN)
        for j, s in enumerate(SHOW):
            if s in cols and seg >= 1:
                g = GEN[s][i]
                d.text((570 + j * 170, y), g, font=fm, fill=GREEN if g == TRUTH[i] else RED)
    if seg >= 1:
        for j, s in enumerate(SHOW):
            if s in cols:
                d.text((570 + j * 170, 560), f"{N_OK[s]}/5", font=vk.font(24, mono=True), fill=CYAN)


def sc_curve(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, f"测试准确率（{len(TE):,} 道未见过的题）")
    n = len(STEPS) if seg >= 1 else max(2, int(len(STEPS) * vk.ease(u * 1.6 if u < .62 else 1)))
    tv.line_chart(d, (150, 150, 1130, 560), STEPS, [("完全正确率", ACC * 100, CYAN)], ylim=(0, 100), upto=n,
                  xlabel="训练步数", ylabel="准确率 %", xticks=[(s, str(s)) for s in range(0, 601, 100)])
    for s in [120, 200]:
        i = int(np.where(STEPS == s)[0][0])
        if i < n:
            x = 150 + s / 600 * 980
            y = 560 - ACC[i] * 410
            d.text((x + 14, y + 10), f"第 {s} 步：{ACC[i]:.1%}", font=F_SMALL, fill=AMBER)


def sc_generalize(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "学会了规律，而不是背答案")
    tv.line_chart(d, (150, 150, 1130, 560), STEPS, [("训练集损失", TRL, CYAN), ("测试集损失（未见过）", TEL, VIOLET)],
                  ylog=True, xlabel="训练步数", ylabel="交叉熵（对数坐标）", xticks=[(s, str(s)) for s in range(0, 601, 100)],
                  legend_xy=(820, 170))
    if seg >= 1:
        vk.text_c(d, (640, 630), f"第 200 步之后两条曲线的最大差距 {GAP:.3f}", F_SMALL, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("从零训练", f"{NP:,} 参数，600 步学会加法"), ("突然出现", "准确率在很短时间内跳升"),
                      ("泛化", "测试损失 ≈ 训练损失：学会规律")], seg, "下一期：高效生成 —— KV 缓存")


RENDER = {"title": sc_title, "setup": sc_setup, "samples": sc_samples, "curve": sc_curve, "generalize": sc_generalize,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "training")
