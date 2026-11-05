"""State Space Models 05 — 选择性：Mamba 的关键一步.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] SSM_CACHE=<dir> python ssm05_select.py <out_dir>

Real computations (ssmx.exp_selcopy, JAX): selective copying - 64 positions of blanks with 4 data tokens (8 kinds)
at random positions, then 4 "go" markers; the model must output the data tokens in order. Three 2-layer models of
width 64, trained 1,500 steps (batch 32): softmax attention, the fixed-parameter (LTI) state space layer, and the
selective (Mamba-like) layer whose step size, B and C are computed from the input. Accuracy on 512 test sequences
(per output token, and all 4 correct), every 150 steps.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import ssmx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "State Space Models"
CACHE = os.environ.get("SSM_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "ssmcache"))

R = ssmx.cached(CACHE, "selcopy", ssmx.exp_selcopy)
F = {k: R[k]["final"] for k in R}
H = {k: R[k]["hist"] for k in R}
ex_x, ex_y, ex_w = ssmx.selcopy_batch()(np.random.default_rng(5), 1)
print({k: (round(v["tok_acc"], 3), round(v["seq_acc"], 3)) for k, v in F.items()})
assert round(F["mamba"]["seq_acc"], 3) == 0.996 and round(F["lti"]["seq_acc"], 3) == 0.564 and round(F["attn"]["seq_acc"], 3) == 0.982
assert round(F["lti"]["tok_acc"], 2) == 0.86 and round(H["mamba"][3]["seq_acc"], 2) == 0.84 and round(H["attn"][3]["seq_acc"], 2) == 0.41

SCENES = [
    ("title", [
        "前几期的状态空间模型有一个共同点：参数不随输入变化。不管读到的是重要的词还是无关的填充，状态都按同样的方式更新。二零二三年的曼巴模型改变了这一点，让模型自己决定记住什么、忘掉什么。这一期看这个选择性机制。",
    ]),
    ("task", [
        "我们用一个叫选择性复制的任务来检验。一段六十四个位置的序列里，大部分是空白，只有四个位置随机放着要记住的词元。读完以后，模型要按顺序把这四个词元说出来。",
        "难点在于，要记住的词元出现在随机的位置。一个不随输入变化的模型，对每个位置的处理都一样，很难把有用的内容和空白区分开。",
    ]),
    ("mechanism", [
        "选择性的做法是：让步长、输入矩阵和输出矩阵都由当前的输入算出来。步长大，就让新输入大量写入、旧状态快速遗忘；步长小，就几乎跳过这个词元，保持原来的记忆。",
        "这样，遇到空白时模型可以让步长很小，直接跳过；遇到要记住的词元时再写入。代价是参数随时间变化，第三期的卷积技巧就不能用了，这也是下一期并行扫描要解决的问题。",
    ]),
    ("result", [
        f"我们训练三个两层的小模型，各一千五百步。固定参数的状态空间模型，每个词元的正确率是{pct_zh(F['lti']['tok_acc'])}，但四个全对只有{pct_zh(F['lti']['seq_acc'])}。",
        f"选择性的模型四个全对达到{pct_zh(F['mamba']['seq_acc'], 1)}，而且学得很快：四百五十步时就有{pct_zh(H['mamba'][3]['seq_acc'])}，同一时间注意力模型只有{pct_zh(H['attn'][3]['seq_acc'])}。注意力最后也达到了{pct_zh(F['attn']['seq_acc'], 1)}。",
    ]),
    ("real", [
        "曼巴论文报告，在选择性复制和类似的任务上，选择性的状态空间模型远超不随输入变化的版本，并且在语言建模上，它是第一个在多个规模上追平同等大小变换器的线性时间模型，推理吞吐量还高出好几倍。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，固定参数的状态空间模型对每个位置一视同仁，难以挑选要记住的内容。",
        "第二，曼巴让步长和输入输出矩阵由输入决定，可以选择写入或跳过。",
        "第三，在我们的实验里，选择性模型在选择性复制上接近全对，固定参数的模型只有一半多。下一期，我们看怎样并行训练这种模型。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 5, "选择性", "Mamba 的关键一步", SERIES)


def sc_task(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "选择性复制：记住随机位置上的 4 个词元")
    x = ex_x[0]
    cell = 16
    for i, t in enumerate(x):
        xx = 80 + i * cell
        col = (30, 41, 59) if t == 0 else (AMBER if t == 9 else GREEN)
        d.rectangle([xx, 260, xx + cell - 3, 300], fill=col)
        if t not in (0, 9):
            d.text((xx + 6, 240), str(t), font=vk.font(16, mono=True), fill=GREEN, anchor="mm")
    d.text((80, 320), "空白 · 要记住的词元（绿）· 输出位置（橙）", font=F_SMALL, fill=MUTED)
    d.text((80, 380), "目标输出：" + " ".join(str(v) for v in ex_y[0][ex_w[0] > 0]), font=F_BODY, fill=AMBER)
    if seg >= 1:
        vk.text_c(d, (640, 520), "位置随机：参数固定的模型对每个位置一视同仁", F_BODY, RED)


def sc_mechanism(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "选择性：参数由输入决定")
    tv.fpaste(c, r"$\Delta_t=\mathrm{softplus}(W_\Delta x_t),\quad B_t=W_B x_t,\quad C_t=W_C x_t$", 640, 200, 32)
    tv.fpaste(c, r"$h_t=e^{\Delta_t A}\,h_{t-1}+\Delta_t B_t\,x_t,\qquad y_t=C_t\,h_t$", 640, 290, 34)
    if seg >= 0:
        d.text((180, 380), "Δ 大：写入新内容、忘掉旧状态", font=F_SMALL, fill=GREEN)
        d.text((180, 420), "Δ 小：几乎跳过这个词元", font=F_SMALL, fill=CYAN)
    if seg >= 1:
        d.text((180, 500), "代价：参数随时间变化 → 不能再用卷积 → 需要并行扫描", font=F_SMALL, fill=AMBER)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "四个全对的比例（512 条测试序列）")
    cols = {"attn": CYAN, "lti": RED, "mamba": GREEN}
    names = {"attn": "注意力", "lti": "固定参数 SSM", "mamba": "选择性 SSM"}
    keys = ["lti"] if seg == 0 else ["attn", "lti", "mamba"]
    xs = [h["step"] for h in H["lti"]]
    ser = [(names[k], [h["seq_acc"] * 100 for h in H[k]], cols[k]) for k in keys]
    tv.line_chart(d, (140, 200, 800, 580), xs, ser, ylim=(0, 102), xticks=[(0, "0"), (750, "750"), (1500, "1500")], xlabel="训练步数", legend_xy=(860, 220))
    d.text((140, 165), "全对（%）", font=F_SMALL, fill=MUTED)
    for i, k in enumerate(keys):
        d.text((860, 340 + i * 45), f"{names[k]}：{F[k]['seq_acc']:.1%}", font=F_SMALL, fill=cols[k])


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "Mamba（Gu & Dao, 2023）")
    for i, (t, col) in enumerate([("选择性复制、联想回忆：远超时不变的 SSM", GREEN), ("语言建模：多个规模上追平同等大小的 Transformer", CYAN),
                                  ("推理吞吐量高出数倍（无 KV 缓存）", AMBER)]):
        vk.text_c(d, (640, 230 + i * 80), t, F_BODY, col)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("固定参数", "一视同仁，难以挑选"), ("选择性", "Δ、B、C 由输入决定"),
                      ("实验", "选择性复制 99.6% vs 56%")], seg, "下一期：并行扫描")


RENDER = {"title": sc_title, "task": sc_task, "mechanism": sc_mechanism, "result": sc_result, "real": sc_real, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "ssm-select")
