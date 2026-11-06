"""State Space Models 08 — 混合架构：注意力与状态空间模型.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] SSM_CACHE=<dir> python ssm08_hybrid.py <out_dir>

Real computations: (1) state space duality (numpy): a recurrence with a scalar decay a_t per step,
h_t = a_t h_{t-1} + k_t v_t^T, y_t = q_t^T h_t, versus the matrix form y = (L o Q K^T) V with the decay mask
L_ij = a_{j+1} ... a_i (j <= i); T = 32, d = 8, random data; largest difference. (2) ssmx.exp_lm / exp_lm_linpos
(JAX): 2-layer width-64 models on Tiny Shakespeare, 2,000 steps, context 128 - softmax attention, linear attention
(with learned positions), fixed-parameter SSM, selective SSM, and a hybrid (selective SSM layer + attention layer).
(3) ssmx.exp_recall: multi-query associative recall (episode 07), hybrid vs selective SSM with state size 16.
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

# state space duality: scalar-decay recurrence == masked (decayed) attention
r = np.random.default_rng(1)
T, D = 32, 8
q, k, v = r.normal(size=(3, T, D))
a = r.uniform(0.7, 1.0, T)
h = np.zeros((D, D))
y_rec = np.zeros((T, D))
for t in range(T):
    h = a[t] * h + np.outer(k[t], v[t])
    y_rec[t] = q[t] @ h
cum = np.cumsum(np.log(a))
L = np.tril(np.exp(cum[:, None] - cum[None, :]))
y_mat = (L * (q @ k.T)) @ v
DIFF = float(np.abs(y_rec - y_mat).max())

LM = ssmx.cached(CACHE, "lm", ssmx.exp_lm)
LP = ssmx.cached(CACHE, "lm_linpos", ssmx.exp_lm_linpos)
RC = ssmx.cached(CACHE, "recall", ssmx.exp_recall)
VAL = {"attn": LM["attn"]["final"]["val"], "linattn": LP["final"]["val"], "lti": LM["lti"]["final"]["val"],
       "mamba": LM["mamba"]["final"]["val"], "hybrid": LM["hybrid"]["final"]["val"]}
RA = {"mamba": RC[("mamba", 16)]["final"]["tok_acc"], "hybrid": RC["hybrid"]["final"]["tok_acc"], "attn": RC["attn"]["final"]["tok_acc"]}
print(DIFF, {k_: round(x, 3) for k_, x in VAL.items()}, RA)
assert 1e-15 < DIFF < 2e-14
assert [round(VAL[k_], 3) for k_ in ("attn", "linattn", "lti", "mamba", "hybrid")] == [1.771, 1.895, 1.597, 1.574, 1.614]
assert RA["hybrid"] > 0.99 and RA["mamba"] < 0.9

SCENES = [
    ("title", [
        "前面几期，线性注意力和状态空间模型是分开讲的。这一期看它们怎样统一起来，以及今天越来越常见的做法：把注意力和状态空间模型混在一个模型里。",
    ]),
    ("ssd", [
        "二零二四年的曼巴第二代提出了状态空间对偶：如果每一步的状态衰减只是一个数，那么选择性状态空间模型的递推，就等价于一种带衰减掩码的线性注意力。递推形式适合生成，矩阵形式可以用矩阵乘法高效训练。",
        "我们用随机数据验证：一步一步递推，和先算查询与键的乘积、乘上衰减掩码、再乘以值，两种算法的结果最大只差约十的负十四次方。",
    ]),
    ("family", [
        "从这个角度看，很多模型都是同一个家族：线性注意力是不衰减的情形；保留网络用固定的衰减；门控线性注意力和曼巴让衰减随输入变化；增量规则在写入前先擦掉旧的内容。它们的区别主要在于状态怎样衰减、怎样写入。",
    ]),
    ("lm", [
        f"在我们的莎士比亚小实验里，五个两层的小模型训练两千步。注意力的验证损失是{dec_zh(VAL['attn'], 2)}，线性注意力{dec_zh(VAL['linattn'], 2)}，固定参数的状态空间模型{dec_zh(VAL['lti'], 2)}，选择性的{dec_zh(VAL['mamba'], 2)}，一层状态空间加一层注意力的混合模型是{dec_zh(VAL['hybrid'], 2)}。",
        "在这么小的字符级模型上，状态空间模型反而比注意力好，它的局部归纳偏置很适合这类任务。这不代表大模型上的结论，但说明状态空间层在建模局部结构上很有效率。",
    ]),
    ("recall", [
        f"回忆任务则正好相反。上一期，状态大小十六的选择性模型答对{pct_zh(RA['mamba'])}；只要把其中一层换成注意力，混合模型就答对了{pct_zh(RA['hybrid'])}。一层注意力就补上了精确回忆的能力。",
    ]),
    ("jamba", [
        "这正是混合架构的思路：大部分层用状态空间模型，便宜地处理长序列和局部结构；少量层用注意力，负责精确地取回信息。二零二四年的佳巴模型每八层里只有一层注意力，其余是曼巴层，上下文达到二十五万六千个词元，缓存比同规模的变换器小得多。之后又有不少模型采用了类似的混合设计。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，状态衰减是一个数时，选择性状态空间模型等价于带衰减掩码的线性注意力。",
        "第二，线性注意力、保留网络、门控线性注意力和曼巴，是同一个家族的不同成员。",
        "第三，混合架构用少量注意力层补上精确回忆，其余用状态空间层。下一期，我们看整个系列的权衡和前沿。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 8, "混合架构", "注意力与状态空间模型", SERIES)


def sc_ssd(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "状态空间对偶（Mamba-2, Dao & Gu, 2024）")
    tv.fpaste(c, r"$h_t=a_t\,h_{t-1}+k_t v_t^\top,\qquad y_t=q_t^\top h_t$", 640, 210, 34)
    vk.text_c(d, (640, 275), "=", F_BODY, MUTED)
    tv.fpaste(c, r"$Y=\left(L\circ QK^\top\right)V,\qquad L_{ij}=a_{j+1}\cdots a_i\ \ (j\leq i)$", 640, 340, 34)
    d.text((240, 420), "递推：适合生成", font=F_SMALL, fill=GREEN)
    d.text((760, 420), "矩阵乘法：适合训练", font=F_SMALL, fill=CYAN)
    if seg >= 1:
        d.text((240, 500), "T = 32，d = 8，随机数据：最大差别", font=F_BODY, fill=TEXT)
        d.text((880, 500), f"{DIFF:.0e}", font=vk.font(30, mono=True), fill=GREEN)


def sc_family(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "同一个家族：状态怎样衰减、怎样写入")
    rows = [("线性注意力", "不衰减", r"$S_t=S_{t-1}+k_tv_t^\top$", MUTED), ("RetNet", "固定衰减", r"$S_t=\gamma S_{t-1}+k_tv_t^\top$", VIOLET),
            ("GLA / Mamba-2", "随输入变化的衰减", r"$S_t=a(x_t)\,S_{t-1}+k_tv_t^\top$", CYAN),
            ("DeltaNet", "先擦后写", r"$S_t=S_{t-1}-\beta_t k_t\,(S_{t-1}^\top k_t-v_t)^\top$", GREEN)]
    for i, (n, dec, f, col) in enumerate(rows):
        y = 200 + i * 90
        d.rounded_rectangle([140, y, 1140, y + 70], 12, outline=col, width=2, fill=(12, 19, 34))
        d.text((170, y + 35), n, font=F_BODY, fill=col, anchor="lm")
        d.text((440, y + 35), dec, font=F_SMALL, fill=TEXT, anchor="lm")
        tv.fpaste(c, f, 900, y + 35, 22 if "beta" in f else 26)


def sc_lm(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "莎士比亚：验证损失（2 层，2,000 步）")
    ks = ["attn", "linattn", "lti", "mamba", "hybrid"]
    labs = ["注意力", "线性注意力", "固定 SSM", "选择性 SSM", "混合"]
    tv.bar_chart(d, (160, 220, 960, 560), labs, [VAL[k_] for k_ in ks], [CYAN, AMBER, VIOLET, GREEN, RED], fmt="{:.3f}", vmax=2.1)
    if seg >= 1:
        vk.text_c(d, (640, 620), "小的字符级模型：局部结构为主，不代表大模型上的结论", F_SMALL, AMBER)


def sc_recall(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "联想回忆：一层注意力就够了")
    tv.bar_chart(d, (260, 220, 1020, 560), ["选择性 SSM（N = 16）", "混合（SSM + 注意力）", "注意力"],
                 [RA["mamba"] * 100, RA["hybrid"] * 100, RA["attn"] * 100], [GREEN, RED, CYAN], fmt="{:.1f}%", vmax=105)
    d.text((260, 180), "查询答对的比例", font=F_SMALL, fill=MUTED)


def sc_jamba(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "混合架构：Jamba（Lieber et al., 2024）")
    for i in range(8):
        col = CYAN if i == 4 else GREEN
        d.rounded_rectangle([200 + i * 112, 230, 300 + i * 112, 300], 10, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (250 + i * 112, 265), "注意力" if i == 4 else "Mamba", F_SMALL, col)
    d.text((200, 320), "每 8 层：1 层注意力 + 7 层 Mamba（并加入 MoE）", font=F_SMALL, fill=MUTED)
    for i, (t, col) in enumerate([("SSM 层：便宜地处理长序列与局部结构", GREEN), ("注意力层：精确取回", CYAN),
                                  ("上下文 256K，缓存远小于同规模 Transformer", AMBER)]):
        vk.text_c(d, (640, 410 + i * 65), t, F_BODY, col)


def sc_summary(c, u, seg, su):
    tv.summary(c, 8, [("对偶", "标量衰减 SSM = 带掩码线性注意力"), ("家族", "区别在衰减与写入"),
                      ("混合", "少量注意力补上回忆")], seg, "下一期：权衡与前沿")


RENDER = {"title": sc_title, "ssd": sc_ssd, "family": sc_family, "lm": sc_lm, "recall": sc_recall, "jamba": sc_jamba, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "ssm-hybrid")
