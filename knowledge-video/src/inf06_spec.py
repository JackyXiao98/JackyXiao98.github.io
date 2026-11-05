"""LLM Inference 06 — 投机解码：先猜后验.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] INF_CACHE=<dir> python inf06_spec.py <out_dir>

Real computations (infx.exp_spec, numpy): target = the trained 2-layer toy model (106,496 parameters), draft = a
1-layer d = 32 model (12,288 parameters) trained on the same data. Speculative sampling (draft proposes gamma tokens,
target scores all of them in one forward pass, token i accepted with probability min(1, p/q), on rejection sample
from max(p - q, 0) normalised, on full acceptance one bonus token from the target):
(1) exactness of one step - the output distribution min(p, q) + (1 - sum min(p, q)) * norm(max(p - q, 0)) versus p,
    600 positions;
(2) 40 prompts x 40 new tokens for gamma = 1, 2, 4, 8: tokens produced per target call and draft acceptance rate,
    compared with (1 - a^(gamma+1)) / (1 - a);
(3) 3,000 two-token continuations of one prompt sampled plainly (twice) and speculatively; total-variation distance
    to the exact two-token distribution.
Speed-up estimate: tokens per target call / (1 + gamma x draft cost), with draft cost = parameter ratio (an
assumption). Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import infx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "LLM Inference"
CACHE = os.environ.get("INF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "infcache"))

SP = infx.cached(CACHE, "spec", infx.exp_spec, cache_dir=CACHE)
G = [1, 2, 4, 8]
PC = {g: SP["gen"][g]["per_call"] for g in G}
A = float(np.mean([SP["gen"][g]["accept"] for g in G]))
THEORY = {g: (1 - A ** (g + 1)) / (1 - A) for g in G}
COST = SP["draft_params"] / SP["target_params"]
SPEED = {g: PC[g] / (1 + g * COST) for g in G}
BEST = max(G, key=lambda g: SPEED[g])
print(SP["maxdiff"], round(A, 3), PC, THEORY, round(COST, 3), SPEED, BEST, SP["tv_plain"], SP["tv_plain2"], SP["tv_spec"])
assert SP["maxdiff"] < 1e-12 and round(A, 2) == 0.63 and round(COST, 2) == 0.12
assert [round(PC[g], 1) for g in G] == [1.6, 1.9, 2.3, 2.6] and all(abs(PC[g] - THEORY[g]) < 0.16 for g in G)
assert BEST == 2 and round(SPEED[2], 1) == 1.6 and round(SPEED[8], 1) == 1.3
assert round(SP["tv_spec"], 2) == 0.08 and round(SP["tv_plain"], 2) == 0.08 and round(SP["tv_plain2"], 2) == 0.08

SCENES = [
    ("title", [
        "解码慢，是因为每一步都要把大模型的权重读一遍，却只生成一个词元。如果能让大模型一次确认好几个词元呢？这一期讲投机解码：让一个小模型先猜，大模型再一次性验证。",
    ]),
    ("idea", [
        "做法是这样的。先用一个小得多的草稿模型，快速地往后猜几个词元。然后把这几个猜测一起送进大模型，跑一次前向，大模型就同时得到了每个位置自己的概率分布。",
        "再从左到右逐个检查。如果大模型对某个猜测的概率不低于草稿模型的，就直接接受；否则按两者概率之比随机决定是否接受。一旦拒绝，就从修正后的分布里重新采一个词元，后面的猜测全部作废。",
    ]),
    ("exact", [
        "这个接受和拒绝的规则有一个重要的性质：最终生成的词元，分布和直接从大模型采样完全一样。投机解码不是近似，它不改变大模型的输出分布，只改变速度。",
        f"我们在小模型上验证。对六百个位置，按这个规则算出的输出分布，和大模型自己的分布最大只差约十的负十六次方。再从同一个开头各采样三千次，比较前两个词元的分布：直接采样和投机采样，与精确分布的距离都约是{dec_zh(SP['tv_spec'], 2)}，这正是三千个样本本身的抽样误差。",
    ]),
    ("speed", [
        f"那么能快多少？草稿模型的参数只有大模型的约九分之一，它的猜测平均有约{pct_zh(A)}被接受。每次只猜一个时，大模型每跑一次平均得到{dec_zh(PC[1], 1)}个词元；猜两个是{dec_zh(PC[2], 1)}个，猜四个是{dec_zh(PC[4], 1)}个，猜八个是{dec_zh(PC[8], 1)}个，和理论公式基本一致。",
        f"但猜得越多，草稿模型自己的开销也越大，而且后面的猜测越来越容易被拒绝。如果按参数量估计草稿模型的开销，猜两个时最划算，大约快{dec_zh(SPEED[2], 1)}倍；猜八个反而只快{dec_zh(SPEED[8], 1)}倍。",
    ]),
    ("real", [
        "真实系统里，二零二二到二零二三年谷歌和深度思维的两篇论文同时提出了这个方法，报告了两到三倍的加速。后来的工作又有很多变化：不用单独的草稿模型，而是在大模型上加几个额外的预测头，一次猜多个位置；或者用大模型自己的中间特征来做草稿。",
        "加速多少，关键在于草稿和大模型有多一致。对于代码、固定格式的文本，猜对的比例很高，加速很明显；对于开放式的创作，加速就小一些。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，投机解码让小模型先猜几个词元，大模型一次前向验证。",
        "第二，接受和拒绝的规则保证输出分布和大模型完全一样，我们的实验也验证了这一点。",
        "第三，加速取决于接受率和草稿的开销，猜的个数要折中。下一期，我们看怎样同时服务很多请求。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 6, "投机解码", "先猜后验", SERIES)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "草稿模型先猜，大模型一次验证")
    for i in range(3):
        tv.token(d, 120 + i * 90, 200, f"x{i + 1}", MUTED, w=70, font=vk.font(20, mono=True))
    for i in range(4):
        col = AMBER
        if seg >= 1:
            col = GREEN if i < 2 else (RED if i == 2 else (71, 85, 105))
        tv.token(d, 420 + i * 90, 200, f"d{i + 1}", col, w=70, font=vk.font(20, mono=True))
    d.text((420, 170), "草稿模型猜 4 个", font=F_SMALL, fill=AMBER)
    d.rounded_rectangle([120, 300, 760, 370], 12, outline=CYAN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (440, 335), "大模型：一次前向，得到每个位置的分布 p", F_SMALL, CYAN)
    if seg >= 1:
        tv.fpaste(c, r"$\mathrm{accept}\ \ \mathrm{w.p.}\ \min\!\left(1,\ p(x)/q(x)\right)$", 440, 450, 28)
        tv.fpaste(c, r"$\mathrm{reject}:\ x\sim \mathrm{norm}\left(\max(p-q,0)\right)$", 440, 520, 28)
        d.text((860, 210), "d1 接受 · d2 接受 · d3 拒绝 · d4 作废", font=F_SMALL, fill=TEXT)
        d.text((860, 270), "接受 2 个 + 重采 1 个", font=F_SMALL, fill=GREEN)


def sc_exact(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "输出分布不变")
    tv.fpaste(c, r"$\min(p,q)+\left(1-\sum\min(p,q)\right)\frac{\max(p-q,0)}{\sum\max(p-q,0)}=p$", 640, 200, 34)
    d.text((160, 270), "600 个位置，最大差别", font=F_SMALL, fill=TEXT)
    d.text((520, 268), f"{SP['maxdiff']:.1e}", font=vk.font(26, mono=True), fill=GREEN)
    if seg >= 1:
        labs = ["直接采样 ①", "直接采样 ②", "投机采样"]
        tv.bar_chart(d, (200, 360, 800, 620), labs, [SP["tv_plain"], SP["tv_plain2"], SP["tv_spec"]], [MUTED, MUTED, GREEN], fmt="{:.3f}", vmax=0.12)
        d.text((200, 325), "与精确分布的距离（3,000 个样本，前两个词元）", font=F_SMALL, fill=MUTED)
        d.text((860, 470), "都在抽样误差之内", font=F_SMALL, fill=AMBER)


def sc_speed(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, f"草稿 1.2 万参数 / 目标 10.6 万参数 · 平均接受率 {A:.0%}")
    labs = [f"猜 {g}" for g in G]
    tv.bar_chart(d, (110, 220, 590, 560), labs, [PC[g] for g in G], [CYAN] * 4, fmt="{:.2f}", vmax=3.0)
    d.text((110, 180), "大模型每跑一次得到的词元数", font=F_SMALL, fill=MUTED)
    for i, g in enumerate(G):
        x = 110 + (i + 0.5) * 480 / 4
        y = 560 - THEORY[g] / 3.0 * 340
        d.line([(x - 40, y), (x + 40, y)], fill=AMBER, width=3)
    d.text((110, 618), "橙线：理论 (1 - a^(γ+1)) / (1 - a)", font=F_SMALL, fill=AMBER)
    if seg >= 1:
        tv.bar_chart(d, (690, 220, 1170, 560), labs, [SPEED[g] for g in G], [GREEN if g == BEST else VIOLET for g in G], fmt="{:.2f}×", vmax=2.0)
        d.text((690, 180), "估计加速（草稿开销按参数比）", font=F_SMALL, fill=MUTED)


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "真实系统")
    rows = [("Leviathan et al.（2022）· Chen et al.（2023）", "投机解码 / 投机采样：2–3 倍", CYAN),
            ("Medusa（2024）", "在大模型上加多个预测头", GREEN), ("EAGLE（2024）", "用大模型的特征做草稿", AMBER)]
    for i, (a, b, col) in enumerate(rows):
        y = 180 + i * 95
        d.rounded_rectangle([120, y, 1160, y + 70], 12, outline=col, width=2, fill=(12, 19, 34))
        d.text((150, y + 35), a, font=F_SMALL, fill=col, anchor="lm")
        d.text((760, y + 35), b, font=F_SMALL, fill=TEXT, anchor="lm")
    if seg >= 1:
        vk.text_c(d, (640, 520), "代码、固定格式：接受率高 · 开放式创作：加速较小", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("做法", "小模型猜，大模型一次验证"), ("精确", "输出分布与大模型完全一样"),
                      ("加速", "取决于接受率与草稿开销")], seg, "下一期：批处理与 PagedAttention")


RENDER = {"title": sc_title, "idea": sc_idea, "exact": sc_exact, "speed": sc_speed, "real": sc_real, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "inf-spec")
