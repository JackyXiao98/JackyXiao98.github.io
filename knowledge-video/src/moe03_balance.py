"""MoE 03 — 负载均衡：专家崩塌与辅助损失.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MOE_CACHE=<dir> python moe03_balance.py <out_dir>

Real computations (moex.exp_balance, JAX): the MoE of episode 01 (8 experts of hidden 128, top-1) trained 2,000
steps three ways: no balancing; Switch auxiliary loss E * sum_i f_i P_i with weight 0.01; DeepSeek-V3-style
auxiliary-loss-free balancing (a per-expert bias added to the router scores for selection only, updated after
every step by 1e-3 * sign(1/E - f_i)). Load = share of tokens routed to each expert on 256 held-out sequences;
"max load" = largest share x E (1 = perfectly even).
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import moex
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Mixture of Experts"
CACHE = os.environ.get("MOE_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "moecache"))

R = moex.cached(CACHE, "balance", moex.exp_balance)
NO, AX, BI = R["none"], R["aux"], R["bias"]
FN = NO["hist"][-1]["frac"]          # (layers, E)
TOP = float(FN.max())
STEPS = [h["step"] for h in NO["hist"]]
LM = {k: [h["load_max"] for h in v["hist"]] for k, v in R.items()}
BAL_STEP = next(h["step"] for h in BI["hist"] if h["load_max"] < 1.5)
print(round(TOP, 3), {k: round(v["final"], 4) for k, v in R.items()}, {k: round(v[-1], 2) for k, v in LM.items()}, BAL_STEP)
assert round(TOP, 2) == 0.77 and FN[0].argmax() == 3 and round(LM["none"][-1], 1) == 6.2
assert round(NO["final"], 2) == 1.74 and round(AX["final"], 2) == 1.69 and round(BI["final"], 2) == 1.73
assert LM["aux"][-1] < 1.6 and LM["bias"][-1] < 1.9 and BAL_STEP == 1000 and 1.8 < max(LM["bias"][14:]) < 2.1

SCENES = [
    ("title", [
        "上一期我们看了路由器怎样为每个词元选专家。但如果让路由器自由地选，常常会出现一个问题：几乎所有词元都挤到少数几个专家那里。这一期讲负载均衡。",
    ]),
    ("collapse", [
        f"我们把第一期的模型去掉任何均衡手段，重新训练。训练完，第一层里有一个专家接收了{pct_zh(TOP)}的词元，其余七个专家加起来只分到不到四分之一。",
        f"损失也变差了，从{dec_zh(AX['final'], 2)}升到{dec_zh(NO['final'], 2)}。这叫专家崩塌：模型虽然有八个专家，实际上大部分时候只用了一两个。",
    ]),
    ("why", [
        "为什么会这样？这是一个富者越富的过程。某个专家一开始碰巧多分到一些词元，它就得到更多训练，变得更好；路由器发现它更好，就把更多词元交给它。其他专家得不到训练，越来越没人用。",
        "而且负载不均还有硬件上的代价。专家分布在不同的显卡上时，最忙的那张卡决定了整体的速度。",
    ]),
    ("aux", [
        "最常用的办法是加一个辅助损失。对每个专家，算它实际分到的词元比例，和路由器给它的平均概率，两者相乘再求和。分布越不均，这个值越大，最均匀时它等于一。",
        f"加上权重零点零一的辅助损失，最忙的专家只比平均多一半左右，损失回到{dec_zh(AX['final'], 2)}。",
    ]),
    ("bias", [
        "辅助损失也有缺点，它的梯度会和语言建模的目标互相拉扯。深度求索第三代用了另一种办法，不加损失，而是给每个专家一个偏置，只在挑选专家时加到分数上，不影响输出的权重。",
        "每一步之后，分到太多词元的专家，偏置减小一点；分得太少的，偏置增大一点。",
        f"在我们的实验里，这种方法也让负载变均匀了，但偏置每步只改一点，要到第{int_zh(BAL_STEP)}步左右才接近均匀，之后最忙的专家在平均的两倍左右波动。两千步时损失是{dec_zh(BI['final'], 2)}，在两者之间。调整的步长决定了它跟得多快、抖得多厉害。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，不加约束时，路由会富者越富，导致专家崩塌。",
        "第二，辅助损失按分到的比例乘以平均概率来惩罚不均衡。",
        "第三，无辅助损失的偏置方法只影响选择，不干扰训练目标。下一期，我们看每个专家能接收多少词元，以及超出的词元怎么办。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 3, "负载均衡", "专家崩塌与辅助损失", SERIES)


def frac_bars(d, box, fr, col, title):
    tv.bar_chart(d, box, [f"E{i + 1}" for i in range(8)], [f * 100 for f in fr], [col] * 8, fmt="{:.0f}%", vmax=90)
    d.text((box[0], box[1] - 40), title, font=F_SMALL, fill=MUTED)


def sc_collapse(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "不加均衡：专家崩塌（第一层，每个专家分到的词元比例）")
    frac_bars(d, (110, 220, 620, 580), FN[0], RED, "不加均衡")
    frac_bars(d, (700, 220, 1210, 580), AX["hist"][-1]["frac"][0], GREEN, "辅助损失 0.01（对比）")
    if seg >= 1:
        vk.text_c(d, (640, 650), f"损失：{AX['final']:.3f} → {NO['final']:.3f}", F_BODY, RED)


def sc_why(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "富者越富")
    pts = [(640, 200), (900, 360), (640, 520), (380, 360)]
    labs = ["多分到词元", "得到更多训练", "变得更好", "路由器更爱选它"]
    for i, ((x, y), t) in enumerate(zip(pts, labs)):
        d.rounded_rectangle([x - 130, y - 32, x + 130, y + 32], 12, outline=AMBER, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x, y), t, F_SMALL, AMBER)
        a, b = pts[i], pts[(i + 1) % 4]
        vk.arrow(d, (a[0] + (b[0] - a[0]) * 0.35, a[1] + (b[1] - a[1]) * 0.35), (a[0] + (b[0] - a[0]) * 0.65, a[1] + (b[1] - a[1]) * 0.65), MUTED, 3)
    if seg >= 1:
        vk.text_c(d, (640, 630), "硬件上：最忙的那张卡决定整体速度", F_BODY, RED)


def sc_aux(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "辅助损失（Switch Transformer）")
    tv.fpaste(c, r"$\mathcal{L}_{aux}=E\sum_{i=1}^{E} f_i\,P_i$", 400, 250, 44)
    tv.fpaste(c, r"$f_i$", 160, 395, 28)
    d.text((190, 380), "：实际分到专家 i 的词元比例", font=F_SMALL, fill=TEXT)
    tv.fpaste(c, r"$P_i$", 160, 440, 28)
    d.text((190, 425), "：路由器给专家 i 的平均概率", font=F_SMALL, fill=TEXT)
    d.text((140, 470), "完全均匀时 = 1", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        frac_bars(d, (720, 220, 1200, 560), AX["hist"][-1]["frac"][0], GREEN, "加 0.01 辅助损失后（第一层）")
        vk.text_c(d, (400, 560), f"最大负载 {LM['aux'][-1]:.2f} × 平均 · 损失 {AX['final']:.3f}", F_SMALL, GREEN)


def sc_bias(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "无辅助损失的偏置均衡（DeepSeek-V3）")
    tv.fpaste(c, r"$\mathrm{select}:\ \mathrm{top}k(s_i+b_i)\qquad \mathrm{gate}:\ s_i$", 640, 160, 32)
    if seg >= 1:
        tv.fpaste(c, r"$b_i \leftarrow b_i+\gamma\,\mathrm{sign}(1/E-f_i)$", 640, 215, 32)
    if seg >= 2:
        ser = [("不加均衡", np.minimum(LM["none"], 6.5), RED), ("辅助损失", np.minimum(LM["aux"], 6.5), GREEN),
               ("偏置（γ = 0.001）", np.minimum(LM["bias"], 6.5), CYAN)]
        tv.line_chart(d, (120, 320, 760, 620), STEPS, ser, ylim=(0.8, 6.6), xlabel="训练步数",
                      xticks=[(0, "0"), (1000, "1000"), (2000, "2000")], legend_xy=(800, 320))
        d.text((120, 280), "最大负载（× 平均）", font=F_SMALL, fill=MUTED)
        f = vk.font(24, mono=True)
        for i, (n, r, col) in enumerate([("不加均衡", NO, RED), ("辅助损失", AX, GREEN), ("偏置", BI, CYAN)]):
            d.text((800, 450 + i * 45), n, font=F_SMALL, fill=col)
            d.text((960, 450 + i * 45), f"损失 {r['final']:.3f}", font=F_SMALL, fill=TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("崩塌", "富者越富，少数专家包揽"), ("辅助损失", "分到的比例 × 平均概率"),
                      ("偏置", "只影响选择，不干扰目标")], seg, "下一期：容量、丢词元与无丢弃路由")


RENDER = {"title": sc_title, "collapse": sc_collapse, "why": sc_why, "aux": sc_aux, "bias": sc_bias, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "moe-balance")
