"""MoE 04 — 容量、丢词元与无丢弃路由.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MOE_CACHE=<dir> python moe04_capacity.py <out_dir>

Real computations (moex.exp_capacity, JAX): the balanced MoE of episode 01 (8 x 128, top-1, auxiliary loss 0.01)
with an expert capacity C = ceil(cf * N * k / E) per batch (N tokens, first come first served; tokens over capacity
skip the expert and pass through the residual connection) for capacity factors 0.5 / 1.0 / 1.25 / 2.0, applied in
training (N = 1,024 per step) and evaluation (N = 8,192), 2,000 steps each; compared with no capacity limit
(dropless). Share of dropped tokens on the held-out set at steps 0 / 100 / ... / 2,000 and final loss.
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

CP = moex.cached(CACHE, "capacity", moex.exp_capacity)
DL = moex.cached(CACHE, "scale", moex.exp_scale)["moe8"]
CFS = [0.5, 1.0, 1.25, 2.0]
DROP = {cf: CP[cf]["hist"][-1]["dropped"] for cf in CFS}
LOSS = {cf: CP[cf]["final"] for cf in CFS}
STEPS = [h["step"] for h in CP[1.0]["hist"]]
DCURVE = {cf: [h["dropped"] for h in CP[cf]["hist"]] for cf in CFS}
print({cf: (round(DROP[cf], 3), round(LOSS[cf], 4)) for cf in CFS}, round(DL["final"], 4), {cf: round(DCURVE[cf][0], 3) for cf in CFS})
assert round(DROP[0.5], 2) == 0.5 and round(DROP[1.0], 2) == 0.04 and DROP[1.25] < 0.005 and DROP[2.0] == 0
assert [round(LOSS[cf], 2) for cf in CFS] == [1.97, 1.73, 1.71, 1.70] and round(DL["final"], 2) == 1.69
assert round(DCURVE[1.0][0], 2) == 0.45 and round(DCURVE[2.0][0], 2) == 0.26 and max(DCURVE[2.0][1:]) < 0.005

SCENES = [
    ("title", [
        "在真实的硬件上，每个专家能处理多少词元，往往要事先定好，这样才能分配固定大小的内存和计算。这一期讲专家容量：当太多词元挤向同一个专家时，会发生什么。",
    ]),
    ("capacity", [
        "专家容量的定义很简单。如果一批有很多词元，均匀分配时每个专家应该分到总数除以专家数那么多。容量就是这个平均值再乘上一个容量系数。",
        "超出容量的词元会被丢掉。不是从序列里删除，而是这一层的专家不处理它，它只通过残差连接直接进入下一层，相当于这一层什么也没做。",
    ]),
    ("sweep", [
        f"我们把第一期平衡好的模型加上容量限制，系数分别取零点五、一、一点二五和二。系数零点五时，一半的词元被丢掉，损失升到{dec_zh(LOSS[0.5], 2)}。",
        f"系数为一，也就是刚好够平均分配时，仍然丢掉约{pct_zh(DROP[1.0])}的词元，损失{dec_zh(LOSS[1.0], 2)}。因为负载不可能完全均匀，总有专家会超出。",
        f"系数一点二五时，几乎不丢了，损失{dec_zh(LOSS[1.25], 2)}；系数为二时，损失{dec_zh(LOSS[2.0], 2)}。不加容量限制时是{dec_zh(DL['final'], 2)}。最后这零点零一的差距很小，在我们的实验里可能只是随机波动。",
    ]),
    ("early", [
        f"丢词元在训练早期最严重。刚开始时路由还很不均匀，系数为一时，丢掉了{pct_zh(DCURVE[1.0][0])}；即使系数为二，也丢掉了{pct_zh(DCURVE[2.0][0])}。随着辅助损失把负载拉平，丢弃很快减少。",
        "所以容量系数是一个取舍：太小会丢词元，太大则要为很少用到的空位预留内存和计算。",
    ]),
    ("dropless", [
        "后来的工作想要彻底避免丢词元。一种方法是用块稀疏的矩阵运算，让每个专家处理任意数量的词元，这叫无丢弃的混合专家。",
        "深度求索第三代也报告说，凭借良好的负载均衡，训练和推理时都不丢弃任何词元。代价是每个专家的工作量不固定，对系统实现提出了更高的要求。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，专家容量等于平均分到的词元数乘以容量系数，超出的词元跳过这一层的专家。",
        "第二，系数为一时仍会丢词元，丢弃主要集中在路由还不均衡的训练早期。",
        "第三，无丢弃的实现避免了这个问题，但需要处理不固定的工作量。下一期，我们看专家到底学到了什么。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 4, "容量、丢词元与无丢弃路由", "专家满了怎么办", SERIES)


def sc_capacity(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "专家容量")
    tv.fpaste(c, r"$C=\mathrm{cf}\times\frac{N\,k}{E}$", 640, 190, 46)
    loads = [3, 6, 2, 4]
    cap = 4
    for i, n in enumerate(loads):
        x = 200 + i * 240
        d.rounded_rectangle([x, 300, x + 160, 300 + cap * 52 + 10], 12, outline=VIOLET, width=2)
        vk.text_c(d, (x + 80, 285), f"专家 {i + 1}（容量 {cap}）", F_SMALL, VIOLET)
        for j in range(n):
            ok = j < cap
            if ok:
                tv.token(d, x + 40, 306 + (cap - 1 - j) * 52, "词", CYAN, w=80, font=F_SMALL)
            elif seg >= 1:
                tv.token(d, x + 40, 300 + cap * 52 + 30 + (j - cap) * 52, "丢", RED, w=80, font=F_SMALL)
    if seg >= 1:
        vk.text_c(d, (640, 660), "超出容量：跳过这一层的专家，只走残差连接", F_BODY, RED)


def sc_sweep(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "容量系数扫描（8 × 128，选 1，辅助损失 0.01）")
    n = [1, 2, 4][seg]
    rows = [(f"cf = {cf}", DROP[cf], LOSS[cf], [RED, AMBER, CYAN, CYAN][i]) for i, cf in enumerate(CFS[:n])]
    if seg >= 2:
        rows.append(("不限容量（无丢弃）", 0.0, DL["final"], GREEN))
    f = vk.font(26, mono=True)
    d.text((200, 170), "", font=F_SMALL, fill=MUTED)
    d.text((620, 170), "丢掉的词元", font=F_SMALL, fill=MUTED)
    d.text((900, 170), "验证损失", font=F_SMALL, fill=MUTED)
    for i, (lab, dr, ls, col) in enumerate(rows):
        y = 225 + i * 75
        d.rounded_rectangle([160, y - 10, 1120, y + 50], 10, outline=col, width=2, fill=(12, 19, 34))
        d.text((190, y + 20), lab, font=F_BODY, fill=col, anchor="lm")
        d.text((640, y + 20), f"{dr:.1%}", font=f, fill=TEXT, anchor="lm")
        d.text((910, y + 20), f"{ls:.3f}", font=f, fill=TEXT, anchor="lm")


def sc_early(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "丢弃集中在训练早期")
    cols = [RED, AMBER, CYAN, GREEN]
    ser = [(f"cf = {cf}", np.array(DCURVE[cf]) * 100, cols[i]) for i, cf in enumerate(CFS)]
    tv.line_chart(d, (120, 200, 760, 580), STEPS, ser, ylim=(0, 70), xlabel="训练步数",
                  xticks=[(0, "0"), (1000, "1000"), (2000, "2000")], legend_xy=(820, 220))
    d.text((120, 165), "丢掉的词元（%）", font=F_SMALL, fill=MUTED)
    if seg >= 1:
        vk.text_c(d, (960, 470), "太小：丢词元", F_SMALL, RED)
        vk.text_c(d, (960, 510), "太大：为空位预留内存与计算", F_SMALL, AMBER)


def sc_dropless(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "无丢弃的 MoE")
    d.rounded_rectangle([140, 190, 1140, 300], 14, outline=GREEN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (640, 225), "MegaBlocks（2022）：块稀疏矩阵运算", F_BODY, GREEN)
    vk.text_c(d, (640, 268), "每个专家处理任意数量的词元，不设容量", F_SMALL, TEXT)
    if seg >= 1:
        d.rounded_rectangle([140, 340, 1140, 450], 14, outline=AMBER, width=2, fill=(12, 19, 34))
        vk.text_c(d, (640, 375), "DeepSeek-V3：训练与推理都不丢词元", F_BODY, AMBER)
        vk.text_c(d, (640, 418), "依靠良好的负载均衡", F_SMALL, TEXT)
        vk.text_c(d, (640, 530), "代价：每个专家的工作量不固定，系统实现更难", F_SMALL, RED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("容量", "平均词元数 × 容量系数"), ("丢弃", "cf = 1 仍会丢，集中在早期"),
                      ("无丢弃", "块稀疏实现，工作量不固定")], seg, "下一期：专家学到了什么")


RENDER = {"title": sc_title, "capacity": sc_capacity, "sweep": sc_sweep, "early": sc_early, "dropless": sc_dropless, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "moe-capacity")
