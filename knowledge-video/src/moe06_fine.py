"""MoE 06 — 细粒度专家与共享专家.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MOE_CACHE=<dir> python moe06_fine.py <out_dir>

Real computations (moex.exp_fine, JAX), each 2,000 steps with auxiliary loss 0.01, about the same compute per
token: 8 experts of hidden 128 with top-1 (episode 01); 32 experts of hidden 32 with top-4 (fine-grained: each
expert split in four); 1 always-on shared expert of hidden 32 plus 28 routed experts of hidden 32 with top-3
(DeepSeekMoE layout). Held-out loss, total and per-token active parameters; number of expert combinations C(E, k).
Facts about real systems are quoted with references in the notes.
"""

import os
import sys
from math import comb

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

FI = moex.cached(CACHE, "fine", moex.exp_fine)
CO, FG, SH = FI["8x128_top1"], FI["32x32_top4"], FI["shared+28x32_top3"]
STEPS = [h["step"] for h in CO["hist"]]
COMB = {"co": comb(8, 1), "fg": comb(32, 4), "sh": comb(28, 3)}
print({k: (round(v["final"], 4), v["params"], v["active"]) for k, v in FI.items()}, COMB)
assert round(CO["final"], 3) == 1.686 and round(FG["final"], 2) == 1.67 and round(SH["final"], 2) == 1.67
assert FG["final"] < CO["final"] - 0.015 and abs(SH["final"] - FG["final"]) < 0.001
assert SH["params"] == 280064 and FG["params"] == 305152 and COMB["fg"] == 35960 and COMB["sh"] == 3276
assert max(CO["active"], FG["active"], SH["active"]) - min(CO["active"], FG["active"], SH["active"]) < 3200

SCENES = [
    ("title", [
        "前几期的专家都比较大，每个词元只选一个。近年的大模型走向了另一个方向：更多更小的专家，每个词元选好几个，再加上一个所有词元都用的共享专家。这一期看这两个设计。",
    ]),
    ("fine", [
        "细粒度专家的做法是，把每个专家切成更小的几块。比如原来八个宽度一百二十八的专家选一个，现在变成三十二个宽度三十二的专家选四个。总参数和每个词元的计算量都基本不变。",
        f"变的是组合的数量。八个里选一个，只有八种选法；三十二个里选四个，有{big_zh(COMB['fg'])}种。每个词元可以拼出更合适自己的一组专家，专家也可以分得更细。",
    ]),
    ("shared", [
        "第二个设计是共享专家。有些知识几乎每个词元都需要，比如常见的语法。如果让路由专家各自都学一遍，就是重复。",
        "共享专家不经过路由，每个词元都会用到，负责这些共同的知识；路由专家就可以专心学各自不同的部分。",
    ]),
    ("result", [
        f"我们在小任务上比较三种结构，每个词元的计算量相同。八个专家选一个，损失{dec_zh(CO['final'], 3)}；三十二个小专家选四个，损失降到{dec_zh(FG['final'], 3)}。",
        f"再把其中一份换成共享专家，用一个共享加二十八个路由专家选三个，损失是{dec_zh(SH['final'], 3)}，和细粒度几乎一样，而总参数少了约百分之八。",
        "不过要说明，在我们的实现里，辅助损失只统计了每个词元的第一选择，所以细粒度模型的负载并不均匀。真实系统会对所有选择一起做均衡。",
    ]),
    ("real", [
        "深度求索的混合专家论文最早系统地提出了这两个设计。它们的第三代模型每层有一个共享专家和二百五十六个路由专家，每个词元选八个。月之暗面的万亿参数模型是三百八十四个选八个，加一个共享专家。",
        "但也不是所有人都用共享专家。通义千问第三代的混合专家模型就去掉了共享专家，一百二十八个专家选八个。共享专家是否必要，目前还没有定论。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，细粒度专家把大专家切小、多选几个，组合数大大增加，在我们的实验里损失更低。",
        "第二，共享专家负责共同的知识，在我们的实验里用更少的参数达到了同样的效果。",
        "第三，真实的大模型普遍采用细粒度专家，共享专家则因模型而异。下一期，我们看混合专家的训练稳定性。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 6, "细粒度专家与共享专家", "更多更小的专家", SERIES)


def sc_fine(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "细粒度：把专家切小，多选几个")
    d.text((120, 170), "8 × 128，选 1", font=F_BODY, fill=CYAN)
    for i in range(8):
        x = 120 + i * 130
        col = GREEN if i == 2 else VIOLET
        d.rounded_rectangle([x, 220, x + 110, 290], 10, outline=col, width=3 if i == 2 else 2, fill=(12, 19, 34))
    d.text((120, 340), "32 × 32，选 4", font=F_BODY, fill=CYAN)
    sel = {3, 9, 10, 22}
    for i in range(32):
        x = 120 + i * 32.5
        col = GREEN if i in sel else VIOLET
        d.rounded_rectangle([x, 390, x + 26, 460], 5, outline=col, width=3 if i in sel else 1, fill=(12, 19, 34))
    if seg >= 1:
        f = vk.font(26, mono=True)
        d.text((300, 560), "组合数", font=F_BODY, fill=AMBER, anchor="rm")
        vk.text_c(d, (660, 560), f"C(8, 1) = {COMB['co']}    →    C(32, 4) = {COMB['fg']:,}", f, AMBER)


def sc_shared(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "共享专家：每个词元都用")
    d.rounded_rectangle([120, 220, 360, 330], 14, outline=AMBER, width=3, fill=(12, 19, 34))
    vk.text_c(d, (240, 260), "共享专家", F_BODY, AMBER)
    vk.text_c(d, (240, 300), "不经过路由", F_SMALL, TEXT)
    for i in range(14):
        x = 440 + i * 52
        col = GREEN if i in (2, 7, 11) else VIOLET
        d.rounded_rectangle([x, 230, x + 42, 320], 6, outline=col, width=3 if i in (2, 7, 11) else 1, fill=(12, 19, 34))
    d.text((440, 340), "路由专家（选 k 个）", font=F_SMALL, fill=VIOLET)
    tv.fpaste(c, r"$y=S(h)+\sum_{i\in\mathrm{top}k}\,g_i\,E_i(h)$", 640, 450, 40)
    if seg >= 1:
        vk.text_c(d, (640, 560), "共享：共同知识（如常见语法） · 路由：各自不同的部分", F_SMALL, GREEN)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "相同计算量下的三种结构（2,000 步）")
    rows = [("8 × 128，选 1", CO, MUTED), ("32 × 32，选 4", FG, CYAN)] + ([("共享 1 + 28 × 32，选 3", SH, GREEN)] if seg >= 1 else [])
    ser = [(n, np.minimum([h["loss"] for h in r["hist"]], 2.08), col) for n, r, col in rows]
    tv.line_chart(d, (110, 200, 640, 560), STEPS[5:], [(n, y[5:], col) for n, y, col in ser], ylim=(1.64, 2.1),
                  xlabel="训练步数", xticks=[(500, "500"), (1000, "1000"), (2000, "2000")])
    d.text((110, 165), "验证损失（第 500 步起）", font=F_SMALL, fill=MUTED)
    f = vk.font(22, mono=True)
    d.text((700, 190), "结构", font=F_SMALL, fill=MUTED)
    d.text((1010, 190), "损失", font=F_SMALL, fill=MUTED)
    d.text((1120, 190), "总参数", font=F_SMALL, fill=MUTED)
    for i, (n, r, col) in enumerate(rows):
        y = 240 + i * 55
        d.text((700, y), n, font=F_SMALL, fill=col)
        d.text((1010, y), f"{r['final']:.3f}", font=f, fill=TEXT)
        d.text((1120, y), f"{r['params'] / 1e3:.0f}k", font=f, fill=TEXT)
    if seg >= 2:
        vk.text_c(d, (640, 640), "注意：我们的辅助损失只统计第一选择，细粒度模型的负载不均匀", F_SMALL, AMBER)


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "真实模型")
    rows = [("DeepSeekMoE（2024）", "提出细粒度专家 + 共享专家", CYAN), ("DeepSeek-V3", "1 共享 + 256 路由，选 8", AMBER),
            ("Kimi K2", "1 共享 + 384 路由，选 8", GREEN)]
    if seg >= 1:
        rows.append(("Qwen3 MoE", "不用共享专家，128 选 8", RED))
    for i, (a, b, col) in enumerate(rows):
        y = 190 + i * 95
        d.rounded_rectangle([140, y, 1140, y + 70], 12, outline=col, width=2, fill=(12, 19, 34))
        d.text((170, y + 35), a, font=F_BODY, fill=col, anchor="lm")
        d.text((620, y + 35), b, font=F_SMALL, fill=TEXT, anchor="lm")


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("细粒度", "切小、多选，组合数大增"), ("共享专家", "共同知识，参数更省"),
                      ("真实模型", "普遍细粒度，共享因模型而异")], seg, "下一期：训练稳定性")


RENDER = {"title": sc_title, "fine": sc_fine, "shared": sc_shared, "result": sc_result, "real": sc_real, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "moe-fine")
