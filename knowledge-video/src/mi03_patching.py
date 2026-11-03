"""Interpretability 03 — 激活修补：用因果干预定位信息.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MI_CACHE=<dir> TF_CACHE=<dir> python mi03_patching.py <out_dir>

Real computations (mix.patching): 1,577 test problems paired with a corrupted version that changes only
the units digit of b so that the carry flips (hundreds digit unchanged). The clean residual stream at one
(layer, position) is copied into the corrupted run; score = (LD_patched - LD_corrupted) / (LD_clean -
LD_corrupted), LD = logit(clean tens digit) - logit(corrupted tens digit) at position 6.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import mix
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Interpretability"
CACHE = os.environ.get("MI_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "micache"))
TF_CACHE = os.environ.get("TF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "tfcache"))

PA = mix.cached(CACHE, "patch", mix.patching, tf_cache=TF_CACHE)
SC = PA["score"]
TOK = ["3", "7", "+", "4", "5", "=", "0"]
print(SC, PA["n"])
assert PA["n"] == 1577 and PA["acc_clean"] == 1.0 and PA["acc_corr"] == 1.0
assert SC[0, 4] == 1.0 and abs(SC[1, 4] - 0.869) < 1e-3 and abs(SC[1, 6] - 0.094) < 1e-3 and abs(SC[2, 6] - 1.0) < 1e-6
assert np.abs(SC[2, :6]).max() < 1e-6 and np.abs(SC[0, :4]).max() < 1e-6
assert 37 + 45 == 82 and 37 + 42 == 79

SCENES = [
    ("title", [
        "上一期说，探针只能说明信息在那里。这一期用因果干预，直接检验模型在哪里用到了这个信息，方法叫激活修补。",
    ]),
    ("idea", [
        "做法是准备两个只差一点的输入。干净的输入是三十七加四十五，等于八十二；损坏的输入把第二个数的个位换成二，三十七加四十二，等于七十九。",
        "两道题唯一的区别是个位有没有进位，所以十位数字一个是八，一个是七。",
        "先分别运行两道题，记下每一层、每个位置的激活。然后运行损坏的输入，但在某一个层和位置，把激活换成干净输入的。如果模型的答案从七变回八，说明这个位置的这一层，承载了进位的信息。",
    ]),
    ("score", [
        f"我们在{big_zh(PA['n'])}对这样的题目上做实验，用十位数字上两个答案的对数几率之差来打分。分数是一，表示完全恢复了干净的答案；是零，表示没有影响。",
    ]),
    ("map", [
        "结果画成一张图。在嵌入层，只有第二个数的个位这个位置分数是一，这很自然，两道题只有这个词元不同。",
        f"第一层之后，信息仍然主要留在这个位置，分数是{dec_zh(SC[1, 4], 2)}，开始有一小部分搬到了后面的位置。",
        "到了第二层之后，这个位置的分数变成了零，而要写十位数字的位置变成了一。进位信息在第二层，从第二个个位数字的位置，被搬运到了答案的位置。",
    ]),
    ("compare", [
        "和第一期的探针表对照，两者讲的是同一个故事。但激活修补更有说服力，因为它证明了模型真的依赖这些激活来给出答案。",
        "下一步自然要问，是哪一个具体的部件完成了搬运，是哪个注意力头，还是哪个前馈层。这个问题留到第八期的电路。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，激活修补把干净输入的激活换进损坏的运行，看答案能否恢复。",
        "第二，加法模型的进位在第二个个位数字的位置算出，到第二层被搬到答案的位置。",
        "第三，修补是因果的证据，比探针更强。下一期，我们看一种在很多语言模型里都会出现的电路，归纳头。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 3, "激活修补", "用因果干预定位信息", SERIES)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "两个只差一点的输入")
    f = vk.font(34, mono=True)
    rows = [("干净", "37+45=082", GREEN), ("损坏", "37+42=079", RED)]
    for r, (lab, txt, col) in enumerate(rows):
        y = 190 + r * 60
        d.text((180, y), lab, font=F_BODY, fill=col, anchor="lm")
        for i, ch in enumerate(txt):
            vk.text_c(d, (330 + i * 46, y), ch, f, TEXT)
    if seg >= 1:
        for i in (4, 7):
            x = 330 + i * 46
            d.rounded_rectangle([x - 20, 160, x + 20, 280], 8, outline=AMBER, width=2)
        d.text((760, 220), "只差个位是否进位 → 十位 8 vs 7", font=F_SMALL, fill=AMBER, anchor="lm")
    if seg >= 2:
        steps = [("运行损坏的输入", RED), ("在某层某位置换上\n干净输入的激活", AMBER), ("十位变回 8？", GREEN)]
        for i, (t, col) in enumerate(steps):
            x = 110 + i * 370
            d.rounded_rectangle([x, 360, x + 320, 470], 14, outline=col, width=2, fill=(12, 19, 34))
            lines = t.split("\n")
            for j, line in enumerate(lines):
                vk.text_c(d, (x + 160, 415 + (j - (len(lines) - 1) / 2) * 34), line, F_SMALL, col)
            if i < 2:
                vk.arrow(d, (x + 325, 415), (x + 365, 415), MUTED, 3)


def sc_score(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, f"打分（{PA['n']:,} 对题目平均）")
    tv.fpaste(c, r"$\mathrm{score}=\frac{LD_{\mathrm{patched}}-LD_{\mathrm{corrupt}}}{LD_{\mathrm{clean}}-LD_{\mathrm{corrupt}}}$", 640, 260, 38)
    vk.text_c(d, (640, 360), "LD = 十位数字上：logit(干净答案) - logit(损坏答案)", F_SMALL, MUTED)
    vk.text_c(d, (640, 440), "1 = 完全恢复 · 0 = 没有影响", F_BODY, AMBER)


def sc_map(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "修补分数：行 = 层，列 = 位置")
    rows = seg + 1
    M = np.full((3, 7), np.nan)
    M[:rows] = SC[:rows]
    tv.heat(d, (260, 190, 1060, 520), M[::-1], 0, 1, labels_x=TOK, fmt="{:.2f}", font_size=22, cmap="green")
    for l, lab in enumerate(["第 2 层后", "第 1 层后", "嵌入"]):
        d.text((245, 190 + (l + .5) * 110), lab, font=F_SMALL, fill=MUTED, anchor="rm")
    msg = ["只有“5”这个位置不同 → 分数 1", "信息仍主要在“5”，少量开始后移", "第 2 层：进位被搬到要写十位的位置"][min(seg, 2)]
    vk.text_c(d, (640, 590), msg, F_BODY, AMBER)


def sc_compare(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "探针 vs 修补：同一个故事")
    for i, (a, b, col) in enumerate([("探针", "信息在那里（相关）", CYAN), ("修补", "模型依赖它给出答案（因果）", GREEN)]):
        x = 120 + i * 540
        d.rounded_rectangle([x, 190, x + 500, 310], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 250, 230), a, F_BODY, col)
        vk.text_c(d, (x + 250, 275), b, F_SMALL, TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 420), "下一步：是哪个注意力头、哪个前馈层完成了搬运？", F_BODY, AMBER)
        vk.text_c(d, (640, 475), "→ 第 08 期：电路", F_SMALL, MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("激活修补", "换上干净的激活，看答案能否恢复"), ("加法模型", "进位：“5”处算出 → 第 2 层搬到答案处"),
                      ("因果", "比探针更强的证据")], seg, "下一期：注意力头与归纳头")


RENDER = {"title": sc_title, "idea": sc_idea, "score": sc_score, "map": sc_map, "compare": sc_compare, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "mi-patching")
