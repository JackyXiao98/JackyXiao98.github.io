"""Interpretability 09 — 用可解释性做什么：引导、安全与局限.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MI_CACHE=<dir> TF_CACHE=<dir> python mi09_steering.py <out_dir>

Real computations (mix.steering): "carry direction" = mean residual after layer 1 at the units-of-b
position for carry problems minus that for no-carry problems (8,000 training problems). alpha x direction
is added to no-carry test problems (subtracted from carry problems); teacher-forced predictions of the
tens digit (does it shift by +1 / -1?) and of the units digit (unchanged?).
Facts about real systems are quoted with references in the notes.
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

SR = mix.cached(CACHE, "steer", mix.steering, tf_cache=TF_CACHE)
AL = [x["alpha"] for x in SR["up"]]
UP = {x["alpha"]: x for x in SR["up"]}
DN = {x["alpha"]: x for x in SR["down"]}
print(UP[1.0], DN[1.0], UP[2.0], DN[2.0])
assert AL == [0.0, 0.5, 1.0, 1.5, 2.0] and UP[0.0]["tens_same"] == 1.0 and DN[0.0]["tens_same"] == 1.0
assert UP[1.0]["tens_shift"] == 0.75 and abs(UP[1.0]["units_same"] - 0.792) < 1e-3
assert abs(DN[1.0]["tens_shift"] - 0.832) < 1e-3 and abs(DN[1.0]["units_same"] - 0.985) < 1e-3
assert max(UP[2.0]["tens_shift"], DN[2.0]["tens_shift"]) < 0.86

SCENES = [
    ("title", [
        "这是机制可解释性系列的最后一期。理解了模型内部之后，能用它做什么？这一期看三件事，引导模型、帮助安全检查，以及这个领域还没有解决的问题。",
    ]),
    ("steer", [
        "如果我们知道某个概念对应激活空间里的一个方向，就可以在推理时直接往残差流里加上这个方向，改变模型的行为，而不需要重新训练。这叫激活引导。",
        "我们在加法模型上试一试。取所有有进位的题目，在第一层之后、第二个数的个位这个位置，求激活的平均；再对没有进位的题目求平均。两者之差，就是进位方向。",
    ]),
    ("result", [
        f"把这个方向加到没有进位的题目上，十位数字有{pct_zh(UP[1.0]['tens_shift'])}的情况多加了一。比如三十二加四十五，模型会写出八十七，而不是七十七。",
        f"反过来，从有进位的题目里减去这个方向，{pct_zh(DN[1.0]['tens_shift'])}的情况十位数字少了一。个位数字基本不受影响，减去方向时有{pct_zh(DN[1.0]['units_same'], 1)}保持不变。",
        f"不过，引导并不完美。加上方向时，个位数字也有约{pct_zh(1 - UP[1.0]['units_same'])}被改变了，再加大强度，成功率也只升到约百分之八十。一个方向只是对模型内部的近似描述。",
    ]),
    ("golden", [
        "在大模型上，二零二四年有一个著名的演示。研究者从稀疏自编码器里找到了金门大桥的特征，把它调得很强，模型就在几乎每个回答里都提到金门大桥，甚至声称自己就是金门大桥。",
        "这说明找到的特征确实在因果上影响模型的行为。引导也被用来研究和调节诚实、拒绝回答、讨好用户等行为。",
    ]),
    ("safety", [
        "可解释性对安全的意义在于，有些问题从外部的行为很难发现。比如模型可能学会了一个隐藏的目标，或者在某些情况下欺骗。如果能直接看到内部对应的特征，就多了一种检查手段。",
        "已经有研究让多个团队去审计一个被故意植入了隐藏目标的模型，可解释性工具帮助其中一些团队找到了这个目标。",
    ]),
    ("limits", [
        "但这个领域还有很多没有解决的问题。第一是忠实性，解释是否真的反映了模型的计算。一个例子是，大模型做三十六加五十九时，内部是一条估计大小的路径和一条计算个位的路径并行，可是当你问它怎么算的，它会说出教科书上的列竖式进位。",
        "第二是覆盖率，目前的归因图方法只在一部分提示上能给出满意的解释，而且每一个都要花人力去读。第三是规模，最大的模型有海量的特征和电路，离完整理解还很远。",
    ]),
    ("recap", [
        "回顾整个系列。我们从探针出发，学会了用对照实验和激活修补做因果验证，看到了归纳头和叠加，用稀疏自编码器拆开特征，从权重里读出了模加法的正弦波算法。",
        "最后拼出了加法模型的进位电路，并用一个方向引导了它的输出。打开黑箱的工作才刚刚开始。感谢大家的观看。",
    ]),
]
assert 32 + 45 == 77 and 36 + 59 == 95


def _example():
    import tfm
    P, _ = mix.addition_model(TF_CACHE)
    TR, _, _, _ = tfm.dataset()
    car = (TR[:, 1] + TR[:, 4]) >= 10
    _, r, _, _ = mix.run(P, TR[:, :8])
    v = r[1][car, 4].mean(0) - r[1][~car, 4].mean(0)
    x = np.array([tfm.encode(32, 45)], np.int32)[:, :8]
    dec = lambda L: "".join(tfm.VOCAB[i] for i in L[0, 5:8].argmax(-1))
    return dec(mix.run(P, x)[0]), dec(mix.run(P, x, add={(1, 4): v})[0])


EX = _example()
print(EX)
assert EX == ("077", "087")


def sc_title(c, u, seg, su):
    tv.title_card(c, 9, "用可解释性做什么", "引导、安全与局限", SERIES)


def sc_steer(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "激活引导：往残差流里加一个方向")
    tv.fpaste(c, r"$h\ \leftarrow\ h+\alpha\,v$", 640, 200, 40)
    vk.text_c(d, (640, 270), "推理时直接改变行为，不需要重新训练", F_SMALL, MUTED)
    if seg >= 1:
        tv.fpaste(c, r"$v=\mathrm{mean}(h\mid\mathrm{carry})-\mathrm{mean}(h\mid\mathrm{no\ carry})$", 640, 400, 32)
        vk.text_c(d, (640, 480), "位置：第二个数的个位 · 第 1 层之后 · 8,000 道训练题", F_SMALL, AMBER)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "加上 / 减去“进位方向”")
    series = [("加上：十位 +1", [UP[a]["tens_shift"] * 100 for a in AL], GREEN)]
    if seg >= 1:
        series += [("减去：十位 -1", [DN[a]["tens_shift"] * 100 for a in AL], CYAN), ("减去：个位不变", [DN[a]["units_same"] * 100 for a in AL], VIOLET)]
    if seg >= 2:
        series += [("加上：个位不变", [UP[a]["units_same"] * 100 for a in AL], RED)]
    tv.line_chart(d, (140, 170, 760, 560), AL, series, ylim=(0, 105), xlabel="强度 α", ylabel="%", legend_xy=(820, 330),
                  xticks=[(a, f"{a:g}") for a in AL])
    vk.text_c(d, (1000, 200), "32 + 45 = 077", vk.font(30, mono=True), TEXT)
    d.text((850, 250), "+ 进位方向", font=F_SMALL, fill=GREEN)
    d.text((1010, 248), "→ 087", font=vk.font(30, mono=True), fill=GREEN)
    if seg >= 2:
        d.text((820, 490), "并不完美：", font=F_BODY, fill=AMBER)
        d.text((820, 540), f"α = 1 时个位也被改变 {1 - UP[1.0]['units_same']:.0%}", font=F_SMALL, fill=TEXT)
        d.text((820, 580), "加大强度，成功率也只到约 80%", font=F_SMALL, fill=TEXT)


def sc_golden(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "Golden Gate Claude（Anthropic, 2024）")
    d.rounded_rectangle([200, 180, 1080, 300], 14, outline=AMBER, width=2, fill=(12, 19, 34))
    vk.text_c(d, (640, 220), "SAE 找到“金门大桥”特征 → 调得很强", F_BODY, AMBER)
    vk.text_c(d, (640, 265), "几乎每个回答都提到金门大桥，甚至自称是金门大桥", F_SMALL, TEXT)
    if seg >= 1:
        vk.text_c(d, (640, 400), "特征在因果上影响行为", F_BODY, GREEN)
        vk.text_c(d, (640, 455), "引导也用于研究：诚实 · 拒绝回答 · 讨好用户", F_SMALL, MUTED)


def sc_safety(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "安全：从内部检查")
    for i, t in enumerate(["隐藏的目标", "某些情况下的欺骗", "外部行为难以发现"]):
        vk.text_c(d, (640, 180 + i * 55), t, F_BODY, [RED, RED, MUTED][i])
    if seg >= 1:
        d.rounded_rectangle([160, 400, 1120, 510], 14, outline=CYAN, width=2, fill=(12, 19, 34))
        vk.text_c(d, (640, 435), "审计实验：模型被故意植入隐藏目标（Marks et al., 2025）", F_SMALL, CYAN)
        vk.text_c(d, (640, 475), "可解释性工具帮助部分团队找到了它", F_SMALL, TEXT)


def sc_limits(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "尚未解决的问题")
    d.rounded_rectangle([100, 160, 1180, 330], 14, outline=AMBER, width=2, fill=(12, 19, 34))
    d.text((130, 185), "忠实性：解释是否反映真实的计算？", font=F_BODY, fill=AMBER)
    d.text((130, 240), "36 + 59：内部 = 估计大小 ∥ 计算个位（两条路径并行）", font=F_SMALL, fill=TEXT)
    d.text((130, 280), "问它怎么算 → 说出教科书式的列竖式进位", font=F_SMALL, fill=RED)
    if seg >= 1:
        for i, (a, b) in enumerate([("覆盖率", "归因图只在一部分提示上有满意的解释 · 需要人力解读"), ("规模", "最大的模型：海量特征与电路，离完整理解还很远")]):
            y = 370 + i * 110
            d.rounded_rectangle([100, y, 1180, y + 85], 14, outline=MUTED, width=2, fill=(12, 19, 34))
            d.text((130, y + 42), a, font=F_BODY, fill=CYAN, anchor="lm")
            d.text((300, y + 42), b, font=F_SMALL, fill=TEXT, anchor="lm")


EPS = ["什么是可解释性", "探针", "激活修补", "归纳头", "叠加", "稀疏自编码器", "模加法与顿悟", "电路与归因图", "引导与局限"]


def sc_recap(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 6, "系列回顾")
    k = 7 if seg == 0 else 9
    for i, t in enumerate(EPS):
        x, y = 110 + (i % 5) * 215, 180 + (i // 5) * 170
        on = i < k
        d.rounded_rectangle([x, y, x + 195, y + 120], 14, outline=CYAN if on else (30, 41, 59), width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 97, y + 36), f"{i + 1:02d}", vk.font(26, mono=True), CYAN if on else MUTED)
        vk.text_c(d, (x + 97, y + 82), t, F_SMALL, TEXT if on else MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 600), "感谢观看 · Interpretability 系列完", F_BODY, VIOLET)


RENDER = {"title": sc_title, "steer": sc_steer, "result": sc_result, "golden": sc_golden, "safety": sc_safety, "limits": sc_limits, "recap": sc_recap}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "mi-steering")
