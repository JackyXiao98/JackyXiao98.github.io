"""Recursive Self-Improvement 02 — 反馈回路：瓶颈在哪里.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python rsi02_bottleneck.py <out_dir>

Real computations: research progress R = (0.5 I^rho + 0.5 C^rho)^(1/rho) combining AI capability I
(which grows by its own output, dI/dt = R) with compute C = e^{0.2 t} that grows at a fixed rate;
Euler integration to t = 20 for no bottleneck (R = I), substitutes (rho = 0.5) and complements
(rho = -0.5, -2); the growth rate at the end of the run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Recursive Self-Improvement"

G, DT, T = 0.2, 1e-3, 20.0
CASES = [("无瓶颈", None, RED), ("可替代 ρ = 0.5", 0.5, AMBER), ("互补 ρ = -0.5", -0.5, CYAN), ("强互补 ρ = -2", -2.0, VIOLET)]


def run(rho):
    I, out = 1.0, []
    for k in range(int(round(T / DT))):
        C = np.exp(G * k * DT)
        R = I if rho is None else (0.5 * I ** rho + 0.5 * C ** rho) ** (1 / rho)
        I += DT * R
        out.append(I)
    return np.array(out)


TS = np.arange(1, int(round(T / DT)) + 1) * DT
CURVES = {name: run(rho) for name, rho, _ in CASES}
RATE = {name: float(np.log(v[-1] / v[-2001]) / 2.0) for name, v in CURVES.items()}
print(RATE)
r = list(RATE.values())
assert 0.99 < r[0] < 1.01 and 0.29 < r[1] < 0.33 and 0.2 < r[2] < 0.22 and 0.2 < r[3] < 0.21

SCENES = [
    ("title", [
        "上一期的模型里，只有能力这一个变量。但现实中的研究，还需要算力、数据和做实验的时间。这一期看这些瓶颈怎样改变自我改进的速度。",
    ]),
    ("inputs", [
        "做人工智能研究，可以粗略地看成两种投入的组合。一种是研究者的智力劳动，在自我改进里，它由人工智能自己提供，随能力增长。",
        "另一种是算力，用来训练模型和跑实验。算力受芯片和电力限制，只能按固定的速度增长，我们假设每单位时间增长百分之二十。",
    ]),
    ("ces", [
        "两种投入怎样组合，用一个参数柔来描述。柔大于零，说明两者可以互相替代，智力不够可以多用算力，反之亦然。",
        "柔小于零，说明两者互补，缺了哪一个都不行，就像只有很多研究员、却没有显卡做实验。",
    ]),
    ("result", [
        "我们数值模拟到时间二十，看最后阶段的增长速度。",
        f"没有瓶颈时，增长率约为{dec_zh(RATE['无瓶颈'], 1)}，就是上一期的指数增长。可替代的情况下，增长率降到约{dec_zh(RATE['可替代 ρ = 0.5'])}。",
        f"互补的情况下，增长率只有{dec_zh(RATE['互补 ρ = -0.5'])}左右，几乎被锁定在算力的增长速度零点二。智力再强，也要等算力跟上。",
    ]),
    ("meaning", [
        "这个简单的模型说明，自我改进能不能加速，很大程度上取决于智力和算力是替代还是互补。",
        "这也是研究者争论的焦点之一。如果人工智能能用更聪明的方法减少对实验的依赖，瓶颈就会松动；如果每个想法都必须靠大量实验验证，算力就会成为上限。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，研究进展同时需要智力劳动和算力等资源。",
        "第二，两者可替代时，自我改进可以持续加速；两者互补时，增长被锁定在最慢投入的速度。",
        "第三，瓶颈的性质是判断自我改进速度的关键。下一期，我们看第一种具体的方法，自举数据的自我训练。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 2, "反馈回路：瓶颈在哪里", "智力劳动 × 算力", SERIES)


def sc_inputs(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "研究的两种投入")
    d.rounded_rectangle([140, 180, 560, 300], 14, outline=CYAN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (350, 220), "智力劳动 L", vk.font(30), CYAN)
    vk.text_c(d, (350, 265), "由 AI 自己提供：L = I", F_SMALL, TEXT)
    if seg >= 1:
        d.rounded_rectangle([720, 180, 1140, 300], 14, outline=AMBER, width=2, fill=(12, 19, 34))
        vk.text_c(d, (930, 220), "算力 C", vk.font(30), AMBER)
        vk.text_c(d, (930, 265), "固定速度增长：C = e^(0.2t)", F_SMALL, TEXT)
        tv.fpaste(c, r"$\frac{dI}{dt}=R(I,\,C)$", 640, 420, 36)


def sc_ces(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "两种投入怎样组合")
    tv.fpaste(c, r"$R=\left(0.5\,I^{\rho}+0.5\,C^{\rho}\right)^{1/\rho}$", 640, 190, 36)
    items = [("ρ > 0", "可替代：缺一个可用另一个补", AMBER, 0), ("ρ < 0", "互补：缺了哪个都不行", CYAN, 1)]
    for i, (a, b, col, s) in enumerate(items):
        if seg >= s:
            y = 300 + i * 100
            d.rounded_rectangle([240, y, 1040, y + 76], 12, outline=col, width=2, fill=(12, 19, 34))
            d.text((270, y + 38), a, font=vk.font(30), fill=col, anchor="lm")
            d.text((420, y + 38), b, font=F_BODY, fill=TEXT, anchor="lm")


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "模拟：能力随时间的增长（对数坐标）")
    step = 200
    k = 1 if seg <= 1 else 4
    series = [(n, CURVES[n][::step], col) for n, _, col in CASES][: (2 if seg == 1 else k)]
    tv.line_chart(d, (130, 160, 760, 560), TS[::step], series, ylog=True, ylim=(0, 9), xlabel="时间 t", ylabel="能力 I",
                  legend_xy=(150, 215), xticks=[(t, str(t)) for t in (0, 5, 10, 15, 20)])
    if seg >= 1:
        d.text((820, 170), "最后阶段的增长率", font=F_SMALL, fill=MUTED)
        for i, (n, _, col) in enumerate(CASES[: (2 if seg == 1 else 4)]):
            y = 220 + i * 70
            d.text((820, y), n, font=F_SMALL, fill=col)
            d.text((1150, y), f"{RATE[n]:.2f}", font=vk.font(26, mono=True), fill=col, anchor="ra")
    if seg >= 2:
        vk.text_c(d, (980, 540), "互补：被锁定在算力增速 0.2", F_BODY, CYAN)


def sc_meaning(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "争论的焦点")
    vk.text_c(d, (640, 200), "智力与算力：替代还是互补？", vk.font(34), AMBER)
    if seg >= 1:
        for i, (t, col) in enumerate([("用更聪明的方法减少实验 → 瓶颈松动", GREEN), ("每个想法都要大量实验 → 算力是上限", RED)]):
            vk.text_c(d, (640, 330 + i * 70), t, F_BODY, col)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("两种投入", "智力劳动 + 算力等资源"), ("替代 / 互补", "可替代可加速；互补被锁定"),
                      ("关键", "瓶颈的性质决定速度")], seg, "下一期：自举数据 —— 自训练与 STaR")


RENDER = {"title": sc_title, "inputs": sc_inputs, "ces": sc_ces, "result": sc_result, "meaning": sc_meaning,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rsi-bottleneck")
