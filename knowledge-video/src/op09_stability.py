"""Optimization 09 — 训练的稳定性：损失尖峰、梯度裁剪与低精度.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] OP_CACHE=<dir> python op09_stability.py <out_dir>

Real computations:
- opx.stability (JAX): 2-layer Transformer on the 12-token Markov language, Adam (no warmup), batch 64,
  1,000 steps, lr 3e-3 / 1e-2 / 3e-2, without and with global-norm gradient clipping at 1.0; per-step gradient
  norm (before clipping) and final test loss.
- opx.bf16_updates: linear regression (16 weights, 512 examples), full-batch GD with lr 1e-3 for 2,000 steps,
  weights stored in fp32, stored in bf16 (each update rounded into bf16), or bf16 gradients with an fp32 master
  copy; 1.0 + 0.001 rounded to bf16.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import opx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Optimization"
CACHE = os.environ.get("OP_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "opcache"))

SB = opx.cached(CACHE, "stab", opx.stability)
FL = {k: float(SB[k]["hist"][-1, 1]) for k in SB}
G = {k: SB[k]["gnorm"] for k in SB}
HI, LO = G[(0.03, None)], G[(3e-3, None)]
SPK = lambda g: int((g > 5 * np.median(g)).sum())
B = opx.cached(CACHE, "bf16", opx.bf16_updates)
print(FL, np.median(HI), HI.max(), HI.argmax(), SPK(HI), SPK(LO), B["fp32"][-1], B["bf16"][-1], B["master"][-1])
assert abs(HI.max() - 18.32) < 0.01 and HI.argmax() == 624 and abs(np.median(HI) - 0.35) < 0.005 and SPK(HI) == 22 and SPK(LO) == 4
assert all(FL[(lr, 1.0)] < FL[(lr, None)] for lr in (3e-3, 1e-2, 3e-2)) and abs(FL[(0.03, None)] - 1.516) < 0.001
assert abs(FL[(0.03, 1.0)] - 1.4585) < 1e-3 and abs(FL[(3e-3, 1.0)] - 1.0904) < 1e-3 and abs(FL[(3e-3, None)] - 1.0944) < 1e-3
assert B["one_plus"] == 1.0 and B["eps_bf16"] == 2 ** -7 and abs(B["bf16"][-1] - 3.598) < 1e-3 and B["fp32"][-1] < 0.02 and B["master"][-1] < 0.02
assert np.all(B["bf16"][50:] == B["bf16"][-1])
ST = np.arange(1, 1001)

SCENES = [
    ("title", [
        "这是深度学习优化系列的最后一期。训练一个大模型往往要几个月，中途任何一次崩溃都代价巨大。这一期看训练的稳定性，损失尖峰、梯度裁剪，以及低精度计算带来的问题。",
    ]),
    ("spikes", [
        "大模型训练中常见的一种现象是损失尖峰，损失突然跳高，有时能自己恢复，有时就再也回不去了。它往往伴随着梯度范数的突然暴涨。",
        f"我们在人造语言上训练一个小变换器。学习率是零点零零三时，一千步里梯度范数超过中位数五倍的只有{int_zh(SPK(LO))}次，都在训练刚开始。学习率提高到零点零三，这样的尖峰有{int_zh(SPK(HI))}次，在第{int_zh(HI.argmax())}步，梯度范数达到中位数的约{int_zh(round(HI.max() / np.median(HI)))}倍。",
    ]),
    ("clip", [
        "最常用的保护手段是梯度裁剪。如果整体梯度的范数超过一个阈值，就把梯度按比例缩小到阈值，方向不变。这样一次异常的梯度，不会把参数一下子推得太远。",
        f"在三个学习率下，加了裁剪，最终的损失都略低一些。学习率零点零三时，从{dec_zh(FL[(0.03, None)], 3)}降到{dec_zh(FL[(0.03, 1.0)], 3)}。",
        f"但裁剪救不了根本上太大的学习率。同样加了裁剪，学习率零点零零三的损失是{dec_zh(FL[(3e-3, 1.0)], 3)}，好得多。裁剪是安全网，不是万能药。",
    ]),
    ("other", [
        "大模型训练还有很多稳定手段。比如在损失里加一项，惩罚输出概率分布归一化因子的对数，防止输出的数值无限变大；对注意力的查询和键做归一化，防止注意力的数值爆炸。",
        "如果尖峰还是发生了，常见的做法是回退到之前的检查点，跳过引发问题的那一批数据，再继续训练。",
    ]),
    ("bf16", [
        "最后是低精度。为了省显存、提速度，大模型用十六位甚至八位的浮点数计算。常用的半精度格式叫脑浮点十六，它只有七位小数精度，在一附近，相邻两个数相差一百二十八分之一。",
        "这带来一个问题。一点零加上零点零零一，在这种格式里结果还是一点零，小的更新被直接舍入掉了。",
        f"我们用梯度下降做一个线性回归。权重用三十二位存储时，损失从{dec_zh(B['fp32'][0], 1)}降到{dec_zh(B['fp32'][-1], 3)}。权重直接用十六位存储时，降到{dec_zh(B['bf16'][-1], 2)}就再也不动了，每一步的更新都被舍入掉。",
        "解决办法是混合精度。计算用低精度，但另外保留一份三十二位的主权重，用它来累积更新。这样损失同样降到了零点零一七。今天的大模型，包括一些使用八位浮点数训练的模型，都依赖这类技巧。",
    ]),
    ("recap", [
        "回顾整个系列。我们从梯度下降和条件数出发，看到了随机梯度和临界批量，动量和自适应方法怎样加速，学习率调度怎样降噪。",
        "然后看了初始化和归一化怎样让深层网络能训练，最大更新参数化怎样迁移超参数，穆昂等新一代优化器，以及训练的稳定性。感谢大家的观看。",
    ]),
]
assert round(B["master"][-1], 3) == 0.017


def sc_title(c, u, seg, su):
    tv.title_card(c, 9, "训练的稳定性", "损失尖峰、梯度裁剪与低精度", SERIES)


def sc_spikes(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "梯度范数的尖峰（人造语言 · Adam · 无预热）")
    series = [("学习率 0.003", LO, CYAN)] + ([("学习率 0.03", HI, RED)] if seg >= 1 else [])
    tv.line_chart(d, (140, 170, 820, 580), ST, series, ylog=True, ylim=(-1.3, 1.4), xlabel="步数", ylabel="梯度范数（对数）",
                  legend_xy=(880, 470), xticks=[(s, str(s)) for s in (1, 250, 500, 750, 1000)])
    vk.text_c(d, (1030, 220), "损失尖峰 ↔ 梯度范数暴涨", F_SMALL, AMBER)
    if seg >= 1:
        d.text((880, 280), "> 5 倍中位数的步数", font=F_SMALL, fill=MUTED)
        d.text((880, 320), f"0.003：{SPK(LO)} 次（都在开头）", font=F_SMALL, fill=CYAN)
        d.text((880, 360), f"0.03：{SPK(HI)} 次", font=F_SMALL, fill=RED)
        d.text((880, 400), f"第 {HI.argmax()} 步：{HI.max():.1f}", font=F_SMALL, fill=RED)


def sc_clip(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "梯度裁剪")
    tv.fpaste(c, r"$g\leftarrow g\cdot\min\left(1,\ \frac{c}{\|g\|}\right)$", 330, 200, 36)
    vk.text_c(d, (330, 270), "方向不变，长度最多为 c", F_SMALL, MUTED)
    if seg >= 1:
        lrs = [3e-3, 1e-2, 3e-2]
        d.text((700, 160), "最终测试损失", font=F_SMALL, fill=MUTED)
        d.text((880, 160), "不裁剪", font=F_SMALL, fill=RED)
        d.text((1040, 160), "裁剪 1.0", font=F_SMALL, fill=GREEN)
        for i, lr in enumerate(lrs):
            y = 205 + i * 50
            d.text((700, y), f"η = {lr:g}", font=vk.font(22, mono=True), fill=TEXT)
            d.text((880, y), f"{FL[(lr, None)]:.3f}", font=vk.font(22, mono=True), fill=RED)
            d.text((1040, y), f"{FL[(lr, 1.0)]:.3f}", font=vk.font(22, mono=True), fill=GREEN)
    if seg >= 2:
        vk.text_c(d, (640, 480), "裁剪是安全网，不是万能药：太大的学习率仍然差得多", F_BODY, AMBER)


def sc_other(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "其他稳定手段")
    items = [("z-loss（PaLM, 2022）", "惩罚 log Z，防止输出数值无限变大", CYAN), ("QK-norm（Dehghani et al., 2023）", "对查询和键做归一化，防止注意力数值爆炸", VIOLET)]
    for i, (a, b, col) in enumerate(items):
        y = 170 + i * 120
        d.rounded_rectangle([100, y, 1180, y + 95], 14, outline=col, width=2, fill=(12, 19, 34))
        d.text((130, y + 30), a, font=F_BODY, fill=col, anchor="lm")
        d.text((130, y + 68), b, font=F_SMALL, fill=TEXT, anchor="lm")
    if seg >= 1:
        vk.text_c(d, (640, 480), "尖峰发生后：回退检查点 · 跳过引发问题的数据批次", F_BODY, AMBER)


def sc_bf16(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "低精度：bf16 只有 7 位小数精度")
    if seg <= 1:
        vk.text_c(d, (640, 200), "bf16：8 位指数 + 7 位尾数 · 1 附近的间隔 = 1/128", F_BODY, CYAN)
        if seg >= 1:
            vk.text_c(d, (640, 330), f"1.0 + 0.001  →  {B['one_plus']:.1f}", vk.font(40, mono=True), RED)
            vk.text_c(d, (640, 400), "小的更新被直接舍入掉", F_BODY, RED)
        return
    X = np.arange(len(B["fp32"])) * 20
    series = [("fp32 权重", B["fp32"], CYAN), ("bf16 权重", B["bf16"], RED)] + ([("bf16 计算 + fp32 主权重", B["master"], GREEN)] if seg >= 3 else [])
    tv.line_chart(d, (140, 170, 820, 580), X, series, ylog=True, ylim=(-2, 1.4), xlabel="步数", ylabel="损失（对数）", legend_xy=(880, 470),
                  xticks=[(s, str(s)) for s in (0, 1000, 2000)])
    d.text((880, 220), f"fp32：{B['fp32'][-1]:.3f}", font=F_SMALL, fill=CYAN)
    d.text((880, 260), f"bf16：停在 {B['bf16'][-1]:.2f}", font=F_SMALL, fill=RED)
    if seg >= 3:
        d.text((880, 300), f"主权重：{B['master'][-1]:.3f}", font=F_SMALL, fill=GREEN)
        d.text((880, 370), "混合精度：低精度计算", font=F_SMALL, fill=AMBER)
        d.text((880, 405), "fp32 主权重累积更新", font=F_SMALL, fill=AMBER)


EPS = ["梯度下降", "随机梯度与批量", "动量", "Adam", "学习率调度", "初始化与归一化", "μP", "Muon 等", "稳定性"]


def sc_recap(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "系列回顾")
    k = 5 if seg == 0 else 9
    for i, t in enumerate(EPS):
        x, y = 110 + (i % 5) * 215, 180 + (i // 5) * 170
        on = i < k
        d.rounded_rectangle([x, y, x + 195, y + 120], 14, outline=CYAN if on else (30, 41, 59), width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 97, y + 36), f"{i + 1:02d}", vk.font(26, mono=True), CYAN if on else MUTED)
        vk.text_c(d, (x + 97, y + 82), t, F_SMALL, TEXT if on else MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 600), "感谢观看 · Optimization 系列完", F_BODY, VIOLET)


RENDER = {"title": sc_title, "spikes": sc_spikes, "clip": sc_clip, "other": sc_other, "bf16": sc_bf16, "recap": sc_recap}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "op-stability")
