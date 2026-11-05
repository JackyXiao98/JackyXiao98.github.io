"""Scaling Laws 03 — 数据量的缩放与过拟合.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] SCL_CACHE=<dir> SHAKESPEARE=<file> python scl03_data.py <out_dir>

Real computations (sclx.exp_data, JAX): the width-64 model (98,304 non-embedding parameters) trained 3,000 steps
(6.1 M characters) on the first 10k / 30k / 100k / 300k / 1.0 M characters of the training set (so 614 ... 6 passes
over the data); training and validation loss every 150 steps; final validation loss and the best validation loss
along the way (what early stopping would keep).
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import sclx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Scaling Laws"
CACHE = os.environ.get("SCL_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "sclcache"))

R = sclx.cached(CACHE, "data", sclx.exp_data)
NS = sorted(R)
FIN = {n: R[n]["val"] for n in NS}
TRN = {n: R[n]["hist"][-1]["train"] for n in NS}
BEST = {n: min(h["val"] for h in R[n]["hist"]) for n in NS}
EPOCHS = {n: 3000 * 32 * 64 / n for n in NS}
H100 = R[100_000]["hist"]
print(FIN, TRN, BEST, EPOCHS)
assert [round(FIN[n], 2) for n in NS] == [6.28, 4.55, 2.41, 2.02, 1.70] and round(TRN[10_000], 2) == 0.14
assert [round(BEST[n], 2) for n in NS] == [2.82, 2.5, 2.27, 2.02, 1.7] and round(EPOCHS[10_000]) == 614

SCENES = [
    ("title", [
        "上一期我们看到，大模型开始过拟合。这一期专门看数据：固定模型的大小，改变训练数据的多少，看损失怎样变化。",
    ]),
    ("setup", [
        "我们用宽度六十四的模型，参数约十万。训练数据分别只取前一万、三万、十万、三十万个字符，和全部约一百万个字符。每个都训练三千步，读过的字符总数一样，所以数据越少，重复的遍数越多：一万个字符要重复六百多遍。",
    ]),
    ("curves", [
        f"看训练过程。只有一万个字符时，训练损失一路降到{dec_zh(TRN[10_000], 2)}，模型几乎把训练文本背了下来；但验证损失在开头短暂下降之后一路上升，最后高达{dec_zh(FIN[10_000], 2)}，比完全不训练还差。",
        "十万个字符时，验证损失先降后升，中间有一个最低点。只有用全部数据时，验证损失到最后仍在下降。",
    ]),
    ("scaling", [
        f"如果在验证损失最低时停下来，最好的结果随数据量增加而改善：一万个字符是{dec_zh(BEST[10_000], 2)}，十万个是{dec_zh(BEST[100_000], 2)}，全部数据是{dec_zh(BEST[NS[-1]], 2)}。数据越多越好，和参数量一样，也接近一条幂律。",
        "而如果不提前停下来，数据少的时候，训练越久反而越糟。这说明模型和数据要一起增长：模型太大、数据太少，多出来的参数只会用来死记硬背。",
    ]),
    ("real", [
        "缩放定律的论文把参数量和数据量放进同一个公式，给出了一个经验规则：要避免过拟合的惩罚，数据量大致要随参数量的零点七四次方增长。后来的大模型训练，往往只把数据过一遍到几遍，过拟合不是主要问题，问题变成了在固定的计算下，参数和数据该怎样分配。这是下一期的内容。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，固定模型时，数据越多，能达到的最好损失越低。",
        "第二，数据太少时模型会死记硬背，训练越久验证损失越高。",
        "第三，模型和数据要一起增长。下一期，我们看计算最优的分配。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 3, "数据量的缩放", "与过拟合", SERIES)


def sc_setup(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "宽度 64 · 3,000 步 · 只改训练数据量")
    f = vk.font(24, mono=True)
    d.text((200, 180), "训练字符数", font=F_SMALL, fill=MUTED)
    d.text((560, 180), "重复遍数", font=F_SMALL, fill=MUTED)
    for i, n in enumerate(NS):
        y = 230 + i * 60
        d.text((200, y), f"{n:,}", font=f, fill=CYAN)
        d.text((560, y), f"{EPOCHS[n]:.0f}", font=f, fill=RED if EPOCHS[n] > 50 else TEXT)


def sc_curves(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "训练损失与验证损失")
    show = [10_000] if seg == 0 else [10_000, 100_000, NS[-1]]
    cols = {10_000: RED, 100_000: AMBER, NS[-1]: GREEN}
    ser = []
    for n in show:
        h = R[n]["hist"]
        ser.append((f"{n:,} 验证", np.minimum([x["val"] for x in h], 6.5), cols[n]))
        if n == 10_000:
            ser.append((f"{n:,} 训练", [x["train"] for x in h], CYAN))
    xs = [h["step"] for h in R[10_000]["hist"]]
    tv.line_chart(d, (140, 200, 820, 580), xs, ser, ylim=(0, 6.6), xlabel="训练步数", xticks=[(0, "0"), (1500, "1500"), (3000, "3000")], legend_xy=(880, 220))
    d.text((140, 165), "损失", font=F_SMALL, fill=MUTED)


def sc_scaling(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "提前停止时的最好验证损失 vs 数据量")
    tv.line_chart(d, (140, 200, 820, 580), [float(n) for n in NS], [("最好（提前停止）", [BEST[n] for n in NS], GREEN), ("训练到底", np.minimum([FIN[n] for n in NS], 6.5), RED)],
                  xlog=True, ylim=(1.5, 6.6), xticks=[(1e4, "1e4"), (1e5, "1e5"), (1e6, "1e6")], xlabel="训练字符数（对数）", legend_xy=(880, 220))
    f = vk.font(22, mono=True)
    for i, n in enumerate(NS):
        d.text((880, 330 + i * 40), f"{n:>9,}  {BEST[n]:.2f}", font=f, fill=TEXT)
    if seg >= 1:
        vk.text_c(d, (1030, 580), "模型与数据要一起增长", F_SMALL, AMBER)


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "真实模型")
    vk.text_c(d, (640, 220), "Kaplan et al.（2020）：避免过拟合，D 约随 N^0.74 增长", F_BODY, CYAN)
    vk.text_c(d, (640, 320), "大模型通常只把数据过 1 – 几遍：过拟合不是主要问题", F_SMALL, TEXT)
    vk.text_c(d, (640, 400), "→ 固定计算下，参数与数据怎样分配？", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("数据越多", "最好损失越低"), ("数据太少", "死记硬背，越训越差"),
                      ("一起增长", "模型与数据")], seg, "下一期：计算最优")


RENDER = {"title": sc_title, "setup": sc_setup, "curves": sc_curves, "scaling": sc_scaling, "real": sc_real, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "scl-data")
