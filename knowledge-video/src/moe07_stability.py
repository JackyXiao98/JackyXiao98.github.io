"""MoE 07 — 训练稳定性：路由器 z-loss 与精度.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] MOE_CACHE=<dir> python moe07_stability.py <out_dir>

Real computations (moex.exp_stability, JAX):
- the MoE of episode 01 (8 x 128, top-1, auxiliary loss 0.01) at a higher learning rate 1e-2, without and with a
  router z-loss mean(logsumexp(logits)^2) of weight 1e-3, 2,000 steps: largest |router logit| on held-out data and
  final loss;
- bfloat16 rounding of router logits: the largest absolute rounding error for values in [m - 1, m] (m = 16.4, 11.8,
  2.5) and the resulting relative error exp(err) - 1 of an un-normalised softmax weight;
- upcycling: a dense model (FFN 128, 2,000 steps, loss 1.794) continued for 1,000 steps (lr 1e-3) either as the
  same dense model or as a MoE whose 8 experts are copies of its FFN (fresh router); and a MoE trained from scratch
  for 1,000 steps (lr 3e-3).
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

ST = moex.cached(CACHE, "stability", moex.exp_stability)
HI, HZ, UP, DC, SCR = ST["lr1e-2"], ST["lr1e-2_z"], ST["upcycle"], ST["dense_cont"], ST["moe_scratch"]
BASE = moex.cached(CACHE, "scale", moex.exp_scale)["moe8"]
STEPS = [h["step"] for h in HI["hist"]]
ML = {k: [h["maxlogit"] for h in r["hist"]] for k, r in (("hi", HI), ("z", HZ))}


def bf16_err(m):
    import jax.numpy as jnp
    x = np.linspace(m - 1, m, 10001).astype(np.float32)
    b = np.asarray(jnp.asarray(x).astype(jnp.bfloat16).astype(jnp.float32))
    e = float(np.abs(b - x).max())
    return e, float(np.exp(e) - 1)


BF = {m: bf16_err(m) for m in (16.4, 11.8, 2.5)}
U0 = UP["hist"][0]["loss"]
print(round(ML["hi"][-1], 1), round(ML["z"][-1], 1), round(BASE["hist"][-1]["maxlogit"], 1), round(HI["final"], 4), round(HZ["final"], 4),
      BF, ST["dense_start"], round(U0, 3), round(UP["final"], 4), round(DC["final"], 4), round(SCR["final"], 4), round(UP["hist"][-1]["load_max"], 2))
assert round(ML["hi"][-1], 1) == 16.4 and round(ML["z"][-1], 1) == 11.8 and round(BASE["hist"][-1]["maxlogit"], 1) == 9.8
assert round(HI["final"], 2) == 1.71 and round(HZ["final"], 2) == 1.71 and abs(HI["final"] - HZ["final"]) < 0.002
assert abs(BF[16.4][0] - 0.0625) < 1e-6 and round(BF[16.4][1], 3) == 0.064 and round(BF[11.8][1], 3) == 0.032 and round(BF[2.5][1], 3) == 0.008
assert round(ST["dense_start"], 2) == 1.79 and round(U0, 2) == 2.96 and round(UP["final"], 2) == 1.75 and round(DC["final"], 2) == 1.76
assert round(SCR["final"], 2) == 1.79 and UP["final"] < DC["final"] < SCR["final"] and round(UP["hist"][-1]["load_max"], 1) == 4.0

SCENES = [
    ("title", [
        "大规模训练混合专家模型时，人们经常遇到损失突然飙升、训练崩掉的情况，而且比稠密模型更常见。这一期看路由器为什么容易出问题，以及两个实用的技巧：给路由器分数加一个惩罚，以及从稠密模型升级。",
    ]),
    ("logits", [
        "问题常常出在路由器的输出上。路由器的分数会在训练中不断变大，因为分数越大，柔性最大值越接近一个专家独占，路由也越确定。",
        f"我们把学习率提高到零点零一。训练结束时，路由器分数的最大绝对值达到{dec_zh(ML['hi'][-1], 1)}，而第一期正常学习率下是{dec_zh(BASE['hist'][-1]['maxlogit'], 1)}。",
    ]),
    ("precision", [
        "分数大了为什么是问题？大规模训练常用十六位的脑浮点数，它的有效数字只有八个二进制位。数越大，能表示的数之间的间隔就越大。",
        f"在十六附近，舍入误差最大有零点零六二五，取指数以后，相当于概率有约{pct_zh(BF[16.4][1], 1)}的相对误差；在十二附近约{pct_zh(BF[11.8][1], 1)}；而在二点五附近只有{pct_zh(BF[2.5][1], 1)}。分数越大，路由的概率越不准，梯度也越不准。",
    ]),
    ("zloss", [
        "稀疏专家变换器的论文提出了一个路由器惩罚项，论文里用一个字母命名，我们叫它分数惩罚：把每个词元的路由分数先做对数求和指数，再平方，作为一个惩罚项。它直接把分数往小里拉。",
        f"加上权重千分之一的分数惩罚，最大分数从{dec_zh(ML['hi'][-1], 1)}降到{dec_zh(ML['z'][-1], 1)}，而最终损失几乎一样，都是{dec_zh(HZ['final'], 2)}。",
        "要说明的是，我们的小模型在这个学习率下并没有真的崩掉，所以这里只能看到分数变小，看不到它挽救一次崩溃。在大模型上，论文报告这个惩罚明显减少了训练的不稳定。",
    ]),
    ("upcycle", [
        "另一个降低训练难度的办法是升级，也就是不从头训练混合专家，而是从一个训练好的稠密模型出发，把它的前馈网络复制成每一个专家，只有路由器是新的。",
        f"我们拿一个训练了两千步、损失{dec_zh(ST['dense_start'], 2)}的稠密模型。升级后再训练一千步，损失降到{dec_zh(UP['final'], 2)}；同样的一千步继续训练稠密模型，只到{dec_zh(DC['final'], 2)}；而从头训练一千步的混合专家只有{dec_zh(SCR['final'], 2)}。",
        f"有两个细节。一开始损失反而跳到{dec_zh(U0, 2)}，因为专家的输出被路由概率缩小到约八分之一，两百步左右就恢复了。另外，复制出来的专家一开始完全一样，路由器没有理由区分它们，所以负载很不均匀，最忙的专家是平均的四倍。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，路由器的分数会越训越大，在低精度下会让概率和梯度变得不准。",
        "第二，路由器的分数惩罚把分数拉小，在我们的实验里不影响最终的损失。",
        "第三，从稠密模型升级，比从头训练更快达到好的效果。下一期，我们看这么多专家在硬件上怎么放。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 7, "训练稳定性", "路由器 z-loss 与精度", SERIES)


def sc_logits(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "路由器的分数越训越大")
    ser = [("学习率 0.003", [h["maxlogit"] for h in BASE["hist"]], MUTED)]
    if seg >= 1:
        ser.append(("学习率 0.01", ML["hi"], RED))
    tv.line_chart(d, (120, 200, 760, 580), STEPS, ser, ylim=(0, 18), xlabel="训练步数",
                  xticks=[(0, "0"), (1000, "1000"), (2000, "2000")], legend_xy=(820, 230))
    d.text((120, 165), "最大 |路由分数|", font=F_SMALL, fill=MUTED)
    vk.text_c(d, (980, 420), "分数越大 → softmax 越接近独占", F_SMALL, TEXT)


def sc_precision(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "bfloat16：有效数字只有 8 个二进制位")
    vk.text_c(d, (640, 190), "数越大，相邻可表示数的间隔越大", F_BODY, TEXT)
    if seg >= 1:
        f = vk.font(24, mono=True)
        d.text((260, 270), "分数附近", font=F_SMALL, fill=MUTED)
        d.text((560, 270), "最大舍入误差", font=F_SMALL, fill=MUTED)
        d.text((860, 270), "概率的相对误差", font=F_SMALL, fill=MUTED)
        for i, m in enumerate((2.5, 11.8, 16.4)):
            y = 330 + i * 70
            col = [GREEN, AMBER, RED][i]
            d.rounded_rectangle([220, y - 12, 1100, y + 44], 10, outline=col, width=2, fill=(12, 19, 34))
            d.text((270, y + 16), f"{m}", font=f, fill=col, anchor="lm")
            d.text((580, y + 16), f"{BF[m][0]:.4f}", font=f, fill=TEXT, anchor="lm")
            d.text((900, y + 16), f"{BF[m][1]:.1%}", font=f, fill=col, anchor="lm")


def sc_zloss(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "路由器 z-loss（ST-MoE）")
    tv.fpaste(c, r"$\mathcal{L}_z=\frac{1}{N}\sum_{n}\left(\log\sum_{i} e^{x_{n,i}}\right)^2$", 640, 200, 40)
    if seg >= 1:
        ser = [("学习率 0.01", ML["hi"], RED), ("+ z-loss 0.001", ML["z"], GREEN)]
        tv.line_chart(d, (120, 300, 700, 620), STEPS, ser, ylim=(0, 18), xlabel="训练步数",
                      xticks=[(0, "0"), (1000, "1000"), (2000, "2000")], legend_xy=(760, 320))
        d.text((120, 268), "最大 |路由分数|", font=F_SMALL, fill=MUTED)
        d.text((760, 420), f"最终损失 {HI['final']:.3f} vs {HZ['final']:.3f}", font=F_SMALL, fill=TEXT)
    if seg >= 2:
        d.text((760, 480), "我们的小模型没有真的崩掉：", font=F_SMALL, fill=AMBER)
        d.text((760, 515), "只看到分数变小", font=F_SMALL, fill=AMBER)


def sc_upcycle(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "从稠密模型升级（sparse upcycling）")
    if seg == 0:
        d.rounded_rectangle([140, 260, 400, 360], 14, outline=MUTED, width=2, fill=(12, 19, 34))
        vk.text_c(d, (270, 310), "训练好的 FFN", F_SMALL, TEXT)
        vk.arrow(d, (420, 310), (500, 310), MUTED, 3)
        for i in range(8):
            x = 530 + i * 80
            d.rounded_rectangle([x, 270, x + 64, 350], 10, outline=VIOLET, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 32, 310), "复制", F_SMALL, VIOLET)
        vk.text_c(d, (850, 420), "只有路由器是新的", F_BODY, AMBER)
        return
    xs = [h["step"] for h in UP["hist"]]
    ser = [("升级为 MoE", np.minimum([h["loss"] for h in UP["hist"]], 3.0), GREEN),
           ("继续训练稠密", [h["loss"] for h in DC["hist"]], MUTED),
           ("从头训练 MoE", np.minimum([h["loss"] for h in SCR["hist"]], 3.0), RED)]
    tv.line_chart(d, (120, 200, 720, 580), xs, ser, ylim=(1.7, 3.05), xlabel="再训练的步数",
                  xticks=[(0, "0"), (500, "500"), (1000, "1000")], legend_xy=(780, 220))
    d.text((120, 165), f"验证损失（起点：稠密模型 {ST['dense_start']:.3f}）", font=F_SMALL, fill=MUTED)
    f = vk.font(22, mono=True)
    for i, (n, r, col) in enumerate([("升级", UP, GREEN), ("稠密", DC, MUTED), ("从头", SCR, RED)]):
        d.text((780, 340 + i * 42), n, font=F_SMALL, fill=col)
        d.text((880, 340 + i * 42), f"{r['final']:.3f}", font=f, fill=TEXT)
    if seg >= 2:
        d.text((780, 490), f"第 0 步跳到 {U0:.2f}（输出 × 约 1/8）", font=F_SMALL, fill=AMBER)
        d.text((780, 530), f"专家相同 → 最大负载 {UP['hist'][-1]['load_max']:.1f} × 平均", font=F_SMALL, fill=AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("分数变大", "低精度下概率与梯度不准"), ("z-loss", "把分数拉小，不伤损失"),
                      ("升级", "从稠密模型出发更快")], seg, "下一期：并行与推理")


RENDER = {"title": sc_title, "logits": sc_logits, "precision": sc_precision, "zloss": sc_zloss, "upcycle": sc_upcycle, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "moe-stability")
