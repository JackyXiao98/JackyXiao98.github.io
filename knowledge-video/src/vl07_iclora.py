"""Video LoRA 07 — 上下文 LoRA：LTX-2 的 IC-LoRA 训练（AVControl）.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] VL_CACHE=<dir> python vl07_iclora.py <out_dir>

Real computations (vlx.ic_lora, JAX): in-context LoRA on the toy video DiT. Training pairs (source, target):
target = source with the object recoloured (red -> blue, green -> yellow, blue -> red, yellow -> green). The
source tokens are prepended in time as clean reference tokens (no noise, timestep 0, excluded from the loss;
temporal positions 0-7 like the target), all labels "none". LoRA rank 16 on all attention + MLP matrices
(135,168 parameters), Adam 1e-3, batch 16, 1,500 steps; evaluated every 100 steps on 48 held-out sources (object
colour = mapped colour, trajectory error vs. source, background PSNR). Baselines: the same reference tokens
without LoRA; regenerating from noise with the target labels.
Facts about AVControl and the LTX-2 trainer are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import vlx
import vlviz
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Video LoRA Edit"
CACHE = os.environ.get("VL_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "vlcache"))

IC = vlx.cached(CACHE, "ic", vlx.ic_lora, cache_dir=CACHE)
INV = vlx.cached(CACHE, "inv", vlx.inversion_edit, cache_dir=CACHE)
LOG = IC["log"]
ST = np.array([s for s, _ in LOG])
TRJ = np.array([m["traj"] for _, m in LOG])
BGP = np.array([m["bg_psnr"] for _, m in LOG])
SUC = np.array([m["success"] for _, m in LOG])
F, N0, L0 = IC["final"], IC["no_lora"], IC["labels_only"]
print(IC["params"], F, N0, L0, TRJ[[0, -1]], BGP[[0, -1]])
assert IC["params"] == 135168 and np.all(SUC == 1.0) and ST[0] == 100 and ST[-1] == 1500
assert abs(TRJ[0] - 0.591) < 1e-3 and abs(TRJ[-1] - 0.201) < 1e-3 and abs(BGP[0] - 24.5) < 0.05 and abs(BGP[-1] - 28.65) < 0.05
assert N0["success"] == 0.0 and abs(L0["traj"] - 3.16) < 0.01 and abs(L0["bg_psnr"] - 17.7) < 0.05 and L0["success"] == 1.0
assert abs(INV["sweep"][0.8]["edit"]["traj"] - 1.56) < 0.01

SCENES = [
    ("title", [
        "这一期介绍 LTX 视频模型的上下文 LoRA 训练，以及二零二六年三月的一篇论文，它用这种方法，为深度、姿态、相机和音频等控制，各自训练了一个 LoRA。",
    ]),
    ("idea", [
        "上下文 LoRA 的想法很简单。把参考视频，比如深度图视频，或者要编辑的原视频，先编码成潜变量，再接到要生成的目标视频的词元序列前面，一起送进变换器。",
        "参考词元和目标词元在自注意力里互相可见，但参考词元不加噪声，时间步记为零，也不计算损失。模型的结构完全不用改，需要学的只是一个 LoRA，让模型学会怎样读取参考、生成对应的目标。",
    ]),
    ("avcontrol", [
        "这篇叫作视听控制的论文，就是在 LTX 二代模型上用这种方式，把每一种控制训练成一个独立的 LoRA，包括深度、姿态、相机轨迹和音频的变换。深度图用一个视频深度估计模型从真实视频中自动生成。",
        "每种控制只需要训练两百到一万五千步。论文也提到，音频的质量受限于模型自带的声码器。",
    ]),
    ("recipe", [
        "LTX 的开源训练器给出了一套推荐的起点配置。LoRA 的秩和缩放系数都取三十二，学习率万分之一，训练两千步，批量为一，加在两种注意力的查询、键、值、输出矩阵和前馈层上。",
        "参考视频和目标视频的帧数必须相同，但参考视频可以降低分辨率，训练器会相应地调整位置编码，以节省计算。官方建议用八十吉字节显存的显卡，三十二吉字节的显卡可以用八位量化的低显存配置。",
    ]),
    ("toy", [
        f"我们在自己的小模型上复现这种训练方式。任务是给视频换颜色，红变蓝、绿变黄、蓝变红、黄变绿，物体的形状、轨迹和背景都不能变。原视频作为参考词元接在前面，训练一个秩为十六的 LoRA，共{big_zh(IC['params'])}个参数。",
        f"训练一百步时，四十八个测试视频的颜色就全部改对了。继续训练，保真度不断提高，轨迹偏差从{dec_zh(TRJ[0], 2)}个像素降到{dec_zh(TRJ[-1], 2)}个像素，背景的峰值信噪比从{dec_zh(BGP[0], 1)}分贝升到{dec_zh(BGP[-1], 1)}分贝。",
    ]),
    ("compare", [
        f"作为对照，同样接上参考词元但不加 LoRA，基础模型从没学过怎么用参考，颜色一个也没改对。只用描述从噪声重新生成，颜色虽然对，但轨迹偏差{dec_zh(L0['traj'], 2)}个像素，背景只有{dec_zh(L0['bg_psnr'], 1)}分贝。",
        f"第四期的反演方法在全部改对颜色时，轨迹偏差是{dec_zh(INV['sweep'][0.8]['edit']['traj'], 2)}个像素。上下文 LoRA 用一千五百步训练，就同时做到了改得对和保得住。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，上下文 LoRA 把参考视频作为不加噪、不算损失的词元，接在目标序列前面，模型结构不用改。",
        "第二，视听控制论文为每种控制训练一个 LoRA，只需两百到一万五千步。",
        "第三，在我们的小模型上，一千五百步的上下文 LoRA 同时做到了编辑正确和高保真。下一期，我们看把生成模型变成按指令编辑的模型。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 7, "上下文 LoRA（IC-LoRA）", "LTX-2 与 AVControl", SERIES)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "参考词元 + 目标词元，一起进变换器")
    for i in range(6):
        x = 140 + i * 80
        d.rounded_rectangle([x, 210, x + 66, 270], 8, outline=CYAN, width=2, fill=(12, 19, 34))
    for i in range(6):
        x = 660 + i * 80
        d.rounded_rectangle([x, 210, x + 66, 270], 8, outline=AMBER, width=2, fill=(40, 30, 10))
    vk.text_c(d, (370, 185), "参考（干净）", F_SMALL, CYAN)
    vk.text_c(d, (890, 185), "目标（加噪、算损失）", F_SMALL, AMBER)
    d.rounded_rectangle([140, 320, 1140, 400], 14, outline=VIOLET, width=2, fill=(12, 19, 34))
    vk.text_c(d, (640, 360), "变换器 + LoRA（双向自注意力）", F_BODY, VIOLET)
    if seg >= 1:
        for k, t in enumerate(["参考：不加噪声 · 时间步 = 0 · 不计算损失", "模型结构不变，只训练 LoRA"]):
            vk.text_c(d, (640, 480 + k * 55), t, F_BODY if k == 0 else F_SMALL, TEXT if k == 0 else GREEN)


def sc_avcontrol(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "AVControl（arXiv 2603.24793）：每种控制一个 LoRA")
    mods = [("深度", CYAN), ("姿态", VIOLET), ("相机轨迹", GREEN), ("音频变换", AMBER)]
    for i, (t, col) in enumerate(mods):
        x = 140 + i * 260
        d.rounded_rectangle([x, 200, x + 220, 280], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 110, 240), t + " LoRA", F_BODY, col)
    vk.text_c(d, (640, 330), "基础模型：LTX-2 · 深度图由 Video Depth Anything 自动生成", F_SMALL, MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 430), "每种控制训练 200 – 15,000 步", F_BODY, AMBER)
        vk.text_c(d, (640, 490), "音频质量受限于 LTX-2 的声码器", F_SMALL, MUTED)


def sc_recipe(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "LTX 训练器的推荐起点配置")
    rows = [("LoRA 秩 / alpha", "32 / 32"), ("学习率", "1e-4（AdamW）"), ("步数 / 批量", "2,000 / 1"),
            ("加在哪里", "attn1、attn2 的 q k v out + 前馈层"), ("梯度裁剪", "1.0")]
    for i, (a, b) in enumerate(rows):
        d.text((180, 165 + i * 48), a, font=F_SMALL, fill=MUTED)
        d.text((480, 165 + i * 48), b, font=F_SMALL, fill=CYAN)
    if seg >= 1:
        d.text((180, 440), "参考与目标帧数相同 · 参考可降低分辨率（位置编码相应调整）", font=F_SMALL, fill=TEXT)
        d.text((180, 490), "显存：推荐 80 GB · 32 GB 用 INT8 低显存配置", font=F_SMALL, fill=AMBER)


def sc_toy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "复现：上下文 LoRA 学“换颜色”（秩 16）")
    if seg == 0:
        for i in range(2):
            vlviz.film(c, IC["src"][i], 300, 180 + i * 200, cell=56, label="参考（原视频）", color=CYAN)
            vlviz.film(c, IC["tgt"][i], 300, 245 + i * 200, cell=56, label="目标", color=AMBER)
        return
    tv.line_chart(d, (140, 170, 760, 540), ST, [("轨迹偏差（像素）× 50", TRJ * 50, AMBER), ("背景 PSNR（dB）", BGP, CYAN)],
                  ylim=(0, 35), xlabel="训练步数", legend_xy=(820, 520), xticks=[(s, str(s)) for s in (100, 500, 1000, 1500)])
    d.text((820, 200), "颜色改对：每个检查点都是 100%", font=F_SMALL, fill=GREEN)
    d.text((820, 250), f"轨迹偏差 {TRJ[0]:.2f} → {TRJ[-1]:.2f}", font=F_SMALL, fill=AMBER)
    d.text((820, 290), f"背景 PSNR {BGP[0]:.1f} → {BGP[-1]:.1f} dB", font=F_SMALL, fill=CYAN)
    vlviz.film(c, IC["src"][0], 860, 380, cell=36, label="参考")
    vlviz.film(c, IC["samples"][0], 860, 430, cell=36, label="生成", color=GREEN)


def sc_compare(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "对比（48 个测试视频）")
    rows = [("参考词元，不加 LoRA", N0, MUTED), ("只用描述，从噪声生成", L0, RED), ("反演（强度 0.8）", INV["sweep"][0.8]["edit"], AMBER),
            ("上下文 LoRA（1,500 步）", F, GREEN)]
    n = 2 if seg == 0 else 4
    for j, h in enumerate(["方法", "颜色改对", "轨迹偏差", "背景 PSNR"]):
        d.text((140 + [0, 520, 720, 920][j], 170), h, font=F_SMALL, fill=MUTED)
    for i, (a, m, col) in enumerate(rows[:n]):
        y = 220 + i * 70
        d.rounded_rectangle([120, y - 12, 1160, y + 44], 10, outline=col, width=2, fill=(12, 19, 34))
        d.text((140, y + 16), a, font=F_SMALL, fill=col, anchor="lm")
        f = vk.font(22, mono=True)
        d.text((660, y + 16), f"{m['success']:.0%}", font=f, fill=TEXT, anchor="lm")
        d.text((860, y + 16), f"{m['traj']:.2f}", font=f, fill=TEXT, anchor="lm")
        d.text((1060, y + 16), f"{m['bg_psnr']:.1f}", font=f, fill=TEXT, anchor="lm")


def sc_summary(c, u, seg, su):
    tv.summary(c, 6, [("IC-LoRA", "参考词元：不加噪、不算损失"), ("AVControl", "每种控制一个 LoRA，200–15,000 步"),
                      ("复现", f"100% 改对 · 轨迹 {TRJ[-1]:.2f} px · {BGP[-1]:.1f} dB")], seg, "下一期：指令编辑", label_w=430)


RENDER = {"title": sc_title, "idea": sc_idea, "avcontrol": sc_avcontrol, "recipe": sc_recipe, "toy": sc_toy, "compare": sc_compare, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vl-iclora")
