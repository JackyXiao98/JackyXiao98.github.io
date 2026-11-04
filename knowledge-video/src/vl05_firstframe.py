"""Video LoRA 05 — 首帧编辑 + 传播：用图生视频模型做编辑.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] VL_CACHE=<dir> python vl05_firstframe.py <out_dir>

Real computations (vlx.first_frame_edit, JAX): the same 48 source clips as episode 04. The first frame is edited
exactly (object pixels recoloured to the target colour) and given as clean conditioning (image-to-video); the
model generates frames 1-7 with labels (source shape, source motion, colour = none) or with the target colour
label. Metrics as in episode 04, plus the share of videos whose object colour equals the target in each frame.
Facts about real systems are quoted with references in the notes.
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

FF = vlx.cached(CACHE, "first", vlx.first_frame_edit, cache_dir=CACHE)
I = vlx.cached(CACHE, "inv", vlx.inversion_edit, cache_dir=CACHE)
PN, PT = np.array(FF["label_none_per_frame"]), np.array(FF["label_target_per_frame"])
N, TG = FF["label_none"], FF["label_target"]
print(PN, PT, N, TG)
assert PN[0] == 1.0 and abs(PN[3] - 0.75) < 1e-3 and abs(PN[6] - 0.3125) < 1e-3 and abs(PN[7] - 0.375) < 1e-3 and np.all(PT == 1.0)
assert abs(N["success"] - 0.729) < 1e-3 and abs(N["traj"] - 1.83) < 0.01 and abs(TG["traj"] - 1.63) < 0.01
assert abs(N["bg_psnr"] - 22.6) < 0.05 and abs(I["fresh"]["traj"] - 3.43) < 0.01

SCENES = [
    ("title", [
        "上一期的反演方法，用文字描述来指定修改。但很多修改用一张图来表达更准确。这一期看另一条路线，先编辑第一帧，再让图生视频模型把修改传播到整段视频。",
    ]),
    ("idea", [
        "图像编辑工具已经非常成熟，改颜色、换物体、加东西都很容易。于是可以先用它们把第一帧改好，再交给一个图生视频模型，让它以改好的第一帧为起点，生成后面的帧。",
        "代表性的方法有二零二四年的任意视频到视频的方法，它不需要训练，还会把原视频反演得到的特征注入进去，让运动跟着原视频走。",
    ]),
    ("experiment", [
        "我们的小模型在训练时，有一部分样本把第一帧当作已知条件，所以它本身就是一个图生视频模型。我们把四十八个原视频的第一帧里的物体改成新的颜色，告诉模型形状和运动方向，但不告诉它颜色，看修改能不能传下去。",
        f"轨迹偏差是{dec_zh(N['traj'], 2)}个像素，比从噪声重新生成的{dec_zh(I['fresh']['traj'], 2)}个像素好得多，因为第一帧固定了物体的起始位置。",
    ]),
    ("decay", [
        f"但看每一帧，第一帧当然全对，第二帧还有{pct_zh(PN[1])}，到第四帧降到{pct_zh(PN[3])}，最后两帧只有约三成到四成还保持新颜色。离第一帧越远，修改越容易被模型自己的习惯冲掉。",
        "如果同时把新颜色也写进描述里，每一帧都能保持。可见首帧传播依赖模型能从第一帧推断出修改的含义，修改越特殊、视频越长，越容易漂移。",
    ]),
    ("limits", [
        "这条路线还有一个根本的限制，第一帧之后发生的事情很难控制。如果要修改的物体在第三秒才出现，或者希望后面的某几帧有不同的变化，只改第一帧就无能为力了。",
        "二零二五年的一项工作正是从这个问题出发，用一个针对原视频训练的、带掩码的 LoRA，来控制第一帧之后的内容，这是下一期的主题。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，首帧编辑把视频编辑转化成图像编辑加图生视频，可以利用成熟的图像编辑工具。",
        "第二，在我们的实验里，首帧固定了物体的位置，但离第一帧越远，修改越容易丢失。",
        "第三，第一帧之后的内容难以控制，需要更强的手段。下一期，我们看掩码感知的 LoRA。感谢大家的观看。",
    ]),
]
assert round(PN[1] * 100) == 98 and round(PN[3] * 100) == 75


def sc_title(c, u, seg, su):
    tv.title_card(c, 5, "首帧编辑 + 传播", "用图生视频模型做编辑", SERIES)


def sc_idea(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "先编辑第一帧，再传播")
    steps = [("原视频第一帧", CYAN), ("图像编辑工具", AMBER), ("改好的第一帧", GREEN), ("图生视频模型", VIOLET), ("编辑后的视频", GREEN)]
    for i, (t, col) in enumerate(steps):
        x = 70 + i * 235
        d.rounded_rectangle([x, 220, x + 200, 300], 12, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 100, 260), t, F_SMALL, col)
        if i < 4:
            vk.arrow(d, (x + 203, 260), (x + 232, 260), MUTED, 2)
    if seg >= 1:
        vk.text_c(d, (640, 420), "AnyV2V（Ku et al., 2024）：免训练 · 注入原视频反演得到的特征", F_BODY, TEXT)
        vk.text_c(d, (640, 480), "I2VEdit（2024）等", F_SMALL, MUTED)


def sc_experiment(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "实验：改第一帧的颜色，不告诉模型新颜色")
    for i in range(2):
        y = 160 + i * 220
        vlviz.film(c, FF["src"][i], 260, y, cell=56, label="原视频")
        vlviz.film(c, FF["edited_first"][i], 260, y + 64, cell=56, frames=[0], label="改好的首帧", color=AMBER)
        vlviz.film(c, FF["label_none_samples"][i], 260, y + 128, cell=56, label="生成", color=GREEN)
    if seg >= 1:
        d.text((860, 620), f"轨迹偏差：首帧传播 {N['traj']:.2f} · 重新生成 {I['fresh']['traj']:.2f}", font=F_SMALL, fill=GREEN, anchor="mm")


def sc_decay(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "每一帧还保持新颜色的比例（48 个视频）")
    series = [("描述里不写颜色", PN * 100, AMBER)] + ([("描述里写上新颜色", PT * 100, GREEN)] if seg >= 1 else [])
    tv.line_chart(d, (160, 170, 820, 560), np.arange(1, 9), series, ylim=(0, 105), xlabel="第几帧", ylabel="%", legend_xy=(180, 470),
                  xticks=[(k, str(k)) for k in range(1, 9)])
    d.text((880, 230), f"第 2 帧：{PN[1]:.0%}", font=F_BODY, fill=AMBER)
    d.text((880, 280), f"第 4 帧：{PN[3]:.0%}", font=F_BODY, fill=AMBER)
    d.text((880, 330), f"第 7 帧：{PN[6]:.0%}", font=F_BODY, fill=AMBER)
    d.text((880, 400), "离第一帧越远，越容易漂移", font=F_SMALL, fill=RED)


def sc_limits(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "根本限制：第一帧之后难以控制")
    for i, t in enumerate(["物体第 3 秒才出现？", "希望后面几帧有不同的变化？"]):
        vk.text_c(d, (640, 200 + i * 60), t, F_BODY, RED)
    if seg >= 1:
        vk.text_c(d, (640, 420), "LoRA-Edit（Gao et al., 2025）：针对原视频训练、带掩码的 LoRA", F_BODY, GREEN)
        vk.text_c(d, (640, 480), "→ 下一期", F_SMALL, MUTED)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("首帧编辑", "图像编辑 + 图生视频"), ("实验", f"第 2 帧 {PN[1]:.0%} → 第 7 帧 {PN[6]:.0%}"),
                      ("限制", "第一帧之后难以控制")], seg, "下一期：掩码感知的 LoRA")


RENDER = {"title": sc_title, "idea": sc_idea, "experiment": sc_experiment, "decay": sc_decay, "limits": sc_limits, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "vl-firstframe")
