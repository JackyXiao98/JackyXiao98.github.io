"""Long Context 08 — 评估：大海捞针与更难的测试.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] LC_CACHE=<dir> python lc08_eval.py <out_dir>

Real computations (lcx.exp_needle): mean retrieval accuracy over the five depths at the training length 128 and at
length 512 for the RoPE / ALiBi / no-position models ("claimed" vs "effective" length in miniature).
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import lcx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Long Context"
CACHE = os.environ.get("LC_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "lccache"))

ND = lcx.cached(CACHE, "needle", lcx.exp_needle)
MEAN = {k: {T: float(np.mean(ND[k][T])) for T in (128, 512)} for k in ND}
print(MEAN)
assert all(MEAN[k][128] > 0.99 for k in MEAN) and round(MEAN["rope"][512], 2) == 0.91 and round(MEAN["alibi"][512], 2) == 0.88

SCENES = [
    ("title", [
        "模型宣称支持十几万个词元，怎样检验它是不是真的能用？这一期看长上下文的评估：最有名的大海捞针测试，它为什么不够，以及更难的测试。",
    ]),
    ("needle", [
        "大海捞针的做法和我们的检索实验一样：在一大段无关的文本里插入一句话，比如一个特定的数字，再问模型这个数字是多少。改变文本的长度和插入的深度，画出一张正确率的网格图。",
        f"在我们的小实验里，三个模型在训练长度上平均都接近百分之百；到了四倍长度，旋转位置编码平均是{pct_zh(MEAN['rope'][512])}，线性偏置是{pct_zh(MEAN['alibi'][512])}。宣称的长度和有效的长度之间，已经有了差距。",
    ]),
    ("easy", [
        "但大海捞针对真实的大模型来说太容易了。插入的那句话和周围的文本格格不入，模型只要找到这个异常就行，不需要理解整段文本。很多模型在这个测试上几乎全绿，却在更实际的长文本任务上表现平平。",
    ]),
    ("ruler", [
        "二零二四年的一个基准测试设计了更难的任务：插入多根针并要求全部找出，在多个相似的干扰项中选出正确的那个，沿着一串变量赋值多跳追踪，以及统计整段文本中最常出现的词。这些任务需要真正读完并整合整段上下文。",
        "测试发现，几乎所有模型的成绩都随长度明显下降，很多宣称支持三万两千甚至更长上下文的模型，真正能保持良好表现的有效长度要短得多。",
    ]),
    ("real", [
        "更贴近应用的评估，会用真实的长文档问答、长篇摘要、整个代码仓库的修改等任务。评估长上下文时，最好同时看三件事：能不能找到信息，能不能整合多处信息，以及在短任务上有没有退化。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，大海捞针检验的是能否找到插入的信息，在我们的实验里，四倍长度就出现了差距。",
        "第二，它对大模型太容易，不需要理解上下文。",
        "第三，多针、多跳、聚合类的测试更能反映有效上下文长度。最后一期，我们看系统层面的方法。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 8, "评估", "大海捞针与更难的测试", SERIES)


def sc_needle(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "大海捞针：长度 × 深度的正确率网格")
    names = {"rope": "RoPE", "alibi": "ALiBi", "nope": "无位置编码"}
    for row, k in enumerate(("rope", "alibi", "nope")):
        y = 210 + row * 120
        d.text((120, y + 40), names[k], font=F_SMALL, fill=TEXT)
        for ti, T in enumerate((128, 512)):
            for j, acc in enumerate(ND[k][T]):
                x = 320 + ti * 420 + j * 75
                col = GREEN if acc > 0.97 else (AMBER if acc > 0.85 else RED)
                d.rectangle([x, y, x + 65, y + 80], fill=col)
                d.text((x + 32, y + 40), f"{acc * 100:.0f}", font=vk.font(18, mono=True), fill=(10, 14, 22), anchor="mm")
    d.text((320, 180), "长度 128（训练长度）· 深度 0 → 1", font=F_SMALL, fill=MUTED)
    d.text((740, 180), "长度 512 · 深度 0 → 1", font=F_SMALL, fill=MUTED)


def sc_easy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "对大模型太容易")
    for i, (t, col) in enumerate([("插入的句子与周围格格不入：只需找到异常", AMBER), ("不需要理解整段文本", RED), ("全绿 ≠ 会用长上下文", CYAN)]):
        vk.text_c(d, (640, 230 + i * 80), t, F_BODY, col)


def sc_ruler(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "RULER（Hsieh et al., 2024）")
    items = [("多根针", CYAN), ("相似干扰项", VIOLET), ("多跳追踪", AMBER), ("聚合统计", GREEN)]
    for i, (t, col) in enumerate(items):
        x = 140 + i * 260
        d.rounded_rectangle([x, 200, x + 220, 280], 14, outline=col, width=2, fill=(12, 19, 34))
        vk.text_c(d, (x + 110, 240), t, F_BODY, col)
    if seg >= 1:
        vk.text_c(d, (640, 400), "几乎所有模型随长度明显下降", F_BODY, RED)
        vk.text_c(d, (640, 470), "有效上下文长度 远短于 宣称长度", F_BODY, AMBER)


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "评估长上下文：三件事")
    for i, (t, col) in enumerate([("能不能找到信息", CYAN), ("能不能整合多处信息", GREEN), ("短任务有没有退化", AMBER)]):
        vk.text_c(d, (640, 230 + i * 80), t, F_BODY, col)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("大海捞针", "4 倍长度已出现差距"), ("太容易", "只需找到异常"),
                      ("更难的测试", "多针、多跳、聚合")], seg, "最后一期：系统与前沿")


RENDER = {"title": sc_title, "needle": sc_needle, "easy": sc_easy, "ruler": sc_ruler, "real": sc_real, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "lc-eval")
