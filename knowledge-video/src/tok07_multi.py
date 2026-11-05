"""Tokenization 07 — 多语言的不公平.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] TOK_DATA=<dir> TOK_CACHE=<dir> python tok07_multi.py <out_dir>

Real computations (tokx.exp_multi): a byte-level BPE (2,000 merges) learned on Tiny Shakespeare + Tang poems
(English and Chinese only) encodes the same sentence ("The weather is nice today, so let's go for a walk in the
park.") translated into Chinese, Spanish, Russian, Japanese, Arabic and Hindi (our own translations); characters,
UTF-8 bytes and tokens per sentence, and the token ratio to English.
Facts about real systems are quoted with references in the notes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import tokx
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Tokenization"
CACHE = os.environ.get("TOK_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "tokcache"))

M = tokx.cached(CACHE, "multi", tokx.exp_multi)
ZH = {"English": "英语", "中文": "中文", "Español": "西班牙语", "Русский": "俄语", "日本語": "日语", "العربية": "阿拉伯语", "हिन्दी": "印地语"}
ORDER = sorted(M, key=lambda k: M[k]["tokens"])
EN = M["English"]["tokens"]
RAT = {k: M[k]["tokens"] / EN for k in M}
print({k: (M[k]["chars"], M[k]["bytes"], M[k]["tokens"], round(RAT[k], 2)) for k in ORDER})
assert EN == 24 and M["中文"]["tokens"] == 21 and M["Español"]["tokens"] == 37 and M["Русский"]["tokens"] == 86
assert M["日本語"]["tokens"] == 49 and M["العربية"]["tokens"] == 79 and M["हिन्दी"]["tokens"] == 121 and round(RAT["हिन्दी"], 2) == 5.04
assert all(M[k]["tokens"] == M[k]["bytes"] for k in ("Русский", "العربية", "हिन्दी"))

SCENES = [
    ("title", [
        "同一句话，用不同的语言说，会被切成多少个词元？答案可能差好几倍。这一期看分词带来的多语言不公平。",
    ]),
    ("setup", [
        "我们用第五期的字节级分词器，它只在英文的莎士比亚和中文的唐诗上学过。然后把同一句话，今天天气很好，我们去公园散步吧，翻译成七种语言，看每种要多少个词元。",
    ]),
    ("result", [
        f"英语用了{int_zh(EN)}个词元，中文{int_zh(M['中文']['tokens'])}个，都很省，因为分词器见过这两种语言。西班牙语用拉丁字母，能借用很多英语的片段，用了{int_zh(M['Español']['tokens'])}个；日语和中文共用汉字，用了{int_zh(M['日本語']['tokens'])}个。",
        f"而俄语、阿拉伯语和印地语，分词器完全没见过，只能退回到一个字节一个词元。俄语用了{int_zh(M['Русский']['tokens'])}个，阿拉伯语{int_zh(M['العربية']['tokens'])}个，印地语{int_zh(M['हिन्दी']['tokens'])}个，是英语的五倍多。",
    ]),
    ("cost", [
        "词元多意味着什么？按词元收费的接口，同样的内容要多付好几倍的钱；同样长的上下文窗口，能装下的内容少了好几倍；生成同样的回答，要多走好几倍的步数。模型在这些语言上学到的每个词元，承载的信息也更少。",
    ]),
    ("real", [
        "真实的商用分词器训练语料大得多、语言也多得多，但问题依然存在。二零二三年的一项研究比较了多种分词器，发现同样内容的文本，在不同语言之间的词元数最多相差约十五倍。",
        "缓解的办法包括：用更平衡的多语言语料训练分词器，扩大词表让更多语言的常用词成为单独的词元。近年的大模型把词表扩大到十几万甚至二十几万，部分就是为了多语言。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，分词器在没见过的语言上会退回字节，词元数可能是英语的好几倍。",
        "第二，在我们的实验里，印地语用的词元是英语的五倍多。",
        "第三，这直接带来成本、上下文和速度上的不公平。最后一期，我们看能不能干脆不分词。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 7, "多语言的不公平", "同一句话，不同的词元数", SERIES)


def sc_setup(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "同一句话，七种语言（字节级 BPE，只在英文和中文上训练）")
    for i, k in enumerate(["English", "中文", "Español", "Русский", "日本語"]):
        s = tokx.PARALLEL[k]
        d.text((120, 180 + i * 64), ZH[k], font=F_SMALL, fill=MUTED)
        d.text((300, 180 + i * 64), s, font=F_SMALL, fill=TEXT)
    d.text((120, 180 + 5 * 64), "阿拉伯语、印地语", font=F_SMALL, fill=MUTED)
    d.text((300, 180 + 5 * 64), "（同一句话的译文，字体所限不在画面上显示）", font=F_SMALL, fill=MUTED)


def sc_result(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "每句的词元数")
    ks = ORDER
    show = [k for k in ks if seg >= 1 or k in ("English", "中文", "Español", "日本語")]
    cols = {k: (GREEN if RAT[k] < 1.1 else (AMBER if RAT[k] < 2.5 else RED)) for k in ks}
    tv.bar_chart(d, (110, 220, 860, 580), [ZH[k] for k in ks], [M[k]["tokens"] if k in show else 0 for k in ks], [cols[k] for k in ks], fmt="{:.0f}", vmax=135)
    if seg >= 1:
        f = vk.font(22, mono=True)
        d.text((920, 220), "相对英语", font=F_SMALL, fill=MUTED)
        for i, k in enumerate(ks):
            d.text((920, 260 + i * 42), ZH[k], font=F_SMALL, fill=cols[k])
            d.text((1080, 260 + i * 42), f"{RAT[k]:.1f}×", font=f, fill=TEXT)
        vk.text_c(d, (480, 630), "没见过的文字：一个字节一个词元", F_SMALL, RED)


def sc_cost(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "词元多意味着")
    for i, (t, col) in enumerate([("按词元收费：多付几倍的钱", RED), ("上下文窗口：装下的内容少几倍", AMBER),
                                  ("生成：多走几倍的步数", VIOLET), ("每个词元承载的信息更少", CYAN)]):
        vk.text_c(d, (640, 210 + i * 80), t, F_BODY, col)


def sc_real(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "真实分词器")
    vk.text_c(d, (640, 220), "Petrov et al.（NeurIPS 2023）：同样内容，不同语言的词元数最多相差约 15 倍", F_SMALL, RED)
    if seg >= 1:
        vk.text_c(d, (640, 330), "更平衡的多语言语料 · 更大的词表（十几万到二十几万）", F_BODY, GREEN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("退回字节", "没见过的语言词元数成倍增加"), ("实验", "印地语是英语的 5 倍多"),
                      ("不公平", "成本、上下文与速度")], seg, "最后一期：不分词行不行")


RENDER = {"title": sc_title, "setup": sc_setup, "result": sc_result, "cost": sc_cost, "real": sc_real, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "tok-multi")
