"""Recursive Self-Improvement 08 — 2026 年的最新进展：自动化 AI 研究.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] RSI_CACHE=<dir> TF_CACHE=<dir> python rsi08_latest.py <out_dir>

Facts about 2026 work (survey of Chen, Wang & Qu; AIDE2 by Srikanth et al. at Weco AI; OpenAI's
research-intern milestone) are quoted with references in the notes; the only computed numbers are
the episode-05 best-of-32 results, reused to illustrate the survey's verification hierarchy.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw

import rsix
import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Recursive Self-Improvement"
CACHE = os.environ.get("RSI_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "rsicache"))
TF_CACHE = os.environ.get("TF_CACHE", os.path.join(os.path.dirname(__file__), "..", "out", "tfcache"))

V = rsix.cached(CACHE, "ver", rsix.verifier_bestofn, tf_cache=TF_CACHE)
VP, VW, VM = V["perfect"][-1], V["weak"][-1], V["majority"][-1]
assert VP > VW > VM

# reported figures (see notes)
N_PAPERS, RUN_DAYS, N_ACCEPTED, N_HELDOUT = 1250, 8, 7, 4
HACK0, HACK1 = 55, 32

SCENES = [
    ("title", [
        "前面几期讲了自我改进的基本方法和失效模式。这一期看二零二六年的研究走到了哪里，重点是让人工智能自动做人工智能研究。",
    ]),
    ("survey", [
        f"二零二六年七月的一篇综述，整理了二零二四到二零二六年间约{big_zh(N_PAPERS)}篇关于人工智能自我改进的论文。",
        "它区分了两类工作。一类是有界的自我完善，依靠固定的外部评估器，会收敛，已经在工业界广泛使用；另一类是开放式的递归自我改进，仍然受到数据来源、模型崩溃和算力的限制。",
        "它还按照闭环的程度分成三级，人在回路之中，人在回路之上，以及完全闭环。",
    ]),
    ("hierarchy", [
        "综述里一个重要的观察是验证的层级。最强的信号来自形式化的验证器，比如证明检查和单元测试；最弱的是模型对自己的主观评价。已经被证实的自我改进强度，大致和这个层级一致。",
        f"这和我们第五期的实验完全吻合。同样采样三十二次，完美检验的准确率是{pct_zh(VP)}，只检查个位数是{pct_zh(VW)}，没有检验只靠投票是{pct_zh(VM)}。",
    ]),
    ("aide", [
        "二零二六年九月的一项工作，让一个智能体去改写另一个人工智能研究智能体的框架代码，也就是控制它怎样搜索、管理上下文和检查结果的代码。",
        f"在一次持续{int_zh(RUN_DAYS)}天的自动运行中，系统接受了{int_zh(N_ACCEPTED)}次改写。每一次都必须在被改写的智能体看不到的隐藏数据上更好，才会被接受。",
        f"改进后的智能体在{int_zh(N_HELDOUT)}个没有参与挑选的基准上也更好，其中包括天气预报这样完全不同的任务。同时，在另一组任务上，钻奖励空子的比例从百分之{int_zh(HACK0)}降到了百分之{int_zh(HACK1)}。",
        "作者自己的表述很谨慎，这证明了框架的净改进，但还没有证明改进者本身在持续变强。",
    ]),
    ("industry", [
        "产业界也在推进。二零二六年九月，开放人工智能公司宣布达到了自动化研究实习生的目标，系统能在人的指导下，完成熟练研究员需要几天的明确任务。它的下一个目标，是在二零二八年三月前实现自动化的人工智能研究员。",
        "综合来看，自我改进的循环正在变强，但还不能自我持续，大部分环节仍然需要人来设定目标和把关。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，有界的自我完善已经实用，开放式的递归自我改进仍受数据、崩溃和算力限制。",
        "第二，验证信号越可靠，自我改进越强，最新系统都依赖隐藏的独立评估。",
        "第三，自动化人工智能研究正在快速推进，但还离不开人的把关。最后一期，我们讨论怎样让自我改进安全可控。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 8, "2026 年的最新进展", "自动化 AI 研究", SERIES)


def sc_survey(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "综述（Chen, Wang & Qu, 2026）")
    vk.text_c(d, (640, 150), f"约 {N_PAPERS:,} 篇论文（2024–2026）", F_BODY, TEXT)
    if seg >= 1:
        for i, (a, b, col) in enumerate([("有界的自我完善", "固定外部评估器 · 收敛 · 已在工业界使用", GREEN),
                                         ("开放式 RSI", "受数据来源、模型崩溃、算力限制", AMBER)]):
            x = 130 + i * 520
            d.rounded_rectangle([x, 210, x + 500, 310], 14, outline=col, width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 250, 245), a, F_BODY, col)
            vk.text_c(d, (x + 250, 285), b, F_SMALL, TEXT)
    if seg >= 2:
        for i, t in enumerate(["人在回路之中", "人在回路之上", "完全闭环"]):
            x = 220 + i * 300
            d.rounded_rectangle([x, 400, x + 240, 460], 12, outline=[CYAN, VIOLET, RED][i], width=2, fill=(12, 19, 34))
            vk.text_c(d, (x + 120, 430), t, F_BODY, [CYAN, VIOLET, RED][i])
        vk.text_c(d, (640, 500), "闭环程度（× 4 类改进对象）", F_SMALL, MUTED)


def sc_hierarchy(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "验证的层级")
    levels = [("形式化验证器（证明检查、单元测试）", GREEN), ("外部评估 / 测试集", CYAN), ("其他模型评判", AMBER), ("自我评价", RED)]
    for i, (t, col) in enumerate(levels):
        w = 520 - i * 90
        d.rectangle([370 - w / 2, 150 + i * 60, 370 + w / 2, 200 + i * 60], fill=(12, 19, 34), outline=col, width=2)
        vk.text_c(d, (370, 175 + i * 60), t, F_SMALL, col)
    d.text((60, 155), "强", font=F_BODY, fill=GREEN)
    d.text((60, 335), "弱", font=F_BODY, fill=RED)
    vk.text_c(d, (370, 410), "两端来自综述；中间两层为示意", F_SMALL, MUTED)
    if seg >= 1:
        tv.bar_chart(d, (700, 170, 1180, 450), ["完美检验", "只查个位", "投票"], [VP * 100, VW * 100, VM * 100], [GREEN, AMBER, RED],
                     fmt="{:.0f}%", vmax=105)
        vk.text_c(d, (940, 520), "第 5 期实验：best-of-32 的准确率", F_SMALL, MUTED)


def sc_aide(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "AIDE²（Srikanth et al., Weco AI, 2026）")
    d.rounded_rectangle([140, 150, 520, 230], 12, outline=VIOLET, width=2, fill=(12, 19, 34))
    vk.text_c(d, (330, 190), "外层智能体：改写框架代码", F_SMALL, VIOLET)
    vk.arrow(d, (525, 190), (655, 190), MUTED, 3)
    d.rounded_rectangle([660, 150, 1140, 230], 12, outline=CYAN, width=2, fill=(12, 19, 34))
    vk.text_c(d, (900, 190), "研究智能体：搜索 · 上下文 · 结果检查", F_SMALL, CYAN)
    if seg >= 1:
        vk.text_c(d, (640, 290), f"自动运行 {RUN_DAYS} 天 · 接受 {N_ACCEPTED} 次改写", F_BODY, TEXT)
        vk.text_c(d, (640, 330), "每次都必须在被改写者看不到的隐藏数据上更好", F_SMALL, AMBER)
    if seg >= 2:
        vk.text_c(d, (400, 420), f"{N_HELDOUT} 个未参与挑选的基准上更好", F_BODY, GREEN)
        vk.text_c(d, (400, 455), "（含天气预报等分布外任务）", F_SMALL, MUTED)
        vk.text_c(d, (900, 420), f"钻奖励空子：{HACK0}% → {HACK1}%", F_BODY, GREEN)
    if seg >= 3:
        vk.text_c(d, (640, 560), "作者：框架有净改进；尚未证明“改进者本身在持续变强”", F_BODY, AMBER)


def sc_industry(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "产业界：自动化 AI 研究")
    pts = [("2026 年 9 月", "自动化研究实习生：在人的指导下完成几天的明确任务（已宣布达到）", GREEN),
           ("2028 年 3 月（目标）", "自动化 AI 研究员", AMBER)]
    for i, (a, b, col) in enumerate(pts):
        y = 180 + i * 110
        d.ellipse([170, y + 10, 190, y + 30], fill=col)
        d.text((220, y), a, font=F_BODY, fill=col)
        d.text((220, y + 40), b, font=F_SMALL, fill=TEXT)
    d.line([(180, 220), (180, 300)], fill=MUTED, width=2)
    vk.text_c(d, (640, 430), "OpenAI（2026 年 9 月公告，据报道）", F_SMALL, MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 520), "循环在变强，但还不能自我持续；目标设定与把关仍需人", F_BODY, AMBER)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("两类", "有界完善已实用；开放式 RSI 受限"), ("验证", "信号越可靠改进越强；隐藏评估"),
                      ("现状", "自动化研究推进中，仍需人把关")], seg, "最后一期：安全与可控")


RENDER = {"title": sc_title, "survey": sc_survey, "hierarchy": sc_hierarchy, "aide": sc_aide, "industry": sc_industry,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rsi-latest")
