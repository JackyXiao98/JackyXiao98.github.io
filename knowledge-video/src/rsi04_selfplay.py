"""Recursive Self-Improvement 04 — 自我博弈：从 AlphaGo Zero 说起.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] RSI_CACHE=<dir> python rsi04_selfplay.py <out_dir>

Real computations (rsix.tictactoe_selfplay): a tabular tic-tac-toe player that learns afterstate
values only by playing against itself (TD updates, 10% exploration); after 0 ... 60,000 self-play
games it plays 400 games against a perfect minimax player and 400 against a random player.
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

H = rsix.cached(CACHE, "ttt", rsix.tictactoe_selfplay)
G = [h["games"] for h in H]
by = {h["games"]: h for h in H}
print(H)
assert by[0]["p_loss"] > 0.85 and by[10000]["p_loss"] < 0.31 and by[30000]["p_loss"] <= 0.02
assert by[60000]["p_loss"] == 0 and by[60000]["p_draw"] == 1 and by[60000]["r_loss"] == 0 and by[0]["r_loss"] > 0.25

SCENES = [
    ("title", [
        "上一期，模型用自己生成的数据训练自己。这一期看另一种经典的自我改进方式，和自己对弈。",
    ]),
    ("alphazero", [
        "二零一七年，阿尔法围棋零完全不看人类棋谱，只和自己下棋，从随机下法开始学习，最后以一百比零战胜了击败李世石的那个版本。",
        "自我博弈的妙处在于，对手永远和自己一样强。自己变强了，对手也随之变强，训练难度自动跟着提高。",
    ]),
    ("setup", [
        "我们用井字棋来实际做一遍。玩家只记住一张表，表里是每个局面对自己有多好。",
        "一开始所有局面的评分都一样，玩家基本是乱下。它和自己对弈，每下完一步，就用下一个局面的评分修正上一个局面的评分，这叫时间差分学习。",
    ]),
    ("progress", [
        "为了衡量实力，我们让它和一个永远不犯错的完美玩家各下四百局。井字棋双方都不犯错的话，结果一定是平局。",
        f"训练之前，它输掉了{pct_zh(by[0]['p_loss'])}的棋局。自我对弈一万局以后，输棋的比例降到{pct_zh(by[10000]['p_loss'])}，三万局时只有{pct_zh(by[30000]['p_loss'])}。",
        f"到六万局时，四百局全部是平局，一局都没有输，已经和完美玩家一样强了。面对随机下棋的对手，胜率也从{pct_zh(by[0]['r_win'])}升到{pct_zh(by[60000]['r_win'])}。整个过程没有用到任何人类的棋谱。",
    ]),
    ("limits", [
        "自我博弈有它的前提。规则必须清楚，胜负必须能自动判定，这相当于一个完美的检验器。",
        "在围棋、国际象棋这样的游戏里，这些前提都成立。但在写代码、做研究这样的开放任务里，什么算赢往往说不清楚，这就是下一期要讨论的验证问题。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，自我博弈让对手和自己一起变强，训练难度自动提升。",
        "第二，在井字棋上，只靠和自己对弈六万局，就学会了从不输棋。",
        "第三，自我博弈依赖清楚的规则和自动的胜负判定。下一期，我们看验证者怎样驱动改进。感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 4, "自我博弈", "从 AlphaGo Zero 说起", SERIES)


def sc_alphazero(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "AlphaGo Zero（Silver et al., 2017）")
    rows = [("不用人类棋谱", "从随机下法开始，只和自己对弈", CYAN), ("结果", "100 : 0 战胜 AlphaGo Lee", GREEN)]
    for i, (a, b, col) in enumerate(rows):
        d.text((200, 180 + i * 70), a, font=F_BODY, fill=col)
        d.text((480, 180 + i * 70), b, font=F_BODY, fill=TEXT)
    if seg >= 1:
        cx, cy = 640, 470
        d.ellipse([cx - 200, cy - 80, cx - 40, cy + 80], outline=CYAN, width=3)
        d.ellipse([cx + 40, cy - 80, cx + 200, cy + 80], outline=VIOLET, width=3)
        vk.text_c(d, (cx - 120, cy), "自己", F_BODY, CYAN)
        vk.text_c(d, (cx + 120, cy - 14), "对手", F_BODY, VIOLET)
        vk.text_c(d, (cx + 120, cy + 20), "= 自己", F_SMALL, VIOLET)
        vk.arrow(d, (cx - 40, cy - 20), (cx + 40, cy - 20), MUTED, 2)
        vk.arrow(d, (cx + 40, cy + 20), (cx - 40, cy + 20), MUTED, 2)


def board(d, x, y, cells, s=50):
    for i in range(1, 3):
        d.line([(x + i * s, y), (x + i * s, y + 3 * s)], fill=MUTED, width=2)
        d.line([(x, y + i * s), (x + 3 * s, y + i * s)], fill=MUTED, width=2)
    for k, v in enumerate(cells):
        cx, cy = x + (k % 3) * s + s / 2, y + (k // 3) * s + s / 2
        if v == 1:
            d.line([(cx - 14, cy - 14), (cx + 14, cy + 14)], fill=CYAN, width=4)
            d.line([(cx - 14, cy + 14), (cx + 14, cy - 14)], fill=CYAN, width=4)
        elif v == -1:
            d.ellipse([cx - 15, cy - 15, cx + 15, cy + 15], outline=VIOLET, width=4)


def sc_setup(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "井字棋：一张“局面评分表”")
    board(d, 180, 170, [1, 0, -1, 0, 1, 0, 0, 0, 0])
    vk.text_c(d, (255, 345), "评分 V(局面)", F_SMALL, AMBER)
    if seg >= 1:
        tv.fpaste(c, r"$V(s)\leftarrow V(s)+\alpha\,[\,V(s')-V(s)\,]$", 820, 230, 32)
        vk.text_c(d, (820, 300), "下一局面 s′ 的评分修正上一局面 s（时间差分学习）", F_SMALL, TEXT)
        vk.text_c(d, (820, 340), "初始所有评分 = 0.5；10% 的步数随机探索", F_SMALL, MUTED)


def sc_progress(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "对完美玩家（400 局）")
    k = len(H) if seg >= 1 else 1
    xs = np.arange(len(H))
    loss = [h["p_loss"] * 100 for h in H]
    tv.line_chart(d, (130, 170, 700, 530), xs, [("输棋比例", loss, RED), ("平局比例", [h["p_draw"] * 100 for h in H], GREEN)],
                  ylim=(0, 100), upto=k, xlabel="自我对弈的局数", ylabel="%", legend_xy=(520, 280),
                  xticks=[(i, f"{g // 1000}K" if g >= 1000 else str(g)) for i, g in enumerate(G)])
    if seg >= 2:
        d.text((820, 200), "对随机玩家", font=F_SMALL, fill=MUTED)
        for i, (lab, h) in enumerate([("训练前", by[0]), ("6 万局后", by[60000])]):
            y = 250 + i * 90
            d.text((820, y), lab, font=F_SMALL, fill=TEXT)
            d.text((820, y + 30), f"胜 {h['r_win']:.0%} · 负 {h['r_loss']:.0%}", font=F_BODY, fill=CYAN if i else MUTED)
        vk.text_c(d, (980, 470), "6 万局后：对完美玩家", F_BODY, GREEN)
        vk.text_c(d, (980, 510), "400 局全部平局", F_BODY, GREEN)
        vk.text_c(d, (980, 560), "没有用到任何人类棋谱", F_SMALL, MUTED)


def sc_limits(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "自我博弈的前提")
    for i, t in enumerate(["规则清楚", "胜负可自动判定 = 完美的检验器"]):
        vk.text_c(d, (640, 200 + i * 60), f"前提 {i + 1}：{t}", F_BODY, GREEN)
    if seg >= 1:
        vk.text_c(d, (640, 380), "开放任务（写代码、做研究）：“赢”往往说不清楚", F_BODY, AMBER)
        vk.text_c(d, (640, 440), "→ 下一期：验证者", F_BODY, TEXT)


def sc_summary(c, u, seg, su):
    tv.summary(c, 5, [("自我博弈", "对手和自己一起变强"), ("井字棋", "6 万局自我对弈 → 从不输棋"),
                      ("前提", "清楚的规则 + 自动的胜负判定")], seg, "下一期：验证比生成容易")


RENDER = {"title": sc_title, "alphazero": sc_alphazero, "setup": sc_setup, "progress": sc_progress, "limits": sc_limits,
          "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "rsi-selfplay")
