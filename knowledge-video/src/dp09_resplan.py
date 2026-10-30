"""Differential Privacy 09 — 残差规划器（ResidualPlanner）：边际查询的最优高斯机制.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python dp09_resplan.py <out_dir>

Real computations on a gender (2) x income (3) table with the paper's counts
x = (200, 300, 100, 220, 260, 80): the four residual queries and their orthogonality; the exact
privacy cost and variance of every residual mechanism; the closed-form noise plan versus two
baselines and versus a general strategy optimiser (HDMM-style, L-BFGS); one private release with
exact standard deviations; residual counts and variance ratio for all <=3-way marginals of up to
14 attributes with 10 values each. Mirrors the resplan package and experiments/visualize.py in
github.com/JackyXiao98/resplan_journal.
"""

import os
import sys
from functools import reduce
from itertools import combinations
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import ImageDraw
from scipy.optimize import minimize

import tviz as tv
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL
from zhnum import big_zh, dec_zh, int_zh, pct_zh

AMBER, RED = tv.AMBER, tv.RED
SERIES = "Differential Privacy"

# ---------------------------------------------------------------------------
# 1. the table and the workload (every marginal of gender x income)
DOM = [2, 3]
COUNTS = np.array([200, 300, 100, 220, 260, 80], float)
MARGS = [(), (0,), (1,), (0, 1)]
NAMES = {(): "总数", (0,): "性别", (1,): "收入", (0, 1): "性别 × 收入"}
CELLS = ["男·低", "男·中", "男·高", "女·低", "女·中", "女·高"]


def sub_mat(k):
    """Residual (subtraction) matrix: rows e_i - e_{i+1}, orthogonal to the all-ones vector."""
    return np.eye(k)[:-1] - np.eye(k, k=1)[:-1]


def kron_full(s, mats, sum_row=True):
    return reduce(np.kron, [mats[i] if i in s else (np.ones((1, k)) if sum_row else np.ones((k, 1)) / k)
                            for i, k in enumerate(DOM)])


C = {i: sub_mat(k) for i, k in enumerate(DOM)}
RES = {s: kron_full(s, C) for s in MARGS}                    # residual query on the full table
for a in MARGS:
    for b in MARGS:
        if a != b:
            assert np.abs(RES[a] @ RES[b].T).max() < 1e-12      # different residuals are orthogonal


def marg_mat(a):
    return reduce(np.kron, [np.eye(DOM[i]) if i in a else np.ones((1, DOM[i])) for i in range(2)])


# base mechanism of residual s: y = S x + (kron_i C_i) z, z ~ N(0, sigma^2 I)
def noise_cov(s):
    return reduce(np.kron, [C[i] @ C[i].T for i in s], np.ones((1, 1)))


P = {s: np.max(np.diag(RES[s].T @ np.linalg.pinv(noise_cov(s)) @ RES[s])) for s in MARGS}   # privacy cost / sigma^-2


def recon(a, s):
    """Maps residual s's answer to the marginal on a."""
    return reduce(np.kron, [np.linalg.pinv(C[i]) if i in s else np.ones((DOM[i], 1)) / DOM[i] for i in a],
                  np.ones((1, 1)))


V = {s: sum(np.trace(recon(a, s) @ noise_cov(s) @ recon(a, s).T) for a in MARGS if set(s) <= set(a)) for s in MARGS}
assert np.allclose([V[s] for s in MARGS], [2, 4 / 3, 3, 2]) and np.allclose([P[s] for s in MARGS], [1, .5, 2 / 3, 1 / 3])
# reconstruction is exact: marginal(a) = sum_{s subset a} recon(a, s) @ residual_s
for a in MARGS:
    rebuilt = sum(recon(a, s) @ (RES[s] @ COUNTS).reshape(-1, 1) for s in MARGS if set(s) <= set(a))
    assert np.allclose(rebuilt.ravel(), marg_mat(a) @ COUNTS)

# 2. closed-form plan: min sum v_j s_j  s.t. sum p_j / s_j = 1  ->  s_j = T sqrt(p_j / v_j) / ..., T = (sum sqrt(v p))^2
VP = np.array([np.sqrt(V[s] * P[s]) for s in MARGS])
TOTAL = VP.sum() ** 2
SIG = {s: np.sqrt(TOTAL * P[s] / V[s]) for s in MARGS}
assert abs(sum(P[s] / SIG[s] for s in MARGS) - 1) < 1e-12
assert abs(sum(V[s] * SIG[s] for s in MARGS) - TOTAL) < 1e-9
DIRECT = sum(np.sqrt(marg_mat(a).shape[0]) for a in MARGS) ** 2      # noise each marginal, optimal split
IDENT = sum(np.prod(DOM) for _ in MARGS)                               # noise each cell once (var 1), sum up
W = np.vstack([marg_mat(a) for a in MARGS])
WTW = W.T @ W
MASK = np.tri(6, k=-1, dtype=bool)


def hdmm_loss(x):
    """Total variance of strategy A with A^T A = X, diag(X) = 1 (privacy cost 1)."""
    X = np.eye(6)
    X[MASK] = x
    X = X + np.tril(X, -1).T
    try:
        np.linalg.cholesky(X)
    except np.linalg.LinAlgError:
        return 1e6
    return float(np.sum(np.linalg.inv(X) * WTW))


OPT = min(minimize(hdmm_loss, np.random.default_rng(r).uniform(-.1, .1, 15), method="L-BFGS-B").fun for r in range(3))
print(f"resplan {TOTAL:.4f} direct {DIRECT:.4f} identity {IDENT} generic {OPT:.4f}")
assert abs(TOTAL - 19.90) < 0.005 and abs(DIRECT - 43.50) < 0.005 and IDENT == 24 and abs(OPT - TOTAL) < 1e-3
SAVE = 1 - TOTAL / DIRECT
assert 0.54 < SAVE < 0.55

# 3. one private release (seed 0) and the exact standard deviation of each 2-way cell
rng = np.random.default_rng(0)
NOISY = {s: RES[s] @ COUNTS + np.sqrt(SIG[s]) * reduce(np.kron, [C[i] for i in s], np.ones((1, 1))) @ rng.normal(size=int(np.prod([DOM[i] for i in s])))
         for s in MARGS}
REL = {a: sum(recon(a, s) @ NOISY[s].reshape(-1, 1) for s in MARGS if set(s) <= set(a)).ravel() for a in MARGS}
COV2 = sum(SIG[s] * recon((0, 1), s) @ noise_cov(s) @ recon((0, 1), s).T for s in MARGS)
SD2 = np.sqrt(np.diag(COV2))
assert np.allclose(SD2, SD2[0]) and abs(SD2[0] - 1.10) < 0.01
assert np.abs(REL[(0, 1)] - COUNTS).max() < 2
assert COUNTS.sum() == 1160 and COUNTS[:3].sum() - COUNTS[3:].sum() == 40

# 4. scaling: all <=3-way marginals, d attributes with n = 10 values each (closed form for identity marginals)
N_VAL, KMAX = 10, 3
DS = np.arange(2, 15)


def scale(d, n=N_VAL, kmax=KMAX):
    resid = sum(comb(d, j) * (n - 1) ** j for j in range(kmax + 1))
    vp = 0.0
    for j in range(min(kmax, d) + 1):
        p = ((n - 1) / n) ** j
        v = sum(comb(d - j, a - j) * (n - 1) ** j / n ** (a - j) for a in range(j, kmax + 1))
        vp += comb(d, j) * np.sqrt(v * p)
    direct = sum(comb(d, a) * n ** (a / 2) for a in range(kmax + 1)) ** 2
    return resid, direct / vp ** 2


# cross-check with resplan.Plan in the resplan_journal repo (experiments/visualize.py output for d = 2..5)
assert [scale(d)[0] for d in (2, 3, 4, 5)] == [100, 1000, 3439, 8146]
assert np.allclose([scale(d)[1] for d in (2, 3, 4, 5)], [1.304, 1.489, 1.569, 1.609], atol=5e-4)
RESID = np.array([scale(d)[0] for d in DS], float)
RATIO = np.array([scale(d)[1] for d in DS])
assert RESID[-1] == 272854 and abs(RATIO[-1] - 1.70) < 0.005 and abs(RATIO[0] - 1.30) < 0.005

SCENES = [
    ("title", [
        "这一期是差分隐私系列的进阶篇。我们介绍残差规划器，一个为边际查询设计的高斯机制。它用一个闭式公式，就能给出理论上最优的加噪方案。",
    ]),
    ("problem", [
        "先看一个小例子。一张表按性别和收入，把一千一百六十个人分成六个格子。",
        "我们想用差分隐私公开这张表的所有边际：总人数，按性别的人数，按收入的人数，以及完整的六个格子。",
        f"最直接的做法，是给每个边际分别加噪声。在隐私代价等于一的条件下，四个边际的方差加起来是{dec_zh(DIRECT)}。",
        f"另一种做法，是只给六个格子各加一次噪声，再把格子加起来得到其他边际，总方差是{int_zh(IDENT)}。能不能做得更好？",
    ]),
    ("decompose", [
        "残差规划器的第一步，是把边际拆成残差查询。属性集合的每一个子集，对应一个残差。",
        "空集对应总人数。性别对应男女之差，收入对应相邻两档之差，性别乘收入对应差的差。",
        "这四组查询两两正交，互不重叠，所以可以分别独立地测量，各自加各自的噪声。",
        "任何一个边际，都能由它的子集对应的残差精确地还原出来。比如按性别的人数，就由总数和男女之差还原。",
    ]),
    ("plan", [
        "第二步，决定每个残差加多大的噪声。每个残差有两个系数，一个衡量它对总方差的贡献，一个衡量它消耗的隐私代价。",
        "在总隐私代价固定的条件下，最小化总方差，由柯西不等式可以直接写出最优解，不需要迭代求解。",
        f"在这个例子里，最优的总方差是{dec_zh(TOTAL)}，比给每个边际分别加噪声少了{pct_zh(SAVE)}。",
        "我们再用一个通用的矩阵机制优化器，在全部六个格子上直接搜索最优策略，得到的方差也是十九点九零。闭式解就是最优解。",
    ]),
    ("reconstruct", [
        "第三步是重建。把带噪声的残差，用伪逆映射回各个边际，再把同一个边际的各项相加。",
        f"这是一次真实的发布。六个格子的误差都在两个人以内，而且每个格子的标准差可以精确算出，约为{dec_zh(SD2[0])}。",
    ]),
    ("scale", [
        "残差规划器真正的优势在于规模。它从不需要构造完整的数据表。",
        f"假设有十四个属性，每个属性有十个取值，我们要公开所有三维以内的边际。完整的表有十的十四次方个格子，而残差规划器只需要测量{big_zh(int(RESID[-1]))}个数。",
        f"同时，给每个边际分别加噪声的总方差，是残差规划器的{dec_zh(RATIO[-1], 1)}倍。属性越多，优势越明显。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，每个边际都可以拆成若干正交的残差，每个残差独立测量。",
        "第二，每个残差的噪声大小有闭式最优解，方差和通用优化器找到的最优值相同。",
        "第三，它只处理残差，不构造完整的表，能扩展到很多属性。代码和可视化放在了视频下方的链接里，感谢大家的观看。",
    ]),
]


def sc_title(c, u, seg, su):
    tv.title_card(c, 9, "残差规划器 ResidualPlanner", "边际查询的最优高斯机制 · 拆分 · 规划 · 重建", SERIES)


def table(d, x, y, vals, title, col=CYAN, w=110, h=56, fmt="{:.0f}"):
    d.text((x, y - 34), title, font=F_SMALL, fill=MUTED)
    for g in range(2):
        d.text((x - 14, y + g * h + h / 2), "男女"[g], font=F_SMALL, fill=MUTED, anchor="rm")
        for i in range(3):
            if g == 0:
                d.text((x + i * w + w / 2, y + 2 * h + 18), "低中高"[i], font=F_SMALL, fill=MUTED, anchor="mm")
            box = [x + i * w, y + g * h, x + (i + 1) * w - 6, y + (g + 1) * h - 6]
            d.rounded_rectangle(box, 8, outline=col, width=2, fill=(12, 19, 34))
            d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), fmt.format(vals[g * 3 + i]),
                   font=vk.font(24, mono=True), fill=TEXT, anchor="mm")


def sc_problem(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "要公开的：一张表的所有边际")
    table(d, 120, 170, COUNTS, "人数（性别 × 收入）")
    if seg >= 1:
        rows = [("总数", f"{COUNTS.sum():.0f}"), ("性别", "600 · 560"), ("收入", "420 · 560 · 180"), ("性别 × 收入", "6 个格子")]
        for j, (k, v) in enumerate(rows):
            y = 400 + j * 44
            d.text((120, y), k, font=F_SMALL, fill=CYAN)
            d.text((300, y), v, font=F_SMALL, fill=TEXT)
    if seg >= 2:
        labels, vals, cols = ["每个边际分别加噪"], [DIRECT], [MUTED]
        if seg >= 3:
            labels, vals, cols = labels + ["每格加噪后求和"], vals + [IDENT], cols + [VIOLET]
        tv.bar_chart(d, (640, 170, 1160, 450), labels, vals, cols, fmt="{:.2f}", vmax=50)
        vk.text_c(d, (900, 525), "总方差（隐私代价 = 1，越低越好）", F_SMALL, MUTED)


def sc_decompose(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "第一步：拆成正交的残差查询")
    boxes = [((), (80, 150, 560, 190)), ((0,), (700, 150, 1180, 190)), ((1,), (80, 330, 560, 410)), ((0, 1), (700, 330, 1180, 410))]
    labels = {(): "空集 → 总数", (0,): "性别 → 男 - 女", (1,): "收入 → 相邻两档之差", (0, 1): "性别 × 收入 → 差的差"}
    show = 4 if seg >= 1 else 1
    for s, box in boxes[:show]:
        d.text((box[0], box[1] - 30), labels[s], font=F_SMALL, fill=CYAN)
        tv.heat(d, box, RES[s] + 0.0, vmin=-1, vmax=1, fmt="{:.0f}", font_size=18)
    for j, cell in enumerate(CELLS):
        for x0 in (80, 700):
            d.text((x0 + (j + .5) * 80, 425), cell, font=vk.font(18), fill=MUTED, anchor="mm")
    if seg == 2:
        vk.text_c(d, (560, 500), "不同残差的行两两正交：", F_BODY, GREEN)
        tv.fpaste(c, r"$R_a R_b^{\top}=0$", 820, 500, 28)
        vk.text_c(d, (640, 550), "→ 每个残差独立测量，各加各的噪声", F_SMALL, MUTED)
    if seg >= 3:
        tv.fpaste(c, r"$m_{\mathrm{gender}}=\frac{1160}{2}\,(1,\ 1)+\frac{40}{2}\,(1,\,-1)=(600,\ 560)$", 640, 530, 26)


def sc_plan(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "第二步：闭式最优的噪声规划")
    tv.fpaste(c, r"$\min_{\sigma^2}\ \sum_j v_j\,\sigma_j^2\quad\mathrm{s.t.}\quad\sum_j \frac{p_j}{\sigma_j^2}=1$", 360, 150, 26)
    if seg == 0:
        rows = [(NAMES[s], f"v = {V[s]:.2f}", f"p = {P[s]:.2f}") for s in MARGS]
        for j, (a, b, e) in enumerate(rows):
            y = 240 + j * 60
            d.text((120, y), a, font=F_BODY, fill=CYAN)
            d.text((340, y), b, font=vk.font(24, mono=True), fill=TEXT)
            d.text((500, y), e, font=vk.font(24, mono=True), fill=TEXT)
    if seg >= 1:
        tv.fpaste(c, r"$\sigma_j^2\ \propto\ \sqrt{p_j/v_j},\qquad \mathrm{Var}_{\min}=\Big(\sum_j\sqrt{v_j\,p_j}\Big)^2$".replace(r"\Big", ""), 360, 230, 24)
        tv.bar_chart(d, (110, 330, 620, 500), [NAMES[s] for s in MARGS], [SIG[s] for s in MARGS], [CYAN] * 4, fmt="{:.2f}", vmax=4)
        d.text((110, 300), "每个残差的噪声方差 σ²", font=F_SMALL, fill=MUTED)
    if seg >= 2:
        labels = ["每个边际", "每个格子", "残差规划"] + (["通用优化"] if seg >= 3 else [])
        vals = [DIRECT, IDENT, TOTAL] + ([OPT] if seg >= 3 else [])
        tv.bar_chart(d, (680, 150, 1220, 450), labels, vals, [MUTED, MUTED, CYAN, VIOLET][:len(vals)], fmt="{:.2f}", vmax=50)
        vk.text_c(d, (950, 515), "总方差（隐私代价 = 1）", F_SMALL, MUTED)
        if seg >= 3:
            vk.text_c(d, (950, 555), "闭式解 = 数值最优", F_BODY, GREEN)


def sc_reconstruct(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "第三步：用伪逆重建，再相加")
    tv.fpaste(c, r"$\hat m_A=\sum_{S\subseteq A}\ \Big(\bigotimes_{i\in S} C_i^{+}\ \otimes\bigotimes_{i\in A\setminus S}\frac{\mathbf{1}}{k_i}\Big)\ y_S$".replace(r"\Big", ""), 640, 145, 26)
    if seg >= 1:
        table(d, 120, 270, COUNTS, "真实人数", col=MUTED)
        table(d, 760, 270, REL[(0, 1)], "一次私密发布（隐私代价 = 1）", col=CYAN, fmt="{:.1f}")
        err = REL[(0, 1)] - COUNTS
        d.text((300, 500), "误差", font=F_BODY, fill=AMBER, anchor="lm")
        d.text((390, 500), "  ".join(f"{e:+.1f}" for e in err), font=vk.font(24, mono=True), fill=AMBER, anchor="lm")
        vk.text_c(d, (640, 550), f"每个格子的标准差（精确计算）= {SD2[0]:.2f}", F_SMALL, GREEN)


def sc_scale(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "扩展到很多属性：所有 ≤3 维边际，每个属性 10 个取值")
    if seg >= 1:
        n = max(2, int(len(DS) * vk.ease(su * 1.4))) if seg == 1 else len(DS)
        tv.line_chart(d, (110, 150, 640, 540), DS, [("完整的表 10^d 个格子", 10.0 ** DS, AMBER), ("残差答案的个数", RESID, CYAN)],
                      ylog=True, xlabel="属性个数 d（纵轴为对数）", upto=n, legend_xy=(130, 165),
                      xticks=[(x, str(x)) for x in (2, 6, 10, 14)])
        if seg >= 1 and n == len(DS):
            d.text((660, 200), "d = 14:", font=F_SMALL, fill=MUTED)
            tv.fpaste(c, r"$10^{14}$", 700, 255, 26, "#fbbf24")
            d.text((745, 255), "个格子", font=F_BODY, fill=AMBER, anchor="lm")
            d.text((660, 290), f"{int(RESID[-1]):,} 个残差答案", font=F_BODY, fill=CYAN)
    if seg >= 2:
        tv.line_chart(d, (820, 425, 1170, 560), DS, [("方差倍数", RATIO, GREEN)], ylim=(1, 1.9), xlabel="d",
                      xticks=[(2, "2"), (14, "14")])
        d.text((820, 350), "分别加噪的方差 ÷ 残差规划器", font=F_SMALL, fill=GREEN)
        d.text((820, 385), f"{RATIO[0]:.2f}× → {RATIO[-1]:.2f}×", font=vk.font(24, mono=True), fill=GREEN)


def sc_summary(c, u, seg, su):
    tv.summary(c, 6, [("拆分", "边际 = 各子集的正交残差之和"), ("规划", "噪声有闭式最优解，等于数值最优"),
                      ("规模", "只处理残差，不构造完整的表")], seg, "代码：github.com/JackyXiao98/resplan_journal", label_w=400)


RENDER = {"title": sc_title, "problem": sc_problem, "decompose": sc_decompose, "plan": sc_plan,
          "reconstruct": sc_reconstruct, "scale": sc_scale, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "resplan")
