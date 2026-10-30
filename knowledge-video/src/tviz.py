"""Drawing helpers for the Transformer series: tokens, matrices, line and bar charts."""

import numpy as np
from PIL import ImageDraw

import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN

RED = (248, 113, 113)
AMBER = (251, 191, 36)


def fpaste(c, tex, cx, cy, size=28, color="#e2e8f0"):
    vk.paste_center(c, vk.formula(tex, size, color), cx, cy)


def token(d, x, y, text, col=CYAN, w=None, h=44, font=None, fill=(12, 19, 34)):
    """Rounded token box with its top-left at (x, y); returns its width."""
    font = font or vk.font(24, mono=True)
    w = w or max(44, int(font.getlength(text)) + 22)
    d.rounded_rectangle([x, y, x + w, y + h], 8, outline=col, width=2, fill=fill)
    d.text((x + w / 2, y + h / 2), text, font=font, fill=col, anchor="mm")
    return w


def tokens_row(d, x, y, texts, col=CYAN, gap=8, font=None):
    for t in texts:
        x += token(d, x, y, t, col, font=font) + gap
    return x


def heat(d, box, M, vmin=None, vmax=None, labels_x=None, labels_y=None, fmt=None, cmap="cyan", font_size=18):
    """Matrix heatmap in box; optional row/column labels and numbers in cells."""
    x0, y0, x1, y1 = box
    n, m = M.shape
    cw, ch = (x1 - x0) / m, (y1 - y0) / n
    lo = np.nanmin(M) if vmin is None else vmin
    hi = np.nanmax(M) if vmax is None else vmax
    base = np.array(vk.BG, float)
    col = np.array({"cyan": CYAN, "violet": VIOLET, "green": GREEN}[cmap], float)
    f = vk.font(font_size, mono=True)
    for i in range(n):
        for j in range(m):
            v = M[i, j]
            if np.isnan(v):
                rgb = (20, 26, 38)
            else:
                t = 0 if hi == lo else (v - lo) / (hi - lo)
                rgb = tuple(int(c) for c in base + (col - base) * (0.12 + 0.88 * t))
            d.rectangle([x0 + j * cw, y0 + i * ch, x0 + (j + 1) * cw - 2, y0 + (i + 1) * ch - 2], fill=rgb)
            if fmt and not np.isnan(v):
                t = 0 if hi == lo else (v - lo) / (hi - lo)
                d.text((x0 + (j + .5) * cw, y0 + (i + .5) * ch), fmt.format(v), font=f,
                       fill=(10, 14, 22) if t > 0.55 else TEXT, anchor="mm")
    fl = vk.font(20)
    for j, lab in enumerate(labels_x or []):
        d.text((x0 + (j + .5) * cw, y0 - 16), lab, font=fl, fill=MUTED, anchor="mm")
    for i, lab in enumerate(labels_y or []):
        d.text((x0 - 10, y0 + (i + .5) * ch), lab, font=fl, fill=MUTED, anchor="rm")


def line_chart(d, box, xs, series, xlog=False, ylog=False, xlabel=None, ylabel=None, upto=None,
               xticks=None, ylim=None, legend_xy=None):
    """series: list of (name, ys, color). upto: number of points to draw (animation)."""
    x0, y0, x1, y1 = box
    d.rectangle(box, outline=(30, 41, 59))
    X = np.log10(xs) if xlog else np.asarray(xs, float)
    allys = np.concatenate([np.asarray(s[1], float) for s in series])
    Yall = np.log10(np.maximum(allys, 1e-12)) if ylog else allys
    lo, hi = (ylim if ylim else (Yall.min(), Yall.max()))
    if hi == lo:
        hi = lo + 1
    px = lambda x: x0 + (x - X.min()) / (X.max() - X.min()) * (x1 - x0)
    py = lambda y: y1 - (y - lo) / (hi - lo) * (y1 - y0)
    n = len(xs) if upto is None else max(1, min(len(xs), upto))
    for j, (name, ys, col) in enumerate(series):
        Y = np.log10(np.maximum(np.asarray(ys, float), 1e-12)) if ylog else np.asarray(ys, float)
        pts = [(px(X[i]), py(Y[i])) for i in range(n)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=3)
        for p in pts[-1:]:
            d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=col)
        if legend_xy:
            d.text((legend_xy[0], legend_xy[1] + j * 28), "● " + name, font=vk.F_SMALL, fill=col)
    for xt, lab in (xticks or []):
        d.text((px(np.log10(xt) if xlog else xt), y1 + 18), lab, font=vk.font(18, mono=True), fill=MUTED, anchor="mm")
    if xlabel:
        d.text((x1, y1 + 44), xlabel, font=vk.F_SMALL, fill=MUTED, anchor="rm")
    if ylabel:
        d.text((x0 + 8, y0 + 8), ylabel, font=vk.F_SMALL, fill=MUTED)


def bar_chart(d, box, labels, values, colors, fmt="{:.2f}", vmax=None):
    x0, y0, x1, y1 = box
    k = len(values)
    gw = (x1 - x0) / k
    vmax = vmax or max(values) * 1.15
    d.line([(x0, y1), (x1, y1)], fill=(71, 85, 105), width=2)
    for i, (lab, v, col) in enumerate(zip(labels, values, colors)):
        bx = x0 + i * gw + gw * 0.2
        top = y1 - v / vmax * (y1 - y0)
        d.rectangle([bx, top, bx + gw * 0.6, y1], fill=col)
        d.text((bx + gw * 0.3, top - 16), fmt.format(v), font=vk.font(20, mono=True), fill=col, anchor="mm")
        d.text((bx + gw * 0.3, y1 + 22), lab, font=vk.F_SMALL, fill=MUTED, anchor="mm")


def summary(c, idx, items, seg, next_text=None, label_w=470):
    d = ImageDraw.Draw(c)
    vk.header(d, idx, "总结")
    for i, (k, v) in enumerate(items):
        if seg >= i + 1:
            y = 130 + i * 95
            d.rounded_rectangle([140, y, 1140, y + 76], 14, outline=(30, 41, 59), fill=(12, 19, 34))
            d.text((180, y + 38), f"0{i + 1}", font=vk.font(30, mono=True), fill=CYAN, anchor="lm")
            d.text((260, y + 38), k, font=vk.F_BODY, fill=TEXT, anchor="lm")
            d.text((label_w, y + 38), v, font=vk.F_BODY, fill=MUTED, anchor="lm")
    if next_text and seg >= len(items) + 1:
        vk.text_c(d, (640, 130 + len(items) * 95 + 30), next_text, vk.F_SMALL, VIOLET)


def title_card(c, n, title, sub):
    d = ImageDraw.Draw(c)
    vk.text_c(d, (640, 190), f"Transformer · {n:02d}", vk.font(24, mono=True), MUTED)
    vk.text_c(d, (640, 270), title, vk.F_TITLE, CYAN)
    vk.text_c(d, (640, 340), sub, vk.F_BODY, TEXT)
