"""2D plotting helpers for the Knowledge Video episodes (drawn with PIL on the vidkit canvas)."""

import numpy as np
from PIL import Image, ImageDraw

import vidkit as vk

EXT = 4.0  # plots show [-EXT, EXT]^2


def to_px(box, pts, ext=EXT):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sc = (x1 - x0) / (2 * ext)
    return np.stack([cx + pts[:, 0] * sc, cy - pts[:, 1] * sc], 1)


def frame(d, box):
    d.rectangle(box, outline=(30, 41, 59))


def heatmap(box, logp_fn, ext=EXT, color=(34, 211, 238)):
    """Density image (RGB) for box; logp_fn maps (N,2) -> log density."""
    w, h = box[2] - box[0], box[3] - box[1]
    xs = np.linspace(-ext, ext, w)
    ys = np.linspace(ext, -ext, h)
    gx, gy = np.meshgrid(xs, ys)
    lp = logp_fn(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(h, w)
    p = np.exp(lp - lp.max())
    v = np.sqrt(p)[..., None]
    bg = np.array(vk.BG, float)
    rgb = bg * (1 - 0.85 * v) + np.array(color, float) * 0.85 * v
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def arrows(d, box, field_fn, n=15, ext=EXT, color=(226, 232, 240), max_len=None):
    """Arrow field: field_fn maps (N,2) -> (N,2) vectors; lengths are compressed for display."""
    g = np.linspace(-ext * 0.9, ext * 0.9, n)
    gx, gy = np.meshgrid(g, g)
    p = np.stack([gx.ravel(), gy.ravel()], 1)
    v = field_fn(p)
    mag = np.linalg.norm(v, axis=1, keepdims=True)
    cell = (box[2] - box[0]) / n
    max_len = max_len or cell * 0.85
    disp = v / np.maximum(mag, 1e-9) * (max_len * np.tanh(mag / np.median(mag + 1e-9) * 0.6))
    a = to_px(box, p, ext)
    for (x, y), (dx, dy) in zip(a, disp):
        ex, ey = x + dx, y - dy
        d.line([(x, y), (ex, ey)], fill=color, width=2)
        L = np.hypot(dx, dy)
        if L > 4:
            ux, uy = dx / L, -dy / L
            for sgn in (-1, 1):
                hx = ex - 6 * ux + sgn * 4 * uy
                hy = ey - 6 * uy - sgn * 4 * ux
                d.line([(ex, ey), (hx, hy)], fill=color, width=2)


def scatter(d, box, pts, color, r=1.6, ext=EXT):
    x0, y0, x1, y1 = box
    for x, y in to_px(box, pts, ext):
        if x0 < x < x1 and y0 < y < y1:
            d.ellipse([x - r, y - r, x + r, y + r], fill=color)


def trails(d, box, traj, upto, color, ext=EXT, width=1):
    """traj: (steps, n, 2); draw each particle's path up to step index `upto`."""
    upto = max(1, min(upto, traj.shape[0] - 1))
    for j in range(traj.shape[1]):
        pts = to_px(box, traj[: upto + 1, j], ext)
        d.line([tuple(p) for p in pts], fill=color, width=width)


def bars(d, box, groups, labels, colors, ymax=0.7, legend=None):
    """Grouped bar chart. groups: list of arrays (one per series) of equal length."""
    x0, y0, x1, y1 = box
    k = len(labels)
    gw = (x1 - x0) / k
    bw = gw * 0.7 / len(groups)
    d.line([(x0, y1), (x1, y1)], fill=(71, 85, 105), width=2)
    for i, lab in enumerate(labels):
        for j, (vals, col) in enumerate(zip(groups, colors)):
            v = vals[i]
            bx = x0 + i * gw + gw * 0.15 + j * bw
            top = y1 - v / ymax * (y1 - y0)
            d.rectangle([bx, top, bx + bw - 4, y1], fill=col)
            d.text((bx + (bw - 4) / 2, top - 14), f"{v:.0%}", font=vk.font(18, mono=True), fill=col, anchor="mm")
        d.text((x0 + i * gw + gw / 2, y1 + 20), lab, font=vk.F_SMALL, fill=vk.MUTED, anchor="mm")
    if legend:
        for j, (name, col) in enumerate(zip(legend, colors)):
            d.rectangle([x1 - 210, y0 + j * 30, x1 - 194, y0 + 16 + j * 30], fill=col)
            d.text((x1 - 186, y0 + 8 + j * 30), name, font=vk.F_SMALL, fill=col, anchor="lm")
