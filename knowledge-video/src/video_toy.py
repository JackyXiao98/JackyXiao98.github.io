"""Toy 8-frame 'videos' of a centred ellipse whose axes change linearly in time."""
import numpy as np
from PIL import Image, ImageDraw
import latent as L

F = 8


def make_videos(n, seed):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, F, 32 * 32), np.float32)
    for i in range(n):
        a0, b0, a1, b1 = rng.uniform(3, 13, 4)
        v = rng.uniform(150, 255)
        for f in range(F):
            s = f / (F - 1)
            a, b = (1 - s) * a0 + s * a1, (1 - s) * b0 + s * b1
            img = Image.new("L", (128, 128), 0)
            ImageDraw.Draw(img).ellipse([64 - 4 * a, 64 - 4 * b, 64 + 4 * a, 64 + 4 * b], fill=int(v))
            out[i, f] = np.asarray(img.resize((32, 32), Image.LANCZOS), np.float32).ravel() / 255.0
    return out


def flicker(v):
    """mean RMS difference between consecutive frames."""
    return float(np.sqrt(((v[:, 1:] - v[:, :-1]) ** 2).mean(-1)).mean())
