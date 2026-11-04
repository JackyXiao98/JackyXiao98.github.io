"""Drawing helpers for the Video LoRA Editing series: filmstrips of the 8-frame toy videos."""
import numpy as np
from PIL import Image

import vidkit as vk


def film(c, v, x, y, cell=56, gap=4, frames=None, label=None, color=vk.MUTED, font=None):
    """Paste the frames of one video (T, 16, 16, 3) in a row; returns the right edge x."""
    frames = range(v.shape[0]) if frames is None else frames
    for k, t in enumerate(frames):
        img = Image.fromarray((np.clip(v[t], 0, 1) * 255).astype(np.uint8)).resize((cell, cell), Image.NEAREST)
        c.paste(img, (int(x + k * (cell + gap)), int(y)))
    if label:
        from PIL import ImageDraw
        ImageDraw.Draw(c).text((x - 12, y + cell / 2), label, font=font or vk.F_SMALL, fill=color, anchor="rm")
    return x + len(list(frames)) * (cell + gap)
