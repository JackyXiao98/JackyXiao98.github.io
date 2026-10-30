import sys, importlib
sys.path.insert(0, sys.argv[1]); mod = importlib.import_module(sys.argv[2]); import vidkit as vk
from PIL import Image
out = sys.argv[3]; frames = []
for spec in sys.argv[4:]:
    name, u, seg, su = spec.split(":"); c = vk.BG_IMG.copy(); mod.RENDER[name](c, float(u), int(seg), float(su)); frames.append(c.resize((640, 360)))
cols = 2; rows = (len(frames) + 1) // 2; sheet = Image.new("RGB", (1280, 360 * rows))
for i, f in enumerate(frames): sheet.paste(f, ((i % 2) * 640, (i // 2) * 360))
sheet.save(out)
