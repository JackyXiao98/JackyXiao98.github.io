#!/usr/bin/env python3
"""Copy a rendered episode into the website and add/update its entry in the catalog.

usage: python publish_episode.py --out out/ep07 --video flow-matching.mp4 --id flow-matching \\
         --title "Flow Matching 与 Rectified Flow" --summary "..." --tags 生成模型,Flow Matching \\
         --notes notes/flow-matching.html --thumb-scene reflow \\
         --chapters '{"title": "开场", "idea": "直线路径", ...}'

Writes assets/videos/<id>/{video.mp4, thumb.jpg, content.html} and updates
assets/videos/catalog.json (chapters come from <out>/timeline.json; titles from --chapters,
falling back to the scene names).
"""
import argparse
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "..", "assets", "videos")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--id", required=True)
ap.add_argument("--title", required=True)
ap.add_argument("--summary", required=True)
ap.add_argument("--tags", default="")
ap.add_argument("--notes", help="HTML notes shown under the video (MathJax works)")
ap.add_argument("--chapters", default="{}", help="JSON object: scene name -> chapter title")
ap.add_argument("--thumb-scene", help="scene to take the cover frame from (default: middle of the video)")
ap.add_argument("--thumb-frac", type=float, default=0.9, help="position inside that scene, 0..1")
ap.add_argument("--site", default=SITE)
a = ap.parse_args()

video = os.path.join(a.out, a.video)
tl = json.load(open(os.path.join(a.out, "timeline.json")))
dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "csv=p=0", video]).decode())
names = json.loads(a.chapters)
scenes = tl["scenes"]

dest = os.path.join(a.site, a.id)
os.makedirs(dest, exist_ok=True)
shutil.copyfile(video, os.path.join(dest, "video.mp4"))
if a.notes:
    shutil.copyfile(a.notes, os.path.join(dest, "content.html"))

t = dur / 2
if a.thumb_scene:
    i = [s["name"] for s in scenes].index(a.thumb_scene)
    t0 = scenes[i]["start"]
    t1 = scenes[i + 1]["start"] if i + 1 < len(scenes) else dur
    t = t0 + a.thumb_frac * (t1 - t0)
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1",
                "-vf", "crop=1280:500:0:10,scale=640:-1", "-q:v", "4", os.path.join(dest, "thumb.jpg")], check=True)

cat_path = os.path.join(a.site, "catalog.json")
catalog = json.load(open(cat_path)) if os.path.exists(cat_path) else {"intro": "", "videos": []}
entry = {
    "id": a.id, "title": a.title, "summary": a.summary,
    "duration": f"{int(dur // 60)}:{int(dur % 60):02d}",
    "tags": [x for x in a.tags.split(",") if x],
    "chapters": [{"t": s["start"], "title": names.get(s["name"], s["name"])} for s in scenes],
}
ids = [v["id"] for v in catalog["videos"]]
if a.id in ids:
    catalog["videos"][ids.index(a.id)] = entry
else:
    catalog["videos"].append(entry)
json.dump(catalog, open(cat_path, "w"), ensure_ascii=False, indent=1)
print(f"published {a.id} ({entry['duration']}, {len(entry['chapters'])} chapters) -> {dest}")
