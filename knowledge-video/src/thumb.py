import sys, json, subprocess
out_dir, video, scene, frac, dest = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
sc = json.load(open(f"{out_dir}/timeline.json"))["scenes"]
names = [s["name"] for s in sc]; i = names.index(scene)
t0 = sc[i]["start"]; t1 = sc[i + 1]["start"] if i + 1 < len(sc) else t0 + 10
t = t0 + frac * (t1 - t0)
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", f"{out_dir}/{video}", "-frames:v", "1",
                "-vf", "crop=1280:560:0:10,scale=640:-1", "-q:v", "4", dest], check=True)
print(dest, round(t, 1))
