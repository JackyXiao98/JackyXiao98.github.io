# Knowledge Video generator

Source code for the [Knowledge Video](https://jackyxiao98.github.io/blog/knowledge-video/) series:
short Chinese-narrated explainer videos where every example is computed for real (exact
denoisers / scores / velocity fields on toy data), rendered frame by frame and narrated with an
offline TTS voice. The workflow is described step by step in
[`.claude/skills/knowledge-video/SKILL.md`](../.claude/skills/knowledge-video/SKILL.md).

## Setup

```sh
cd knowledge-video
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # also needs ffmpeg
./setup_models.sh                                                     # ~330 MB of offline models
```

## Render, check, publish

```sh
PYTHON=.venv/bin/python ./render.sh src/ep07_flow.py out/ep07       # video + subtitle check
.venv/bin/python src/preview.py src ep07_flow out/p.png reflow:1:3:1 # (optional) preview frames

.venv/bin/python publish_episode.py --out out/ep07 --video flow-matching.mp4 --id flow-matching \
  --title "Flow Matching 与 Rectified Flow" --summary "..." --tags 生成模型,Flow\ Matching \
  --notes ../assets/videos/flow-matching/content.html --thumb-scene reflow \
  --chapters '{"title":"开场","idea":"直线路径","marginal":"学到的是平均速度","compare":"流匹配 vs 扩散 ODE","reflow":"整流流：重新配对","summary":"总结"}'
```

`publish_episode.py` writes `assets/videos/<id>/` and adds the episode to a series
(`--series diffusion`, `--series transformer`, …) in `assets/videos/catalog.json`. Each series has its own page `/blog/<series>/`
(`_pages/<series>.md`, linked from a card on the blog page). Each series has `title`, `subtitle`, `summary`, `videos` and an
optional `upcoming` list of planned episodes.

## Files

| file | purpose |
|---|---|
| `src/vidkit.py` | drawing helpers, formulas (matplotlib mathtext), subtitles, `render()`: narration → frames → MP4 |
| `src/narrate.py` | offline TTS (Matcha / Kokoro / Melo via sherpa-onnx); per-scene paragraph synthesis + subtitle timing verified with ASR |
| `src/diff2d.py` | exact 2D Gaussian-mixture diffusion: denoiser, score, DDPM / DDIM / DPM-Solver++(2M), metrics |
| `src/fm.py`, `src/latent.py`, `src/video_toy.py`, `src/p2d.py` | flow matching, toy latent diffusion (PCA + GMM), toy videos, 2D plots |
| `src/ep01_intro.py` … `src/ep09_dit.py` | Diffusion series, nine episodes (narration `SCENES`, renderers `RENDER`, asserted measurements) |
| `src/tf01_tokens.py` … `src/tf08_scaling.py` | Transformer series, eight episodes (`--series transformer`) |
| `src/dp01_intro.py` … `src/dp09_resplan.py` | Differential Privacy series, nine episodes (`--series differential-privacy`); `dp09` mirrors ResidualPlanner from [resplan_journal](https://github.com/JackyXiao98/resplan_journal) |
| `src/dpx.py` | DP toolkit: Gaussian (ε, δ) curve, RDP accountant for subsampled Gaussian, DP-SGD on sklearn digits (JAX, cached via `DP_CACHE`), canaries + membership inference |
| `src/vg01_overview.py` … `src/vg08_sla.py` | Video Generation series, eight episodes (`--series video-generation`) |
| `src/vgx.py` | video-generation toys: ring teacher, one-step regression and DMD distillation, Self-Forcing toy, INT8 attention with K smoothing, sparse + linear attention (cached via `VG_CACHE`) |
| `src/rsi01_explosion.py` … `src/rsi09_safety.py` | Recursive Self-Improvement series, nine episodes (`--series recursive-self-improvement`) |
| `src/rsix.py` | RSI toys: STaR self-training of the addition model, tic-tac-toe self-play, verifier best-of-n, model collapse, Goodhart, evolved bin-packing heuristics (cached via `RSI_CACHE`) |
| `src/tfm.py`, `src/tviz.py`, `src/zhnum.py` | small JAX Transformer trained on 2-digit addition (cached via `TF_CACHE`), chart/matrix helpers, numbers → spoken Chinese |
| `src/check_align.py` | transcribe each subtitle span with ASR and compare to the script |
| `src/preview.py`, `src/thumb.py`, `src/tts_bench.py` | frame previews, cover frames, voice comparison |
