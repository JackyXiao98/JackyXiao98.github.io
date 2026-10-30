---
name: knowledge-video
description: Make a short Chinese-narrated explainer video for an ML concept (Knowledge Video style) — real computed examples, offline TTS narration with verified subtitles, rendered to MP4 and published to the site's video library. Use when asked to make, redo or publish a knowledge/explainer video, add an episode, change the narration voice, or reuse this video workflow elsewhere.
---

# Knowledge Video workflow

Code lives in `knowledge-video/` (see its README for setup). Each episode is one script
`src/epNN_<name>.py` that defines `SCENES` (narration) and `RENDER` (one drawing function per
scene) and calls `vidkit.render(SCENES, RENDER, out_dir, name)`.

## 1. Plan from measurements, not from memory

1. Pick a toy setting where the quantity of interest is **exactly computable** (2D/1D Gaussian
   mixtures, finite datasets, PCA latents). `src/diff2d.py`, `src/fm.py`, `src/latent.py` give exact
   denoisers, scores, velocity fields and samplers.
2. Write a small benchmark first and look at the numbers **before** writing narration. In this
   series several textbook claims did not hold in the toy and were rewritten:
   - "DDIM 50 steps ≈ DDPM 1000" did not reproduce → attributed to the paper instead.
   - "a 2nd-order solver is better at 5 steps" was false (all methods fail at 5) → shown at 20 steps.
   - "flow-matching paths are straight" — the learned marginal paths curve; only the conditional
     paths are straight (that motivates reflow).
   - A metric dominated by sampling noise (sliced Wasserstein at 4k samples) was dropped; check
     the noise floor (true-vs-true distance) before trusting a metric.
3. Put every spoken number behind an `assert` in the episode script so the render fails if the
   narration and the measurement disagree. Label example calculations as examples on screen.
   Facts about real systems (Stable Diffusion, DiT…) go in the notes with their references.

## 2. Narration rules (Matcha zh-baker voice)

- Chinese only: the voice drops Latin letters, so say 去噪扩散隐式模型 not "DDIM"; put English
  names on screen and in the notes. Greek letters: 贝塔, 阿尔法拔, 艾普西隆, 德尔塔.
- Avoid very short standalone sentences (e.g. 谢谢观看。 gets truncated) — merge them.
- Each scene is synthesized as one paragraph (natural intonation). Sentence-by-sentence synthesis
  with fixed gaps sounded choppy (“顿挫”) and was rejected.
- Voices are compared with `src/tts_bench.py` (pause count/length + ASR character error rate);
  the user chose Matcha (female) over Kokoro and Melo.

## 3. Render and verify

```sh
cd knowledge-video
.venv/bin/python src/preview.py src ep07_flow out/p.png idea:1:2:1 reflow:1:3:1   # scene:u:seg:su
PYTHON=.venv/bin/python ./render.sh src/ep07_flow.py out/ep07
```

- Always preview a few frames per scene and look at them: overlapping labels, text running off
  the 1280×720 frame, and **missing glyphs** (□) are common. The CJK font lacks ᾱ, ₀/ₜ subscripts,
  ▇; the mono font lacks CJK — render math with `vk.formula(r"$...$")` (matplotlib mathtext has no
  `\big`), use `F_SMALL`/`F_BODY` for Chinese text.
- `render.sh` checks every subtitle span with ASR. Lines marked `??` are usually homophones
  (噪→造, 它→他, 词元→词源); a wrong boundary shows as text from the neighbouring sentence.
- Renderer signature: `f(canvas, u, seg, su)` — `u` scene progress, `seg` current sentence,
  `su` progress within that sentence (sync animations to the narration with `seg`/`su`).

## 4. Publish

```sh
.venv/bin/python publish_episode.py --out out/ep07 --video flow-matching.mp4 --id flow-matching \
  --title "..." --summary "..." --tags a,b --notes notes.html --thumb-scene reflow \
  --chapters '{"title":"开场", ...}'
```

Add `--series <id>` (default `diffusion`; e.g. `transformer`). This writes
`assets/videos/<id>/{video.mp4,thumb.jpg,content.html}` and adds the episode to that series in
`assets/videos/catalog.json` (removing a matching planned entry from `upcoming`);
`/blog/knowledge-video/` (`assets/js/knowledge-video.js`) renders one section per series.
Notes are HTML with MathJax (`\( \)`, `$$ $$`); escape `<` as `&lt;` inside formulas.

## 5. Build, test, deploy

- `JEKYLL_ENV=production bundle exec jekyll build`, serve `_site`, and test with Playwright:
  cards load, each video reaches `readyState >= 1`, chapters and notes appear.
- Commit on the working branch, then fast-forward `gh-pages` (the deploy workflow runs on it) and
  check the "Deploy site" run.
- Keep each MP4 ≈ 5–8 MB (720p, CRF 26). Re-rendering an episode adds another copy to git history,
  so finalize before committing.
