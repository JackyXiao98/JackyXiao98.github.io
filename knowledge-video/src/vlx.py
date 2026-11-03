"""Toy experiments for the Video LoRA Editing series (cached via VL_CACHE).

Sprite videos: 8 frames of 16x16 RGB, a static smooth textured background and one object (square / disk /
cross; red / green / blue / yellow) moving 1 px per frame (right / left / down / up). A small video DiT with
factorised spatial + temporal attention (4x4 patches, 16 tokens per frame) trained with flow matching,
conditioned on (shape, color, motion) labels. Every token carries a "clean" flag: clean tokens get their
true value and timestep 0 and are excluded from the loss. The same mechanism gives first-frame conditioning,
mask-preserving inpainting and in-context reference tokens (IC-LoRA). LoRA adapters can be attached to any
subset of {spatial attention, temporal attention, MLP} matrices.
"""

import os
import pickle

import numpy as np

T, H, W, C, P = 8, 16, 16, 3, 4
NP = (H // P) * (W // P)              # patches per frame
PD = P * P * C                       # values per patch
COLORS = np.array([[0.95, 0.25, 0.2], [0.2, 0.85, 0.3], [0.25, 0.45, 1.0], [0.95, 0.85, 0.15], [0.75, 0.3, 0.95]])
COLOR_NAMES = ["红", "绿", "蓝", "黄", "紫"]
SHAPES = ["方块", "圆盘", "十字"]
MOTIONS = ["向右", "向左", "向下", "向上", "斜向", "之字"]
VEL = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1), 4: (1, 1)}


def cached(cache_dir, name, fn, **kw):
    key = "_".join(f"{k}{v}" for k, v in sorted(kw.items()) if k != "cache_dir")
    path = os.path.join(cache_dir, f"{name}_{key}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    res = fn(**kw)
    os.makedirs(cache_dir, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


# ---------------------------------------------------------------- data
def _mask(shape, cx, cy, r=2.6):
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    dx, dy = xx - cx, yy - cy
    if shape == 0:
        return (np.abs(dx) <= r) & (np.abs(dy) <= r)
    if shape == 1:
        return dx * dx + dy * dy <= (r + 0.4) ** 2
    return ((np.abs(dx) <= 1.0) & (np.abs(dy) <= r + 0.5)) | ((np.abs(dy) <= 1.0) & (np.abs(dx) <= r + 0.5))


def background(rng, kind="dark"):
    """Smooth random texture, static in time."""
    g = rng.normal(size=(4, 4, 3))
    from scipy.ndimage import zoom
    tex = zoom(g, (H / 4, W / 4, 1), order=3)
    base = 0.18 if kind == "dark" else 0.85
    return np.clip(base + 0.07 * tex, 0, 1)


def trajectory(motion, x0, y0):
    if motion in VEL:
        vx, vy = VEL[motion]
        return [(x0 + vx * t, y0 + vy * t) for t in range(T)]
    # zigzag: right every frame, up/down alternating 2 px
    return [(x0 + t, y0 + (2 if t % 2 else -0)) for t in range(T)]


def render(shape, color, traj, bg):
    v = np.repeat(bg[None], T, 0).copy()
    col = COLORS[color] if np.ndim(color) == 0 else np.asarray(color)
    for t, (cx, cy) in enumerate(traj):
        m = _mask(shape, cx, cy)
        v[t][m] = col
    return v.astype(np.float32)


def sample_clip(rng, shapes=(0, 1, 2), colors=(0, 1, 2, 3), motions=(0, 1, 2, 3), bg_kind="dark"):
    s, c, m = rng.choice(shapes), rng.choice(colors), rng.choice(motions)
    margin = 3.5
    while True:
        x0, y0 = rng.uniform(margin, W - margin), rng.uniform(margin, H - margin)
        tr = trajectory(m, x0, y0)
        if all(margin - 0.5 <= x <= W - margin + 0.5 and margin - 0.5 <= y <= H - margin + 0.5 for x, y in tr):
            break
        # restart from a position that keeps the object inside
        if m in VEL:
            vx, vy = VEL[m]
            x0 = W / 2 - vx * (T - 1) / 2 + rng.uniform(-1.5, 1.5)
            y0 = H / 2 - vy * (T - 1) / 2 + rng.uniform(-1.5, 1.5)
        else:
            x0, y0 = W / 2 - (T - 1) / 2, H / 2 - 1
        tr = trajectory(m, x0, y0)
        break
    bg = background(rng, bg_kind)
    return render(s, c, tr, bg), (int(s), int(c), int(m)), tr, bg


def batch(rng, n, **kw):
    vids, labs = [], []
    for _ in range(n):
        v, l, _, _ = sample_clip(rng, **kw)
        vids.append(v)
        labs.append(l)
    return np.stack(vids), np.array(labs, np.int32)


def to_tokens(v):
    """(N, T, H, W, C) -> (N, T, NP, PD)."""
    N = v.shape[0]
    x = v.reshape(N, T, H // P, P, W // P, P, C).transpose(0, 1, 2, 4, 3, 5, 6)
    return x.reshape(N, T, NP, PD) * 2 - 1


def from_tokens(x):
    N, Tn = x.shape[:2]
    v = (x.reshape(N, Tn, H // P, W // P, P, P, C).transpose(0, 1, 2, 4, 3, 5, 6).reshape(N, Tn, H, W, C) + 1) / 2
    return np.clip(v, 0, 1)


# ---------------------------------------------------------------- model
D, LAYERS, HEADS = 96, 4, 4
MODULES = ("s_qkv", "s_o", "t_qkv", "t_o", "mlp1", "mlp2")


def init(key):
    import jax
    ks = iter(jax.random.split(key, 64))
    n = lambda shape, sc: jax.random.normal(next(ks), shape) * sc
    p = {"in": n((PD, D), 1 / np.sqrt(PD)), "out": n((D, PD), 0.02), "spos": n((NP, D), 0.1), "tpos": n((16, D), 0.1),
         "lab": [n((len(SHAPES) + 1, D), 0.1), n((len(COLORS) + 1, D), 0.1), n((len(MOTIONS) + 1, D), 0.1)],
         "t1": n((32, D), 1 / np.sqrt(32)), "t2": n((D, D), 1 / np.sqrt(D)), "inst": n((8, D), 0.1), "layers": []}
    for _ in range(LAYERS):
        p["layers"].append({"s_qkv": n((D, 3 * D), 1 / np.sqrt(D)), "s_o": n((D, D), 0.5 / np.sqrt(D)),
                            "t_qkv": n((D, 3 * D), 1 / np.sqrt(D)), "t_o": n((D, D), 0.5 / np.sqrt(D)),
                            "mlp1": n((D, 4 * D), 1 / np.sqrt(D)), "mlp2": n((4 * D, D), 0.5 / np.sqrt(4 * D))})
    return p


def lora_init(key, rank, modules, layers=range(LAYERS)):
    import jax
    import jax.numpy as jnp
    shapes = {"s_qkv": (D, 3 * D), "s_o": (D, D), "t_qkv": (D, 3 * D), "t_o": (D, D), "mlp1": (D, 4 * D), "mlp2": (4 * D, D)}
    ks = iter(jax.random.split(key, 64))
    out = {}
    for l in layers:
        for m in modules:
            a, b = shapes[m]
            out[f"{l}.{m}"] = (jax.random.normal(next(ks), (a, rank)) / np.sqrt(a), jnp.zeros((rank, b)))
    return out


def lora_params(lora):
    return int(sum(A.size + B.size for A, B in lora.values()))


def _ln(x):
    import jax.numpy as jnp
    m = x.mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(((x - m) ** 2).mean(-1, keepdims=True) + 1e-5)


def _attn(x, qkv, o):
    import jax
    import jax.numpy as jnp
    *b, n, d = x.shape
    q, k, v = jnp.split(x @ qkv, 3, -1)
    sh = lambda t: t.reshape(*b, n, HEADS, d // HEADS).swapaxes(-2, -3)
    q, k, v = sh(q), sh(k), sh(v)
    a = jax.nn.softmax(q @ k.swapaxes(-1, -2) / np.sqrt(d // HEADS), -1)
    return (a @ v).swapaxes(-2, -3).reshape(*b, n, d) @ o


def forward(p, x, t, labels, clean, tpos=None, lora=None, lora_scale=1.0, inst=None):
    """x: (B, Tn, NP, PD) noisy tokens; t: (B,) noise level; clean: (B, Tn, NP) 1 = clean token (time 0);
    tpos: (Tn,) temporal position ids; labels: (B, 3) ints (use the last index = 'none'); inst: (B,) ints or None.
    Returns predicted velocity (B, Tn, NP, PD)."""
    import jax.numpy as jnp
    B, Tn = x.shape[:2]
    tpos = jnp.arange(Tn) if tpos is None else tpos
    W_ = lambda l, name: p["layers"][l][name] + (lora_scale * lora[f"{l}.{name}"][0] @ lora[f"{l}.{name}"][1]
                                                    if lora and f"{l}.{name}" in lora else 0)
    tt = jnp.where(clean > 0, 0.0, t[:, None, None])                      # (B, Tn, NP)
    freqs = jnp.exp(jnp.linspace(0, np.log(100), 16))
    temb = jnp.concatenate([jnp.sin(tt[..., None] * freqs), jnp.cos(tt[..., None] * freqs)], -1) @ p["t1"]
    temb = jnp.tanh(temb) @ p["t2"]
    cond = sum(p["lab"][i][labels[:, i]] for i in range(3))[:, None, None, :]
    if inst is not None:
        cond = cond + p["inst"][inst][:, None, None, :]
    h = x @ p["in"] + p["spos"][None, None] + p["tpos"][tpos][None, :, None] + temb + cond
    for l in range(LAYERS):
        h = h + _attn(_ln(h), W_(l, "s_qkv"), W_(l, "s_o"))                              # within each frame
        ht = h.swapaxes(1, 2)                                                           # (B, NP, Tn, D)
        ht = ht + _attn(_ln(ht), W_(l, "t_qkv"), W_(l, "t_o"))                           # across frames
        h = ht.swapaxes(1, 2)
        h = h + jnp.maximum(0, _ln(h) @ W_(l, "mlp1")) @ W_(l, "mlp2")
    return _ln(h) @ p["out"]


def fm_loss(p, x0, labels, clean, key, lora=None, tpos=None, inst=None, ref=None):
    """Flow matching: x_t = (1 - t) x0 + t eps, target v = eps - x0, loss on non-clean tokens. ref: optional
    clean reference tokens (B, Tr, NP, PD) prepended in time (IC-LoRA); their positions are 0..Tr-1."""
    import jax
    import jax.numpy as jnp
    k1, k2 = jax.random.split(key)
    B = x0.shape[0]
    t = jax.random.uniform(k1, (B,))
    eps = jax.random.normal(k2, x0.shape)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * eps
    xt = jnp.where(clean[..., None] > 0, x0, xt)
    if ref is not None:
        Tr = ref.shape[1]
        xin = jnp.concatenate([ref, xt], 1)
        cl = jnp.concatenate([jnp.ones(ref.shape[:3]), clean], 1)
        tp = jnp.concatenate([jnp.arange(Tr), jnp.arange(x0.shape[1])]) if tpos is None else tpos
        v = forward(p, xin, t, labels, cl, tp, lora, inst=inst)[:, Tr:]
    else:
        v = forward(p, xt, t, labels, clean, tpos, lora, inst=inst)
    w = 1 - clean[..., None]
    return (((v - (eps - x0)) ** 2) * w).sum() / (w.sum() * x0.shape[-1] + 1e-8)


def sample(p, labels, key, steps=20, clean_x=None, clean=None, lora=None, ref=None, inst=None, x_init=None, t_start=1.0,
           lora_scale=1.0):
    """Euler sampler from t = t_start to 0. clean_x / clean: tokens kept fixed (first frame, preserved region).
    x_init: start point at t_start (for inversion-based editing); otherwise Gaussian noise."""
    import jax
    import jax.numpy as jnp
    B = labels.shape[0]
    x = jax.random.normal(key, (B, T, NP, PD)) if x_init is None else x_init
    cl = jnp.zeros((B, T, NP)) if clean is None else clean
    cx = jnp.zeros((B, T, NP, PD)) if clean_x is None else clean_x
    ts = jnp.linspace(t_start, 0, steps + 1)
    for i in range(steps):
        xin = jnp.where(cl[..., None] > 0, cx, x)
        tvec = jnp.full((B,), ts[i])
        if ref is not None:
            Tr = ref.shape[1]
            v = forward(p, jnp.concatenate([ref, xin], 1), tvec, labels, jnp.concatenate([jnp.ones(ref.shape[:3]), cl], 1),
                        jnp.concatenate([jnp.arange(Tr), jnp.arange(T)]), lora, lora_scale, inst)[:, Tr:]
        else:
            v = forward(p, xin, tvec, labels, cl, None, lora, lora_scale, inst)
        x = x + (ts[i + 1] - ts[i]) * v
    return jnp.where(cl[..., None] > 0, cx, x)


def invert(p, x0, labels, steps=20, t_end=1.0):
    """Euler ODE from the clean video to noise level t_end (deterministic inversion)."""
    import jax.numpy as jnp
    B = x0.shape[0]
    x = x0
    ts = jnp.linspace(0, t_end, steps + 1)
    cl = jnp.zeros((B, T, NP))
    for i in range(steps):
        v = forward(p, x, jnp.full((B,), ts[i]), labels, cl)
        x = x + (ts[i + 1] - ts[i]) * v
    return x


# ---------------------------------------------------------------- training
def _random_clean(rng, n, p_first=0.25, p_mask=0.25):
    """Per-example clean pattern: none, first frame, or a random set of spatial patches kept in all frames."""
    cl = np.zeros((n, T, NP), np.float32)
    for i in range(n):
        u = rng.random()
        if u < p_first:
            cl[i, 0] = 1
        elif u < p_first + p_mask:
            keep = rng.random(NP) < rng.uniform(0.3, 0.8)
            cl[i, :, keep] = 1
    return cl


def train_base(steps=6000, batch_size=64, lr=2e-3, seed=0, drop_label=0.1):
    import jax
    import jax.numpy as jnp
    import optax
    p = init(jax.random.PRNGKey(seed))
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, lr, min(300, steps // 4), steps, lr * 0.05), weight_decay=0.01)
    st = opt.init(p)

    @jax.jit
    def step(p, st, x, lab, cl, key):
        l, g = jax.value_and_grad(fm_loss)(p, x, lab, cl, key)
        g, _ = optax.clip_by_global_norm(1.0).update(g, optax.EmptyState())
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed + 1)
    hist = []
    for it in range(steps):
        v, lab = batch(rng, batch_size)
        lab = np.where(rng.random(lab.shape) < drop_label, [len(SHAPES), len(COLORS), len(MOTIONS)], lab)
        cl = _random_clean(rng, batch_size)
        key, k = jax.random.split(key)
        p, st, l = step(p, st, jnp.array(to_tokens(v)), jnp.array(lab), jnp.array(cl), k)
        if it % 100 == 0:
            hist.append((it, float(l)))
    return dict(p=jax.tree_util.tree_map(np.asarray, p), hist=hist)


# ---------------------------------------------------------------- measurement
def measure(v, bg=None):
    """Per video: object color (mean RGB of the object pixels), centroid per frame, object pixel count.
    Object pixels = saturated pixels (max - min channel > 0.35; backgrounds are near-grey)."""
    out = []
    for i in range(v.shape[0]):
        sat = v[i].max(-1) - v[i].min(-1)
        cents, cols, cnt = [], [], []
        for t in range(T):
            m = sat[t] > 0.35
            cnt.append(int(m.sum()))
            if m.sum() == 0:
                cents.append((np.nan, np.nan))
                continue
            yy, xx = np.nonzero(m)
            cents.append((xx.mean() + 0.5, yy.mean() + 0.5))
            cols.append(v[i, t][m].mean(0))
        out.append(dict(cent=np.array(cents), color=np.mean(cols, 0) if cols else np.full(3, np.nan), count=np.array(cnt)))
    return out


def color_id(rgb):
    return int(np.argmin(((COLORS - rgb) ** 2).sum(1))) if np.all(np.isfinite(rgb)) else -1


def motion_id(cent):
    d = np.nanmean(np.diff(cent, axis=0), 0)
    if not np.all(np.isfinite(d)):
        return -1
    best = min(VEL, key=lambda m: (VEL[m][0] - d[0]) ** 2 + (VEL[m][1] - d[1]) ** 2)
    return best


# ---------------------------------------------------------------- LoRA / full fine-tuning
def base_model(cache_dir):
    import jax
    return jax.tree_util.tree_map(lambda a: a, cached(cache_dir, "base", train_base)["p"])


def finetune(p, data_fn, steps, lr, rank=None, modules=MODULES, seed=0, full=False, train_inst=False, log_every=None,
             eval_fn=None):
    """Fine-tune either LoRA adapters (rank, modules) or all weights (full=True). data_fn(rng) -> dict with x0, labels,
    clean, optional ref, inst. Returns adapters (or new params) and an optional log of eval_fn(state)."""
    import jax
    import jax.numpy as jnp
    import optax
    key = jax.random.PRNGKey(seed)
    if full:
        train = p
    else:
        train = {"lora": lora_init(key, rank, modules)}
        if train_inst:
            train["inst"] = p["inst"]
    opt = optax.adam(lr)
    st = opt.init(train)

    def loss(tr, b, k):
        if full:
            return fm_loss(tr, b["x0"], b["labels"], b["clean"], k, ref=b.get("ref"), inst=b.get("inst"))
        pp = {**p, "inst": tr["inst"]} if train_inst else p
        return fm_loss(pp, b["x0"], b["labels"], b["clean"], k, lora=tr["lora"], ref=b.get("ref"), inst=b.get("inst"))

    @jax.jit
    def step(tr, st, b, k):
        l, g = jax.value_and_grad(loss)(tr, b, k)
        u, st = opt.update(g, st, tr)
        return optax.apply_updates(tr, u), st, l

    rng = np.random.default_rng(seed)
    log = []
    for it in range(1, steps + 1):
        b = {k: jnp.array(v) for k, v in data_fn(rng).items()}
        key, k = jax.random.split(key)
        train, st, l = step(train, st, b, k)
        if log_every and it % log_every == 0 and eval_fn:
            log.append((it, eval_fn(train)))
    return jax.tree_util.tree_map(np.asarray, train), log


def eval_loss(p, data, keys=8, lora=None):
    """Average flow-matching loss on fixed data over fixed noise keys."""
    import jax
    import jax.numpy as jnp
    f = jax.jit(lambda k: fm_loss(p, data["x0"], data["labels"], data["clean"], k, lora=lora, ref=data.get("ref"), inst=data.get("inst")))
    return float(np.mean([f(jax.random.PRNGKey(1000 + i)) for i in range(keys)]))


def psnr(a, b, mask=None):
    d = (a - b) ** 2
    if mask is not None:
        d = d[mask]
    return float(10 * np.log10(1.0 / max(d.mean(), 1e-10)))


def traj_err(c1, c2):
    """Mean distance between two centroid trajectories (pixels), relative to their first frame."""
    a, b = c1 - c1[:1], c2 - c2[:1]
    d = np.sqrt(((a - b) ** 2).sum(-1))
    return float(np.nanmean(d))


def abs_traj_err(c1, c2):
    return float(np.nanmean(np.sqrt(((c1 - c2) ** 2).sum(-1))))


NONE = (len(SHAPES), len(COLORS), len(MOTIONS))


# ---------------------------------------------------------------- E1: rank sweep on a new domain (light backgrounds)
def _domain_fn(bg_kind, n=32):
    def f(rng):
        v, lab = batch(rng, n, bg_kind=bg_kind)
        return dict(x0=to_tokens(v), labels=lab, clean=np.zeros((n, T, NP), np.float32))
    return f


def _gen_eval(p, lora, labels, seed=0, **kw):
    import jax
    import jax.numpy as jnp
    x = sample(p, jnp.array(labels), jax.random.PRNGKey(seed), lora=lora, **kw)
    return from_tokens(np.asarray(x))


def _gen_stats(v, labels):
    m = measure(v)
    col = np.mean([color_id(x["color"]) == l[1] for x, l in zip(m, labels)])
    mot = np.mean([motion_id(x["cent"]) == l[2] for x, l in zip(m, labels)])
    sat = v.max(-1) - v.min(-1)
    bgb = float(np.mean([v[i][sat[i] < 0.2].mean() for i in range(len(v))]))
    return dict(color_acc=float(col), motion_acc=float(mot), bg_brightness=bgb)


def rank_sweep(cache_dir, ranks=(1, 2, 4, 8, 16), steps=600, seed=0):
    import jax
    p = base_model(cache_dir)
    rng = np.random.default_rng(123)
    v, lab = batch(rng, 64, bg_kind="light")
    ev = dict(x0=to_tokens(v), labels=lab, clean=np.zeros((64, T, NP), np.float32))
    glab = batch(np.random.default_rng(7), 64)[1]
    out = dict(base_loss=eval_loss(p, ev), base_gen=_gen_stats(_gen_eval(p, None, glab), glab), lora={}, params={})
    vd, labd = batch(np.random.default_rng(124), 64)
    out["base_loss_dark"] = eval_loss(p, dict(x0=to_tokens(vd), labels=labd, clean=np.zeros((64, T, NP), np.float32)))
    out["n_params"] = int(sum(a.size for a in jax.tree_util.tree_leaves(p)))
    for r in ranks:
        tr, _ = finetune(p, _domain_fn("light"), steps, 1e-3, rank=r, seed=seed)
        out["lora"][r] = dict(loss=eval_loss(p, ev, lora=tr["lora"]), gen=_gen_stats(_gen_eval(p, tr["lora"], glab), glab))
        out["params"][r] = lora_params(tr["lora"])
    full, _ = finetune(p, _domain_fn("light"), steps, 2e-4, full=True, seed=seed)
    out["full"] = dict(loss=eval_loss(full, ev), gen=_gen_stats(_gen_eval(full, None, glab), glab))
    # spectrum of the full fine-tuning update
    fr = {}
    for l in range(LAYERS):
        for m in MODULES:
            dW = np.asarray(full["layers"][l][m]) - np.asarray(p["layers"][l][m])
            s = np.linalg.svd(dW, compute_uv=False) ** 2
            fr[f"{l}.{m}"] = np.cumsum(s) / s.sum()
    out["energy"] = fr
    out["energy_mean"] = np.mean([v[:16] for v in fr.values()], 0)
    out["samples_base"] = _gen_eval(p, None, glab[:4])
    tr8, _ = finetune(p, _domain_fn("light"), steps, 1e-3, rank=8, seed=seed)
    out["samples_lora8"] = _gen_eval(p, tr8["lora"], glab[:4])
    out["lora8"] = tr8["lora"]
    out["glab"] = glab[:4]
    return out


# ---------------------------------------------------------------- E3: one video, one LoRA (motion customisation)
def ref_clip(seed=5, shape=0, color=0, motion=5, bg_kind="dark"):
    rng = np.random.default_rng(seed)
    x0, y0 = W / 2 - (T - 1) / 2, H / 2 - 1
    tr = trajectory(motion, x0, y0)
    bg = background(rng, bg_kind)
    return render(shape, color, tr, bg), np.array(tr), bg


GROUPS = {"temporal": ("t_qkv", "t_o"), "spatial": ("s_qkv", "s_o", "mlp1", "mlp2"), "all": MODULES}


def motion_lora(cache_dir, steps=300, rank=8, n_eval=48, seed=0):
    p = base_model(cache_dir)
    v, tr, bg = ref_clip()
    lab = np.array([[0, 0, NONE[2]]] * 8, np.int32)
    x0 = np.repeat(to_tokens(v[None]), 8, 0)

    def data(rng):
        return dict(x0=x0, labels=lab, clean=np.zeros((8, T, NP), np.float32))

    rng = np.random.default_rng(11)
    elab = np.stack([rng.choice([0, 1, 2], n_eval), rng.choice([1, 2, 3], n_eval), np.full(n_eval, NONE[2])], 1).astype(np.int32)

    def stats(vid):
        m = measure(vid)
        te = [traj_err(x["cent"], tr) for x in m]
        return dict(traj=float(np.nanmean(te)), color_acc=float(np.mean([color_id(x["color"]) == l[1] for x, l in zip(m, elab)])),
                    copy=float(np.mean([color_id(x["color"]) == 0 for x in m])), traj_all=np.array(te))

    out = dict(ref=v, ref_traj=tr, base=stats(_gen_eval(p, None, elab)), runs={}, params={})
    for name, mods in GROUPS.items():
        trn, _ = finetune(p, data, steps, 1e-3, rank=rank, modules=mods, seed=seed)
        vid = _gen_eval(p, trn["lora"], elab)
        out["runs"][name] = stats(vid)
        out["runs"][name]["samples"] = vid[:4]
        out["params"][name] = lora_params(trn["lora"])
        if name == "temporal":
            out["lora_temporal"] = trn["lora"]
    out["elab"] = elab[:4]
    out["base_samples"] = _gen_eval(p, None, elab[:4])
    return out


# ---------------------------------------------------------------- E4: inversion-based editing
def edit_set(n=48, seed=21, motions=(0, 1, 2, 3)):
    """Source clips + target colour (a different colour)."""
    rng = np.random.default_rng(seed)
    vids, labs, trs, bgs = [], [], [], []
    for _ in range(n):
        v, l, tr, bg = sample_clip(rng, motions=motions)
        vids.append(v); labs.append(l); trs.append(np.array(tr)); bgs.append(bg)
    labs = np.array(labs, np.int32)
    tgt = (labs[:, 1] + 1 + rng.integers(0, 3, n)) % 4
    return np.stack(vids), labs, np.array(trs), np.stack(bgs), tgt.astype(np.int32)


def edit_metrics(out_v, src_v, src_tr, tgt_color):
    m = measure(out_v)
    msrc = measure(src_v)
    sat = src_v.max(-1) - src_v.min(-1)
    res = dict(success=float(np.mean([color_id(x["color"]) == c for x, c in zip(m, tgt_color)])),
               traj=float(np.nanmean([abs_traj_err(x["cent"], s["cent"]) for x, s in zip(m, msrc)])),
               bg_psnr=float(np.mean([psnr(out_v[i], src_v[i], sat[i] < 0.2) for i in range(len(out_v))])),
               flicker=float(np.mean(np.sqrt(((out_v[:, 1:] - out_v[:, :-1]) ** 2)[np.repeat((sat < 0.2)[:, 1:, ..., None], 3, -1) & np.repeat((sat < 0.2)[:, :-1, ..., None], 3, -1)].mean()))))
    return res


def inversion_edit(cache_dir, t_ends=(0.2, 0.4, 0.6, 0.8, 1.0), steps=20):
    import jax.numpy as jnp
    import jax
    p = base_model(cache_dir)
    v, lab, tr, bg, tgt = edit_set()
    x0 = jnp.array(to_tokens(v))
    tl = lab.copy(); tl[:, 1] = tgt
    out = dict(sweep={}, src=v[:4], tgt=tgt[:4], lab=lab[:4])
    for te in t_ends:
        n = max(2, int(round(steps * te)))
        xi = invert(p, x0, jnp.array(lab), steps=n, t_end=te)
        rec = from_tokens(np.asarray(sample(p, jnp.array(lab), jax.random.PRNGKey(0), steps=n, x_init=xi, t_start=te)))
        ed = from_tokens(np.asarray(sample(p, jnp.array(tl), jax.random.PRNGKey(0), steps=n, x_init=xi, t_start=te)))
        out["sweep"][te] = dict(edit=edit_metrics(ed, v, tr, tgt), recon=psnr(rec, v), samples=ed[:4])
    # regenerate from scratch with the target labels (no inversion)
    fresh = from_tokens(np.asarray(sample(p, jnp.array(tl), jax.random.PRNGKey(3))))
    out["fresh"] = edit_metrics(fresh, v, tr, tgt)
    out["fresh_samples"] = fresh[:4]
    return out


# ---------------------------------------------------------------- E5: first-frame edit + propagation
def recolor(v, color):
    out = v.copy()
    sat = v.max(-1) - v.min(-1)
    out[sat > 0.35] = COLORS[color]
    return out


def first_frame_edit(cache_dir):
    import jax
    import jax.numpy as jnp
    p = base_model(cache_dir)
    v, lab, tr, bg, tgt = edit_set()
    ed = np.stack([recolor(v[i], tgt[i]) for i in range(len(v))])
    cl = np.zeros((len(v), T, NP), np.float32); cl[:, 0] = 1
    out = {}
    for name, cols in (("label_none", NONE[1]), ("label_target", None)):
        L = lab.copy()
        L[:, 1] = tgt if cols is None else cols
        g = from_tokens(np.asarray(sample(p, jnp.array(L), jax.random.PRNGKey(0), clean_x=jnp.array(to_tokens(ed)), clean=jnp.array(cl))))
        out[name] = edit_metrics(g, v, tr, tgt)
        out[name + "_samples"] = g[:4]
        # per-frame colour success (does the edit survive to the last frames?)
        m = [measure(g[:, t:t + 1].repeat(T, 1)) for t in range(T)]
        out[name + "_per_frame"] = [float(np.mean([color_id(x["color"]) == c for x, c in zip(mm, tgt)])) for mm in m]
    out["src"] = v[:4]
    out["edited_first"] = ed[:4]
    return out


# ---------------------------------------------------------------- E6: mask-aware editing with a per-video LoRA
def object_patch_mask(v, dilate=0):
    sat = v.max(-1) - v.min(-1)
    m = (sat > 0.35).any(0)                                                     # (H, W) pixels ever touched
    pm = m.reshape(H // P, P, W // P, P).any((1, 3))
    if dilate:
        from scipy.ndimage import binary_dilation
        pm = binary_dilation(pm, iterations=dilate)
    return pm.reshape(NP)                                                       # True = regenerate


def mask_lora_edit(cache_dir, n=12, steps=200, rank=8):
    import jax
    import jax.numpy as jnp
    p = base_model(cache_dir)
    rng = np.random.default_rng(31)
    res = {"none": [], "lora": []}
    samples = {}
    for i in range(n):
        src, tr, bg = ref_clip(seed=100 + i, shape=int(rng.integers(0, 3)), color=int(rng.integers(0, 4)), motion=5)
        c_src = int(np.argmin(((COLORS[:4] - src[0][(src[0].max(-1) - src[0].min(-1)) > 0.35].mean(0)) ** 2).sum(1)))
        tgt = int((c_src + 1 + rng.integers(0, 3)) % 4)
        pm = object_patch_mask(src)
        keep = np.repeat((~pm)[None], T, 0).astype(np.float32)
        keep[0] = 1                                                             # edited first frame is also given
        ed = recolor(src, tgt)
        cx = jnp.array(to_tokens(ed[None]))
        L = jnp.array([[NONE[0], NONE[1], NONE[2]]], jnp.int32)
        g0 = from_tokens(np.asarray(sample(p, L, jax.random.PRNGKey(i), clean_x=cx, clean=jnp.array(keep[None]))))
        # per-video LoRA trained on the source with mask-aware clean patterns
        x0 = np.repeat(to_tokens(src[None]), 8, 0)

        def data(r):
            cl = np.repeat(np.repeat((~pm)[None], T, 0)[None], 8, 0).astype(np.float32)
            first = r.random(8) < 0.5
            cl[first, 0] = 1
            return dict(x0=x0, labels=np.array([list(NONE)] * 8, np.int32), clean=cl)

        trn, _ = finetune(p, data, steps, 1e-3, rank=rank, seed=i)
        g1 = from_tokens(np.asarray(sample(p, L, jax.random.PRNGKey(i), clean_x=cx, clean=jnp.array(keep[None]), lora=trn["lora"])))
        for name, g in (("none", g0), ("lora", g1)):
            res[name].append(edit_metrics(g, src[None], tr[None], np.array([tgt])))
        if i < 2:
            samples[i] = dict(src=src, edited=ed, none=g0[0], lora=g1[0], mask=pm)
    agg = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in res.items()}
    return dict(agg=agg, per=res, samples=samples)


# ---------------------------------------------------------------- E7: in-context LoRA (reference tokens)
CMAP = np.array([2, 3, 0, 1])          # red->blue, green->yellow, blue->red, yellow->green


def _ic_pairs(rng, n, task=0):
    src, lab = batch(rng, n)
    if task == 0:
        tgt = np.stack([recolor(src[i], CMAP[lab[i, 1]]) for i in range(n)])
    elif task == 1:                       # change shape: shape -> (shape + 1) % 3, same colour / trajectory / background
        tgt = []
        for i in range(n):
            m = measure(src[i:i + 1])[0]
            bg = _bg_of(src[i])
            tgt.append(render((lab[i, 0] + 1) % 3, lab[i, 1], [tuple(c) for c in m["cent"]], bg))
        tgt = np.stack(tgt)
    elif task == 2:                       # reverse time
        tgt = src[:, ::-1].copy()
    else:                                 # remove the object
        tgt = np.stack([np.repeat(_bg_of(src[i])[None], T, 0) for i in range(n)])
    return src, tgt, lab


def _bg_of(v):
    """Recover the static background: for each pixel take a frame where it is not saturated."""
    sat = v.max(-1) - v.min(-1)
    out = np.zeros(v.shape[1:])
    for y in range(H):
        for x in range(W):
            ok = np.where(sat[:, y, x] < 0.35)[0]
            out[y, x] = v[ok[0], y, x] if len(ok) else v[:, y, x].mean(0)
    return out


def ic_lora(cache_dir, steps=1500, rank=16, batch_size=16, seed=0):
    import jax
    import jax.numpy as jnp
    p = base_model(cache_dir)
    src, tgt, lab = _ic_pairs(np.random.default_rng(77), 48)
    tcol = CMAP[lab[:, 1]]
    tr = np.array([measure(src[i:i + 1])[0]["cent"] for i in range(48)])
    L = jnp.array([list(NONE)] * 48, jnp.int32)
    ref = jnp.array(to_tokens(src))

    def data(rng):
        s, t, _ = _ic_pairs(rng, batch_size)
        return dict(x0=to_tokens(t), labels=np.array([list(NONE)] * batch_size, np.int32),
                    clean=np.zeros((batch_size, T, NP), np.float32), ref=to_tokens(s))

    def evaluate(trn):
        g = from_tokens(np.asarray(sample(p, L, jax.random.PRNGKey(5), ref=ref, lora=trn["lora"])))
        return edit_metrics(g, src, tr, tcol)

    trn, log = finetune(p, data, steps, 1e-3, rank=rank, seed=seed, log_every=100, eval_fn=evaluate)
    out = dict(final=evaluate(trn), params=lora_params(trn["lora"]), log=log)
    g = from_tokens(np.asarray(sample(p, L, jax.random.PRNGKey(5), ref=ref, lora=trn["lora"])))
    out["samples"] = g[:4]; out["src"] = src[:4]; out["tgt"] = tgt[:4]
    g0 = from_tokens(np.asarray(sample(p, L, jax.random.PRNGKey(5), ref=ref)))
    out["no_lora"] = edit_metrics(g0, src, tr, tcol)
    Lt = lab.copy(); Lt[:, 1] = tcol
    g1 = from_tokens(np.asarray(sample(p, jnp.array(Lt), jax.random.PRNGKey(5))))
    out["labels_only"] = edit_metrics(g1, src, tr, tcol)
    out["lora"] = trn["lora"]
    return out


def shape_id(v):
    """Best IoU between the object mask (per frame) and the three rendered shapes at the same centroid."""
    sat = v.max(-1) - v.min(-1)
    scores = np.zeros(3)
    n = 0
    for t in range(T):
        m = sat[t] > 0.35
        if m.sum() < 4:
            continue
        yy, xx = np.nonzero(m)
        cx, cy = xx.mean() + 0.5, yy.mean() + 0.5
        for s in range(3):
            r = _mask(s, cx, cy)
            scores[s] += (m & r).sum() / (m | r).sum()
        n += 1
    return int(np.argmax(scores)) if n else -1


# ---------------------------------------------------------------- E8: instruction editing (one LoRA, four edits)
INSTR = ["改颜色", "改形状", "倒放", "移除物体"]


def instruct_lora(cache_dir, steps=2500, rank=16, batch_size=16, seed=0, n_eval=32):
    import jax
    import jax.numpy as jnp
    p = base_model(cache_dir)

    def data(rng):
        task = rng.integers(0, 4, batch_size)
        S, Tg = [], []
        for k in range(4):
            idx = np.where(task == k)[0]
            if len(idx) == 0:
                continue
            s, t, _ = _ic_pairs(rng, len(idx), task=k)
            S.append(s); Tg.append(t)
        S, Tg = np.concatenate(S), np.concatenate(Tg)
        inst = np.sort(task)
        return dict(x0=to_tokens(Tg), labels=np.array([list(NONE)] * batch_size, np.int32),
                    clean=np.zeros((batch_size, T, NP), np.float32), ref=to_tokens(S), inst=inst.astype(np.int32))

    trn, _ = finetune(p, data, steps, 1e-3, rank=rank, seed=seed, train_inst=True)
    pp = {**p, "inst": trn["inst"]}
    out = {"per_task": {}, "samples": {}}
    for k in range(4):
        src, tgt, lab = _ic_pairs(np.random.default_rng(500 + k), n_eval, task=k)
        g = from_tokens(np.asarray(sample(pp, jnp.array([list(NONE)] * n_eval, jnp.int32), jax.random.PRNGKey(k),
                                          ref=jnp.array(to_tokens(src)), lora=trn["lora"], inst=jnp.full((n_eval,), k, jnp.int32))))
        ms, mg = measure(src), measure(g)
        if k == 0:
            ok = [color_id(x["color"]) == CMAP[l[1]] for x, l in zip(mg, lab)]
        elif k == 1:
            ok = [shape_id(g[i]) == (lab[i, 0] + 1) % 3 for i in range(n_eval)]
        elif k == 2:
            ok = [abs_traj_err(x["cent"], s["cent"][::-1]) < 1.0 for x, s in zip(mg, ms)]
        else:
            ok = [x["count"].sum() < 8 for x in mg]
        out["per_task"][k] = dict(success=float(np.mean(ok)), psnr=float(np.mean([psnr(g[i], tgt[i]) for i in range(n_eval)])))
        out["samples"][k] = dict(src=src[:2], out=g[:2], tgt=tgt[:2])
    out["params"] = lora_params(trn["lora"])
    return out


# ---------------------------------------------------------------- E9: combining LoRAs
def combine(loras, scales):
    """Sum of several adapters: concatenate along the rank dimension (B rows scaled)."""
    import jax.numpy as jnp
    out = {}
    for lo, sc in zip(loras, scales):
        for k, (A, B) in lo.items():
            if k in out:
                out[k] = (jnp.concatenate([out[k][0], A], 1), jnp.concatenate([out[k][1], sc * B], 0))
            else:
                out[k] = (A, sc * B)
    return out


def merge_loras(cache_dir, scales=(0.5, 1.0, 1.5), n_eval=48):
    import jax
    import jax.numpy as jnp
    p = base_model(cache_dir)
    R1 = cached(cache_dir, "rank", rank_sweep, cache_dir=cache_dir)
    R3 = cached(cache_dir, "motion", motion_lora, cache_dir=cache_dir)
    bgL, mvL = R1["lora8"], R3["lora_temporal"]
    _, ref_tr, _ = ref_clip()
    rng = np.random.default_rng(13)
    elab = np.stack([rng.choice([0, 1, 2], n_eval), rng.choice([1, 2, 3], n_eval), np.full(n_eval, NONE[2])], 1).astype(np.int32)

    def stats(lo):
        v = _gen_eval(p, lo, elab)
        m = measure(v)
        sat = v.max(-1) - v.min(-1)
        return dict(bg=float(np.mean([v[i][sat[i] < 0.2].mean() for i in range(n_eval)])),
                    traj=float(np.nanmean([traj_err(x["cent"], ref_tr) for x in m])),
                    color_acc=float(np.mean([color_id(x["color"]) == l[1] for x, l in zip(m, elab)])), samples=v[:4])

    out = {"base": stats(None), "bg": stats(bgL), "motion": stats(mvL), "both": {}}
    for a in scales:
        for b in scales:
            out["both"][(a, b)] = stats(combine([bgL, mvL], [a, b]))
    return out

