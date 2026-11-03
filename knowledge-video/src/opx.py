"""Toy experiments for the Optimization for Deep Learning series (all computed, cached via OP_CACHE).

- quadratic bowls: gradient descent, learning-rate thresholds, condition number, heavy-ball / Nesterov
- mini-batch SGD on the 8x8 digits: steps to a target loss vs batch size
- the tfm.py addition Transformer trained with different optimizers / schedules / normalisation
  placements / Muon / gradient clipping (generic trainer below)
- signal propagation in deep MLPs, muP vs standard parametrisation, bf16 weight updates
"""

import os
import pickle

import numpy as np

import tfm


def cached(cache_dir, name, fn, **kw):
    key = "_".join(f"{k}{v}" for k, v in sorted(kw.items()) if k != "tf_cache")
    path = os.path.join(cache_dir, f"{name}_{key}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    res = fn(**kw)
    os.makedirs(cache_dir, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


# ---------------------------------------------------------------- quadratic bowls
def gd_quadratic(lam=(1.0, 10.0), lrs=(0.05, 0.15, 0.19, 0.21), x0=(1.0, 1.0), steps=60):
    """f(x) = 0.5 * sum lam_i x_i^2; gradient descent paths for several learning rates."""
    lam = np.array(lam)
    out = {}
    for lr in lrs:
        x = np.array(x0, float)
        path = [x.copy()]
        for _ in range(steps):
            x = x - lr * lam * x
            path.append(x.copy())
        path = np.array(path)
        out[lr] = dict(path=path, f=0.5 * (lam * path ** 2).sum(1))
    return dict(lam=lam, runs=out)


def steps_to(eps, f):
    idx = np.where(np.asarray(f) <= eps)[0]
    return int(idx[0]) if len(idx) else None


def condition(kappas=(10, 100, 1000, 10000), eps=1e-6, max_steps=400000):
    """Steps for f to drop from f(x0) to eps * f(x0), x0 = (1, 1), eigenvalues (1/kappa... scaled so L = 1):
    lam = (1/kappa, 1). GD with lr = 2/(L+mu); heavy ball with the optimal (alpha, beta); Nesterov with
    lr = 1/L, beta = (sqrt(k)-1)/(sqrt(k)+1)."""
    res = {}
    for k in kappas:
        lam = np.array([1.0 / k, 1.0])
        f = lambda x: 0.5 * (lam * x * x).sum()
        f0 = f(np.ones(2))
        out = {}
        # GD
        lr = 2 / (lam.max() + lam.min())
        x = np.ones(2)
        n = None
        for t in range(1, max_steps):
            x = x - lr * lam * x
            if f(x) <= eps * f0:
                n = t
                break
        out["gd"] = n
        # heavy ball
        L, mu = lam.max(), lam.min()
        a = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2
        b = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
        x, xp = np.ones(2), np.ones(2)
        n = None
        for t in range(1, max_steps):
            x, xp = x - a * lam * x + b * (x - xp), x
            if f(x) <= eps * f0:
                n = t
                break
        out["hb"] = n
        # Nesterov
        q = (np.sqrt(L / mu) - 1) / (np.sqrt(L / mu) + 1)
        x, y = np.ones(2), np.ones(2)
        n = None
        for t in range(1, max_steps):
            xn = y - lam * y / L
            y = xn + q * (xn - x)
            x = xn
            if f(x) <= eps * f0:
                n = t
                break
        out["nag"] = n
        res[k] = out
    return res


def valley_paths(k=50.0, steps=60):
    """2-D paths on lam = (1/k, 1) from x0 = (10, 1): GD (lr = 2/(L+mu)) vs heavy ball (optimal)."""
    lam = np.array([1.0 / k, 1.0])
    x0 = np.array([10.0, 1.0])
    lr = 2 / (lam.max() + lam.min())
    x = x0.copy()
    gd = [x.copy()]
    for _ in range(steps):
        x = x - lr * lam * x
        gd.append(x.copy())
    L, mu = lam.max(), lam.min()
    a = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    b = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
    x, xp = x0.copy(), x0.copy()
    hb = [x.copy()]
    for _ in range(steps):
        x, xp = x - a * lam * x + b * (x - xp), x
        hb.append(x.copy())
    f = lambda P: 0.5 * (lam * np.array(P) ** 2).sum(1)
    return dict(lam=lam, gd=np.array(gd), hb=np.array(hb), f_gd=f(gd), f_hb=f(hb), a=a, b=b, lr=lr)


# ---------------------------------------------------------------- bf16 weight updates
def bf16_updates(lr=1e-3, steps=2000, seed=0):
    """Linear regression y = X w*, w* ~ N(0,1) (16 dims), weights initialised at 1. Full-batch GD with
    weights stored in fp32, in bf16 (update rounded into bf16 each step), or bf16 compute with an fp32
    master copy. Also: 1.0 + 1e-3 in bf16."""
    import ml_dtypes
    bf = ml_dtypes.bfloat16
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(512, 16)).astype(np.float32)
    ws = rng.normal(size=16).astype(np.float32)
    y = X @ ws
    loss = lambda w: float(((X @ w.astype(np.float32) - y) ** 2).mean())
    out = {}
    for mode in ("fp32", "bf16", "master"):
        w = np.ones(16, np.float32)
        wb = w.astype(bf)
        hist = []
        for t in range(steps + 1):
            cur = w if mode == "fp32" else (wb.astype(np.float32) if mode == "bf16" else w.astype(bf).astype(np.float32))
            if t % 20 == 0:
                hist.append(loss(cur))
            g = (2 / len(X)) * X.T @ (X @ cur - y)
            if mode == "fp32":
                w = w - lr * g
            elif mode == "bf16":
                wb = (wb.astype(np.float32) - lr * g).astype(bf)
            else:
                w = w - lr * g.astype(bf).astype(np.float32)
        out[mode] = np.array(hist)
    one = np.array(1.0, bf)
    out["one_plus"] = float((one.astype(np.float32) + np.float32(1e-3)).astype(bf).astype(np.float32))
    out["eps_bf16"] = float(2.0 ** -7)
    return out


# ---------------------------------------------------------------- deep MLP signal propagation
def signal(depth=50, width=256, n=512, seed=0):
    """Std of activations at each layer of a deep MLP at initialisation: tanh / ReLU with weight std 1/sqrt(n)
    scaled by gain; 'too small' (0.5x Xavier), Xavier for tanh, He for ReLU, 'too large'."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n, width))
    cfg = {"tanh_small": (np.tanh, 0.5), "tanh_xavier": (np.tanh, 1.0), "relu_xavier": (lambda z: np.maximum(z, 0), 1.0),
           "relu_he": (lambda z: np.maximum(z, 0), np.sqrt(2)), "relu_large": (lambda z: np.maximum(z, 0), 2.0)}
    out = {}
    for name, (act, gain) in cfg.items():
        r = np.random.default_rng(seed + 1)
        h = x0
        stds = [float(h.std())]
        for _ in range(depth):
            W = r.normal(size=(width, width)) * gain / np.sqrt(width)
            h = act(h @ W)
            stds.append(float(h.std()))
        out[name] = np.array(stds)
    return out


# ---------------------------------------------------------------- generic Transformer trainer (addition task)
def _forward(p, idx, heads, post_ln=False):
    import jax.numpy as jnp
    B, T = idx.shape
    x = p["tok"][idx] + p["pos"][:T]
    d = x.shape[-1]
    mask = jnp.tril(jnp.ones((T, T))) > 0
    ln = lambda x, g: tfm._ln(x, g, jnp)
    for L in p["layers"]:
        y = x if post_ln else ln(x, L["g1"])
        q, k, v = jnp.split(y @ L["qkv"], 3, axis=-1)
        sh = lambda t: t.reshape(B, T, heads, d // heads).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        s = jnp.where(mask, q @ k.transpose(0, 1, 3, 2) / np.sqrt(d // heads), -1e9)
        a = jnp.exp(s - s.max(-1, keepdims=True))
        a = a / a.sum(-1, keepdims=True)
        x = x + (a @ v).transpose(0, 2, 1, 3).reshape(B, T, d) @ L["o"]
        if post_ln:
            x = ln(x, L["g1"])
        y = x if post_ln else ln(x, L["g2"])
        x = x + jnp.maximum(0, y @ L["w1"]) @ L["w2"]
        if post_ln:
            x = ln(x, L["g2"])
    return ln(x, p["gf"]) @ p["out"]


def _muon(lr, beta=0.95, ns_steps=5, adam_lr=3e-3, wd=0.0):
    """Muon for the 2-D hidden matrices (qkv, o, w1, w2): momentum + Nesterov, then Newton-Schulz
    orthogonalisation; AdamW for embeddings, unembedding and gains."""
    import jax
    import jax.numpy as jnp
    import optax

    def ns(G):
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G / (jnp.linalg.norm(G) + 1e-7)
        tr = X.shape[0] > X.shape[1]
        if tr:
            X = X.T
        for _ in range(ns_steps):
            A = X @ X.T
            X = a * X + (b * A + c * A @ A) @ X
        return X.T if tr else X

    def init(params):
        return {"m": jax.tree_util.tree_map(jnp.zeros_like, params), "adam": optax.adamw(adam_lr, weight_decay=wd).init(params)}

    def is_hidden(path):
        return any(getattr(k, "key", None) in ("qkv", "o", "w1", "w2") for k in path)

    def update(g, st, params):
        m = jax.tree_util.tree_map(lambda m, g: beta * m + g, st["m"], g)
        au, ast = optax.adamw(adam_lr, weight_decay=wd).update(g, st["adam"], params)

        def one(path, gi, mi, ai):
            if is_hidden(path):
                u = ns(gi + beta * mi)
                return -lr * u * max(1.0, gi.shape[0] / gi.shape[1]) ** 0.5
            return ai
        upd = jax.tree_util.tree_map_with_path(one, g, m, au)
        return upd, {"m": m, "adam": ast}

    return optax.GradientTransformation(init, update)


def train_tf(opt="adam", lr=3e-3, steps=600, sched="constant", warmup=0, layers=2, d=64, heads=4, post_ln=False,
             batch=256, clip=None, seed=0, eval_every=20, wd=0.0, momentum=0.9):
    """Train the addition Transformer; returns history of (step, train_loss, test_loss, test_acc)."""
    import jax
    import jax.numpy as jnp
    import optax
    tr, te, _, _ = tfm.dataset()
    p = tfm.init(jax.random.PRNGKey(seed), d, heads, layers)
    p.pop("heads")
    if sched == "constant":
        s = optax.join_schedules([optax.linear_schedule(0, lr, warmup), optax.constant_schedule(lr)], [warmup]) if warmup else lr
    elif sched == "cosine":
        s = optax.warmup_cosine_decay_schedule(0, lr, max(warmup, 1), steps, 0.0)
    elif sched == "wsd":
        dec = int(0.2 * steps)
        s = optax.join_schedules([optax.linear_schedule(0, lr, max(warmup, 1)), optax.constant_schedule(lr),
                                  optax.linear_schedule(lr, 0.0, dec)], [max(warmup, 1), steps - dec])
    if opt == "sgd":
        o = optax.sgd(s)
    elif opt == "momentum":
        o = optax.sgd(s, momentum=momentum)
    elif opt == "adam":
        o = optax.adamw(s, weight_decay=wd)
    elif opt == "muon":
        o = _muon(s if not callable(s) else s, adam_lr=3e-3)
    if clip:
        o = optax.chain(optax.clip_by_global_norm(clip), o)
    st = o.init(p)

    def lf(p, b):
        lg = _forward(p, b[:, :-1], heads, post_ln)
        lp = jax.nn.log_softmax(lg[:, tfm.ANS - 1:], -1)
        return -jnp.take_along_axis(lp, b[:, tfm.ANS:, None], -1).mean()

    @jax.jit
    def step(p, st, b):
        l, g = jax.value_and_grad(lf)(p, b)
        gn = optax.global_norm(g)
        u, st = o.update(g, st, p)
        return optax.apply_updates(p, u), st, l, gn

    @jax.jit
    def evaluate(p):
        lg = _forward(p, te[:, :-1], heads, post_ln)
        pred = lg[:, tfm.ANS - 1:].argmax(-1)
        lp = jax.nn.log_softmax(lg[:, tfm.ANS - 1:], -1)
        loss = -jnp.take_along_axis(lp, te[:, tfm.ANS:, None], -1).mean()
        return loss, (pred == te[:, tfm.ANS:]).all(1).mean()

    rng = np.random.default_rng(seed)
    hist, losses, gns = [], [], []
    te = jnp.array(te)
    for i in range(steps + 1):
        if i % eval_every == 0:
            l, a = evaluate(p)
            hist.append((i, float(l), float(a)))
        if i < steps:
            p, st, l, gn = step(p, st, jnp.array(tr[rng.choice(len(tr), batch)]))
            losses.append(float(l))
            gns.append(float(gn))
    return dict(hist=np.array(hist), train=np.array(losses), gnorm=np.array(gns))


def _first(hist, thr=0.99):
    idx = np.where(hist[:, 2] >= thr)[0]
    return int(hist[idx[0], 0]) if len(idx) else None


def optimizers(steps=600):
    """Each optimizer with a learning-rate grid (constant lr, no warmup); keep every run."""
    grid = {"sgd": (0.1, 0.3, 1.0, 3.0), "momentum": (0.03, 0.1, 0.3, 1.0), "adam": (1e-3, 3e-3, 1e-2, 3e-2),
            "muon": (0.005, 0.01, 0.02, 0.05)}
    out = {}
    for opt, lrs in grid.items():
        out[opt] = {lr: train_tf(opt=opt, lr=lr, steps=steps) for lr in lrs}
    return out


def schedules(lr=3e-3, steps=1000, warmup=50):
    out = {s: train_tf(opt="adam", lr=lr, steps=steps, sched=s, warmup=warmup, eval_every=10) for s in ("constant", "cosine", "wsd")}
    return out


def warmup_test(lr=3e-2, steps=600):
    return {w: train_tf(opt="adam", lr=lr, steps=steps, warmup=w, eval_every=10) for w in (0, 100)}


def postln(lr=3e-3, steps=800, layers=6):
    out = {}
    for post in (False, True):
        for w in (0, 200):
            out[(post, w)] = train_tf(opt="adam", lr=lr, steps=steps, warmup=w, layers=layers, post_ln=post)
    return out


def clipping(lr=3e-2, steps=600, clip=1.0):
    return {c: train_tf(opt="adam", lr=lr, steps=steps, clip=c, eval_every=10) for c in (None, clip)}


# ---------------------------------------------------------------- batch size (MLP on digits)
def _mlp_sgd_steps(B, lr, Xtr, ytr, target, max_steps, seed=0):
    import jax
    import jax.numpy as jnp
    sizes = [64, 128, 10]
    k = jax.random.split(jax.random.PRNGKey(seed), 2)
    p = [(jax.random.normal(k[i], (a, b)) / np.sqrt(a), jnp.zeros(b)) for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]
    X, y = jnp.array(Xtr), jnp.array(ytr)

    def f(p, X):
        h = jnp.maximum(0, X @ p[0][0] + p[0][1])
        return h @ p[1][0] + p[1][1]

    def lf(p, X, y):
        return -jnp.take_along_axis(jax.nn.log_softmax(f(p, X)), y[:, None], 1).mean()

    @jax.jit
    def run(p, key):
        def body(c, k):
            p = c
            idx = jax.random.choice(k, len(Xtr), (B,))
            g = jax.grad(lf)(p, X[idx], y[idx])
            p = jax.tree_util.tree_map(lambda a, b: a - lr * b, p, g)
            return p, lf(p, X, y)
        return jax.lax.scan(body, p, jax.random.split(key, max_steps))

    _, L = run(p, jax.random.PRNGKey(seed + 1))
    L = np.asarray(L)
    ok = np.where(L <= target)[0]
    return (int(ok[0]) + 1 if len(ok) else None), L


def batch_size(batches=(4, 8, 16, 32, 64, 128, 256, 512, 1024), lrs=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0), target=0.1, max_steps=20000):
    """Plain SGD, MLP [64, 128, 10] on the 1,197 digits training images: for each batch size the fewest steps
    (best lr from the grid, full-train-set loss checked every step) to reach train loss <= target."""
    import dpx
    Xtr, ytr, _, _ = dpx.digits()
    out = {}
    for B in batches:
        best = None
        for lr in lrs:
            n, L = _mlp_sgd_steps(B, lr, Xtr, ytr, target, max_steps)
            if n is not None and (best is None or n < best[0]):
                best = (n, lr)
        out[B] = best
    return out


# ---------------------------------------------------------------- muP vs standard parametrisation
def mup(widths=(64, 128, 256, 512, 1024, 2048), lrs=tuple(2.0 ** np.arange(-14, -1)), steps=300, base=64, seed=0):
    """MLP [64, w, w, 10] on the digits, Adam, full batch, train loss after `steps`. SP: every layer lr, init
    N(0, 1/fan_in). muP (Adam): hidden and output layers use lr / m (m = w / base) and the logits are
    multiplied by 1/m."""
    import dpx
    import jax
    import jax.numpy as jnp
    import optax
    Xtr, ytr, _, _ = dpx.digits()
    X, y = jnp.array(Xtr), jnp.array(ytr)
    out = {"sp": {}, "mup": {}}
    for w in widths:
        m = w / base
        sizes = [64, w, w, 10]
        ks = jax.random.split(jax.random.PRNGKey(seed), 3)
        p0 = [(jax.random.normal(ks[i], (a, b)) / np.sqrt(a), jnp.zeros(b)) for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]
        for par in ("sp", "mup"):
            mult = 1 / m if par == "mup" else 1.0

            def f(p):
                h = jnp.maximum(0, X @ p[0][0] + p[0][1])
                h = jnp.maximum(0, h @ p[1][0] + p[1][1])
                return (h @ p[2][0] + p[2][1]) * mult

            lf = lambda p: -jnp.take_along_axis(jax.nn.log_softmax(f(p)), y[:, None], 1).mean()
            res = []
            for lr in lrs:
                if par == "mup":
                    tx = optax.multi_transform({"in": optax.adam(lr), "rest": optax.adam(lr / m)}, ["in", "rest", "rest"])
                else:
                    tx = optax.adam(lr)
                st = tx.init(p0)

                @jax.jit
                def run(p, st):
                    def body(c, _):
                        p, st = c
                        g = jax.grad(lf)(p)
                        u, st = tx.update(g, st, p)
                        return (optax.apply_updates(p, u), st), None
                    (p, st), _ = jax.lax.scan(body, (p, st), None, steps)
                    return lf(p)
                v = float(run(p0, st))
                res.append(v if np.isfinite(v) else np.nan)
            out[par][w] = np.array(res)
    out["lrs"] = np.array(lrs)
    return out
