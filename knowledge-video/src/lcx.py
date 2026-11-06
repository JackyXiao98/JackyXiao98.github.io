"""Toy experiments for the Long Context series (cached via LC_CACHE).

Model: 2-layer pre-LN causal Transformer (d = 64, 4 heads of 16, MLP 256) on character-level Tiny Shakespeare
(SHAKESPEARE) with one of five position schemes:
  "learned"  learned absolute position embeddings (a table of 1,024 rows; rows beyond the training length are never
             trained), "sin" fixed sinusoidal absolute embeddings, "rope" rotary position embedding (base 10,000),
  "alibi"    linear distance penalties on the attention scores (head slopes 1/2, 1/4, 1/8, 1/16), "nope" none.
RoPE context extension: position interpolation (positions divided by a factor) and NTK-aware base scaling, with or
without a short fine-tune at the longer length. Synthetic retrieval ("needle"): a key-value pair hidden among
distractor tokens, query at the end.
"""

import os
import pickle

import numpy as np

D, H, HD, LAYERS = 64, 4, 16, 2


def cached(cdir, name, fn, **kw):
    key = "_".join(f"{k}{v}" for k, v in sorted(kw.items()) if k != "cache_dir")
    path = os.path.join(cdir, f"{name}_{key}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    res = fn(**kw)
    os.makedirs(cdir, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


# ---------------------------------------------------------------- numpy check: RoPE scores depend on distance only
def rope_np(x, pos, base=10000.0):
    d = x.shape[-1]
    inv = base ** (-np.arange(0, d, 2) / d)
    ang = pos[:, None] * inv[None]
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2], out[..., 1::2] = x1 * c - x2 * s, x1 * s + x2 * c
    return out


# ---------------------------------------------------------------- model
def init(key, V, pos="rope", maxlen=1024):
    import jax
    ks = iter(jax.random.split(key, 32))
    g = lambda s, sc: jax.random.normal(next(ks), s) * sc
    p = dict(tok=g((V, D), 0.5), out=g((D, V), 1 / np.sqrt(D)), layers=[])
    if pos == "learned":
        p["pos"] = g((maxlen, D), 0.1)
    for _ in range(LAYERS):
        p["layers"].append(dict(qkv=g((D, 3 * D), 1 / np.sqrt(D)), o=g((D, D), 1 / np.sqrt(D) / 2),
                                w1=g((D, 4 * D), 1 / np.sqrt(D)), w2=g((4 * D, D), 1 / np.sqrt(4 * D) / 2)))
    return p


def _sin_table(T):
    pos = np.arange(T)[:, None]
    i = np.arange(0, D, 2)[None]
    t = np.zeros((T, D), np.float32)
    t[:, 0::2], t[:, 1::2] = np.sin(pos / 10000 ** (i / D)), np.cos(pos / 10000 ** (i / D))
    return t


def _rope(x, pos, base):
    import jax.numpy as jnp
    inv = base ** (-jnp.arange(0, HD, 2) / HD)
    ang = pos[:, None] * inv[None]
    c, s = jnp.cos(ang), jnp.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return jnp.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).reshape(x.shape)


def forward(p, x, pos="rope", scale=1.0, base=10000.0, return_attn=False):
    """scale: RoPE position interpolation factor (positions divided by scale); base: RoPE base (NTK scaling)."""
    import jax
    import jax.numpy as jnp
    B, T = x.shape
    ln = lambda h: (h - h.mean(-1, keepdims=True)) / jnp.sqrt(h.var(-1, keepdims=True) + 1e-5)
    h = p["tok"][x]
    if pos == "learned":
        h = h + p["pos"][:T]
    elif pos == "sin":
        h = h + jnp.asarray(_sin_table(T))
    mask = jnp.tril(jnp.ones((T, T))) > 0
    idx = jnp.arange(T, dtype=jnp.float32)
    dist = idx[:, None] - idx[None, :]
    slopes = jnp.array([2.0 ** -(i + 1) for i in range(H)])
    attns = []
    for L in p["layers"]:
        q, k, v = jnp.split(ln(h) @ L["qkv"], 3, -1)
        sh = lambda t: t.reshape(B, T, H, HD).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        if pos == "rope":
            q, k = _rope(q, idx / scale, base), _rope(k, idx / scale, base)
        s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(HD)
        if pos == "alibi":
            s = s - slopes[None, :, None, None] * dist[None, None]
        s = jnp.where(mask, s, -1e9)
        a = jax.nn.softmax(s, -1)
        attns.append(a)
        h = h + (a @ v).transpose(0, 2, 1, 3).reshape(B, T, D) @ L["o"]
        h = h + jnp.maximum(0, ln(h) @ L["w1"]) @ L["w2"]
    out = ln(h) @ p["out"]
    return (out, attns) if return_attn else out


_SH = None


def shakespeare():
    global _SH
    if _SH is None:
        path = os.environ.get("SHAKESPEARE", os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare.txt"))
        text = open(path).read()
        chars = sorted(set(text))
        ids = np.array([chars.index(c) for c in text], np.int32)
        n = int(len(ids) * 0.9)
        _SH = dict(train=ids[:n], val=ids[n:], V=len(chars))
    return _SH


def batch(ids, rng, n, T):
    i = rng.integers(0, len(ids) - T - 1, n)
    return np.stack([ids[j:j + T + 1] for j in i])


def train(pos="rope", T=64, steps=2000, lr=3e-3, seed=0, init_p=None, scale=1.0, base=10000.0, batch_size=32, warmup=100):
    import jax
    import jax.numpy as jnp
    import optax
    Dd = shakespeare()
    p = init(jax.random.PRNGKey(seed), Dd["V"], pos) if init_p is None else jax.tree_util.tree_map(jnp.asarray, init_p)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.warmup_cosine_decay_schedule(0, lr, warmup, steps, lr * 0.1)))
    st = opt.init(p)

    def loss_fn(p, x):
        lp = jax.nn.log_softmax(forward(p, x[:, :-1], pos, scale, base))
        return -jnp.take_along_axis(lp, x[:, 1:, None], -1).mean()

    @jax.jit
    def step(p, st, x):
        l, g = jax.value_and_grad(loss_fn)(p, x)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(seed)
    for _ in range(steps):
        p, st, _ = step(p, st, jnp.array(batch(Dd["train"], rng, batch_size, T)))
    return jax.tree_util.tree_map(np.asarray, p)


def loss_by_position(p, pos, T, scale=1.0, base=10000.0, n=64, seed=999):
    """Mean next-character loss at each position of validation windows of length T."""
    import jax
    import jax.numpy as jnp
    Dd = shakespeare()
    x = batch(Dd["val"], np.random.default_rng(seed), n, T)
    out = []
    for i in range(0, n, 16):
        xb = jnp.array(x[i:i + 16])
        lp = jax.nn.log_softmax(forward(p, xb[:, :-1], pos, scale, base))
        out.append(np.asarray(-jnp.take_along_axis(lp, xb[:, 1:, None], -1)[..., 0]))
    return np.concatenate(out).mean(0)


# ---------------------------------------------------------------- synthetic needle retrieval
# tokens: noise 0..15, keys 16..23, values 24..31, key marker 32, query marker 33; V = 34
def needle_batch(T=128, depth=None):
    def fn(rng, n):
        x = rng.integers(0, 16, (n, T + 1)).astype(np.int32)
        y = np.zeros((n, T), np.int32)
        w = np.zeros((n, T))
        for i in range(n):
            k, v = rng.integers(16, 24), rng.integers(24, 32)
            d = int(rng.integers(0, T - 6)) if depth is None else int(depth * (T - 6))
            x[i, d:d + 3] = (32, k, v)
            x[i, T - 2:T] = (33, k)
            y[i, T - 1], w[i, T - 1] = v, 1
        return x[:, :T], y, w
    return fn


def train_task(batch_fn, V, pos="rope", steps=1500, lr=3e-3, seed=0):
    import jax
    import jax.numpy as jnp
    import optax
    p = init(jax.random.PRNGKey(seed), V, pos)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.warmup_cosine_decay_schedule(0, lr, 100, steps, lr * 0.1)))
    st = opt.init(p)

    def loss_fn(p, x, y, w):
        lp = jax.nn.log_softmax(forward(p, x, pos))
        return (-jnp.take_along_axis(lp, y[..., None], -1)[..., 0] * w).sum() / w.sum()

    @jax.jit
    def step(p, st, x, y, w):
        l, g = jax.value_and_grad(loss_fn)(p, x, y, w)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(seed)
    for _ in range(steps):
        x, y, w = batch_fn(rng, 32)
        p, st, _ = step(p, st, jnp.array(x), jnp.array(y), jnp.array(w, jnp.float32))
    return jax.tree_util.tree_map(np.asarray, p)


def needle_acc(p, pos, T, depth, n=256, seed=7):
    import jax.numpy as jnp
    x, y, w = needle_batch(T, depth)(np.random.default_rng(seed), n)
    pred = np.asarray(forward(p, jnp.array(x), pos))[:, -1].argmax(-1)
    return float((pred == y[:, -1]).mean())


# ---------------------------------------------------------------- experiments
POS = ("learned", "sin", "rope", "alibi", "nope")
DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def exp_extrap(steps=2000):
    out = {}
    for pos in POS:
        p = train(pos=pos, T=64, steps=steps)
        out[pos] = dict(curve=loss_by_position(p, pos, 512), p=p if pos == "rope" else None)
    return out


def exp_extend(cache_dir, ft_steps=200):
    rp = cached(cache_dir, "extrap", exp_extrap)["rope"]["p"]
    T = 256
    ntk_base = 10000.0 * 4 ** (HD / (HD - 2))
    out = {"none": loss_by_position(rp, "rope", T), "pi": loss_by_position(rp, "rope", T, scale=4.0),
           "ntk": loss_by_position(rp, "rope", T, base=ntk_base)}
    ft = train(pos="rope", T=T, steps=ft_steps, lr=1e-3, init_p=rp, warmup=20)
    out["ft"] = loss_by_position(ft, "rope", T)
    ftpi = train(pos="rope", T=T, steps=ft_steps, lr=1e-3, init_p=rp, scale=4.0, warmup=20)
    out["ft_pi"] = loss_by_position(ftpi, "rope", T, scale=4.0)
    out["ntk_base"] = ntk_base
    return out


def exp_needle(steps=1500):
    out = {}
    for pos in ("rope", "alibi", "nope"):
        p = train_task(needle_batch(128), 34, pos=pos, steps=steps)
        out[pos] = {T: [needle_acc(p, pos, T, dp) for dp in DEPTHS] for T in (128, 512)}
    return out
