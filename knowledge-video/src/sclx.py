"""Toy experiments for the Scaling Laws series (cached via SCL_CACHE).

Data: Tiny Shakespeare (Karpathy, char-rnn; 1.1 M characters, 65 distinct characters), character level; the last
10% is held out. Path from the environment variable SHAKESPEARE (default ../data/shakespeare.txt).
Model: pre-LN causal Transformer, 2 layers, width d (heads = d / 16, FFN 4d), context 64, learned positions.
"Parameters" N counts the non-embedding parameters (12 d^2 per layer), as in Kaplan et al.; training compute
C = 6 N D with D = training tokens.
"""

import os
import pickle

import numpy as np

CTX = 64


def cached(cdir, name, fn, **kw):
    key = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    path = os.path.join(cdir, f"{name}_{key}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    res = fn(**kw)
    os.makedirs(cdir, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


_DATA = None


def data():
    global _DATA
    if _DATA is None:
        path = os.environ.get("SHAKESPEARE", os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare.txt"))
        text = open(path).read()
        chars = sorted(set(text))
        ids = np.array([chars.index(c) for c in text], np.int32)
        n = int(len(ids) * 0.9)
        _DATA = dict(train=ids[:n], val=ids[n:], chars=chars, V=len(chars))
    return _DATA


def batch(ids, rng, b):
    i = rng.integers(0, len(ids) - CTX - 1, b)
    return np.stack([ids[j:j + CTX + 1] for j in i])


def init(key, d, layers, V):
    import jax
    import jax.numpy as jnp
    ks = iter(jax.random.split(key, 4 + 6 * layers))
    g = lambda s, sc: jax.random.normal(next(ks), s) * sc
    p = dict(tok=g((V, d), 0.5), pos=g((CTX, d), 0.1), out=g((d, V), 1 / np.sqrt(d)), layers=[])
    for _ in range(layers):
        p["layers"].append(dict(qkv=g((d, 3 * d), 1 / np.sqrt(d)), o=g((d, d), 1 / np.sqrt(d) / np.sqrt(2 * layers)),
                                w1=g((d, 4 * d), 1 / np.sqrt(d)), w2=g((4 * d, d), 1 / np.sqrt(4 * d) / np.sqrt(2 * layers))))
    return p


def forward(p, x):
    import jax.numpy as jnp
    B, T = x.shape
    d = p["tok"].shape[1]
    nh = max(1, d // 16)
    ln = lambda h: (h - h.mean(-1, keepdims=True)) / jnp.sqrt(h.var(-1, keepdims=True) + 1e-5)
    h = p["tok"][x] + p["pos"][:T]
    mask = jnp.tril(jnp.ones((T, T))) > 0
    for L in p["layers"]:
        q, k, v = jnp.split(ln(h) @ L["qkv"], 3, -1)
        sh = lambda t: t.reshape(B, T, nh, d // nh).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        s = jnp.where(mask, q @ k.transpose(0, 1, 3, 2) / np.sqrt(d // nh), -1e9)
        a = jnp.exp(s - s.max(-1, keepdims=True))
        a = a / a.sum(-1, keepdims=True)
        h = h + (a @ v).transpose(0, 2, 1, 3).reshape(B, T, d) @ L["o"]
        h = h + jnp.maximum(0, ln(h) @ L["w1"]) @ L["w2"]
    return ln(h) @ p["out"]


def n_nonemb(d, layers=2):
    return 12 * d * d * layers


def train(d=64, layers=2, steps=3000, batch_size=32, lr=None, seed=0, train_chars=None, eval_every=None, warmup=100,
          eval_batches=8, keep_params=False):
    """lr default: 3e-3 * sqrt(64 / d). train_chars: restrict the training set to its first N characters."""
    import jax
    import jax.numpy as jnp
    import optax
    D = data()
    tr = D["train"] if train_chars is None else D["train"][:train_chars]
    lr = lr if lr is not None else 3e-3 * np.sqrt(64 / d)
    p = init(jax.random.PRNGKey(seed), d, layers, D["V"])
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.warmup_cosine_decay_schedule(0, lr, warmup, steps, lr * 0.1), weight_decay=0.0))
    st = opt.init(p)

    def loss_fn(p, x):
        lg = forward(p, x[:, :-1])
        return -jnp.take_along_axis(jax.nn.log_softmax(lg), x[:, 1:, None], -1).mean()

    @jax.jit
    def step(p, st, x):
        l, g = jax.value_and_grad(loss_fn)(p, x)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    lossj = jax.jit(loss_fn)
    erng = np.random.default_rng(999)
    ev = [jnp.array(batch(D["val"], erng, 64)) for _ in range(eval_batches)]
    trev = [jnp.array(batch(tr, erng, 64)) for _ in range(eval_batches)]
    evalf = lambda p, bs: float(np.mean([float(lossj(p, b)) for b in bs]))
    rng = np.random.default_rng(seed)
    eval_every = eval_every or max(1, steps // 20)
    hist = []
    for it in range(steps + 1):
        if it % eval_every == 0 or it == steps:
            hist.append(dict(step=it, tokens=it * batch_size * CTX, val=evalf(p, ev), train=evalf(p, trev)))
        if it == steps:
            break
        p, st, _ = step(p, st, jnp.array(batch(tr, rng, batch_size)))
    out = dict(hist=hist, val=hist[-1]["val"], d=d, layers=layers, N=n_nonemb(d, layers), lr=lr, steps=steps)
    if keep_params:
        out["p"] = jax.tree_util.tree_map(np.asarray, p)
    return out


# ---------------------------------------------------------------- experiments
WIDTHS = (16, 24, 32, 48, 64, 96, 128)
STEPS = 4000


def exp_size():
    return {d: train(d=d, steps=STEPS, keep_params=d in (16, 32, 64, 128)) for d in WIDTHS}


def exp_data():
    """Fixed width 64; training set restricted to the first N characters; 3,000 steps (many epochs for small N)."""
    return {n: train(d=64, steps=3000, train_chars=n) for n in (10_000, 30_000, 100_000, 300_000, 1_003_854)}


def exp_lr():
    return {(d, lr): train(d=d, steps=1500, lr=lr) for d in (32, 64, 128) for lr in (1e-3, 3e-3, 1e-2, 3e-2)}


def exp_repeat():
    """Width 64, 3,000 steps (6.1 M tokens) drawn from unique pools of decreasing size -> 1 ... 48 epochs."""
    total = 3000 * 32 * CTX
    return {ep: train(d=64, steps=3000, train_chars=int(total / ep) + CTX + 1) for ep in (6.1, 12, 24, 48)}
