"""Toy experiments for the State Space Models series (cached via SSM_CACHE).

Model: a small pre-LN residual network whose sequence mixer is one of
  "attn"    causal softmax attention (4 heads),
  "linattn" causal linear attention with feature map elu(x) + 1 (4 heads),
  "lti"     a gated diagonal state space layer with fixed (input-independent) step size, B and C (S4D-like),
  "mamba"   the same layer with input-dependent step size, B and C (selective scan, Mamba-like),
each followed by an MLP. Linear recurrences h_t = a_t * h_{t-1} + b_t are computed with jax.lax.associative_scan.
Tasks: character-level Tiny Shakespeare (SHAKESPEARE), selective copying, associative recall, and remembering the
first token over a long gap. Exact numpy checks: linear attention parallel == recurrent, SSM recurrence ==
convolution, parallel (Hillis-Steele) scan == sequential scan.
"""

import os
import pickle

import numpy as np


def cached(cdir, name, fn, **kw):
    key = "_".join(f"{k}{v}" for k, v in sorted(kw.items()) if k != "cache_dir")
    path = os.path.join(cdir, f"{name}_{key}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    res = fn(**kw)
    os.makedirs(cdir, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


# ---------------------------------------------------------------- exact numpy checks
def linattn_parallel(q, k, v):
    fq, fk = np.where(q > 0, q + 1, np.exp(q)), np.where(k > 0, k + 1, np.exp(k))
    A = np.tril(fq @ fk.T)
    return (A @ v) / A.sum(1, keepdims=True)


def linattn_recurrent(q, k, v):
    fq, fk = np.where(q > 0, q + 1, np.exp(q)), np.where(k > 0, k + 1, np.exp(k))
    S = np.zeros((q.shape[1], v.shape[1]))
    z = np.zeros(q.shape[1])
    out = []
    for t in range(len(q)):
        S += np.outer(fk[t], v[t])
        z += fk[t]
        out.append(fq[t] @ S / (fq[t] @ z))
    return np.array(out)


def ssm_discretize(A, B, dt):
    """Diagonal continuous system x' = A x + B u, zero-order hold."""
    Ab = np.exp(dt * A)
    Bb = (Ab - 1) / A * B
    return Ab, Bb


def ssm_recurrent(Ab, Bb, C, u):
    h = np.zeros_like(Ab)
    ys = []
    for x in u:
        h = Ab * h + Bb * x
        ys.append((C * h).sum().real)
    return np.array(ys)


def ssm_conv(Ab, Bb, C, u):
    T = len(u)
    K = np.array([(C * Ab ** t * Bb).sum().real for t in range(T)])
    return np.array([(K[:t + 1][::-1] * u[:t + 1]).sum() for t in range(T)]), K


def scan_sequential(a, b):
    h, out = 0.0, []
    for x, y in zip(a, b):
        h = x * h + y
        out.append(h)
    return np.array(out)


def scan_parallel(a, b):
    """Hillis-Steele inclusive scan with the operator (a1, b1) o (a2, b2) = (a1 a2, a2 b1 + b2). Returns result and
    the number of parallel steps."""
    a, b = np.array(a, float), np.array(b, float)
    n, step, d = len(a), 0, 1
    while d < n:
        a2, b2 = a.copy(), b.copy()
        a2[d:] = a[:-d] * a[d:]
        b2[d:] = a[d:] * b[:-d] + b[d:]
        a, b = a2, b2
        d *= 2
        step += 1
    return b, step


# ---------------------------------------------------------------- model
def init(key, mixers, V, d=64, N=16, expand=2, heads=4, dt_range=(1e-3, 1e-1), shift=False):
    import jax
    import jax.numpy as jnp
    E = expand * d
    ks = iter(jax.random.split(key, 64))
    g = lambda s, sc: jax.random.normal(next(ks), s) * sc
    p = dict(tok=g((V, d), 0.5), out=g((d, V), 1 / np.sqrt(d)), layers=[])
    if shift:
        p["prev"] = g((V, d), 0.5)                     # embedding of the previous token, added to the input
    for m in mixers:
        L = dict(w1=g((d, 4 * d), 1 / np.sqrt(d)), w2=g((4 * d, d), 1 / np.sqrt(4 * d) / 2))
        if m in ("attn", "linattn"):
            L.update(qkv=g((d, 3 * d), 1 / np.sqrt(d)), o=g((d, d), 1 / np.sqrt(d) / 2), pos=g((512, d), 0.05))
        else:
            dt = np.exp(np.random.default_rng(0).uniform(np.log(dt_range[0]), np.log(dt_range[1]), E))
            L.update(inp=g((d, 2 * E), 1 / np.sqrt(d)), conv=g((4, E), 0.5), o=g((E, d), 1 / np.sqrt(E) / 2),
                     A_log=jnp.log(jnp.tile(jnp.arange(1, N + 1, dtype=jnp.float32), (E, 1))), Dskip=jnp.ones(E),
                     dt_bias=jnp.asarray(np.log(np.expm1(dt)), jnp.float32))
            if m == "mamba":
                L.update(wB=g((E, N), 1 / np.sqrt(E)), wC=g((E, N), 1 / np.sqrt(E)), wdt=g((E, E), 0.1 / np.sqrt(E)))
            else:
                L.update(B=g((E, N), 1.0), C=g((E, N), 1 / np.sqrt(N)))
        p["layers"].append(L)
    return p


def _scan(a, b):
    import jax
    def comb(x, y):
        return x[0] * y[0], y[0] * x[1] + y[1]
    return jax.lax.associative_scan(comb, (a, b), axis=1)[1]


def mixer(L, m, x):
    import jax
    import jax.numpy as jnp
    Bsz, T, d = x.shape
    if m in ("attn", "linattn"):
        nh = 4
        x = x + L["pos"][:T]
        q, k, v = jnp.split(x @ L["qkv"], 3, -1)
        sh = lambda t: t.reshape(Bsz, T, nh, d // nh).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        mask = jnp.tril(jnp.ones((T, T)))
        if m == "attn":
            s = jnp.where(mask > 0, q @ k.transpose(0, 1, 3, 2) / np.sqrt(d // nh), -1e9)
            a = jax.nn.softmax(s, -1)
        else:
            fq, fk = jax.nn.elu(q) + 1, jax.nn.elu(k) + 1
            a = (fq @ fk.transpose(0, 1, 3, 2)) * mask
            a = a / (a.sum(-1, keepdims=True) + 1e-6)
        return (a @ v).transpose(0, 2, 1, 3).reshape(Bsz, T, d) @ L["o"]
    E = L["o"].shape[0]
    xz = x @ L["inp"]
    u, z = xz[..., :E], xz[..., E:]
    up = jnp.pad(u, ((0, 0), (3, 0), (0, 0)))
    u = sum(up[:, i:i + T] * L["conv"][i] for i in range(4))
    u = jax.nn.silu(u)
    A = -jnp.exp(L["A_log"])                                                  # (E, N)
    if m == "mamba":
        dt = jax.nn.softplus(u @ L["wdt"] + L["dt_bias"])                     # (B, T, E)
        Bm, Cm = u @ L["wB"], u @ L["wC"]                                       # (B, T, N)
        a = jnp.exp(dt[..., None] * A)                                          # (B, T, E, N)
        b = dt[..., None] * Bm[:, :, None, :] * u[..., None]
        h = _scan(a, b)
        y = (h * Cm[:, :, None, :]).sum(-1)
    else:
        dt = jax.nn.softplus(L["dt_bias"])                                      # (E,)
        a = jnp.broadcast_to(jnp.exp(dt[:, None] * A), (Bsz, T) + A.shape)
        b = (dt[:, None] * L["B"])[None, None] * u[..., None]
        h = _scan(a, b)
        y = (h * L["C"][None, None]).sum(-1)
    y = y + u * L["Dskip"]
    return (y * jax.nn.silu(z)) @ L["o"]


def forward(p, x, mixers):
    import jax.numpy as jnp
    ln = lambda h: (h - h.mean(-1, keepdims=True)) / jnp.sqrt(h.var(-1, keepdims=True) + 1e-5)
    h = p["tok"][x]
    if "prev" in p:
        h = h + p["prev"][jnp.pad(x, ((0, 0), (1, 0)))[:, :-1]]
    for L, m in zip(p["layers"], mixers):
        h = h + mixer(L, m, ln(h))
        h = h + jnp.maximum(0, ln(h) @ L["w1"]) @ L["w2"]
    return ln(h) @ p["out"]


def n_params(p):
    import jax
    return int(sum(np.prod(a.shape) for a in jax.tree_util.tree_leaves(p)))


def train(mixers, batch_fn, V, steps=2000, lr=3e-3, d=64, N=16, seed=0, eval_fn=None, eval_every=None, dt_range=(1e-3, 1e-1), batch_size=32, shift=False):
    """batch_fn(rng, n) -> (x, y, w): inputs, targets, loss weights (B, T)."""
    import jax
    import jax.numpy as jnp
    import optax
    p = init(jax.random.PRNGKey(seed), mixers, V, d=d, N=N, dt_range=dt_range, shift=shift)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.warmup_cosine_decay_schedule(0, lr, 100, steps, lr * 0.1), weight_decay=0.0))
    st = opt.init(p)

    def loss_fn(p, x, y, w):
        lp = jax.nn.log_softmax(forward(p, x, mixers))
        nll = -jnp.take_along_axis(lp, y[..., None], -1)[..., 0]
        return (nll * w).sum() / w.sum()

    @jax.jit
    def step(p, st, x, y, w):
        l, g = jax.value_and_grad(loss_fn)(p, x, y, w)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(seed)
    hist = []
    eval_every = eval_every or steps
    for it in range(steps + 1):
        if eval_fn is not None and (it % eval_every == 0):
            hist.append(dict(step=it, **eval_fn(p, mixers)))
        if it == steps:
            break
        x, y, w = batch_fn(rng, batch_size)
        p, st, _ = step(p, st, jnp.array(x), jnp.array(y), jnp.array(w, jnp.float32))
    return dict(hist=hist, final=hist[-1] if hist else None, params=n_params(p), p=jax.tree_util.tree_map(np.asarray, p))


# ---------------------------------------------------------------- tasks
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


def lm_batch(T=128):
    D = shakespeare()

    def fn(rng, n, src="train"):
        ids = D[src]
        i = rng.integers(0, len(ids) - T - 1, n)
        s = np.stack([ids[j:j + T + 1] for j in i])
        return s[:, :-1], s[:, 1:], np.ones((n, T))
    return fn


def lm_eval(T=128):
    import jax
    import jax.numpy as jnp
    fn = lm_batch(T)
    xs = [fn(np.random.default_rng(999 + k), 32, "val") for k in range(8)]

    def ev(p, mixers):
        tot = 0.0
        for x, y, _ in xs:
            lp = jax.nn.log_softmax(forward(p, jnp.array(x), mixers))
            tot += float(-jnp.take_along_axis(lp, jnp.array(y)[..., None], -1).mean())
        return dict(val=tot / len(xs))
    return ev


# selective copying: 0 = blank, 1..K = data tokens, K+1 = "go" marker
def selcopy_batch(L=64, n_data=4, K=8):
    def fn(rng, n):
        x = np.zeros((n, L + n_data), np.int32)
        y = np.zeros_like(x)
        w = np.zeros(x.shape)
        for i in range(n):
            pos = np.sort(rng.choice(L, n_data, replace=False))
            data = rng.integers(1, K + 1, n_data)
            x[i, pos] = data
            x[i, L:] = K + 1
            y[i, L:] = data
            w[i, L:] = 1
        return x, y, w
    return fn


def acc_eval(batch_fn, n=512):
    import jax.numpy as jnp
    x, y, w = batch_fn(np.random.default_rng(12345), n)

    def ev(p, mixers):
        pred = np.asarray(forward(p, jnp.array(x), mixers)).argmax(-1)
        m = w > 0
        tok = float((pred[m] == y[m]).mean())
        seq = float(np.mean([(pred[i][m[i]] == y[i][m[i]]).all() for i in range(len(x))]))
        return dict(tok_acc=tok, seq_acc=seq)
    return ev


# multi-query associative recall: keys 1..K, values K+1..2K; n_pairs (key value) pairs, then n_pairs queries
# (keys in random order), each followed by the answer (teacher forcing); loss at every query position.
def recall_batch(n_pairs=16, n_keys=32):
    def fn(rng, n):
        T = 4 * n_pairs
        x = np.zeros((n, T), np.int32)
        y = np.zeros_like(x)
        w = np.zeros(x.shape)
        for i in range(n):
            keys = rng.choice(n_keys, n_pairs, replace=False) + 1
            vals = rng.integers(0, n_keys, n_pairs) + n_keys + 1
            x[i, 0:2 * n_pairs:2], x[i, 1:2 * n_pairs:2] = keys, vals
            order = rng.permutation(n_pairs)
            qpos = 2 * n_pairs + 2 * np.arange(n_pairs)
            x[i, qpos], x[i, qpos + 1] = keys[order], vals[order]
            y[i, qpos], w[i, qpos] = vals[order], 1
        return x, y, w
    return fn


# ---------------------------------------------------------------- experiments
def _strip(r):
    return {k: v for k, v in r.items() if k != "p"}


def exp_lm(steps=2000):
    D = shakespeare()
    out = {}
    for name, mix in [("attn", ["attn", "attn"]), ("linattn", ["linattn", "linattn"]), ("lti", ["lti", "lti"]),
                      ("mamba", ["mamba", "mamba"]), ("hybrid", ["mamba", "attn"])]:
        out[name] = _strip(train(mix, lm_batch(), D["V"], steps=steps, eval_fn=lm_eval(), eval_every=steps // 10))
    return out


def exp_selcopy(steps=1500):
    bf = selcopy_batch()
    return {name: _strip(train(mix, bf, 10, steps=steps, eval_fn=acc_eval(bf), eval_every=steps // 10))
            for name, mix in [("attn", ["attn", "attn"]), ("lti", ["lti", "lti"]), ("mamba", ["mamba", "mamba"])]}


def exp_recall(steps=2000, n_pairs=16):
    bf = recall_batch(n_pairs=n_pairs)
    out = {"attn": _strip(train(["attn", "attn"], bf, 65, steps=steps, shift=True, eval_fn=acc_eval(bf)))}
    for N in (1, 4, 16, 64):
        out[("mamba", N)] = _strip(train(["mamba", "mamba"], bf, 65, steps=steps, N=N, shift=True, eval_fn=acc_eval(bf)))
    out["hybrid"] = _strip(train(["mamba", "attn"], bf, 65, steps=steps, shift=True, eval_fn=acc_eval(bf)))
    return out



def exp_lm_linpos(steps=2000):
    """Linear attention with the same learned position embedding as softmax attention."""
    D = shakespeare()
    return _strip(train(["linattn", "linattn"], lm_batch(), D["V"], steps=steps, eval_fn=lm_eval(), eval_every=steps // 10))
