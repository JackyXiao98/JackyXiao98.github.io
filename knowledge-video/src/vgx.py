"""Toy experiments for the Video Generation series (all small enough for a CPU).

- ring(): a 2D "data distribution" (16-component Gaussian ring) with an exact teacher (diff2d.GMM)
- distill_regression(): one-step student trained on (noise, 50-step DDIM sample) pairs
- train_dmd(): Distribution Matching Distillation (Yin et al., 2024) of the exact teacher into a
  one-step generator, with a learned "fake" score network
- selfforcing_toy(): next-frame prediction of a point moving on a circle, trained with teacher
  forcing vs. on its own rollouts (the idea behind Self-Forcing, Huang et al., 2025)
- attention helpers: INT8 block quantisation with/without K smoothing (SageAttention), and
  block-sparse + linear attention on toy video tokens (SLA)
Results of the trained toys are cached (VG_CACHE).
"""

import os
import pickle

import numpy as np

import diff2d as D

# ---------------------------------------------------------------- 2D ring data + teacher
RING_K, RING_R, RING_S = 16, 2.0, 0.25


def ring():
    a = np.arange(RING_K) * 2 * np.pi / RING_K
    return D.GMM(np.stack([np.cos(a), np.sin(a)], 1) * RING_R, RING_S)


def cached(cache_dir, name, fn, **kw):
    path = os.path.join(cache_dir, name + "_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items())) + ".pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    res = fn(**kw)
    os.makedirs(cache_dir, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


def _mlp_init(key, i, o, h=128, out_scale=0.1):
    import jax
    ks = jax.random.split(key, 3)
    return [(jax.random.normal(ks[0], (i, h)) / np.sqrt(i), np.zeros(h, np.float32)),
            (jax.random.normal(ks[1], (h, h)) / np.sqrt(h), np.zeros(h, np.float32)),
            (jax.random.normal(ks[2], (h, o)) / np.sqrt(h) * out_scale, np.zeros(o, np.float32))]


def _mlp(p, x):
    import jax
    for w, b in p[:-1]:
        x = jax.nn.silu(x @ w + b)
    return x @ p[-1][0] + p[-1][1]


def generator(p, z):
    return _mlp(p, z) * 10


def distill_regression(n_pairs=20000, steps=4000, seed=0):
    """One-step student fitted to (z, DDIM-50(z)) pairs from the exact teacher."""
    import jax
    import jax.numpy as jnp
    import optax
    g = ring()
    z = np.random.default_rng(seed).standard_normal((n_pairs, 2))
    x = D.ddim(g, z, 50)
    p = _mlp_init(jax.random.PRNGKey(seed), 2, 2)
    opt = optax.adam(optax.cosine_decay_schedule(1e-3, steps, 0.05))
    st = opt.init(p)

    @jax.jit
    def up(p, st, zb, xb):
        l, gr = jax.value_and_grad(lambda p: ((generator(p, zb) - xb) ** 2).sum(1).mean())(p)
        u, st = opt.update(gr, st)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(seed + 1)
    for _ in range(steps):
        i = rng.choice(n_pairs, 512)
        p, st, _ = up(p, st, jnp.array(z[i]), jnp.array(x[i]))
    return {"params": jax.tree_util.tree_map(np.asarray, p), "teacher_evals": n_pairs * 50}


def train_dmd(steps=8000, fake_updates=5, batch=512, seed=0, snaps=(0, 1000, 2000, 4000, 8000)):
    """DMD: generator gradient = (s_fake - s_real)(x_t), weighted as in Yin et al. (2024)."""
    import jax
    import jax.numpy as jnp
    import optax
    g = ring()
    MU = jnp.array(g.mu)
    abar = lambda t: jnp.exp(-(D.B0 * t + 0.5 * (D.B1 - D.B0) * t ** 2))

    def real_score(x, a):
        var = a * g.s2 + 1 - a
        diff = x[:, None, :] - jnp.sqrt(a)[:, None, None] * MU[None]
        r = jax.nn.softmax(-(diff ** 2).sum(-1) / (2 * var[:, None]), 1)
        return -(r[..., None] * diff).sum(1) / var[:, None]

    feat = lambda x, t: jnp.concatenate([x, t[:, None], jnp.sin(6 * t)[:, None]], 1)
    kg, kf = jax.random.split(jax.random.PRNGKey(seed))
    pg, pf = _mlp_init(kg, 2, 2), _mlp_init(kf, 4, 2)
    og, of = optax.adam(optax.cosine_decay_schedule(1e-3, steps, 0.05)), optax.adam(1e-3)
    sg, sf = og.init(pg), of.init(pf)

    @jax.jit
    def fstep(pf, sf, pg, key):
        k1, k2, k3 = jax.random.split(key, 3)
        x = jax.lax.stop_gradient(generator(pg, jax.random.normal(k1, (batch, 2))))
        t = jax.random.uniform(k2, (batch,), minval=0.02, maxval=0.98)
        a, e = abar(t), jax.random.normal(k3, (batch, 2))
        xt = jnp.sqrt(a)[:, None] * x + jnp.sqrt(1 - a)[:, None] * e
        l, gr = jax.value_and_grad(lambda pf: ((_mlp(pf, feat(xt, t)) - e) ** 2).mean())(pf)
        u, sf = of.update(gr, sf)
        return optax.apply_updates(pf, u), sf

    @jax.jit
    def gstep(pg, sg, pf, key):
        k1, k2, k3 = jax.random.split(key, 3)
        z = jax.random.normal(k1, (batch, 2))
        t = jax.random.uniform(k2, (batch,), minval=0.02, maxval=0.98)
        a, e = abar(t), jax.random.normal(k3, (batch, 2))

        def loss(pg):
            x = generator(pg, z)
            xt = jnp.sqrt(a)[:, None] * x + jnp.sqrt(1 - a)[:, None] * e
            sr = real_score(xt, a)
            sfk = -_mlp(pf, feat(xt, t)) / jnp.sqrt(1 - a)[:, None]
            x0r = (xt + (1 - a)[:, None] * sr) / jnp.sqrt(a)[:, None]
            w = 1 / (jnp.abs(x - x0r).mean(1, keepdims=True) + 1e-3)
            grad = jax.lax.stop_gradient((sfk - sr) * w)
            return 0.5 * ((x - jax.lax.stop_gradient(x - grad)) ** 2).sum(1).mean()

        gr = jax.grad(loss)(pg)
        u, sg = og.update(gr, sg)
        return optax.apply_updates(pg, u), sg

    key = jax.random.PRNGKey(seed + 1)
    out = {}
    for it in range(steps + 1):
        if it in snaps:
            out[it] = jax.tree_util.tree_map(np.asarray, pg)
        if it == steps:
            break
        for _ in range(fake_updates):
            key, k = jax.random.split(key)
            pf, sf = fstep(pf, sf, pg, k)
        key, k = jax.random.split(key)
        pg, sg = gstep(pg, sg, pf, k)
    return out


def sample_generator(p, z):
    import jax.numpy as jnp
    return np.asarray(generator(p, jnp.array(z, np.float32)))


# ---------------------------------------------------------------- Self-Forcing toy
def circle_clips(n, T, rng):
    w = rng.uniform(0.2, 0.4, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    a = ph[:, None] + w[:, None] * np.arange(T)
    return np.stack([np.cos(a), np.sin(a)], -1).astype(np.float32), w.astype(np.float32)


def selfforcing_toy(seed=0, steps=3000, rollout=20, test_len=200):
    """Next-frame model f(x_{t-1}, x_t) -> x_{t+1}. Teacher forcing: contexts are real frames.
    Self forcing (fine-tuned from it): contexts are the model's own rollouts; the target for each
    generated frame is what a teacher would produce from that same (self-generated) context:
    the previous frame projected back to the circle and rotated by the clip's speed."""
    import jax
    import jax.numpy as jnp
    import optax
    rng = np.random.default_rng(seed)
    TR, WTR = circle_clips(512, 40, rng)
    TE, WTE = circle_clips(256, test_len + 2, rng)

    def init(k, h=64):
        k1, k2, k3 = jax.random.split(k, 3)
        return dict(w1=jax.random.normal(k1, (4, h)) * .5, b1=jnp.zeros(h), w2=jax.random.normal(k2, (h, h)) / 8,
                    b2=jnp.zeros(h), w3=jax.random.normal(k3, (h, 2)) / 8, b3=jnp.zeros(2))

    def f(p, a, b):
        h = jnp.tanh(jnp.concatenate([a, b], -1) @ p["w1"] + p["b1"])
        h = jnp.tanh(h @ p["w2"] + p["b2"])
        return b + h @ p["w3"] + p["b3"]

    def teacher(b, w):
        u = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
        c, s = jnp.cos(w)[:, None], jnp.sin(w)[:, None]
        return jnp.concatenate([c * u[:, :1] - s * u[:, 1:], s * u[:, :1] + c * u[:, 1:]], -1)

    def tf_loss(p, X, w):
        return ((f(p, X[:, :-2], X[:, 1:-1]) - X[:, 2:]) ** 2).mean()

    def sf_loss(p, X, w):
        def step(c, _):
            a, b = c
            n = f(p, a, b)
            return (b, n), ((n - jax.lax.stop_gradient(teacher(b, w))) ** 2).sum(-1)
        _, ls = jax.lax.scan(step, (X[:, 0], X[:, 1]), None, length=rollout)
        return ls.mean()

    def roll(p, a, b, T):
        def step(c, _):
            a, b = c
            n = f(p, a, b)
            return (b, n), n
        _, ys = jax.lax.scan(step, (a, b), None, length=T)
        return jnp.swapaxes(ys, 0, 1)

    def train(lossf, p, lr):
        opt = optax.adam(optax.cosine_decay_schedule(lr, steps))
        st = opt.init(p)

        @jax.jit
        def up(p, st, X, w):
            g = jax.grad(lossf)(p, X, w)
            u, st = opt.update(g, st)
            return optax.apply_updates(p, u), st

        for _ in range(steps):
            i = rng.choice(len(TR), 64)
            o = rng.integers(0, 18)
            p, st = up(p, st, jnp.array(TR[i, o:]), jnp.array(WTR[i]))
        return p

    p_tf = train(tf_loss, init(jax.random.PRNGKey(seed)), 3e-3)
    p_sf = train(sf_loss, p_tf, 1e-3)
    out = {"test": TE, "w": WTE}
    for name, p in [("tf", p_tf), ("sf", p_sf)]:
        ys = np.asarray(roll(p, jnp.array(TE[:, 0]), jnp.array(TE[:, 1]), test_len))
        out[name] = ys
        out[name + "_1step"] = float(tf_loss(p, jnp.array(TE), None))
    return out


def rollout_metrics(ys, w):
    """Per-frame mean |radius - 1| and |angular step - true speed|."""
    rad = np.abs(np.sqrt((ys ** 2).sum(-1)) - 1).mean(0)
    ang = np.arctan2(ys[..., 1], ys[..., 0])
    da = np.angle(np.exp(1j * np.diff(ang, axis=1)))
    return rad, np.abs(da - w[:, None]).mean(0)


# ---------------------------------------------------------------- attention helpers
def softmax_attn(Q, K, V):
    s = Q @ K.T / np.sqrt(Q.shape[1])
    s -= s.max(1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(1, keepdims=True)
    return p @ V, p


def int8_blocks(x, block=128):
    """Symmetric INT8 quantisation with one scale per block of rows (dequantised back to float)."""
    out = np.empty_like(x)
    for i in range(0, len(x), block):
        b = x[i:i + block]
        sc = np.abs(b).max() / 127
        out[i:i + block] = np.round(b / sc).clip(-127, 127) * sc
    return out


def rel_err(a, ref):
    return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))


def toy_video_tokens(T=8, H=16, W=16, d=64, pos_scale=1.5, seed=0):
    """Tokens of a toy video on a T x H x W grid: queries/keys share random Fourier features of the
    3D position (nearby tokens attend to each other) plus content; values vary smoothly in space."""
    rng = np.random.default_rng(seed)
    t, y, x = np.meshgrid(np.arange(T), np.arange(H), np.arange(W), indexing="ij")
    pos = np.stack([t.ravel() * 2.0, y.ravel(), x.ravel()], 1)
    F = rng.standard_normal((3, d // 2)) / 3.0
    rff = lambda p: np.concatenate([np.cos(p @ F), np.sin(p @ F)], 1)
    N = len(pos)
    C = rng.standard_normal((N, d))
    Q = pos_scale * rff(pos) + 0.3 * C @ rng.standard_normal((d, d)) / 8
    K = pos_scale * rff(pos) + 0.3 * C @ rng.standard_normal((d, d)) / 8
    V = rff(pos * 0.3) @ rng.standard_normal((d, d)) / 4 + 0.3 * rng.standard_normal((N, d))
    return Q, K, V


def block_mass(P, B=64):
    nb = len(P) // B
    return P.reshape(nb, B, nb, B).mean((1, 3))


def sla(Q, K, V, M, keep, skip, B=64, fit_idx=None, seed=0):
    """Per query block: the top `keep` fraction of key blocks exactly (sparse softmax), the bottom
    `skip` fraction dropped, the rest by linear attention (phi = elu + 1) through a projection
    fitted by least squares on `fit_idx` queries (standing in for fine-tuning).
    Returns (sparse-only output, sparse + linear output)."""
    nb = len(M)
    nh, nl = max(1, int(round(keep * nb))), int(round(skip * nb))
    crit, marg = np.zeros_like(M, bool), np.zeros_like(M, bool)
    for i in range(nb):
        o = np.argsort(-M[i])
        crit[i, o[:nh]] = True
        marg[i, o[nh:nb - nl]] = True
    cm, mm = np.kron(crit, np.ones((B, B), bool)), np.kron(marg, np.ones((B, B), bool))
    s = np.where(cm, Q @ K.T / np.sqrt(Q.shape[1]), -np.inf)
    s -= s.max(1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(1, keepdims=True)
    Os = p @ V
    phi = lambda z: np.where(z > 0, z + 1, np.exp(z))
    Wl = (phi(Q) @ phi(K).T) * mm
    Ol = (Wl @ V) / (Wl.sum(1, keepdims=True) + 1e-9)
    O, _ = softmax_attn(Q, K, V)
    X = np.c_[Ol, np.ones(len(Q))]
    Wp, *_ = np.linalg.lstsq(X[fit_idx], (O - Os)[fit_idx], rcond=None)
    return Os, Os + X @ Wp
