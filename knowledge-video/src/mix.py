"""Toy experiments for the Mechanistic Interpretability series (all computed, cached via MI_CACHE).

- the tfm.py addition Transformer ("ab+cd=sss", 2 layers, 4 heads, d = 64): linear probes for the
  carry, random-network and shuffled-label controls, activation patching, head / MLP ablation,
  steering with a "carry" direction
- a 2-layer attention-only Transformer on repeated random sequences (induction heads)
- Toy Models of Superposition (5 features in 2 dimensions)
- sparse autoencoders on synthetic superposed activations
- modular addition: grokking and Fourier features
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


# ---------------------------------------------------------------- hooked forward (numpy)
def run(p, idx, patch=None, add=None, ablate=None, mean_cache=None, patch_comp=None):
    """Forward pass of the tfm model with hooks.
    Returns logits (B, T, V) and the residual stream resid[l] (B, T, d) for l = 0..L
    (0 = embeddings, l = after block l).
    patch: dict {(l, t): array (B, d)} replaces resid[l][:, t] before block l+1 (or before the unembed).
    add:   dict {(l, t): vector (d,)} added to resid[l][:, t].
    ablate: set of ("head", layer, h) or ("mlp", layer) replaced by their mean over mean_cache.
    patch_comp: dict {component: (t, array (B, d))} replaces that component's output at position t."""
    xp = np
    B, T = idx.shape
    H = p["heads"]
    x = p["tok"][idx] + p["pos"][:T]
    d = x.shape[-1]
    mask = np.tril(np.ones((T, T))) > 0
    resid, attns, comps = [], [], {}

    def hook(l, x):
        x = x.copy()
        for (ll, t), v in (patch or {}).items():
            if ll == l:
                x[:, t] = v
        for (ll, t), v in (add or {}).items():
            if ll == l:
                x[:, t] = x[:, t] + v
        return x

    x = hook(0, x)
    resid.append(x)
    for li, L in enumerate(p["layers"]):
        y = tfm._ln(x, L["g1"], xp)
        q, k, v = np.split(y @ L["qkv"], 3, axis=-1)
        sh = lambda t: t.reshape(B, T, H, d // H).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d // H)
        s = np.where(mask, s, -1e9)
        s = s - s.max(-1, keepdims=True)
        a = np.exp(s)
        a = a / a.sum(-1, keepdims=True)
        attns.append(a)
        z = a @ v                                       # (B, H, T, dh)
        Wo = L["o"].reshape(H, d // H, d)
        head_out = np.einsum("bhtk,hkd->bhtd", z, Wo)    # per-head contribution
        for h in range(H):
            comps[("head", li, h)] = head_out[:, h]
            if ablate and ("head", li, h) in ablate:
                head_out[:, h] = mean_cache[("head", li, h)]
            if patch_comp and ("head", li, h) in patch_comp:
                t, v = patch_comp[("head", li, h)]
                head_out[:, h, t] = v
        x = x + head_out.sum(1)
        y = tfm._ln(x, L["g2"], xp)
        m = np.maximum(0, y @ L["w1"]) @ L["w2"]
        comps[("mlp", li)] = m
        if ablate and ("mlp", li) in ablate:
            m = np.broadcast_to(mean_cache[("mlp", li)], m.shape)
        if patch_comp and ("mlp", li) in patch_comp:
            t, v = patch_comp[("mlp", li)]
            m = m.copy()
            m[:, t] = v
        x = x + m
        x = hook(li + 1, x)
        resid.append(x)
    logits = tfm._ln(x, p["gf"], xp) @ p["out"]
    return logits, resid, attns, comps


def addition_model(tf_cache):
    P, hist, snaps = tfm.cached_train(tf_cache, steps=600, log_every=20, snapshots=(0, 60, 120, 200, 300, 600))
    return P, snaps


def _digits(data):
    a = data[:, 0] * 10 + data[:, 1]
    b = data[:, 3] * 10 + data[:, 4]
    return a, b


def _logreg(Xtr, ytr, Xte, yte, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return m.score(sc.transform(Xtr), ytr), m.score(sc.transform(Xte), yte)


def probes(tf_cache, seed=0):
    """Linear probe for the carry (units of a + units of b >= 10) on the residual stream of the
    trained model and of the same architecture at random initialisation, at every layer and position;
    shuffled-label control on the trained model."""
    P, snaps = addition_model(tf_cache)
    TR, TE, _, _ = tfm.dataset()
    car = lambda D: ((D[:, 1] + D[:, 4]) >= 10).astype(int)
    ytr, yte = car(TR), car(TE)
    out = {}
    for name, p in (("trained", P), ("random", snaps[0])):
        _, rtr, _, _ = run(p, TR[:, :8])
        _, rte, _, _ = run(p, TE[:, :8])
        acc = np.zeros((len(rtr), 8))
        for l in range(len(rtr)):
            for t in range(8):
                acc[l, t] = _logreg(rtr[l][:, t], ytr, rte[l][:, t], yte)[1]
        out[name] = acc
    # shuffled labels at the best trained location
    _, rtr, _, _ = run(P, TR[:, :8])
    _, rte, _, _ = run(P, TE[:, :8])
    rng = np.random.default_rng(seed)
    l, t = 2, 6
    out["shuffled_train"], out["shuffled_test"] = _logreg(rtr[l][:, t], rng.permutation(ytr), rte[l][:, t], rng.permutation(yte), C=100.0)
    out["true_train"], out["true_test"] = _logreg(rtr[l][:, t], ytr, rte[l][:, t], yte, C=100.0)
    out["base_rate"] = float(max(yte.mean(), 1 - yte.mean()))
    return out


def _pairs_flip_carry(TE):
    """For each test problem, a corrupted partner with only the units digit of b changed so that the
    carry flips while the hundreds digit of the sum stays the same."""
    a, b = _digits(TE)
    rng = np.random.default_rng(0)
    clean, corr = [], []
    for x, y in zip(a, b):
        c = (x % 10 + y % 10) >= 10
        opts = [u for u in range(10) if ((x % 10 + u) >= 10) != c]
        if not opts:
            continue
        y2 = y - y % 10 + rng.choice(opts)
        if (x + y) // 100 != (x + y2) // 100:
            continue
        clean.append(tfm.encode(x, y))
        corr.append(tfm.encode(x, y2) [:6] + tfm.encode(x, y)[6:])   # same hundreds digit token
    return np.array(clean, np.int32), np.array(corr, np.int32)


def patching(tf_cache):
    """Activation patching for the tens digit (predicted at position 6). Clean and corrupted inputs differ
    only in the units digit of b (the carry flips). Patch the clean residual stream at (layer, position)
    into the corrupted run; score = (LD_patched - LD_corr) / (LD_clean - LD_corr), LD = logit(clean tens)
    - logit(corrupted tens), averaged over pairs."""
    P, _ = addition_model(tf_cache)
    _, TE, _, _ = tfm.dataset()
    cl, co = _pairs_flip_carry(TE)
    tc = cl[:, 7]
    a, b = _digits(co)
    tco = np.array([tfm.STOI[f"{x + y:03d}"[1]] for x, y in zip(a, b)])
    lc, rc, _, _ = run(P, cl[:, :7])
    lk, _, _, _ = run(P, co[:, :7])
    ld = lambda L: L[np.arange(len(L)), 6, tc] - L[np.arange(len(L)), 6, tco]
    LDc, LDk = ld(lc).mean(), ld(lk).mean()
    score = np.zeros((len(rc), 7))
    for l in range(len(rc)):
        for t in range(7):
            lp, _, _, _ = run(P, co[:, :7], patch={(l, t): rc[l][:, t]})
            score[l, t] = (ld(lp).mean() - LDk) / (LDc - LDk)
    _, _, attn_c, comps_c = run(P, cl[:, :7])
    comp = {}
    for key in comps_c:
        t = 6 if key[1] == 1 else 4
        lp, _, _, _ = run(P, co[:, :7], patch_comp={key: (t, comps_c[key][:, t])})
        comp[key] = float((ld(lp).mean() - LDk) / (LDc - LDk))
    attn6 = [a[:, :, 6].mean(0) for a in attn_c]          # where position 6 attends, per layer and head
    attn4 = [a[:, :, 4].mean(0) for a in attn_c]
    acc_clean = float((lc[:, 6].argmax(-1) == tc).mean())
    acc_corr = float((lk[:, 6].argmax(-1) == tco).mean())
    return dict(score=score, n=len(cl), LDc=float(LDc), LDk=float(LDk), acc_clean=acc_clean, acc_corr=acc_corr, comp=comp, attn6=attn6, attn4=attn4)


def ablation(tf_cache):
    """Mean-ablate each attention head and each MLP; teacher-forced accuracy of each answer digit on the
    2,000 test problems (hundreds at position 5, tens at 6, units at 7)."""
    P, _ = addition_model(tf_cache)
    TR, TE, _, _ = tfm.dataset()
    _, _, _, comps = run(P, TR[:2000, :8])
    mean = {k: v.mean(0) for k, v in comps.items()}
    def accs(abl=None):
        L, _, attn, _ = run(P, TE[:, :8], ablate=abl, mean_cache=mean)
        return [float((L[:, 5 + i].argmax(-1) == TE[:, 6 + i]).mean()) for i in range(3)], attn
    base, attn = accs()
    res = {}
    for k in comps:
        res[k] = accs({k})[0]
    return dict(base=base, res=res, attn=[a.mean(0) for a in attn])


def steering(tf_cache, layer=1, pos=4, alphas=(0.0, 0.5, 1.0, 1.5, 2.0)):
    """Carry direction = mean residual (layer, position) of carry problems - of no-carry problems (train
    set). Added (alpha x direction) to no-carry test problems, subtracted from carry problems; teacher-forced
    predictions of the tens digit (position 6) and of the units digit (position 7)."""
    P, _ = addition_model(tf_cache)
    TR, TE, _, _ = tfm.dataset()
    car = lambda D: (D[:, 1] + D[:, 4]) >= 10
    _, r, _, _ = run(P, TR[:, :8])
    v = r[layer][car(TR), pos].mean(0) - r[layer][~car(TR), pos].mean(0)
    out = dict(norm=float(np.linalg.norm(v)), resid_norm=float(np.linalg.norm(r[layer][:, pos], axis=1).mean()), up=[], down=[])
    for sgn, key, m in ((1, "up", ~car(TE)), (-1, "down", car(TE))):
        D = TE[m]
        for al in alphas:
            L, _, _, _ = run(P, D[:, :8], add={(layer, pos): sgn * al * v})
            tens, units = L[:, 6].argmax(-1), L[:, 7].argmax(-1)
            shifted = float((tens == (D[:, 7] + sgn) % 10).mean())
            out[key].append(dict(alpha=al, tens_shift=shifted, tens_same=float((tens == D[:, 7]).mean()),
                                 units_same=float((units == D[:, 8]).mean())))
    return out


# ---------------------------------------------------------------- induction heads
def _attn_only_init(key, V, T, d, H, layers):
    import jax
    ks = iter(jax.random.split(key, 3 + 2 * layers))
    n = lambda shape, sc: jax.random.normal(next(ks), shape) * sc
    return {"tok": n((V, d), 0.1), "pos": n((T, d), 0.1), "out": n((d, V), 1 / np.sqrt(d)),
            "layers": [{"qkv": n((d, 3 * d), 1 / np.sqrt(d)), "o": n((d, d), 1 / np.sqrt(d))} for _ in range(layers)]}


def _attn_only(p, idx, H):
    import jax.numpy as jnp
    B, T = idx.shape
    x = p["tok"][idx] + p["pos"][:T]
    d = x.shape[-1]
    mask = jnp.tril(jnp.ones((T, T))) > 0
    attns = []
    for L in p["layers"]:
        q, k, v = jnp.split(x @ L["qkv"], 3, axis=-1)
        sh = lambda t: t.reshape(B, T, H, d // H).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        s = jnp.where(mask, q @ k.transpose(0, 1, 3, 2) / np.sqrt(d // H), -1e9)
        a = jax_softmax(s)
        attns.append(a)
        x = x + (a @ v).transpose(0, 2, 1, 3).reshape(B, T, d) @ L["o"]
    return x @ p["out"], attns


def jax_softmax(s):
    import jax
    return jax.nn.softmax(s, -1)


def _rep_batch(rng, B, V, T):
    """A random segment of random length k (8..24) tiled to fill the sequence; mask = positions whose next
    token is a copy (everything after the first k tokens) - predictable only by looking back k tokens."""
    X = np.zeros((B, T), np.int64)
    M = np.zeros((B, T), bool)
    for i in range(B):
        k = rng.integers(8, 25)
        seg = rng.integers(0, V, k)
        X[i] = np.tile(seg, T // k + 1)[:T]
        M[i, k - 1:] = True
    return X, M


def induction(layers=2, V=32, T=48, d=64, H=2, steps=4000, lr=3e-3, seed=0):
    """Attention-only Transformer trained on next-token prediction of random sequences that contain one
    repeated segment of random length. Loss on repeated positions vs on random positions; previous-token
    score of layer-0 heads and induction score of the last layer's heads."""
    import jax
    import jax.numpy as jnp
    import optax
    p = _attn_only_init(jax.random.PRNGKey(seed), V, T, d, H, layers)
    opt = optax.adam(lr)
    st = opt.init(p)

    def loss(p, X):
        lg, _ = _attn_only(p, X[:, :-1], H)
        lp = jax.nn.log_softmax(lg, -1)
        return -jnp.take_along_axis(lp, X[:, 1:, None], -1)[..., 0]

    @jax.jit
    def step(p, st, X):
        g = jax.grad(lambda p: loss(p, X).mean())(p)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st

    ev_rng = np.random.default_rng(1234)
    Xe, Me = _rep_batch(ev_rng, 512, V, T)
    ev = jax.jit(loss)
    rng = np.random.default_rng(seed)
    hist = []
    for it in range(steps + 1):
        if it % 50 == 0:
            l = np.asarray(ev(p, jnp.array(Xe)))
            m = Me[:, :-1]
            hist.append(dict(step=it, rep=float(l[m].mean()), other=float(l[~m].mean())))
        if it < steps:
            X, _ = _rep_batch(rng, 64, V, T)
            p, st = step(p, st, jnp.array(X))
    # head scores on a clean repeated sequence: random 24 tokens repeated twice
    Xs = np.random.default_rng(7).integers(0, V, (64, 24))
    Xs = np.concatenate([Xs, Xs], 1)
    _, attns = _attn_only(p, jnp.array(Xs), H)
    attns = [np.asarray(a) for a in attns]
    tt = np.arange(1, 48)
    prev = [[float(attns[0][:, h, tt, tt - 1].mean()) for h in range(H)]]
    t2 = np.arange(24, 48)
    ind = [[float(a[:, h, t2, t2 - 23].mean()) for h in range(H)] for a in attns]
    return dict(hist=hist, prev=prev, ind=ind, attn_example=[a[0] for a in attns], seq_example=Xs[0], layers=layers)


# ---------------------------------------------------------------- superposition
def superposition(n=5, m=2, sparsities=(0.0, 0.7, 0.9, 0.97, 0.99), imp=0.9, steps=10000, seed=0):
    """Toy Models of Superposition: x in [0,1]^n, each feature is 0 with probability S; h = W x (m dims),
    x' = ReLU(W^T h + b); loss weighted by importance imp^i."""
    import jax
    import jax.numpy as jnp
    import optax
    I = imp ** np.arange(n)
    out = {}
    for S in sparsities:
        k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
        p = {"W": jax.random.normal(k1, (m, n)) * 0.3, "b": jnp.zeros(n)}
        opt = optax.adam(1e-2)
        st = opt.init(p)

        @jax.jit
        def step(p, st, key):
            ka, kb = jax.random.split(key)
            x = jax.random.uniform(ka, (1024, n)) * (jax.random.uniform(kb, (1024, n)) > S)
            def lf(p):
                xr = jax.nn.relu(x @ p["W"].T @ p["W"] + p["b"])
                return (I * (x - xr) ** 2).mean()
            g = jax.grad(lf)(p)
            u, st = opt.update(g, st)
            return optax.apply_updates(p, u), st

        key = jax.random.PRNGKey(seed + 1)
        for _ in range(steps):
            key, sk = jax.random.split(key)
            p, st = step(p, st, sk)
        W = np.asarray(p["W"])
        norms = np.linalg.norm(W, axis=0)
        Wn = W / np.maximum(norms, 1e-9)
        inter = np.array([sum((Wn[:, i] @ W[:, j]) ** 2 for j in range(n) if j != i) for i in range(n)])
        out[S] = dict(W=W, b=np.asarray(p["b"]), norms=norms, n_repr=int((norms > 0.5).sum()), interference=inter)
    return out


# ---------------------------------------------------------------- sparse autoencoders
def sae(n_feat=20, m=8, p_active=0.05, n_latent=40, l1s=(0.0, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0), steps=20000, seed=0):
    """Synthetic superposed activations: 20 random unit feature directions in 8 dims, each active with
    probability 5% with magnitude U[0, 1]. SAE: z = ReLU(W_e (x - b_d) + b_e), x' = W_d z + b_d with unit-norm
    decoder columns, loss = MSE + l1 * |z|_1. Recovered = true features whose best decoder column has
    cosine > 0.9. Also: best cosine of each feature with the 8 neuron (standard basis) directions."""
    import jax
    import jax.numpy as jnp
    import optax
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n_feat, m))
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    def data(r, N):
        c = r.uniform(0, 1, (N, n_feat)) * (r.uniform(0, 1, (N, n_feat)) < p_active)
        return c @ F, c
    Xte, Cte = data(np.random.default_rng(99), 20000)
    out = dict(F=F, neuron_cos=np.abs(F).max(1), res={})
    for l1 in l1s:
        k = jax.random.PRNGKey(seed)
        ka, kb = jax.random.split(k)
        Wd0 = jax.random.normal(kb, (n_latent, m))
        Wd0 = Wd0 / jnp.linalg.norm(Wd0, axis=1, keepdims=True)
        p = {"We": Wd0.T, "be": jnp.zeros(n_latent), "Wd": Wd0, "bd": jnp.zeros(m)}
        opt = optax.adam(3e-3)
        st = opt.init(p)

        def fwd(p, X):
            Wd = p["Wd"] / jnp.linalg.norm(p["Wd"], axis=1, keepdims=True)
            z = jax.nn.relu((X - p["bd"]) @ p["We"] + p["be"])
            return z, z @ Wd + p["bd"]

        @jax.jit
        def step(p, st, X):
            def lf(p):
                z, xr = fwd(p, X)
                return ((X - xr) ** 2).sum(1).mean() + l1 * jnp.abs(z).sum(1).mean()
            g = jax.grad(lf)(p)
            u, st = opt.update(g, st)
            return optax.apply_updates(p, u), st

        r = np.random.default_rng(seed + 1)
        for _ in range(steps):
            X, _ = data(r, 512)
            p, st = step(p, st, jnp.array(X))
        z, xr = fwd(p, jnp.array(Xte))
        z, xr = np.asarray(z), np.asarray(xr)
        Wd = np.asarray(p["Wd"]) / np.linalg.norm(np.asarray(p["Wd"]), axis=1, keepdims=True)
        cos = np.abs(F @ Wd.T)
        out["res"][l1] = dict(rel_mse=float(((Xte - xr) ** 2).sum(1).mean() / (Xte ** 2).sum(1).mean()),
                              l0=float((z > 1e-6).sum(1).mean()), recovered=int((cos.max(1) > 0.9).sum()),
                              best_cos=cos.max(1), true_l0=float((Cte > 0).sum(1).mean()), Wd=Wd)
    return out


# ---------------------------------------------------------------- modular addition / grokking
def grokking(P=59, frac=0.3, d=128, width=256, wd=0.3, lr=1e-3, steps=40000, seed=0, log_every=250):
    """(a + b) mod P with an MLP on summed embeddings: logits = W_out ReLU(W_in (E[a] + E[b])). Full-batch AdamW
    on 30% of all P^2 pairs. Train / test accuracy over time; Fourier power of the embedding over the
    token axis (frequencies 1..(P-1)/2) at several checkpoints."""
    import jax
    import jax.numpy as jnp
    import optax
    pairs = np.array([(a, b) for a in range(P) for b in range(P)])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    ntr = int(frac * len(pairs))
    tr, te = pairs[perm[:ntr]], pairs[perm[ntr:]]
    ytr, yte = (tr.sum(1) % P), (te.sum(1) % P)
    ks = jax.random.split(jax.random.PRNGKey(seed), 3)
    p = {"E": jax.random.normal(ks[0], (P, d)) / np.sqrt(d), "Win": jax.random.normal(ks[1], (d, width)) / np.sqrt(d),
         "Wout": jax.random.normal(ks[2], (width, P)) / np.sqrt(width)}
    opt = optax.adamw(lr, b1=0.9, b2=0.98, weight_decay=wd)
    st = opt.init(p)
    f = lambda p, X: jax.nn.relu((p["E"][X[:, 0]] + p["E"][X[:, 1]]) @ p["Win"]) @ p["Wout"]
    def lf(p, X, y):
        return -jnp.take_along_axis(jax.nn.log_softmax(f(p, X)), y[:, None], 1).mean()
    @jax.jit
    def step(p, st):
        l, g = jax.value_and_grad(lf)(p, tr, ytr)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l
    accj = jax.jit(lambda p, X, y: (f(p, X).argmax(1) == y).mean())
    def spectrum(E):
        Fh = np.fft.rfft(np.asarray(E), axis=0)[1:(P - 1) // 2 + 1]
        pw = (np.abs(Fh) ** 2).sum(1)
        return pw / pw.sum()

    def neuron_freq(p):
        """For each hidden neuron, its input as a function of one operand, (E W_in)[a, h], a = 0..P-1:
        share of its (non-constant) Fourier power in the single strongest frequency; and that frequency."""
        G = np.asarray(p["E"] @ p["Win"])
        Fh = np.fft.rfft(G, axis=0)[1:(P - 1) // 2 + 1]
        pw = np.abs(Fh) ** 2
        share = pw.max(0) / pw.sum(0)
        return share, pw.argmax(0) + 1
    hist, specs = [], {}
    tr, ytr, te, yte = jnp.array(tr), jnp.array(ytr), jnp.array(te), jnp.array(yte)
    for it in range(steps + 1):
        if it % log_every == 0:
            hist.append(dict(step=it, train=float(accj(p, tr, ytr)), test=float(accj(p, te, yte)),
                             wnorm=float(sum(float((v ** 2).sum()) for v in p.values()) ** 0.5)))
            specs[it] = spectrum(p["E"])
            sh, fr = neuron_freq(p)
            hist[-1]["neuron_share"] = float(np.median(sh))
            if it == steps:
                final_share, final_freq = sh, fr
        if it < steps:
            p, st, _ = step(p, st)
    return dict(hist=hist, specs=specs, P=P, n_train=int(ntr), n_test=int(len(pairs) - ntr), E=np.asarray(p["E"]),
                final_share=final_share, final_freq=final_freq, G=np.asarray(p["E"] @ p["Win"]))


if __name__ == "__main__":
    import sys
    np.set_printoptions(precision=3, suppress=True)
    which = sys.argv[1]
    if which == "ind":
        for L in (2, 1):
            r = induction(layers=L)
            print(L, [(h["step"], round(h["rep"], 2), round(h["other"], 2)) for h in r["hist"][::4]], r["prev"], r["ind"])
    if which == "sup":
        for S, v in superposition().items():
            print(S, v["n_repr"], v["norms"], v["interference"])
    if which == "sae":
        r = sae()
        print("neuron cos", r["neuron_cos"].mean())
        for l1, v in r["res"].items():
            print(l1, v["rel_mse"], v["l0"], v["recovered"], v["true_l0"])
    if which == "grok":
        r = grokking()
        for h in r["hist"][::8]:
            print(h)
        for k in (0, 1000, 5000, 30000):
            if k in r["specs"]:
                sp = r["specs"][k]
                print(k, np.sort(sp)[::-1][:6], np.argsort(sp)[::-1][:6] + 1)
