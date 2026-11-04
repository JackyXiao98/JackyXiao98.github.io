"""Toy experiments for the Mixture of Experts series (cached via MOE_CACHE).

Data: a mixture of 8 hidden "languages" over a 32-token vocabulary. Each sequence (length 32) is drawn from one
language, a first-order Markov chain with its own sharp random transition table; the model must infer the
language from the context to predict the next token. Floor = the true conditional entropy (given the language).

Model: 2-layer causal Transformer (d = 64, 4 heads); every feed-forward block is either dense (hidden h) or a
mixture of experts (E experts of hidden h_e, top-k routing) with options: Switch auxiliary load-balancing loss,
DeepSeek-V3-style auxiliary-loss-free bias balancing, expert capacity with token dropping, router z-loss, shared
experts, and initialisation from a trained dense model (upcycling). Experts are computed densely and masked (fine
for a toy).
"""

import os
import pickle

import numpy as np

V, LEN, K = 32, 32, 8
D, HEADS, LAYERS = 64, 4, 2


def cached(cdir, name, fn, **kw):
    key = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    path = os.path.join(cdir, f"{name}_{key}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    res = fn(**kw)
    os.makedirs(cdir, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


# ---------------------------------------------------------------- data
def languages(seed=0, beta=3.0):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(K, V, V)) * beta
    P = np.exp(G - G.max(-1, keepdims=True))
    return P / P.sum(-1, keepdims=True)


PT = languages()


def sample(rng, n):
    lang = rng.integers(0, K, n)
    x = np.zeros((n, LEN + 1), np.int32)
    x[:, 0] = rng.integers(0, V, n)
    for t in range(1, LEN + 1):
        p = PT[lang, x[:, t - 1]]
        x[:, t] = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
    return x, lang


def floor_entropy():
    """Average entropy of the next token given the language and the previous token, under the data distribution."""
    rng = np.random.default_rng(999)
    x, lang = sample(rng, 4000)
    p = PT[lang[:, None], x[:, :-1], x[:, 1:]]
    return float(-np.log(p).mean())


# ---------------------------------------------------------------- model
def init(key, ffn="dense", hidden=128, E=8, h_e=128, shared=0, h_s=0):
    import jax
    ks = iter(jax.random.split(key, 64))
    n = lambda shape, sc: jax.random.normal(next(ks), shape) * sc
    p = {"tok": n((V, D), 0.1), "pos": n((LEN, D), 0.1), "out": n((D, V), 1 / np.sqrt(D)), "layers": []}
    for _ in range(LAYERS):
        L = {"qkv": n((D, 3 * D), 1 / np.sqrt(D)), "o": n((D, D), 0.5 / np.sqrt(D))}
        if ffn == "dense":
            L["w1"], L["w2"] = n((D, hidden), 1 / np.sqrt(D)), n((hidden, D), 0.5 / np.sqrt(hidden))
        else:
            L["router"] = n((D, E), 0.01)
            L["e1"], L["e2"] = n((E, D, h_e), 1 / np.sqrt(D)), n((E, h_e, D), 0.5 / np.sqrt(h_e))
            if shared:
                L["s1"], L["s2"] = n((D, h_s * shared), 1 / np.sqrt(D)), n((h_s * shared, D), 0.5 / np.sqrt(h_s * shared))
        p["layers"].append(L)
    return p


def n_params(p, active_k=None):
    """Total parameters; with active_k, parameters touched per token (experts counted k times instead of E)."""
    import jax
    tot = 0
    for path, a in jax.tree_util.tree_flatten_with_path(p)[0]:
        name = getattr(path[-1], "key", "")
        if name in ("e1", "e2") and active_k is not None:
            tot += a.size // a.shape[0] * active_k
        else:
            tot += a.size
    return int(tot)


def _ln(x):
    import jax.numpy as jnp
    m = x.mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(((x - m) ** 2).mean(-1, keepdims=True) + 1e-5)


def moe(L, h, cfg, bias):
    """h: (N, D) tokens. Returns output (N, D) and stats."""
    import jax
    import jax.numpy as jnp
    E, k = L["router"].shape[1], cfg.get("k", 1)
    logits = h @ L["router"]
    probs = jax.nn.softmax(logits, -1)
    sel_score = logits + (bias if bias is not None else 0)
    topv, topi = jax.lax.top_k(sel_score, k)
    mask = jax.nn.one_hot(topi, E).sum(1)                                   # (N, E) 0/1
    gate = probs * mask
    if k > 1:
        gate = gate / (gate.sum(-1, keepdims=True) + 1e-9)
    N = h.shape[0]
    cf = cfg.get("capacity")
    dropped = jnp.zeros(())
    if cf is not None:
        C = jnp.ceil(cf * N * k / E)
        pos = jnp.cumsum(mask, 0) * mask                                     # 1-based position in each expert queue
        keep = (pos <= C) & (mask > 0)
        dropped = 1 - keep.sum() / mask.sum()
        gate = gate * keep
        mask = mask * keep
    hid = jnp.maximum(0, jnp.einsum("nd,edh->neh", h, L["e1"]))
    out_e = jnp.einsum("neh,ehd->ned", hid, L["e2"])
    out = (gate[..., None] * out_e).sum(1)
    if "s1" in L:
        out = out + jnp.maximum(0, h @ L["s1"]) @ L["s2"]
    frac = mask.sum(0) / (mask.sum() + 1e-9)                                # share of routed assignments per expert
    aux = E * (jax.lax.stop_gradient(jax.nn.one_hot(topi[:, 0], E).mean(0)) * probs.mean(0)).sum()
    z = (jax.nn.logsumexp(logits, -1) ** 2).mean()
    return out, dict(frac=frac, aux=aux, z=z, dropped=dropped, maxlogit=jnp.abs(logits).max(), topi=topi[:, 0],
                     maxprob=probs.max(-1).mean())


def forward(p, x, cfg, biases=None):
    import jax.numpy as jnp
    B, T = x.shape
    h = p["tok"][x] + p["pos"][:T]
    mask = jnp.tril(jnp.ones((T, T))) > 0
    stats = []
    for li, L in enumerate(p["layers"]):
        y = _ln(h)
        q, k_, v = jnp.split(y @ L["qkv"], 3, -1)
        sh = lambda t: t.reshape(B, T, HEADS, D // HEADS).transpose(0, 2, 1, 3)
        q, k_, v = sh(q), sh(k_), sh(v)
        s = jnp.where(mask, q @ k_.transpose(0, 1, 3, 2) / np.sqrt(D // HEADS), -1e9)
        a = jnp.exp(s - s.max(-1, keepdims=True))
        a = a / a.sum(-1, keepdims=True)
        h = h + (a @ v).transpose(0, 2, 1, 3).reshape(B, T, D) @ L["o"]
        y = _ln(h)
        if "w1" in L:
            h = h + jnp.maximum(0, y @ L["w1"]) @ L["w2"]
        else:
            o, st = moe(L, y.reshape(B * T, D), cfg, None if biases is None else biases[li])
            h = h + o.reshape(B, T, D)
            stats.append(st)
    return _ln(h) @ p["out"], stats


def train(ffn="dense", hidden=128, E=8, h_e=128, k=1, aux=0.0, bias_rate=0.0, capacity=None, zloss=0.0, shared=0, h_s=0,
          steps=3000, batch=32, lr=3e-3, seed=0, init_from=None, eval_every=100, log_route=False):
    import jax
    import jax.numpy as jnp
    import optax
    p = init(jax.random.PRNGKey(seed), ffn, hidden, E, h_e, shared, h_s)
    if init_from is not None and ffn == "dense":    # continue training a dense model
        p = jax.tree_util.tree_map(jnp.asarray, init_from)
    elif init_from is not None:                     # upcycling: copy a dense model's FFN into every expert
        for li, L in enumerate(p["layers"]):
            src = init_from["layers"][li]
            for key in ("qkv", "o"):
                L[key] = jnp.asarray(src[key])
            L["e1"] = jnp.repeat(jnp.asarray(src["w1"])[None], E, 0)
            L["e2"] = jnp.repeat(jnp.asarray(src["w2"])[None], E, 0)
        for key in ("tok", "pos", "out"):
            p[key] = jnp.asarray(init_from[key])
    cfg = dict(k=k, capacity=capacity)
    use_bias = bias_rate > 0
    opt = optax.adam(optax.warmup_cosine_decay_schedule(0, lr, 100, steps, lr * 0.1))
    st = opt.init(p)

    def loss_fn(p, x, biases):
        lg, stats = forward(p, x[:, :-1], cfg, biases)
        ce = -jnp.take_along_axis(jax.nn.log_softmax(lg), x[:, 1:, None], -1).mean()
        extra = sum(aux * s["aux"] + zloss * s["z"] for s in stats) if stats else 0.0
        return ce + extra, (ce, stats)

    @jax.jit
    def step(p, st, x, biases):
        (l, (ce, stats)), g = jax.value_and_grad(loss_fn, has_aux=True)(p, x, biases)
        u, st = opt.update(g, st)
        fr = jnp.stack([s["frac"] for s in stats]) if stats else jnp.zeros((1, 1))
        ml = jnp.stack([s["maxlogit"] for s in stats]) if stats else jnp.zeros(1)
        dr = jnp.stack([s["dropped"] for s in stats]) if stats else jnp.zeros(1)
        return optax.apply_updates(p, u), st, ce, fr, ml, dr

    rng = np.random.default_rng(seed)
    xe, le = sample(np.random.default_rng(1234), 256)
    evalf = jax.jit(lambda p, b: forward(p, jnp.array(xe[:, :-1]), cfg, b))
    biases = jnp.zeros((LAYERS, E)) if ffn == "moe" else None
    hist = []
    for it in range(steps + 1):
        if it % eval_every == 0:
            lg, stats = evalf(p, biases)
            ce = float(-np.take_along_axis(np.asarray(jax.nn.log_softmax(lg)), xe[:, 1:, None], -1).mean())
            rec = dict(step=it, loss=ce)
            if stats:
                fr = np.stack([np.asarray(s["frac"]) for s in stats])
                rec.update(load_max=float(fr.max() * E), load_cv=float((fr.std(1) / fr.mean(1)).mean()),
                           maxlogit=float(max(float(s["maxlogit"]) for s in stats)), maxprob=float(np.mean([float(s["maxprob"]) for s in stats])),
                           dropped=float(np.mean([float(s["dropped"]) for s in stats])), frac=fr)
            hist.append(rec)
        if it == steps:
            break
        x, _ = sample(rng, batch)
        p, st, ce, fr, ml, dr = step(p, st, jnp.array(x), biases)
        if use_bias:
            fr = np.asarray(fr)
            biases = biases + bias_rate * jnp.sign(1.0 / E - fr)
    out = dict(hist=hist, final=hist[-1]["loss"], params=n_params(p), active=n_params(p, active_k=k) if ffn == "moe" else n_params(p),
               p=jax.tree_util.tree_map(np.asarray, p))
    if ffn == "moe" and log_route:
        lg, stats = evalf(p, biases)
        out["route"] = [np.asarray(s["topi"]).reshape(256, LEN) for s in stats]
        out["eval_x"], out["eval_lang"] = xe, le
    return out


def ffn_flops(hidden):
    """Multiply-adds per token of a dense FFN block: 2 * D * hidden."""
    return 2 * D * hidden


# ---------------------------------------------------------------- experiments
STEPS = 2000
GOOD = dict(ffn="moe", E=8, h_e=128, k=1, aux=0.01)


def _strip(r):
    return {k: v for k, v in r.items() if k != "p"}


def exp_scale():
    """Dense FFN hidden 128 vs dense hidden 1,024 (8x compute) vs MoE 8 x 128 top-1 (same compute as dense-128)."""
    return {"dense128": _strip(train(ffn="dense", hidden=128, steps=STEPS)),
            "dense1024": _strip(train(ffn="dense", hidden=1024, steps=STEPS)),
            "moe8": _strip(train(steps=STEPS, **GOOD))}


def exp_experts():
    out = {}
    for E in (2, 4, 16):
        out[(E, 1)] = _strip(train(steps=STEPS, **{**GOOD, "E": E}))
    out[(8, 2)] = _strip(train(steps=STEPS, **{**GOOD, "k": 2}))
    return out


def exp_balance():
    return {"none": _strip(train(steps=STEPS, ffn="moe", E=8, h_e=128, k=1)),
            "aux": _strip(train(steps=STEPS, **GOOD)),
            "bias": _strip(train(steps=STEPS, ffn="moe", E=8, h_e=128, k=1, bias_rate=1e-3))}


def exp_capacity():
    return {cf: _strip(train(steps=STEPS, capacity=cf, **GOOD)) for cf in (0.5, 1.0, 1.25, 2.0)}


def exp_special():
    return _strip(train(steps=STEPS, log_route=True, **GOOD))


def exp_fine():
    return {"8x128_top1": _strip(train(steps=STEPS, **GOOD)),
            "32x32_top4": _strip(train(steps=STEPS, ffn="moe", E=32, h_e=32, k=4, aux=0.01)),
            "shared+28x32_top3": _strip(train(steps=STEPS, ffn="moe", E=28, h_e=32, k=3, aux=0.01, shared=1, h_s=32))}


def exp_stability():
    out = {"lr1e-2": _strip(train(steps=STEPS, lr=1e-2, **GOOD)),
           "lr1e-2_z": _strip(train(steps=STEPS, lr=1e-2, zloss=1e-3, **GOOD))}
    dense = train(ffn="dense", hidden=128, steps=STEPS)
    out["dense_start"] = dense["final"]
    out["upcycle"] = _strip(train(steps=1000, init_from=dense["p"], lr=1e-3, **GOOD))
    out["dense_cont"] = _strip(train(ffn="dense", hidden=128, steps=1000, init_from=dense["p"], lr=1e-3))
    out["moe_scratch"] = _strip(train(steps=1000, **GOOD))
    return out
