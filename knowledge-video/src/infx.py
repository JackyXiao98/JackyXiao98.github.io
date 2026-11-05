"""Toy experiments for the LLM Inference series (cached via INF_CACHE).

Data: the 8 hidden Markov "languages" of the MoE series (moex.PT, vocabulary 32), sequences of length 64.
Model: small pre-LN causal Transformer with learned absolute positions (d = 64, 4 query heads of dim 16, FFN 256,
2 layers) whose attention can be multi-head (4 KV heads), grouped-query (2), multi-query (1) or a multi-head latent
attention (keys / values reconstructed from a cached 16-dim latent). Trained with JAX; all inference experiments
(KV cache, FlashAttention, quantisation, speculative decoding, cache eviction) run in a plain numpy forward pass so
that every step is explicit.
"""

import os
import pickle
import time

import numpy as np

import moex

V, LEN = moex.V, 64
D, H, HD, F = 64, 4, 16, 256


def cached(cdir, name, fn, **kw):
    key = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    path = os.path.join(cdir, f"{name}_{key}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    res = fn(**kw)
    os.makedirs(cdir, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


def sample(rng, n, length=LEN):
    lang = rng.integers(0, moex.K, n)
    x = np.zeros((n, length + 1), np.int32)
    x[:, 0] = rng.integers(0, V, n)
    for t in range(1, length + 1):
        p = moex.PT[lang, x[:, t - 1]]
        x[:, t] = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
    return x, lang


# ---------------------------------------------------------------- model
def init(seed=0, kv=4, mla=0, layers=2, d=D, heads=H, ffn=F):
    rng = np.random.default_rng(seed)
    hd = d // heads
    g = lambda *s: (rng.normal(size=s) / np.sqrt(s[0])).astype(np.float32)
    p = dict(tok=(rng.normal(size=(V, d)) * 0.5).astype(np.float32), pos=(rng.normal(size=(LEN, d)) * 0.1).astype(np.float32),
             out=g(d, V), layers=[])
    for _ in range(layers):
        L = dict(q=g(d, heads * hd), o=g(heads * hd, d), w1=g(d, ffn), w2=g(ffn, d))
        if mla:
            L.update(dkv=g(d, mla), uk=g(mla, heads * hd), uv=g(mla, heads * hd))
        else:
            L.update(k=g(d, kv * hd), v=g(d, kv * hd))
        p["layers"].append(L)
    p["cfg"] = dict(kv=kv, mla=mla, heads=heads, hd=hd)
    return p


def _ln(x, xp):
    m = x.mean(-1, keepdims=True)
    return (x - m) / xp.sqrt(((x - m) ** 2).mean(-1, keepdims=True) + 1e-5)


def kv_from(L, y, cfg, xp):
    """Keys and values (..., T, kv_heads, hd) and the tensor that has to be cached."""
    if cfg["mla"]:
        c = y @ L["dkv"]
        return c, (c @ L["uk"]), (c @ L["uv"])
    return None, y @ L["k"], y @ L["v"]


def forward(p, x, xp=np):
    """Full-sequence forward. x: (B, T) ints -> logits (B, T, V)."""
    cfg = p["cfg"]
    B, T = x.shape
    nh, hd = cfg["heads"], cfg["hd"]
    nkv = nh if cfg["mla"] else cfg["kv"]
    h = p["tok"][x] + p["pos"][:T]
    mask = xp.tril(xp.ones((T, T))) > 0
    for L in p["layers"]:
        y = _ln(h, xp)
        q = (y @ L["q"]).reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        _, k, v = kv_from(L, y, cfg, xp)
        k = k.reshape(B, T, nkv, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, nkv, hd).transpose(0, 2, 1, 3)
        rep = nh // nkv
        k, v = xp.repeat(k, rep, 1), xp.repeat(v, rep, 1)
        s = xp.where(mask, q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd), -1e9)
        a = xp.exp(s - s.max(-1, keepdims=True))
        a = a / a.sum(-1, keepdims=True)
        h = h + (a @ v).transpose(0, 2, 1, 3).reshape(B, T, nh * hd) @ L["o"]
        h = h + xp.maximum(0, _ln(h, xp) @ L["w1"]) @ L["w2"]
    return _ln(h, xp) @ p["out"]


def train(kv=4, mla=0, layers=2, d=D, heads=H, ffn=F, steps=3000, batch=32, lr=3e-3, seed=0):
    import jax
    import jax.numpy as jnp
    import optax
    p = init(seed, kv, mla, layers, d, heads, ffn)
    cfg = p.pop("cfg")
    p = jax.tree_util.tree_map(jnp.asarray, p)
    opt = optax.adam(optax.warmup_cosine_decay_schedule(0, lr, 100, steps, lr * 0.1))
    st = opt.init(p)

    def loss_fn(p, x):
        lg = forward({**p, "cfg": cfg}, x[:, :-1], jnp)
        return -jnp.take_along_axis(jax.nn.log_softmax(lg), x[:, 1:, None], -1).mean()

    @jax.jit
    def step(p, st, x):
        l, g = jax.value_and_grad(loss_fn)(p, x)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(seed)
    for it in range(steps):
        x, _ = sample(rng, batch)
        p, st, l = step(p, st, jnp.array(x))
    p = jax.tree_util.tree_map(lambda a: np.asarray(a, np.float32), p)
    p["cfg"] = cfg
    return p


def eval_loss(p, n=256, seed=1234):
    x, _ = sample(np.random.default_rng(seed), n)
    lg = forward(p, x[:, :-1])
    lg = lg - lg.max(-1, keepdims=True)
    lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    return float(-np.take_along_axis(lp, x[:, 1:, None], -1).mean())


def n_params(p):
    tot = 0
    for k, v in p.items():
        if k == "cfg":
            continue
        if k == "layers":
            tot += sum(a.size for L in v for a in L.values())
        else:
            tot += v.size
    return int(tot)


def cache_floats_per_token(p):
    cfg = p["cfg"]
    per_layer = cfg["mla"] if cfg["mla"] else 2 * cfg["kv"] * cfg["hd"]
    return per_layer * len(p["layers"])


# ---------------------------------------------------------------- incremental decoding with a KV cache
class Cache:
    def __init__(self, p):
        self.p, self.cfg = p, p["cfg"]
        self.k = [[] for _ in p["layers"]]      # per layer: list of (nkv, hd) or latent vectors
        self.v = [[] for _ in p["layers"]]
        self.pos = []                           # absolute positions kept in the cache
        self.t = 0
        self.flops = 0

    def step(self, tok, keep=None, record_attn=None):
        """Feed one token; returns logits (V,). keep(t, pos_list, attn_scores) -> indices to keep (eviction)."""
        p, cfg = self.p, self.cfg
        nh, hd = cfg["heads"], cfg["hd"]
        nkv = nh if cfg["mla"] else cfg["kv"]
        h = p["tok"][tok] + p["pos"][self.t]
        self.pos.append(self.t)
        attn_all = []
        for li, L in enumerate(p["layers"]):
            y = _ln(h, np)
            q = (y @ L["q"]).reshape(nh, hd)
            if cfg["mla"]:
                c = y @ L["dkv"]
                self.k[li].append(c)
                C = np.stack(self.k[li])
                K = (C @ L["uk"]).reshape(-1, nkv, hd)
                Vv = (C @ L["uv"]).reshape(-1, nkv, hd)
            else:
                self.k[li].append((y @ L["k"]).reshape(nkv, hd))
                self.v[li].append((y @ L["v"]).reshape(nkv, hd))
                K, Vv = np.stack(self.k[li]), np.stack(self.v[li])
            rep = nh // nkv
            K, Vv = np.repeat(K, rep, 1), np.repeat(Vv, rep, 1)          # (n, nh, hd)
            s = np.einsum("hd,nhd->hn", q, K) / np.sqrt(hd)
            a = np.exp(s - s.max(-1, keepdims=True))
            a = a / a.sum(-1, keepdims=True)
            attn_all.append(a)
            self.flops += 4 * nh * hd * K.shape[0]
            h = h + np.einsum("hn,nhd->hd", a, Vv).reshape(-1) @ L["o"]
            h = h + np.maximum(0, _ln(h, np) @ L["w1"]) @ L["w2"]
        self.t += 1
        if record_attn is not None:
            record_attn.append(attn_all)
        if keep is not None:
            idx = keep(self.t, self.pos, attn_all)
            if idx is not None:
                self.pos = [self.pos[i] for i in idx]
                for li in range(len(p["layers"])):
                    self.k[li] = [self.k[li][i] for i in idx]
                    if self.v[li]:
                        self.v[li] = [self.v[li][i] for i in idx]
        return _ln(h, np) @ p["out"]


def attn_flops_nocache(p, T):
    """Attention score + weighted-sum multiply-adds x2 when every step recomputes the whole prefix."""
    cfg = p["cfg"]
    return sum(4 * cfg["heads"] * cfg["hd"] * t * t for t in range(1, T + 1)) * len(p["layers"])


# ---------------------------------------------------------------- FlashAttention-style tiled attention
def attention_naive(q, k, v):
    s = q @ k.T / np.sqrt(q.shape[1])
    s = np.where(np.tril(np.ones(s.shape)) > 0, s, -np.inf)
    a = np.exp(s - s.max(-1, keepdims=True))
    return (a / a.sum(-1, keepdims=True)) @ v


def attention_tiled(q, k, v, bq=16, bk=16):
    """Online-softmax tiled causal attention: never materialises the full score matrix."""
    T, d = q.shape
    out = np.zeros_like(q)
    for i0 in range(0, T, bq):
        qi = q[i0:i0 + bq]
        m = np.full(len(qi), -np.inf)
        l = np.zeros(len(qi))
        acc = np.zeros((len(qi), d))
        for j0 in range(0, i0 + bq, bk):
            s = qi @ k[j0:j0 + bk].T / np.sqrt(d)
            rows = np.arange(i0, i0 + len(qi))[:, None]
            cols = np.arange(j0, j0 + s.shape[1])[None]
            s = np.where(cols <= rows, s, -np.inf)
            m_new = np.maximum(m, s.max(1))
            scale = np.exp(m - m_new)
            pexp = np.exp(s - m_new[:, None])
            l = l * scale + pexp.sum(1)
            acc = acc * scale[:, None] + pexp @ v[j0:j0 + bk]
            m = m_new
        out[i0:i0 + bq] = acc / l[:, None]
    return out


# ---------------------------------------------------------------- quantisation
def quantize(w, bits, per_channel):
    qmax = 2 ** (bits - 1) - 1
    s = (np.abs(w).max(0, keepdims=True) if per_channel else np.abs(w).max()) / qmax
    s = np.where(s == 0, 1, s)
    return np.clip(np.round(w / s), -qmax - 1, qmax) * s


def quantize_model(p, bits, per_channel):
    q = {k: v for k, v in p.items()}
    q["layers"] = [{k: quantize(v, bits, per_channel) for k, v in L.items()} for L in p["layers"]]
    return q


# ---------------------------------------------------------------- speculative sampling
def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def probs_seq(p, prefix):
    """Next-token distribution after each prefix position (T, V) via a full forward (deterministic)."""
    return np.stack([softmax(r) for r in forward(p, np.asarray(prefix)[None])[0]])


def speculative(target, draft, prefix, n_new, gamma, rng):
    """Leviathan / Chen speculative sampling. Returns tokens, number of target calls, accepted drafts."""
    seq = list(prefix)
    calls, acc, prop = 0, 0, 0
    while len(seq) < len(prefix) + n_new:
        dseq, qs = list(seq), []
        for _ in range(gamma):
            qd = probs_seq(draft, dseq)[-1]
            t = int(rng.choice(V, p=qd))
            qs.append(qd)
            dseq.append(t)
        P = probs_seq(target, dseq)[len(seq) - 1:]          # gamma + 1 distributions
        calls += 1
        n_ok = 0
        for i in range(gamma):
            t = dseq[len(seq) + i]
            prop += 1
            if rng.random() < min(1.0, P[i][t] / qs[i][t]):
                n_ok += 1
            else:
                break
        acc += n_ok
        new = dseq[len(seq):len(seq) + n_ok]
        if n_ok < gamma:
            r = np.maximum(P[n_ok] - qs[n_ok], 0)
            new.append(int(rng.choice(V, p=r / r.sum())))
        else:
            new.append(int(rng.choice(V, p=P[gamma])))
        seq += new
    return seq[:len(prefix) + n_new], calls, acc, prop


def spec_one_step_dist(p, q):
    """Exact output distribution of one speculative step for a single position: accept with min(1, p/q)."""
    accept = np.minimum(p, q)
    r = np.maximum(p - q, 0)
    return accept + (1 - accept.sum()) * r / r.sum()


# ---------------------------------------------------------------- experiments
STEPS = 3000


def exp_models():
    out = {}
    for name, kw in [("mha", dict(kv=4)), ("gqa", dict(kv=2)), ("mqa", dict(kv=1)), ("mla", dict(mla=16)),
                     ("draft", dict(layers=1, d=32, heads=2, ffn=64))]:
        p = train(steps=STEPS, **kw)
        out[name] = dict(p=p, loss=eval_loss(p), params=n_params(p), cache=cache_floats_per_token(p))
    return out


def exp_timing():
    """Matrix-vector vs matrix-matrix on this CPU: time per multiply for batch sizes 1..256 (float32, 2048 x 2048)."""
    rng = np.random.default_rng(0)
    W = rng.normal(size=(2048, 2048)).astype(np.float32)
    res = {}
    for B in (1, 4, 16, 64, 256):
        X = rng.normal(size=(B, 2048)).astype(np.float32)
        for _ in range(3):
            X @ W
        n = 50
        t0 = time.perf_counter()
        for _ in range(n):
            X @ W
        res[B] = (time.perf_counter() - t0) / n
    return res


def _models(cdir):
    return cached(cdir, "models", exp_models)


def exp_spec(cache_dir):
    M = _models(cache_dir)
    T, Dm = M["mha"]["p"], M["draft"]["p"]
    rng = np.random.default_rng(7)
    xs, _ = sample(rng, 200, 16)
    # one-step exactness and acceptance rate
    diffs, alphas = [], []
    for x in xs:
        P, Q = probs_seq(T, x[:16]), probs_seq(Dm, x[:16])
        for t in (3, 9, 15):
            diffs.append(np.abs(spec_one_step_dist(P[t], Q[t]) - P[t]).max())
            alphas.append(np.minimum(P[t], Q[t]).sum())
    alpha = float(np.mean(alphas))
    # generation with different gamma
    gen = {}
    for g in (1, 2, 4, 8):
        r2 = np.random.default_rng(100 + g)
        tok, calls, acc, prop = 0, 0, 0, 0
        for x in xs[:40]:
            seq, c, a, pr = speculative(T, Dm, list(x[:8]), 40, g, r2)
            tok += 40
            calls += c
            acc += a
            prop += pr
        gen[g] = dict(per_call=tok / calls, accept=acc / prop)
    # distribution check from a fixed prefix: two-token continuations
    pre = list(xs[0][:8])
    N = 3000

    def plain(r):
        P1 = probs_seq(T, pre)[-1]
        a = int(r.choice(V, p=P1))
        b = int(r.choice(V, p=probs_seq(T, pre + [a])[-1]))
        return a * V + b

    def spec(r):
        s, _, _, _ = speculative(T, Dm, pre, 2, 2, r)
        return s[-2] * V + s[-1]

    def hist(fn, seed):
        r = np.random.default_rng(seed)
        h = np.zeros(V * V)
        for _ in range(N):
            h[fn(r)] += 1
        return h / N

    hp, hp2, hs = hist(plain, 1), hist(plain, 2), hist(spec, 3)
    P1 = probs_seq(T, pre)[-1]
    exact = np.zeros(V * V)
    for a in range(V):
        exact[a * V:(a + 1) * V] = P1[a] * probs_seq(T, pre + [a])[-1]
    tv = lambda u, w: 0.5 * np.abs(u - w).sum()
    return dict(maxdiff=float(max(diffs)), alpha=alpha, gen=gen, tv_plain=tv(hp, exact), tv_plain2=tv(hp2, exact),
                tv_spec=tv(hs, exact), tv_pp=tv(hp, hp2), n=N, draft_params=M["draft"]["params"], target_params=M["mha"]["params"])


def exp_quant(cache_dir):
    M = _models(cache_dir)
    p = M["mha"]["p"]
    base = eval_loss(p)
    res = {"base": base}
    for b in (8, 6, 4, 3, 2):
        for pc in (False, True):
            res[(b, pc)] = eval_loss(quantize_model(p, b, pc))
    # function-preserving outlier: scale one FFN hidden unit up by a, its output weights down by a (ReLU is homogeneous)
    po = {k: v for k, v in p.items()}
    po["layers"] = [dict(L) for L in p["layers"]]
    j = 0
    for L in po["layers"]:
        L["w1"] = L["w1"].copy()
        L["w2"] = L["w2"].copy()
        L["w1"][:, j] *= 30.0
        L["w2"][j, :] /= 30.0
    res["outlier_same"] = eval_loss(po)
    for b in (8, 4):
        for pc in (False, True):
            res[("out", b, pc)] = eval_loss(quantize_model(po, b, pc))
    bytes_ = {b: n_params(p) * b / 8 for b in (32, 16, 8, 4)}
    res["bytes"] = bytes_
    return res


def _evict_loss(p, xs, policy, budget, sink=4, start=16):
    losses = []
    attn_first = []
    for x in xs:
        acc = None

        def keep(t, pos, attn):
            nonlocal acc
            n = len(pos)
            score = sum(a.sum(0) for a in attn)                        # attention received this step, summed over layers/heads
            acc = score if acc is None else np.concatenate([acc, [0.0]])[:n] + score
            if n <= budget:
                return None
            if policy == "window":
                idx = list(range(n - budget, n))
            elif policy == "sink":
                idx = list(range(sink)) + list(range(n - (budget - sink), n))
            else:                                                       # heavy hitters + recent half
                recent = list(range(n - budget // 2, n))
                rest = np.argsort(-acc[:n - budget // 2])[:budget - budget // 2]
                idx = sorted(set(rest.tolist()) | set(recent))
            acc = acc[idx]
            return idx

        c = Cache(p)
        for t in range(LEN):
            lg = c.step(int(x[t]), keep=None if policy == "full" else keep)
            if t >= start:
                losses.append(-np.log(softmax(lg)[x[t + 1]]))
    return float(np.mean(losses))


def exp_evict(cache_dir):
    M = _models(cache_dir)
    p = M["mha"]["p"]
    xs, _ = sample(np.random.default_rng(11), 48)
    res = {"full": _evict_loss(p, xs, "full", LEN)}
    for b in (4, 8, 16):
        for pol in ("window", "sink", "heavy"):
            res[(pol, b)] = _evict_loss(p, xs, pol, b)
    # how much attention goes to the first token (positions >= 16, full cache)
    rec = []
    for x in xs[:16]:
        c = Cache(p)
        r = []
        for t in range(LEN):
            c.step(int(x[t]), record_attn=r)
        rec += [np.mean([a[:, 0].mean() for a in r[t]]) for t in range(16, LEN)]
    res["first_attn"] = float(np.mean(rec))
    res["uniform_attn"] = float(np.mean([1 / (t + 1) for t in range(16, LEN)]))
    return res


def exp_batching(seed=0, n_req=400, slots=16, max_len=2048, page=16):
    rng = np.random.default_rng(seed)
    out_len = np.clip(np.round(rng.lognormal(np.log(200), 0.8, n_req)), 8, max_len).astype(int)
    # static batching: groups of `slots`, each group runs until its longest request finishes
    steps_static = sum(out_len[i:i + slots].max() for i in range(0, n_req, slots))
    useful = out_len.sum()
    # continuous batching: a finished request is replaced at the next step
    rem = list(out_len)
    active, steps_cont, nxt = [], 0, 0
    while nxt < n_req or active:
        while len(active) < slots and nxt < n_req:
            active.append(rem[nxt])
            nxt += 1
        steps_cont += 1
        active = [a - 1 for a in active if a > 1]
    # KV memory: contiguous reservation of max_len per request vs pages
    reserved = n_req * max_len
    paged = int(np.sum(np.ceil(out_len / page) * page))
    return dict(out_len=out_len, steps_static=int(steps_static), steps_cont=int(steps_cont), useful=int(useful), slots=slots,
                util_static=useful / (steps_static * slots), util_cont=useful / (steps_cont * slots),
                mem_contig=useful / reserved, mem_paged=useful / paged, median=float(np.median(out_len)), p99=float(np.percentile(out_len, 99)))
