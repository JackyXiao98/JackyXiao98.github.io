"""Toy experiments for the Recursive Self-Improvement series (CPU-sized; trained results cached in RSI_CACHE).

- star(): STaR-style self-training of the tfm.py addition Transformer from 300 labelled problems,
  with and without a verifier on its own answers
- tictactoe_selfplay(): tabular self-play (afterstate TD learning), evaluated against a perfect
  minimax player and a random player
- verifier_bestofn(): sampling from a half-trained addition model; best-of-n with a perfect checker,
  a weak checker (units digit only) and majority vote
- model_collapse(), goodhart(): refitting a Gaussian on its own samples; best-of-n on a noisy proxy
- binpack_search(): (1+8) evolutionary search over online bin-packing priority functions, selected
  on a small or a larger visible set and scored on held-out instances
"""

import functools
import os
import pickle

import numpy as np

import tfm


def cached(cache_dir, name, fn, **kw):
    path = os.path.join(cache_dir, name + "_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items()) if k != "tf_cache") + ".pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    res = fn(**kw)
    os.makedirs(cache_dir, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


# ---------------------------------------------------------------- 1. STaR on 2-digit addition
def _train_tfm(data, steps=600, seed=0):
    import jax
    import optax
    p = tfm.init(jax.random.PRNGKey(seed))
    heads = p.pop("heads")
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, 3e-3, 100, steps), weight_decay=0.01)
    st = opt.init(p)

    @jax.jit
    def step(p, st, b):
        g = jax.grad(lambda p: tfm.loss_fn({**p, "heads": heads}, b))(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st

    rng = np.random.default_rng(seed)
    for _ in range(steps):
        p, st = step(p, st, data[rng.choice(len(data), 256)])
    return {**tfm.to_numpy(p), "heads": heads}


def _check(seq):
    """External verifier: does answer - second addend equal the first addend?"""
    valid = (seq[:, 6:] < 10).all(1)
    a = seq[:, 0] * 10 + seq[:, 1]
    b = seq[:, 3] * 10 + seq[:, 4]
    ans = seq[:, 6] * 100 + seq[:, 7] * 10 + seq[:, 8]
    return valid & (ans - b == a), valid


def star(n_seed=300, rounds=6, seed=0):
    TR, TE, _, _ = tfm.dataset()
    idx = np.random.default_rng(seed).permutation(len(TR))
    seed_set, pool = TR[idx[:n_seed]], TR[idx[n_seed:]]
    out = {"n_pool": len(pool)}
    for mode in ("verify", "noverify"):
        data, hist = seed_set.copy(), []
        for r in range(rounds):
            p = _train_tfm(data)
            gen = tfm.generate(p, pool[:, :tfm.ANS])
            ok, valid = _check(gen)
            keep = gen[ok] if mode == "verify" else gen[valid]
            hist.append(dict(round=r, acc=tfm.accuracy(p, TE), kept=int(len(keep)), correct=int(ok.sum())))
            data = np.concatenate([seed_set, keep])
        out[mode] = hist
    return out


# ---------------------------------------------------------------- 2. tic-tac-toe self-play
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def _win(b):
    for a, c, d in LINES:
        if b[a] != 0 and b[a] == b[c] == b[d]:
            return b[a]
    return 0


@functools.lru_cache(None)
def _minimax(b, p):
    w = _win(b)
    if w:
        return 1 if w == p else -1
    if 0 not in b:
        return 0
    return max(-_minimax(b[:i] + (p,) + b[i + 1:], -p) for i in range(9) if b[i] == 0)


def tictactoe_selfplay(checkpoints=(0, 100, 300, 1000, 3000, 10000, 30000, 60000), eval_games=200, seed=0):
    rng = np.random.default_rng(seed)
    V = {}

    def val(b, p):
        w = _win(b)
        if w:
            return 1.0 if w == p else 0.0
        if 0 not in b:
            return 0.5
        return V.get((b, p), 0.5)

    def greedy(b, p, eps=0.0):
        moves = [i for i in range(9) if b[i] == 0]
        if rng.random() < eps:
            return rng.choice(moves)
        vs = [val(b[:i] + (p,) + b[i + 1:], p) for i in moves]
        m = max(vs)
        return rng.choice([i for i, v in zip(moves, vs) if v == m])

    def perfect(b, p):
        vals = [(-_minimax(b[:i] + (p,) + b[i + 1:], -p), i) for i in range(9) if b[i] == 0]
        m = max(v for v, _ in vals)
        return rng.choice([i for v, i in vals if v == m])

    def selfplay(alpha=0.2, eps=0.1):
        b, p, last = (0,) * 9, 1, {1: None, -1: None}
        while True:
            i = greedy(b, p, eps)
            nb = b[:i] + (p,) + b[i + 1:]
            if last[p] is not None:
                V[last[p]] = V.get(last[p], 0.5) + alpha * (val(nb, p) - V.get(last[p], 0.5))
            last[p], b = (nb, p), nb
            if _win(b) or 0 not in b:
                for q in (1, -1):
                    if last[q] is not None and last[q][0] != b:
                        V[last[q]] = V.get(last[q], 0.5) + alpha * (val(b, q) - V.get(last[q], 0.5))
                return
            p = -p

    def match(me, opponent):
        b, p = (0,) * 9, 1
        while True:
            i = greedy(b, p) if p == me else opponent(b, p)
            b = b[:i] + (p,) + b[i + 1:]
            w = _win(b)
            if w:
                return 1 if w == me else -1
            if 0 not in b:
                return 0
            p = -p

    rand = lambda b, p: rng.choice([j for j in range(9) if b[j] == 0])
    hist, done = [], 0
    for n in checkpoints:
        while done < n:
            selfplay()
            done += 1
        vp = np.array([match(me, perfect) for me in (1, -1) for _ in range(eval_games)])
        vr = np.array([match(me, rand) for me in (1, -1) for _ in range(eval_games)])
        hist.append(dict(games=n, p_loss=float((vp == -1).mean()), p_draw=float((vp == 0).mean()),
                         r_win=float((vr == 1).mean()), r_loss=float((vr == -1).mean()), states=len(V)))
    return hist


# ---------------------------------------------------------------- 3. verifier-guided best-of-n
def verifier_bestofn(tf_cache, snapshot=120, n_test=1000, n_samples=32, seed=0):
    _, _, snaps = tfm.cached_train(tf_cache, steps=600, log_every=20, snapshots=(0, 60, 120, 200, 300, 600))
    p = snaps[snapshot]
    _, TE, _, _ = tfm.dataset()
    TE = TE[:n_test]
    rng = np.random.default_rng(seed)

    def sample(prompts):
        seq = np.array(prompts)
        for _ in range(3):
            lo = np.asarray(tfm.forward(p, seq, xp=np))[:, -1]
            lo = lo - lo.max(1, keepdims=True)
            pr = np.exp(lo)
            pr /= pr.sum(1, keepdims=True)
            u = rng.random((len(pr), 1))
            nxt = (pr.cumsum(1) < u).sum(1).clip(0, pr.shape[1] - 1)
            seq = np.concatenate([seq, nxt[:, None]], 1)
        return seq[:, 6:]

    samp = np.stack([sample(TE[:, :6]) for _ in range(n_samples)], 1)
    a = TE[:, 0] * 10 + TE[:, 1]
    b = TE[:, 3] * 10 + TE[:, 4]
    valid = (samp < 10).all(-1)
    ans = samp[..., 0] * 100 + samp[..., 1] * 10 + samp[..., 2]
    correct = valid & (ans == (a + b)[:, None])
    weak = valid & ((ans % 10) == ((a + b) % 10)[:, None])
    greedy = float((tfm.generate(p, TE[:, :6])[:, 6:] == TE[:, 6:]).all(1).mean())
    ns = [1, 2, 4, 8, 16, 32]
    res = dict(n=ns, greedy=greedy, single=float(correct.mean()), perfect=[], weak=[], majority=[])
    for n in ns:
        c, w = correct[:, :n], weak[:, :n]
        res["perfect"].append(float(c.any(1).mean()))
        idx = np.where(w.any(1), w.argmax(1), 0)
        res["weak"].append(float(c[np.arange(len(c)), idx].mean()))
        maj = []
        for i in range(len(c)):
            vals, cnt = np.unique(ans[i, :n][valid[i, :n]], return_counts=True)
            maj.append(len(vals) > 0 and vals[cnt.argmax()] == a[i] + b[i])
        res["majority"].append(float(np.mean(maj)))
    return res


# ---------------------------------------------------------------- 4. failure modes
def model_collapse(n=50, generations=300, reps=200, seed=0):
    """Fit a Gaussian (mean, std) to n samples drawn from the previous generation's fit."""
    rng = np.random.default_rng(seed)
    out = {}
    for mode in ("replace", "accumulate"):
        sd_hist = np.zeros((reps, generations + 1))
        for r in range(reps):
            real = rng.standard_normal(n)
            mu, sd, pool = real.mean(), real.std(), [real]
            sd_hist[r, 0] = sd
            for g in range(1, generations + 1):
                x = mu + sd * rng.standard_normal(n)
                if mode == "accumulate":
                    pool.append(x)
                    x = np.concatenate(pool)
                mu, sd = x.mean(), x.std()
                sd_hist[r, g] = sd
        out[mode] = np.median(sd_hist, 0)
    return out


def goodhart(ns=(1, 4, 16, 64, 256, 1024, 4096, 16384), reps=400, seed=0):
    """True quality q ~ N(0,1); proxy = q + error. Pick the best of n by the proxy; report E[q]."""
    rng = np.random.default_rng(seed)
    out = {"n": list(ns)}
    for name, err in (("light", lambda k: rng.standard_normal(k)), ("heavy", lambda k: rng.standard_t(2, k))):
        res = []
        for n in ns:
            tq = []
            for _ in range(reps):
                q = rng.standard_normal(n)
                tq.append(q[np.argmax(q + err(n))])
            res.append(float(np.mean(tq)))
        out[name] = res
    return out


# ---------------------------------------------------------------- 5. evolving a bin-packing heuristic
CAP = 100


def bp_instances(n, seed, m=300):
    r = np.random.default_rng(seed)
    return [np.clip(np.round(r.weibull(3, m) * 45), 1, CAP).astype(int) for _ in range(n)]


def _bp_feats(rem, item):
    a = rem - item
    return np.stack([-a / 10.0, (a == 0) * 1.0, ((a > 0) & (a < 5)) * 1.0, ((a >= 5) & (a < 10)) * 1.0,
                     ((a >= 10) & (a < 20)) * 1.0, ((a >= 20) & (a < 30)) * 1.0], 1)


def bp_pack(items, w):
    bins = []
    for it in items:
        rem = np.array(bins)
        ok = np.where(rem >= it)[0] if len(bins) else np.array([], int)
        if len(ok) == 0:
            bins.append(CAP - it)
            continue
        j = ok[np.argmax(_bp_feats(rem[ok], it) @ w)]
        bins[j] -= it
    return len(bins)


def bp_first_fit(items):
    bins = []
    for it in items:
        for j, b in enumerate(bins):
            if b >= it:
                bins[j] -= it
                break
        else:
            bins.append(CAP - it)
    return len(bins)


def bp_excess(insts, packer):
    """Mean % of bins above the lower bound ceil(sum / capacity)."""
    return float(np.mean([(packer(x) - np.ceil(x.sum() / CAP)) / np.ceil(x.sum() / CAP) * 100 for x in insts]))


BEST_FIT = np.array([1, 0, 0, 0, 0, 0.0])


def binpack_search(n_train, generations=150, children=8, seed=0):
    TR, TE = bp_instances(n_train, 1), bp_instances(50, 2)
    score = lambda insts, w: bp_excess(insts, lambda x: bp_pack(x, w))
    rng = np.random.default_rng(seed)
    w = BEST_FIT.copy()
    best = score(TR, w)
    hist = [(0, best, score(TE, w))]
    for g in range(1, generations + 1):
        cands = [w + rng.standard_normal(6) * abs(rng.standard_normal()) for _ in range(children)]
        sc = [score(TR, c) for c in cands]
        i = int(np.argmin(sc))
        if sc[i] < best:
            w, best = cands[i], sc[i]
            hist.append((g, best, score(TE, w)))
    return dict(w=w, hist=hist, first_fit_test=bp_excess(TE, bp_first_fit), best_fit_train=score(TR, BEST_FIT),
                best_fit_test=score(TE, BEST_FIT), train=best, test=score(TE, w))
