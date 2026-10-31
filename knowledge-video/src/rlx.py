"""Toy experiments for the RL for LLMs series (CPU-sized; trained results cached in RL_CACHE).

- bandit(): 10-armed Gaussian testbed, epsilon-greedy vs. softmax policy gradient with a baseline
- reward_model(): Bradley-Terry reward model fitted to simulated pairwise preferences
- Discrete "response" toys for RLHF: 200 responses with a true reward, a reference policy and a
  learned reward model; the KL-regularised optimum pi* ∝ pi_ref exp(r_hat / beta) (rlhf_curve),
  DPO trained on the same preference pairs (dpo), and a length-biased reward model (length_hack)
- grpo(): GRPO-style RL (group-normalised advantages, KL to the reference) on the tfm.py addition
  model from a half-trained checkpoint, reward = an automatic correctness check
- cot(): direct answer vs. chain of thought (running sums) for "sum of n digits mod 10"
- process_vs_outcome(): a k-step task learned with outcome-only vs. per-step (process) rewards
"""

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


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------- 1. bandits
def bandit(k=10, steps=1000, runs=2000, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((runs, k))
    best = q.argmax(1)
    out = {}
    for eps in (0.0, 0.01, 0.1):
        Q, N = np.zeros((runs, k)), np.zeros((runs, k))
        rew, opt = np.zeros(steps), np.zeros(steps)
        for t in range(steps):
            greedy = np.argmax(Q + 1e-9 * rng.random((runs, k)), 1)
            a = np.where(rng.random(runs) < eps, rng.integers(0, k, runs), greedy)
            r = q[np.arange(runs), a] + rng.standard_normal(runs)
            N[np.arange(runs), a] += 1
            Q[np.arange(runs), a] += (r - Q[np.arange(runs), a]) / N[np.arange(runs), a]
            rew[t], opt[t] = r.mean(), (a == best).mean()
        out[f"eps{eps}"] = (rew, opt)
    H, base = np.zeros((runs, k)), np.zeros(runs)
    rew, opt = np.zeros(steps), np.zeros(steps)
    for t in range(steps):
        pi = softmax(H)
        a = (pi.cumsum(1) > rng.random((runs, 1))).argmax(1)
        r = q[np.arange(runs), a] + rng.standard_normal(runs)
        base += (r - base) / (t + 1)
        onehot = np.zeros_like(H)
        onehot[np.arange(runs), a] = 1
        H += 0.1 * (r - base)[:, None] * (onehot - pi)
        rew[t], opt[t] = r.mean(), (a == best).mean()
    out["pg"] = (rew, opt)
    out["best_mean"] = float(q.max(1).mean())
    return out


# ---------------------------------------------------------------- 2. reward model
def reward_model(d=8, n_items=2000, ns=(30, 100, 300, 1000, 3000, 10000), seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d)
    X = rng.standard_normal((n_items, d))
    true = X @ w / np.sqrt(d) * 2
    tr_items, te_items = np.arange(n_items // 2), np.arange(n_items // 2, n_items)

    def pairs(items, n):
        i, j = rng.choice(items, n), rng.choice(items, n)
        y = rng.random(n) < 1 / (1 + np.exp(-(true[i] - true[j])))
        return i, j, y

    ti, tj, ty = pairs(te_items, 20000)
    oracle = float(((true[ti] > true[tj]) == ty).mean())
    res = []
    for n in ns:
        i, j, y = pairs(tr_items, n)
        Z = X[i] - X[j]
        v = np.zeros(d)
        for _ in range(500):  # logistic regression, Newton steps with a tiny ridge
            p = 1 / (1 + np.exp(-Z @ v))
            g = Z.T @ (p - y) + 1e-3 * v
            Hm = (Z * (p * (1 - p))[:, None]).T @ Z + 1e-3 * np.eye(d)
            v -= np.linalg.solve(Hm, g)
            if np.abs(g).max() < 1e-8:
                break
        acc = float((((X[ti] - X[tj]) @ v > 0) == ty).mean())
        corr = float(np.corrcoef(X[te_items] @ v, true[te_items])[0, 1])
        res.append(dict(n=n, acc=acc, corr=corr))
    return dict(oracle=oracle, res=res)


# ---------------------------------------------------------------- 3-4, 7. discrete response toys
def responses(n=200, seed=0):
    """True reward r, reference policy pi_ref, and a reward model r_hat = r + error, where the error
    is large on a few responses that the reference policy rarely produces (they are poorly covered
    by the preference data)."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n)
    logits = rng.standard_normal(n) * 1.5
    pi_ref = softmax(logits)
    rare = np.argsort(pi_ref)[: n // 10]
    err = rng.standard_normal(n) * 0.3
    err[rare] += rng.exponential(1.5, len(rare))
    return dict(r=r, pi_ref=pi_ref, r_hat=r + err, rare=rare)


def kl(p, q):
    return float((p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum())


def rlhf_curve(betas=None, seed=0):
    R = responses(seed=seed)
    betas = np.logspace(1, -2, 60) if betas is None else betas
    out = []
    for b in betas:
        pi = softmax(np.log(R["pi_ref"]) + R["r_hat"] / b)
        out.append(dict(beta=float(b), kl=kl(pi, R["pi_ref"]), proxy=float(pi @ R["r_hat"]), true=float(pi @ R["r"]),
                        rare=float(pi[R["rare"]].sum())))
    # verify the closed form with plain policy gradient on one beta
    b = 0.3
    th = np.log(R["pi_ref"]).copy()
    for _ in range(3000):  # natural policy gradient on the KL-regularised objective
        pi = softmax(th)
        adv = R["r_hat"] - b * (np.log(pi) - np.log(R["pi_ref"]))
        th += 0.05 * (adv - pi @ adv) / b
    pg = softmax(th)
    star = softmax(np.log(R["pi_ref"]) + R["r_hat"] / b)
    return dict(curve=out, ref_true=float(R["pi_ref"] @ R["r"]), best_true=float(R["r"].max()), pg_check=float(np.abs(pg - star).max()),
                pg_beta=b)


def dpo(n_pairs=5000, beta=0.3, steps=3000, lr=0.5, seed=0):
    """Preference pairs (y1, y2) ~ pi_ref, labelled by Bradley-Terry on the true reward. Compare
    (a) RLHF: fit a reward model on the pairs, then take the KL-optimal policy, and (b) DPO."""
    R = responses(seed=seed)
    rng = np.random.default_rng(seed + 1)
    n = len(R["r"])
    a, b = rng.choice(n, n_pairs, p=R["pi_ref"]), rng.choice(n, n_pairs, p=R["pi_ref"])
    keep = a != b
    a, b = a[keep], b[keep]
    y = rng.random(len(a)) < 1 / (1 + np.exp(-(R["r"][a] - R["r"][b])))
    w, l = np.where(y, a, b), np.where(y, b, a)
    # (a) reward model: one free score per response (L2-regularised), then pi* ∝ pi_ref exp(s / beta)
    s = np.zeros(n)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(s[w] - s[l])))
        g = np.zeros(n)
        np.add.at(g, w, -(1 - p))
        np.add.at(g, l, (1 - p))
        s -= lr * (g / len(w) + 1e-3 * s)
    pi_rlhf = softmax(np.log(R["pi_ref"]) + s / beta)
    # (b) DPO on policy logits theta (log pi up to a constant)
    th = np.log(R["pi_ref"]).copy()
    hist = []
    for t in range(steps + 1):
        lr_ratio = th - np.log(R["pi_ref"])
        z = beta * (lr_ratio[w] - lr_ratio[l])
        p = 1 / (1 + np.exp(-z))
        g = np.zeros(n)
        np.add.at(g, w, -(1 - p) * beta)
        np.add.at(g, l, (1 - p) * beta)
        if t % 300 == 0:
            pi = softmax(th)
            hist.append(dict(step=t, true=float(pi @ R["r"]), kl=kl(pi, R["pi_ref"]), loss=float(-np.log(p + 1e-12).mean())))
        th -= lr * 40 * (g / len(w) + 1e-4 * (th - np.log(R["pi_ref"])))
    pi_dpo = softmax(th)
    return dict(n_pairs=int(len(w)), ref_true=float(R["pi_ref"] @ R["r"]), rlhf_true=float(pi_rlhf @ R["r"]),
                rlhf_kl=kl(pi_rlhf, R["pi_ref"]), dpo_true=float(pi_dpo @ R["r"]), dpo_kl=kl(pi_dpo, R["pi_ref"]),
                agree=float(np.abs(pi_dpo - pi_rlhf).sum() / 2), hist=hist)


def length_hack(n=400, seed=0, betas=None):
    """Responses with a true quality and a length; the reward model also rewards length a little
    (as raters often prefer longer answers). Sweep the KL penalty."""
    rng = np.random.default_rng(seed)
    L = rng.integers(20, 600, n).astype(float)
    base = rng.standard_normal(n)
    q = base - 0.008 * np.maximum(L - 250, 0)  # answers beyond ~250 tokens get worse (padding, rambling)
    r_hat = 0.6 * base + 0.5 * rng.standard_normal(n) + 0.008 * (L - 200)  # noisy quality signal + length bias
    pi_ref = softmax(rng.standard_normal(n) * 0.5 - ((L - 150) / 120) ** 2)
    betas = np.logspace(1, -1.5, 50) if betas is None else betas
    out = []
    for b in betas:
        pi = softmax(np.log(pi_ref) + r_hat / b)
        out.append(dict(beta=float(b), kl=kl(pi, pi_ref), proxy=float(pi @ r_hat), true=float(pi @ q), length=float(pi @ L)))
    return dict(curve=out, ref_len=float(pi_ref @ L), ref_true=float(pi_ref @ q))


# ---------------------------------------------------------------- 5. GRPO on the addition model
def grpo(tf_cache, steps=400, group=8, batch=64, beta=0.02, lr=1e-3, snapshot=120, eval_every=25, seed=0):
    import jax
    import jax.numpy as jnp
    import optax
    _, _, snaps = tfm.cached_train(tf_cache, steps=600, log_every=20, snapshots=(0, 60, 120, 200, 300, 600))
    TR, TE, _, _ = tfm.dataset()
    p0 = dict(snaps[snapshot])
    heads = p0.pop("heads")
    ref = jax.tree_util.tree_map(jnp.asarray, p0)
    p = ref
    fwd = lambda p, x: tfm.forward({**p, "heads": heads}, x)

    @jax.jit
    def sample(p, prompts, key):
        seq = prompts
        for _ in range(3):
            key, k = jax.random.split(key)
            nxt = jax.random.categorical(k, fwd(p, seq)[:, -1])
            seq = jnp.concatenate([seq, nxt[:, None]], 1)
        return seq

    def logp(p, seq):
        lp = jax.nn.log_softmax(fwd(p, seq[:, :-1]), -1)
        return jnp.take_along_axis(lp[:, 5:], seq[:, 6:, None], -1)[..., 0]

    def loss(p, seq, A):
        l, lr_ = logp(p, seq), logp(ref, seq)
        k3 = (jnp.exp(lr_ - l) - (lr_ - l) - 1).sum(1)
        return -(A * l.sum(1)).mean() + beta * k3.mean()

    opt = optax.adam(lr)
    st = opt.init(p)

    @jax.jit
    def update(p, st, seq, A):
        g = jax.grad(loss)(p, seq, A)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st

    acc = lambda p: tfm.accuracy({**jax.tree_util.tree_map(np.asarray, p), "heads": heads}, TE)
    rng, key = np.random.default_rng(seed), jax.random.PRNGKey(seed)
    hist = [dict(step=0, acc=acc(p), reward=None)]
    for step in range(1, steps + 1):
        pr = TR[rng.choice(len(TR), batch), :6]
        key, k = jax.random.split(key)
        seq = sample(p, jnp.array(np.repeat(pr, group, 0)), k)
        s = np.asarray(seq)
        a, b = s[:, 0] * 10 + s[:, 1], s[:, 3] * 10 + s[:, 4]
        ans = s[:, 6] * 100 + s[:, 7] * 10 + s[:, 8]
        r = ((s[:, 6:] < 10).all(1) & (ans == a + b)).astype(np.float32).reshape(batch, group)
        A = ((r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-6)).reshape(-1)
        p, st = update(p, st, seq, jnp.array(A))
        if step % eval_every == 0:
            hist.append(dict(step=step, acc=acc(p), reward=float(r.mean())))
    return dict(hist=hist, group=group, batch=batch)


# ---------------------------------------------------------------- 6. chain of thought
COT_V = list("0123456789=|")


def _cot_enc(x, cot):
    s = "".join(map(str, x)) + "="
    if cot:
        c = 0
        for v in x:
            c = (c + v) % 10
            s += str(c)
        s += "|"
    s += str(sum(x) % 10)
    return [COT_V.index(ch) for ch in s]


def cot(n_digits=12, use_cot=False, steps=1500, eval_every=250, seed=0):
    import jax
    import jax.numpy as jnp
    import optax
    rng = np.random.default_rng(seed)
    X, XT = rng.integers(0, 10, (60000, n_digits)), rng.integers(0, 10, (2000, n_digits))
    D = np.array([_cot_enc(x, use_cot) for x in X])
    DT = np.array([_cot_enc(x, use_cot) for x in XT])
    T, Q = D.shape[1], n_digits + 1
    key = jax.random.PRNGKey(seed)
    p = tfm.init(key)
    k1, k2, k3 = jax.random.split(key, 3)
    p["pos"] = jax.random.normal(k1, (T, 64)) * 0.1
    p["tok"] = jax.random.normal(k2, (len(COT_V), 64)) * 0.1
    p["out"] = jax.random.normal(k3, (64, len(COT_V))) / 8
    heads = p.pop("heads")
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, 3e-3, 200, steps), weight_decay=0.01)
    st = opt.init(p)

    def lf(p, b):
        lp = jax.nn.log_softmax(tfm.forward({**p, "heads": heads}, b[:, :-1])[:, Q - 1:], -1)
        return -jnp.take_along_axis(lp, b[:, Q:, None], -1).mean()

    @jax.jit
    def step(p, st, b):
        g = jax.grad(lf)(p, b)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st

    @jax.jit
    def gen(p, prompts):
        seq = prompts
        for _ in range(T - Q):
            nxt = tfm.forward({**p, "heads": heads}, seq)[:, -1].argmax(-1)
            seq = jnp.concatenate([seq, nxt[:, None]], 1)
        return seq

    hist = []
    for i in range(steps + 1):
        if i % eval_every == 0:
            out = np.asarray(gen(p, jnp.array(DT[:, :Q])))
            hist.append(dict(step=i, acc=float((out[:, -1] == DT[:, -1]).mean())))
        if i < steps:
            p, st = step(p, st, jnp.array(D[rng.choice(len(D), 256)]))
    return dict(hist=hist, extra_tokens=T - Q - 1, seq_len=T)


# ---------------------------------------------------------------- 8. process vs outcome reward
def process_vs_outcome(k=8, n_actions=4, episodes=3000, runs=200, lr=0.2, seed=0):
    """k sequential steps; at each step one of n_actions actions is correct. Outcome reward = 1 only
    if every step is correct; process reward = fraction of correct steps. REINFORCE on per-step
    softmax policies; report the probability of solving the whole task (greedy policy)."""
    rng = np.random.default_rng(seed)
    out = {}
    for mode in ("outcome", "process"):
        curve = np.zeros(episodes // 100 + 1)
        for r_ in range(runs):
            correct = rng.integers(0, n_actions, k)
            H = np.zeros((k, n_actions))
            base = 0.0
            for e in range(episodes + 1):
                if e % 100 == 0:
                    curve[e // 100] += np.prod(softmax(H)[np.arange(k), correct])
                if e == episodes:
                    break
                pi = softmax(H)
                a = (pi.cumsum(1) > rng.random((k, 1))).argmax(1)
                ok = a == correct
                R = float(ok.all()) if mode == "outcome" else float(ok.mean())
                base += 0.05 * (R - base)
                g = -pi
                g[np.arange(k), a] += 1
                H += lr * (R - base) * g
        out[mode] = curve / runs
    out["episodes"] = list(range(0, episodes + 1, 100))
    return out
