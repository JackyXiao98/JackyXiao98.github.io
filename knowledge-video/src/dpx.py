"""Differential-privacy toolkit for the DP series: mechanisms, privacy accounting, DP-SGD, attacks.

Everything here is small and exact enough to narrate from:
- gaussian_delta(): exact (eps, delta) curve of the Gaussian mechanism (Balle & Wang, 2018)
- rdp_subsampled_gaussian(): Renyi DP of the Poisson-subsampled Gaussian for integer orders
  (Mironov, Talwar & Zhang, 2019) and rdp_to_eps() (the standard conversion)
- train_mlp(): a 1-hidden-layer MLP on sklearn's 8x8 digits, plain SGD or DP-SGD (per-example
  clipping + Gaussian noise, Abadi et al., 2016), in JAX; results cached like tfm.cached_train
- loss-threshold membership inference (Yeom et al., 2018)
"""

import os
import pickle
from math import comb, exp, log

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

ORDERS = list(range(2, 257))


# ---------------------------------------------------------------- mechanisms
def gaussian_delta(eps, sigma, sens=1.0):
    """Smallest delta such that the Gaussian mechanism (noise sigma, L2 sensitivity sens) is (eps, delta)-DP."""
    a = sens / (2 * sigma)
    b = eps * sigma / sens
    return norm.cdf(a - b) - np.exp(eps) * norm.cdf(-a - b)


def classic_sigma(eps, delta, sens=1.0):
    """Dwork & Roth Thm A.1 calibration (valid for eps < 1)."""
    return np.sqrt(2 * np.log(1.25 / delta)) * sens / eps


# ---------------------------------------------------------------- accounting
def rdp_gaussian(sigma, orders=ORDERS):
    return np.array([a / (2 * sigma ** 2) for a in orders])


def rdp_subsampled_gaussian(q, sigma, orders=ORDERS):
    """RDP of one step of the Poisson-subsampled Gaussian mechanism (integer orders, exact sum)."""
    if q == 1:
        return rdp_gaussian(sigma, orders)
    out = []
    for a in orders:
        terms = [log(comb(a, k)) + (a - k) * log(1 - q) + k * log(q) + (k * k - k) / (2 * sigma ** 2)
                 for k in range(a + 1)]
        out.append(logsumexp(terms) / (a - 1))
    return np.array(out)


def rdp_to_eps(rdp, delta, orders=ORDERS):
    """eps = min_a rdp(a) + log(1/delta)/(a-1). Returns (eps, best order)."""
    eps = np.array(rdp) + np.log(1 / delta) / (np.array(orders) - 1)
    i = int(np.argmin(eps))
    return float(eps[i]), orders[i]


def dpsgd_eps(q, sigma, steps, delta):
    return rdp_to_eps(steps * rdp_subsampled_gaussian(q, sigma), delta)[0]


def advanced_composition(eps0, k, delta_prime):
    """Dwork, Rothblum & Vadhan (2010)."""
    return np.sqrt(2 * k * np.log(1 / delta_prime)) * eps0 + k * eps0 * (np.exp(eps0) - 1)


# ---------------------------------------------------------------- data + DP-SGD
def digits(seed=0, n_train=None):
    from sklearn.datasets import load_digits
    d = load_digits()
    X = (d.data / 16.0).astype(np.float32)
    y = d.target.astype(np.int32)
    idx = np.random.default_rng(seed).permutation(len(X))
    X, y = X[idx], y[idx]
    n_test = 597
    Xte, yte, Xtr, ytr = X[:n_test], y[:n_test], X[n_test:], y[n_test:]
    if n_train:
        Xtr, ytr = Xtr[:n_train], ytr[:n_train]
    return Xtr, ytr, Xte, yte


def _init(key, d_in=64, hidden=128, n_cls=10):
    import jax
    k1, k2 = jax.random.split(key)
    return {"w1": jax.random.normal(k1, (d_in, hidden)) / np.sqrt(d_in), "b1": np.zeros(hidden, np.float32),
            "w2": jax.random.normal(k2, (hidden, n_cls)) / np.sqrt(hidden), "b2": np.zeros(n_cls, np.float32)}


def logits_fn(p, X):
    import jax.numpy as jnp
    return jnp.tanh(X @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


def per_example_loss(p, X, y):
    import jax
    import jax.numpy as jnp
    lp = jax.nn.log_softmax(logits_fn(p, X))
    return -jnp.take_along_axis(lp, y[:, None], 1)[:, 0]


def canaries(n_canary, seed=0):
    """Audit canaries: held-out digits (the last test images) with a random wrong label; each one is
    inserted into the training set with probability 1/2. Returns (X, y, included mask)."""
    _, _, Xte, yte = digits(seed=0)
    rng = np.random.default_rng(1000 + seed)
    Xc, yc = Xte[-n_canary:], yte[-n_canary:]
    yc = ((yc + rng.integers(1, 10, n_canary)) % 10).astype(np.int32)
    return Xc, yc, rng.random(n_canary) < 0.5


def train_mlp(dp=False, clip=1.0, sigma=1.0, batch=64, epochs=30, lr=0.5, seed=0, n_train=None, wd=0.0, n_canary=0):
    """Plain SGD or DP-SGD (Poisson sampling with rate batch/n, per-example clipping, Gaussian noise).
    Returns dict(params, hist=[(epoch, train_acc, test_acc)], q, steps)."""
    import jax
    import jax.numpy as jnp
    Xtr, ytr, Xte, yte = digits(seed=0, n_train=n_train)
    if n_canary:
        Xc, yc, inc = canaries(n_canary, seed)
        Xte, yte = Xte[:-n_canary], yte[:-n_canary]
        Xtr, ytr = np.concatenate([Xtr, Xc[inc]]), np.concatenate([ytr, yc[inc]])
    n = len(Xtr)
    q = batch / n
    p = _init(jax.random.PRNGKey(seed))
    rng = np.random.default_rng(seed)

    def one(p, x, y):
        return per_example_loss(p, x[None], y[None])[0]

    grad_one = jax.jit(jax.vmap(jax.grad(one), in_axes=(None, 0, 0)))

    @jax.jit
    def mean_grad(p, X, y):
        return jax.grad(lambda p: per_example_loss(p, X, y).mean())(p)

    acc = jax.jit(lambda p, X, y: (logits_fn(p, X).argmax(1) == y).mean())
    steps_per_epoch = int(round(1 / q))
    hist, steps = [], 0
    for ep in range(epochs + 1):
        hist.append((ep, float(acc(p, Xtr, ytr)), float(acc(p, Xte, yte))))
        if ep == epochs:
            break
        for _ in range(steps_per_epoch):
            if dp:
                m = rng.random(n) < q  # Poisson sampling
                g = grad_one(p, Xtr[m], ytr[m]) if m.any() else jax.tree_util.tree_map(lambda a: a[None] * 0, p)
                norms = jnp.sqrt(sum((a.reshape(a.shape[0], -1) ** 2).sum(1) for a in jax.tree_util.tree_leaves(g)))
                f = jnp.minimum(1.0, clip / (norms + 1e-12))
                g = jax.tree_util.tree_map(
                    lambda a: (jnp.tensordot(f, a, 1) + sigma * clip * rng.standard_normal(a.shape[1:])) / batch, g)
            else:
                i = rng.choice(n, batch, replace=False)
                g = mean_grad(p, Xtr[i], ytr[i])
            p = jax.tree_util.tree_map(lambda w, gw: w - lr * (gw + wd * w), p, g)
            steps += 1
    return {"params": jax.tree_util.tree_map(np.asarray, p), "hist": hist, "q": q, "steps": steps}


def cached(cache_dir, fn, **cfg):
    key = fn.__name__ + "_" + "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))
    path = os.path.join(cache_dir, key + ".pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    res = fn(**cfg)
    os.makedirs(cache_dir, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


# ---------------------------------------------------------------- membership inference
def losses(params, X, y):
    return np.asarray(per_example_loss(params, X, y))


def auc(member_scores, nonmember_scores):
    """P(score(member) > score(nonmember)) with ties counted as 1/2."""
    m, n = np.asarray(member_scores)[:, None], np.asarray(nonmember_scores)[None, :]
    return float((m > n).mean() + 0.5 * (m == n).mean())
