"""Exact 2D toy diffusion: Gaussian-mixture data, exact denoiser/score, and samplers.

Discrete VP schedule of DDPM (T=1000, linear beta 1e-4..0.02). For a mixture of isotropic
Gaussians the noised marginal q(x_t) is again a mixture, so E[x0|x_t] and the score are exact.
"""

import numpy as np

T = 1000
BETA = np.concatenate([[0.0], np.linspace(1e-4, 0.02, T)])
ABAR = np.cumprod(1.0 - BETA)  # discrete DDPM schedule, ABAR[0] = 1

# Continuous-time version of the same schedule (VP SDE, beta(t) = 0.1 + 19.9 t, t in [0, 1]).
B0, B1 = 0.1, 20.0
T_MIN = 1e-3


def abar(t):
    return np.exp(-(B0 * t + 0.5 * (B1 - B0) * t ** 2))


def beta(t):
    return B0 + (B1 - B0) * t


class GMM:
    def __init__(self, means, sigma, weights=None):
        self.mu = np.asarray(means, float)
        self.s2 = float(sigma) ** 2
        k = len(self.mu)
        self.w = np.full(k, 1.0 / k) if weights is None else np.asarray(weights, float) / np.sum(weights)

    def sample(self, n, rng):
        k = rng.choice(len(self.mu), n, p=self.w)
        return self.mu[k] + np.sqrt(self.s2) * rng.standard_normal((n, 2))

    def _resp(self, x, a):
        """Responsibilities and per-component variance of q(x_t) with abar = a."""
        var = a * self.s2 + 1 - a
        diff = x[:, None, :] - np.sqrt(a) * self.mu[None]
        logw = np.log(self.w)[None] - (diff ** 2).sum(-1) / (2 * var)
        logw -= logw.max(1, keepdims=True)
        r = np.exp(logw)
        r /= r.sum(1, keepdims=True)
        return r, diff, var

    def x0_hat(self, x, a):
        r, diff, var = self._resp(x, a)
        return (r[..., None] * (self.mu[None] + np.sqrt(a) * self.s2 / var * diff)).sum(1)

    def eps_hat(self, x, a):
        return (x - np.sqrt(a) * self.x0_hat(x, a)) / np.sqrt(1 - a)

    def score(self, x, a=1.0, extra_var=0.0):
        """grad log density of the data convolved to abar=a (plus optional extra isotropic noise)."""
        var = a * self.s2 + 1 - a + extra_var
        diff = x[:, None, :] - np.sqrt(a) * self.mu[None]
        logw = np.log(self.w)[None] - (diff ** 2).sum(-1) / (2 * var)
        logw -= logw.max(1, keepdims=True)
        r = np.exp(logw)
        r /= r.sum(1, keepdims=True)
        return -(r[..., None] * diff).sum(1) / var

    def logpdf(self, x, extra_var=0.0):
        var = self.s2 + extra_var
        d2 = ((x[:, None, :] - self.mu[None]) ** 2).sum(-1)
        comp = np.log(self.w)[None] - d2 / (2 * var) - np.log(2 * np.pi * var)
        m = comp.max(1, keepdims=True)
        return (m + np.log(np.exp(comp - m).sum(1, keepdims=True)))[:, 0]


def grid(n, kind="time"):
    """n+1 times from 1 down to T_MIN; uniform in time or in log-SNR."""
    if kind == "time":
        return np.linspace(1.0, T_MIN, n + 1)
    lam = lambda t: 0.5 * np.log(abar(t) / (1 - abar(t)))
    ls = np.linspace(lam(1.0), lam(T_MIN), n + 1)
    tt = np.linspace(0, 1, 20001)[1:]
    lt = lam(tt)
    return np.interp(ls, lt[::-1], tt[::-1])


def _finish(gmm, x, t):
    """Final denoising: return E[x0 | x_{t_min}]."""
    return gmm.x0_hat(x, abar(t))


def ddpm(gmm, xT, n, rng, keep=None):
    """Ancestral sampling with n steps (respaced posterior steps)."""
    ts = grid(n)
    x, traj = xT.copy(), ([xT[:keep].copy()] if keep else None)
    for t, s in zip(ts[:-1], ts[1:]):
        a, a_s = abar(t), abar(s)
        x0 = gmm.x0_hat(x, a)
        b = 1 - a / a_s
        mean = np.sqrt(a_s) * b / (1 - a) * x0 + np.sqrt(1 - b) * (1 - a_s) / (1 - a) * x
        x = mean + np.sqrt((1 - a_s) / (1 - a) * b) * rng.standard_normal(x.shape)
        if keep:
            traj.append(x[:keep].copy())
    x = _finish(gmm, x, ts[-1])
    return (x, np.array(traj + [x[:keep].copy()])) if keep else x


def ddim(gmm, xT, n, keep=None):
    """Deterministic DDIM (eta = 0)."""
    ts = grid(n)
    x, traj = xT.copy(), ([xT[:keep].copy()] if keep else None)
    for t, s in zip(ts[:-1], ts[1:]):
        a, a_s = abar(t), abar(s)
        x0 = gmm.x0_hat(x, a)
        eps = (x - np.sqrt(a) * x0) / np.sqrt(1 - a)
        x = np.sqrt(a_s) * x0 + np.sqrt(1 - a_s) * eps
        if keep:
            traj.append(x[:keep].copy())
    x = _finish(gmm, x, ts[-1])
    return (x, np.array(traj + [x[:keep].copy()])) if keep else x


def dpm_solver_pp_2m(gmm, xT, n, keep=None):
    """DPM-Solver++(2M): data prediction, second-order multistep, log-SNR uniform steps."""
    ts = grid(n, "logsnr")
    lam = lambda t: 0.5 * np.log(abar(t) / (1 - abar(t)))
    x, traj = xT.copy(), ([xT[:keep].copy()] if keep else None)
    prev = None
    for t, s in zip(ts[:-1], ts[1:]):
        a, a_s = abar(t), abar(s)
        x0 = gmm.x0_hat(x, a)
        h = lam(s) - lam(t)
        if prev is None:
            D = x0
        else:
            r = prev[1] / h
            D = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * prev[0]
        x = np.sqrt((1 - a_s) / (1 - a)) * x - np.sqrt(a_s) * np.expm1(-h) * D
        prev = (x0, h)
        if keep:
            traj.append(x[:keep].copy())
    x = _finish(gmm, x, ts[-1])
    return (x, np.array(traj + [x[:keep].copy()])) if keep else x


def bad_fraction(gmm, x, k=4.0):
    """Share of samples farther than k*sigma from every mode center (i.e. between modes)."""
    d = np.sqrt(((x[:, None, :] - gmm.mu[None]) ** 2).sum(-1)).min(1)
    return float((d > k * np.sqrt(gmm.s2)).mean())


def sliced_w2(x, y, n_proj=200, seed=0):
    """Sliced Wasserstein-2 distance between two equal-size 2D samples."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, np.pi, n_proj)
    dirs = np.stack([np.cos(th), np.sin(th)], 1)
    px, py = np.sort(x @ dirs.T, 0), np.sort(y @ dirs.T, 0)
    return float(np.sqrt(((px - py) ** 2).mean()))
