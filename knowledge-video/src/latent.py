"""Toy image set + PCA autoencoder + GMM latent model for the latent-diffusion episode."""
import numpy as np
from PIL import Image, ImageDraw

R = 32


def make_images(n, seed):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, R, R), np.float32)
    for i in range(n):
        img = Image.new("L", (R * 4, R * 4), 0)
        d = ImageDraw.Draw(img)
        cx, cy = 16 * 4, 16 * 4
        a, b = rng.uniform(3, 13) * 4, rng.uniform(3, 13) * 4
        d.ellipse([cx - a, cy - b, cx + a, cy + b], fill=int(rng.uniform(150, 255)))
        out[i] = np.asarray(img.resize((R, R), Image.LANCZOS), np.float32) / 255.0
    return out.reshape(n, -1)


class PCA:
    def __init__(self, X, d):
        self.mean = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        self.V = Vt[:d]
        self.ev = S ** 2 / (S ** 2).sum()

    def enc(self, X):
        return (X - self.mean) @ self.V.T

    def dec(self, Z):
        return np.clip(Z @ self.V + self.mean, 0, 1)


def fit_gmm(Z, k, seed, iters=200):
    """Diagonal-free (spherical per component) EM; returns means, variances, weights."""
    rng = np.random.default_rng(seed)
    n, d = Z.shape
    mu = Z[rng.choice(n, k, replace=False)]
    var = np.full(k, Z.var())
    w = np.full(k, 1 / k)
    for _ in range(iters):
        d2 = ((Z[:, None] - mu[None]) ** 2).sum(-1)
        logr = np.log(w)[None] - 0.5 * d2 / var[None] - 0.5 * d * np.log(var)[None]
        logr -= logr.max(1, keepdims=True)
        r = np.exp(logr); r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-9
        mu = (r.T @ Z) / nk[:, None]
        var = np.maximum((r * ((Z[:, None] - mu[None]) ** 2).sum(-1)).sum(0) / (nk * d), 1e-4)
        w = nk / n
    return mu, var, w


def x0_hat(x, a, mu, var, w):
    """Exact E[z0 | z_t] for a GMM with per-component isotropic variance (any dimension)."""
    d = x.shape[1]
    v = a * var + 1 - a                                    # (k,)
    diff = x[:, None, :] - np.sqrt(a) * mu[None]           # (n,k,d)
    logr = np.log(w)[None] - 0.5 * (diff ** 2).sum(-1) / v[None] - 0.5 * d * np.log(v)[None]
    logr -= logr.max(1, keepdims=True)
    r = np.exp(logr); r /= r.sum(1, keepdims=True)
    gain = (np.sqrt(a) * var / v)[None, :, None]
    return (r[..., None] * (mu[None] + gain * diff)).sum(1)


def ddim_latent(zT, mu, var, w, n, abar, keep=False):
    ts = np.linspace(1.0, 1e-3, n + 1)
    z, traj = zT.copy(), [zT.copy()]
    for t, s in zip(ts[:-1], ts[1:]):
        a, a_s = abar(t), abar(s)
        z0 = x0_hat(z, a, mu, var, w)
        eps = (z - np.sqrt(a) * z0) / np.sqrt(1 - a)
        z = np.sqrt(a_s) * z0 + np.sqrt(1 - a_s) * eps
        traj.append(z.copy())
    z = x0_hat(z, abar(ts[-1]), mu, var, w)
    traj.append(z.copy())
    return (z, np.array(traj)) if keep else z
