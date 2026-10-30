"""Exact flow matching (linear / rectified-flow path) for a GMM target and N(0, I) source."""
import numpy as np

T_END = 1 - 1e-3


def x1_hat(g, x, t):
    """E[x1 | x_t] for x_t = (1-t) x0 + t x1, x0 ~ N(0,I), x1 ~ GMM g."""
    var = t ** 2 * g.s2 + (1 - t) ** 2
    diff = x[:, None, :] - t * g.mu[None]
    logw = np.log(g.w)[None] - (diff ** 2).sum(-1) / (2 * var)
    logw -= logw.max(1, keepdims=True)
    r = np.exp(logw); r /= r.sum(1, keepdims=True)
    return (r[..., None] * (g.mu[None] + t * g.s2 / var * diff)).sum(1)


def velocity(g, x, t):
    return (x1_hat(g, x, t) - x) / (1 - t)


def fm_euler(g, x0, n, keep=None):
    ts = np.linspace(0, T_END, n + 1)
    x, traj = x0.copy(), ([x0[:keep].copy()] if keep else None)
    for t, s in zip(ts[:-1], ts[1:]):
        x = x + (s - t) * velocity(g, x, t)
        if keep:
            traj.append(x[:keep].copy())
    x = x1_hat(g, x, ts[-1])
    return (x, np.array(traj + [x[:keep].copy()])) if keep else x


def straightness(traj):
    """endpoint displacement / path length, averaged (1 = perfectly straight)."""
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=-1).sum(0)
    disp = np.linalg.norm(traj[-1] - traj[0], axis=-1)
    return float(np.mean(disp / np.maximum(seg, 1e-9)))
