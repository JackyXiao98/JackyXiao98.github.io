"""Toy experiments for the Knowledge Distillation series (CPU-sized; results cached in KD_CACHE).

- MLPs on sklearn's 8x8 digits (dpx.digits): teacher / student training with hard labels, soft
  labels (temperature T) or both; held-out classes; FitNets-style hint loss; born-again generations
- forward vs reverse KL fits of a single Gaussian to a two-mode mixture (exact, by grid search)
- off-policy vs on-policy distillation of a 2nd-order Markov "language" teacher into a 1st-order
  student (capacity mismatch)
- sequence-level distillation of the tfm.py addition model into smaller students
- compression: magnitude pruning and INT8/INT4 weight quantisation of the digit teacher
"""

import os
import pickle

import numpy as np

import dpx
import tfm


def cached(cache_dir, name, fn, **kw):
    path = os.path.join(cache_dir, name + "_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items()) if k != "tf_cache") + ".pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    res = fn(**kw)
    os.makedirs(cache_dir, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


def softmax(z, T=1.0):
    z = np.asarray(z, float) / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------- MLPs on digits
def mlp_init(key, sizes):
    import jax
    ks = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (a, b)) / np.sqrt(a), np.zeros(b, np.float32)) for k, a, b in zip(ks, sizes[:-1], sizes[1:])]


def mlp_apply(p, x, return_hidden=False):
    import jax.numpy as jnp
    hs = []
    for w, b in p[:-1]:
        x = jnp.tanh(x @ w + b)
        hs.append(x)
    out = x @ p[-1][0] + p[-1][1]
    return (out, hs) if return_hidden else out


def train_mlp(sizes, X, y=None, soft=None, T=4.0, alpha=0.0, steps=3000, lr=3e-3, seed=0, hint=None, wd=0.0, init_net=None):
    """Train an MLP. Loss = alpha * CE(y) + (1 - alpha) * T^2 * KL(soft || student_T) (if soft given),
    + optional hint loss ||regressor(h_student) - h_teacher||^2 on a chosen hidden layer."""
    import jax
    import jax.numpy as jnp
    import optax
    key = jax.random.PRNGKey(seed)
    p = {"net": mlp_init(key, sizes) if init_net is None else jax.tree_util.tree_map(jnp.asarray, init_net)}
    if hint is not None:
        H_t, layer_s = hint
        p["reg"] = jax.random.normal(jax.random.split(key)[0], (sizes[layer_s + 1], H_t.shape[1])) / np.sqrt(sizes[layer_s + 1])
    opt = optax.adamw(optax.cosine_decay_schedule(lr, steps), weight_decay=wd)
    st = opt.init(p)
    n = len(X)
    Xj = jnp.array(X)
    yj = None if y is None else jnp.array(y)
    sj = None if soft is None else jnp.array(soft)

    def loss(p, idx):
        out, hs = mlp_apply(p["net"], Xj[idx], return_hidden=True)
        l = 0.0
        if yj is not None and alpha > 0:
            l += alpha * -jnp.take_along_axis(jax.nn.log_softmax(out), yj[idx][:, None], 1).mean()
        if sj is not None and alpha < 1:
            lq = jax.nn.log_softmax(out / T)
            pt = sj[idx]
            l += (1 - alpha) * T * T * (pt * (jnp.log(pt + 1e-12) - lq)).sum(1).mean()
        if hint is not None:
            l += ((hs[layer_s] @ p["reg"] - jnp.array(H_t)[idx]) ** 2).mean()
        return l

    @jax.jit
    def step(p, st, idx):
        g = jax.grad(loss)(p, idx)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st

    rng = np.random.default_rng(seed)
    for _ in range(steps):
        p, st = step(p, st, jnp.array(rng.choice(n, min(128, n))))
    return jax.tree_util.tree_map(lambda a: np.array(a), p["net"])


def acc(p, X, y):
    return float((np.asarray(mlp_apply(p, X)).argmax(1) == y).mean())


def logits(p, X):
    return np.asarray(mlp_apply(p, X))


TEACHER = [64, 512, 512, 10]
STUDENT = [64, 16, 10]


def digits_kd(n_label=100, seeds=(0, 1, 2, 3, 4)):
    """Teacher trained on all 1,197 labelled digits; the small student only sees the first n_label
    training images, either with their hard labels or with the teacher's soft labels (T = 4,
    alpha = 0.1)."""
    Xtr, ytr, Xte, yte = dpx.digits()
    t = train_mlp(TEACHER, Xtr, ytr, alpha=1.0, wd=1e-4)
    Xs, ys = Xtr[:n_label], ytr[:n_label]
    soft = softmax(logits(t, Xs), 4.0)
    res = dict(teacher=acc(t, Xte, yte), hard=[], kd=[], teacher_params=sum(w.size + b.size for w, b in t),
               student_params=sum(a * b + b for a, b in zip(STUDENT[:-1], STUDENT[1:])))
    for s in seeds:
        res["hard"].append(acc(train_mlp(STUDENT, Xs, ys, alpha=1.0, seed=s), Xte, yte))
        res["kd"].append(acc(train_mlp(STUDENT, Xs, ys, soft=soft, alpha=0.1, seed=s), Xte, yte))
    res["teacher_soft_te"] = softmax(logits(t, Xte), 1.0)
    res["teacher_logits_te"] = logits(t, Xte)
    res["yte"] = yte
    return res


def _bias_fix(p, X_unlab, X_test, cls, prior=0.1):
    """Raise the logit of the never-seen class until it is predicted for `prior` of unlabelled inputs
    (uses only the class prior, not test labels); return test predictions and the bias."""
    L = logits(p, X_unlab).copy()
    b = 0.0
    while (np.argmax(L + np.eye(L.shape[1])[cls] * b, 1) == cls).mean() < prior:
        b += 0.25
    Lt = logits(p, X_test).copy()
    Lt[:, cls] += b
    return Lt.argmax(1), b


def dark_knowledge(held_out=3, seeds=(0, 1, 2), T=4.0):
    """Transfer set without any example of class `held_out`: student trained only on the teacher's
    soft labels (no true labels) vs. on hard labels; the never-seen class's logit is then raised
    until it is predicted for 10% of the (unlabelled) training images; accuracy on that class."""
    Xtr, ytr, Xte, yte = dpx.digits()
    t = train_mlp(TEACHER, Xtr, ytr, alpha=1.0, wd=1e-4)
    keep = ytr != held_out
    soft = softmax(logits(t, Xtr[keep]), T)
    m3 = yte == held_out
    out = dict(teacher=acc(t, Xte, yte), soft_all=[], soft_held=[], hard_all=[], hard_held=[], soft_only_full=[],
               soft_held_raw=[], hard_held_raw=[])
    softfull = softmax(logits(t, Xtr), T)
    for s in seeds:
        ps = train_mlp(STUDENT, Xtr[keep], soft=soft, alpha=0.0, T=T, seed=s)
        ph = train_mlp(STUDENT, Xtr[keep], ytr[keep], alpha=1.0, seed=s)
        out["soft_held_raw"].append(acc(ps, Xte[m3], yte[m3]))
        out["hard_held_raw"].append(acc(ph, Xte[m3], yte[m3]))
        for name, p in (("soft", ps), ("hard", ph)):
            pr, _ = _bias_fix(p, Xtr, Xte, held_out)
            out[name + "_all"].append(float((pr == yte).mean()))
            out[name + "_held"].append(float((pr[m3] == held_out).mean()))
        out["soft_only_full"].append(acc(train_mlp(STUDENT, Xtr, soft=softfull, alpha=0.0, T=T, seed=s), Xte, yte))
    lt = logits(t, Xte)
    out["sim"] = np.array([softmax(lt[yte == c], T).mean(0) for c in range(10)])
    out["example_logits"] = lt[np.where(yte == held_out)[0][0]]
    out["n_removed"] = int((~keep).sum())
    return out


def fitnets(seeds=(0, 1, 2), n_label=100, lam=1.0, steps=3000):
    """Student [64, 32, 32, 10] with only n_label labelled images. (a) labels only; (b) labels + matching
    the teacher's first hidden layer (via a learned linear regressor) on all 1,197 unlabelled inputs
    (FitNets-style hint); (c) the teacher's soft labels on all inputs, no true labels."""
    import jax
    import jax.numpy as jnp
    import optax
    Xtr, ytr, Xte, yte = dpx.digits()
    t = train_mlp(TEACHER, Xtr, ytr, alpha=1.0, wd=1e-4)
    _, hs = mlp_apply(t, Xtr, return_hidden=True)
    H1 = jnp.array(np.asarray(hs[0]))
    S = [64, 32, 32, 10]
    X, y = jnp.array(Xtr), jnp.array(ytr)
    soft = softmax(logits(t, Xtr), 4.0)

    def joint(seed, lam_):
        k = jax.random.PRNGKey(seed)
        p = {"net": mlp_init(k, S), "reg": jax.random.normal(k, (32, H1.shape[1])) / np.sqrt(32)}
        opt = optax.adam(optax.cosine_decay_schedule(3e-3, steps))
        st = opt.init(p)

        def loss(p, il, iu):
            out = mlp_apply(p["net"], X[il])
            ce = -jnp.take_along_axis(jax.nn.log_softmax(out), y[il][:, None], 1).mean()
            _, h = mlp_apply(p["net"], X[iu], return_hidden=True)
            return ce + lam_ * ((h[1] @ p["reg"] - H1[iu]) ** 2).mean()

        @jax.jit
        def step(p, st, il, iu):
            g = jax.grad(loss)(p, il, iu)
            u, st = opt.update(g, st)
            return optax.apply_updates(p, u), st

        r = np.random.default_rng(seed)
        for _ in range(steps):
            p, st = step(p, st, jnp.array(r.choice(n_label, 64)), jnp.array(r.choice(len(Xtr), 128)))
        return acc(jax.tree_util.tree_map(np.asarray, p["net"]), Xte, yte)

    out = dict(labels=[], hint=[], soft=[])
    for s in seeds:
        out["labels"].append(joint(s, 0.0))
        out["hint"].append(joint(s, lam))
        out["soft"].append(acc(train_mlp(S, Xtr, soft=soft, alpha=0.0, seed=s), Xte, yte))
    return out


def born_again(generations=4, seeds=(0, 1, 2, 3, 4), n_train=300, smooth=0.1):
    """Same-size networks [64, 64, 10] on 300 labelled digits; generation k+1 is trained on hard labels
    plus the soft labels of generation k (T = 2, alpha = 0.5). Compare label smoothing."""
    Xtr, ytr, Xte, yte = dpx.digits(n_train=n_train)
    S = [64, 64, 10]
    gens = np.zeros((len(seeds), generations))
    ls = []
    for i, s in enumerate(seeds):
        p = train_mlp(S, Xtr, ytr, alpha=1.0, seed=s)
        gens[i, 0] = acc(p, Xte, yte)
        for g in range(1, generations):
            soft = softmax(logits(p, Xtr), 2.0)
            p = train_mlp(S, Xtr, ytr, soft=soft, alpha=0.5, T=2.0, seed=s + 100 * g)
            gens[i, g] = acc(p, Xte, yte)
        onehot = np.eye(10)[ytr] * (1 - smooth) + smooth / 10
        ls.append(acc(train_mlp(S, Xtr, soft=onehot, alpha=0.0, T=1.0, seed=s), Xte, yte))
    return dict(gens=gens, ls=np.array(ls))


# ---------------------------------------------------------------- forward vs reverse KL
def kl_fits():
    """Teacher p = 0.6 N(-2, 0.5^2) + 0.4 N(2, 0.5^2); student q = N(mu, s^2). Grid search over (mu, s)."""
    x = np.linspace(-8, 8, 4001)
    dx = x[1] - x[0]
    n = lambda m, s: np.exp(-(x - m) ** 2 / (2 * s * s)) / (np.sqrt(2 * np.pi) * s)
    p = 0.6 * n(-2, 0.5) + 0.4 * n(2, 0.5)
    mus, ss = np.linspace(-3, 3, 241), np.linspace(0.2, 3, 141)
    fwd, rev = np.zeros((len(mus), len(ss))), np.zeros((len(mus), len(ss)))
    for i, m in enumerate(mus):
        for j, s in enumerate(ss):
            q = n(m, s) + 1e-300
            fwd[i, j] = (p * (np.log(p + 1e-300) - np.log(q))).sum() * dx
            rev[i, j] = (q * (np.log(q) - np.log(p + 1e-300))).sum() * dx
    i, j = np.unravel_index(fwd.argmin(), fwd.shape)
    k, l = np.unravel_index(rev.argmin(), rev.shape)
    mean = (p * x).sum() * dx
    std = np.sqrt((p * (x - mean) ** 2).sum() * dx)
    q_f, q_r = n(mus[i], ss[j]), n(mus[k], ss[l])
    gap = (np.abs(x) < 0.5)
    return dict(x=x, p=p, fwd=(float(mus[i]), float(ss[j])), rev=(float(mus[k]), float(ss[l])), mean=float(mean), std=float(std),
                q_f=q_f, q_r=q_r, gap_mass_p=float(p[gap].sum() * dx), gap_mass_f=float(q_f[gap].sum() * dx),
                gap_mass_r=float(q_r[gap].sum() * dx))


# ---------------------------------------------------------------- off- vs on-policy distillation
def markov_teacher(V=12, seed=0, first=3.0, second=1.5):
    """P(x_t | x_{t-2}, x_{t-1}): a strong first-order part plus a second-order modulation."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((V, V)) * first
    B = rng.standard_normal((V, V, V)) * second
    return softmax(A[None, :, :] + B)


def onpolicy_vs_offpolicy(V=12, L=30, n_seq=4000, steps=600, lr=0.5, seed=0):
    """Teacher: 2nd-order Markov chain. Student: 1st-order (can't represent the teacher).
    Off-policy: minimise forward KL on teacher-generated sequences (closed form: MLE of the bigram).
    On-policy: minimise reverse KL(student || teacher) per token on student-generated sequences
    (GKD-style, no gradient through sampling). Evaluate on student samples: average teacher
    log-probability per token and the share of tokens the teacher considers unlikely (p < 0.02)."""
    rng = np.random.default_rng(seed)
    P = markov_teacher(V, seed)

    def sample_teacher(n):
        s = np.zeros((n, L), int)
        s[:, :2] = rng.integers(0, V, (n, 2))
        for t in range(2, L):
            pr = P[s[:, t - 2], s[:, t - 1]]
            s[:, t] = (pr.cumsum(1) > rng.random((n, 1))).argmax(1)
        return s

    def sample_student(Q, n, start):
        s = np.zeros((n, L), int)
        s[:, :2] = start
        for t in range(2, L):
            pr = Q[s[:, t - 1]]
            s[:, t] = (pr.cumsum(1) > rng.random((n, 1))).argmax(1)
        return s

    def evaluate(Q, n=4000):
        s = sample_student(Q, n, rng.integers(0, V, (n, 2)))
        pt = P[s[:, :-2], s[:, 1:-1], s[:, 2:]]
        return dict(teacher_logp=float(np.log(pt).mean()), unlikely=float((pt < 0.02).mean()))

    def teacher_text_nll(Q, n=4000):
        s = sample_teacher(n)
        return float(-np.log(Q[s[:, 1:-1], s[:, 2:]]).mean())

    # off-policy: bigram MLE on teacher text
    s = sample_teacher(n_seq)
    C = np.ones((V, V)) * 0.1
    np.add.at(C, (s[:, 1:-1].ravel(), s[:, 2:].ravel()), 1)
    Q_off = C / C.sum(1, keepdims=True)
    # on-policy: start from the off-policy student, minimise per-token reverse KL on own samples
    th = np.log(Q_off)
    hist = []
    for it in range(steps + 1):
        Q = softmax(th)
        if it % 100 == 0:
            hist.append(dict(step=it, **evaluate(Q)))
        if it == steps:
            break
        st = sample_student(Q, 256, rng.integers(0, V, (256, 2)))
        prev2, prev1 = st[:, :-1][:, :-1].ravel(), st[:, 1:-1].ravel()
        pt = P[prev2, prev1]                       # teacher next-token dist on student prefixes
        q = Q[prev1]
        lq, lp = np.log(q), np.log(pt)
        kl_terms = q * (lq - lp)
        kl = kl_terms.sum(1, keepdims=True)
        g = q * ((lq - lp) - kl)                    # d KL(q || p) / d logits
        G = np.zeros_like(th)
        np.add.at(G, prev1, g)
        th -= lr * G / len(prev1) * V
    Q_on = softmax(th)
    st_ = sample_teacher(4000)
    pt_ = P[st_[:, :-2], st_[:, 1:-1], st_[:, 2:]]
    return dict(off=evaluate(Q_off), on=evaluate(Q_on), teacher=dict(teacher_logp=float(np.log(pt_).mean()), unlikely=float((pt_ < 0.02).mean())),
                off_nll=teacher_text_nll(Q_off), on_nll=teacher_text_nll(Q_on), hist=hist, V=V)


# ---------------------------------------------------------------- sequence-level KD on addition
def addition_seqkd(tf_cache, sizes=(16, 32, 64), seed=0):
    """Students of width d trained (600 steps) on (a) 300 labelled problems, (b) 7,700 problems labelled
    by the trained teacher, (c) the same problems labelled by a weak teacher (120-step checkpoint)."""
    import jax
    import optax
    P, _, snaps = tfm.cached_train(tf_cache, steps=600, log_every=20, snapshots=(0, 60, 120, 200, 300, 600))
    TR, TE, _, _ = tfm.dataset()
    idx = np.random.default_rng(seed).permutation(len(TR))
    lab, pool = TR[idx[:300]], TR[idx[300:]]
    strong = tfm.generate(P, pool[:, :6])
    weak = tfm.generate(snaps[120], pool[:, :6])
    q_strong = float((strong == pool).all(1).mean())
    q_weak = float((weak == pool).all(1).mean())

    def train(data, d):
        p = tfm.init(jax.random.PRNGKey(seed), d=d)
        heads = p.pop("heads")
        opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, 3e-3, 100, 600), weight_decay=0.01)
        st = opt.init(p)

        @jax.jit
        def step(p, st, b):
            g = jax.grad(lambda p: tfm.loss_fn({**p, "heads": heads}, b))(p)
            u, st = opt.update(g, st, p)
            return optax.apply_updates(p, u), st

        r = np.random.default_rng(seed)
        for _ in range(600):
            p, st = step(p, st, data[r.choice(len(data), 256)])
        return tfm.accuracy({**tfm.to_numpy(p), "heads": heads}, TE)

    out = dict(teacher=tfm.accuracy(P, TE), weak_teacher=tfm.accuracy(snaps[120], TE), q_strong=q_strong, q_weak=q_weak, res={})
    for d in sizes:
        out["res"][d] = dict(labels300=train(lab, d), strong=train(np.concatenate([lab, strong]), d),
                             weak=train(np.concatenate([lab, weak]), d))
    return out


def ood_imitation(tf_cache, seed=0):
    """Student (d = 32) trained only on teacher answers for problems whose first addend < 50;
    tested on held-out problems with first addend < 50 and >= 50."""
    import jax
    import optax
    P, _, _ = tfm.cached_train(tf_cache, steps=600, log_every=20, snapshots=(0, 60, 120, 200, 300, 600))
    TR, TE, _, _ = tfm.dataset()
    a_tr, a_te = TR[:, 0] * 10 + TR[:, 1], TE[:, 0] * 10 + TE[:, 1]
    src = TR[a_tr < 50]
    data = tfm.generate(P, src[:, :6])
    p = tfm.init(jax.random.PRNGKey(seed), d=32)
    heads = p.pop("heads")
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, 3e-3, 100, 600), weight_decay=0.01)
    st = opt.init(p)

    @jax.jit
    def step(p, st, b):
        g = jax.grad(lambda p: tfm.loss_fn({**p, "heads": heads}, b))(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st

    r = np.random.default_rng(seed)
    for _ in range(600):
        p, st = step(p, st, data[r.choice(len(data), 256)])
    ps = {**tfm.to_numpy(p), "heads": heads}
    return dict(n_train=int(len(src)), in_dist=tfm.accuracy(ps, TE[a_te < 50]), ood=tfm.accuracy(ps, TE[a_te >= 50]),
                teacher_in=tfm.accuracy(P, TE[a_te < 50]), teacher_ood=tfm.accuracy(P, TE[a_te >= 50]))


# ---------------------------------------------------------------- compression
def compression(seed=0):
    Xtr, ytr, Xte, yte = dpx.digits()
    t = train_mlp(TEACHER, Xtr, ytr, alpha=1.0, wd=1e-4)
    n_params = sum(w.size + b.size for w, b in t)

    def prune(p, frac):
        allw = np.concatenate([np.abs(w).ravel() for w, _ in p])
        th = np.quantile(allw, frac)
        return [(np.where(np.abs(w) >= th, w, 0), b) for w, b in p]

    def quant(p, bits):
        q = 2 ** (bits - 1) - 1
        return [((np.round(w / (np.abs(w).max(0) / q)) * (np.abs(w).max(0) / q)), b) for w, b in p]  # per output channel

    res = dict(teacher=acc(t, Xte, yte), n_params=n_params, prune={}, quant={})
    for f in (0.5, 0.8, 0.9, 0.95, 0.98):
        res["prune"][f] = acc(prune(t, f), Xte, yte)
    for b in (8, 4, 3, 2):
        res["quant"][b] = acc(quant(t, b), Xte, yte)
    soft = softmax(logits(t, Xtr), 4.0)
    S = [64, 32, 10]
    st = train_mlp(S, Xtr, ytr, soft=soft, alpha=0.1, seed=seed)
    res["student"] = acc(st, Xte, yte)
    res["student_params"] = sum(w.size + b.size for w, b in st)
    res["student_int8"] = acc(quant(st, 8), Xte, yte)
    res["student_int4"] = acc(quant(st, 4), Xte, yte)
    return res
