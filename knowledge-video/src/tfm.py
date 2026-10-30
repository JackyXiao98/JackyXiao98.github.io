"""Small decoder-only Transformer in JAX, trained on 2-digit addition ("34+58=092").

Used by the Transformer series: real training curves, generations, attention maps and the KV
cache / scaling experiments. Training results are cached in <cache_dir>/addition_<cfg>.npz.
"""

import os

import numpy as np

VOCAB = list("0123456789+=")
STOI = {c: i for i, c in enumerate(VOCAB)}
SEQ = 9  # "ab+cd=sss"
ANS = 6  # answer starts at position 6 (predicted from positions 5..7)


def encode(a, b):
    return [STOI[c] for c in f"{a:02d}+{b:02d}={a + b:03d}"]


def dataset(seed=0, test_frac=0.2):
    pairs = [(a, b) for a in range(100) for b in range(100)]
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    n_test = int(len(pairs) * test_frac)
    test, train = pairs[:n_test], pairs[n_test:]
    to = lambda ps: np.array([encode(a, b) for a, b in ps], np.int32)
    return to(train), to(test), train, test


def init(key, d=64, heads=4, layers=2, ff=4):
    import jax
    ks = iter(jax.random.split(key, 4 + 6 * layers))
    n = lambda shape, s: jax.random.normal(next(ks), shape) * s
    p = {"tok": n((len(VOCAB), d), 0.1), "pos": n((SEQ, d), 0.1), "out": n((d, len(VOCAB)), 1 / np.sqrt(d)),
         "layers": []}
    for _ in range(layers):
        p["layers"].append({
            "qkv": n((d, 3 * d), 1 / np.sqrt(d)), "o": n((d, d), 1 / np.sqrt(d) / np.sqrt(2 * layers)),
            "w1": n((d, ff * d), 1 / np.sqrt(d)), "w2": n((ff * d, d), 1 / np.sqrt(ff * d) / np.sqrt(2 * layers)),
            "g1": np.ones(d, np.float32), "g2": np.ones(d, np.float32)})
    p["gf"] = np.ones(d, np.float32)
    p["heads"] = heads
    return p


def _ln(x, g, xp):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / xp.sqrt(v + 1e-5) * g


def forward(p, idx, xp=None, return_attn=False):
    """Pre-LN causal Transformer. idx: (B, T) int. Works with jax.numpy or numpy (xp)."""
    import jax.numpy as jnp
    xp = xp or jnp
    B, T = idx.shape
    h = p["heads"]
    x = p["tok"][idx] + p["pos"][:T]
    d = x.shape[-1]
    mask = xp.tril(xp.ones((T, T))) > 0
    attns = []
    for L in p["layers"]:
        y = _ln(x, L["g1"], xp)
        q, k, v = xp.split(y @ L["qkv"], 3, axis=-1)
        sh = lambda t: t.reshape(B, T, h, d // h).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d // h)
        s = xp.where(mask, s, -1e9)
        s = s - s.max(-1, keepdims=True)
        a = xp.exp(s)
        a = a / a.sum(-1, keepdims=True)
        attns.append(a)
        o = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        x = x + o @ L["o"]
        y = _ln(x, L["g2"], xp)
        x = x + xp.maximum(0, y @ L["w1"]) @ L["w2"]
    logits = _ln(x, p["gf"], xp) @ p["out"]
    return (logits, attns) if return_attn else logits


def loss_fn(p, batch):
    import jax
    import jax.numpy as jnp
    logits = forward(p, batch[:, :-1])
    tgt = batch[:, 1:]
    lp = jax.nn.log_softmax(logits[:, ANS - 1:], -1)
    return -jnp.take_along_axis(lp, tgt[:, ANS - 1:, None], -1).mean()


def generate(p, prompts, xp=np):
    """Greedy decode the 3 answer digits for prompts (N, 6)."""
    seq = np.array(prompts)
    for _ in range(3):
        logits = forward(p, seq, xp=xp)
        nxt = np.asarray(logits[:, -1]).argmax(-1)
        seq = np.concatenate([seq, nxt[:, None]], 1)
    return seq


def accuracy(p, data, xp=np):
    out = generate(p, data[:, :ANS], xp=xp)
    return float((out[:, ANS:] == data[:, ANS:]).all(1).mean())


def to_numpy(p):
    import jax
    q = jax.tree_util.tree_map(lambda a: np.asarray(a) if hasattr(a, "shape") else a, p)
    return q


def train(d=64, heads=4, layers=2, steps=3000, batch=256, lr=3e-3, seed=0, log_every=100, snapshots=()):
    """Returns (params_numpy, history, snapshot_params). history: list of (step, train_loss, test_loss, test_acc)."""
    import jax
    import jax.numpy as jnp
    import optax
    tr, te, _, _ = dataset()
    p = init(jax.random.PRNGKey(seed), d, heads, layers)
    heads_ = p.pop("heads")
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, lr, 100, steps), weight_decay=0.01)
    st = opt.init(p)

    def lf(p, b):
        return loss_fn({**p, "heads": heads_}, b)

    @jax.jit
    def step(p, st, b):
        l, g = jax.value_and_grad(lf)(p, b)
        up, st = opt.update(g, st, p)
        return optax.apply_updates(p, up), st, l

    rng = np.random.default_rng(seed)
    hist, snaps = [], {}
    eval_test = jax.jit(lf)
    for i in range(steps + 1):
        if i in snapshots:
            snaps[i] = {**to_numpy(p), "heads": heads_}
        if i % log_every == 0:
            pn = {**to_numpy(p), "heads": heads_}
            hist.append((i, float(eval_test(p, tr[rng.choice(len(tr), 1000)])), float(eval_test(p, te)), accuracy(pn, te)))
        if i < steps:
            p, st, _ = step(p, st, tr[rng.choice(len(tr), batch)])
    return {**to_numpy(p), "heads": heads_}, hist, snaps


def n_params(p):
    import jax
    leaves = [a for a in jax.tree_util.tree_leaves({k: v for k, v in p.items() if k != "heads"}) if hasattr(a, "size")]
    return int(sum(a.size for a in leaves))


def cached_train(cache_dir, **cfg):
    """Train once and cache history + final params (pickled)."""
    import pickle
    key = "_".join(f"{k}{v}" for k, v in sorted(cfg.items()) if k != "snapshots")
    path = os.path.join(cache_dir, f"addition_{key}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    res = train(**cfg)
    os.makedirs(cache_dir, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res
