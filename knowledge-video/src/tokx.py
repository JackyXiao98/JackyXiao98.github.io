"""Toy experiments for the Tokenization series (cached via TOK_CACHE).

Corpora (paths from TOK_DATA, default ../data): Tiny Shakespeare (shakespeare.txt, 1.1 M characters, English) and
four files of the chinese-poetry project's Complete Tang Poems (poet.tang.{0,1000,2000,3000}.json, traditional
Chinese). Everything here is implemented from scratch in Python / numpy: character-level BPE on pre-split words,
byte-level BPE, WordPiece-style merge scores, a small Unigram language-model tokenizer (EM + pruning), and
tokenizer statistics.
"""

import collections
import heapq
import json
import os
import pickle
import re

import numpy as np


def cached(cdir, name, fn, **kw):
    key = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    path = os.path.join(cdir, f"{name}_{key}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    res = fn(**kw)
    os.makedirs(cdir, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return res


def _dir():
    return os.environ.get("TOK_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))


def english():
    return open(os.path.join(_dir(), "shakespeare.txt")).read()


def chinese():
    out = []
    for i in (0, 1000, 2000, 3000):
        for poem in json.load(open(os.path.join(_dir(), f"tang{i}.json"))):
            out.append("".join(poem["paragraphs"]))
    return "\n".join(out)


PRE = re.compile(r" ?[A-Za-z]+| ?[0-9]|[，。！？、；：「」]| ?[^\sA-Za-z0-9，。！？、；：「」]{1,8}|\s+")


def pretokenize(text):
    """GPT-2-like split: words keep their leading space; digits are split one by one before merging is NOT applied
    here (digits form their own pieces of length 1 only when split_digits=True in bpe_train)."""
    return PRE.findall(text)


# ---------------------------------------------------------------- BPE
def touched_pairs(seqs, idx):
    seen = set()
    for i in idx:
        sq = seqs[i][0]
        seen.update(zip(sq, sq[1:]))
    return seen


def bpe_train(words, n_merges, byte_level=False, log_every=None):
    """words: Counter of pre-tokenised strings. Returns merges [(a, b)], vocab list, and snapshots."""
    seqs = {}
    for w, c in words.items():
        sym = tuple(bytes([b]) for b in w.encode("utf-8")) if byte_level else tuple(w)
        seqs[sym] = seqs.get(sym, 0) + c
    seqs = [(list(s), c) for s, c in seqs.items()]
    base = sorted({s for sq, _ in seqs for s in sq})
    pairs = collections.Counter()
    where = collections.defaultdict(set)
    for i, (sq, c) in enumerate(seqs):
        for a, b in zip(sq, sq[1:]):
            pairs[(a, b)] += c
            where[(a, b)].add(i)
    merges, snaps = [], []
    heap = [(-c, pr) for pr, c in pairs.items()]
    heapq.heapify(heap)
    for m in range(n_merges):
        while heap and pairs.get(heap[0][1], 0) != -heap[0][0]:
            heapq.heappop(heap)                                  # stale entry
        if not heap:
            break
        negc, (a, b) = heapq.heappop(heap)
        cnt = -negc
        if cnt < 2:
            break
        merges.append((a, b, cnt))
        new = a + b
        for i in list(where[(a, b)]):
            sq, c = seqs[i]
            for x, y in zip(sq, sq[1:]):
                pairs[(x, y)] -= c
                if pairs[(x, y)] <= 0:
                    del pairs[(x, y)]
            j, out = 0, []
            while j < len(sq):
                if j < len(sq) - 1 and sq[j] == a and sq[j + 1] == b:
                    out.append(new)
                    j += 2
                else:
                    out.append(sq[j])
                    j += 1
            seqs[i] = (out, c)
            for x, y in zip(out, out[1:]):
                pairs[(x, y)] += c
                where[(x, y)].add(i)
        for x, y in touched_pairs(seqs, where[(a, b)]):
            heapq.heappush(heap, (-pairs[(x, y)], (x, y)))
        where.pop((a, b), None)
        if log_every and (m + 1) % log_every == 0:
            snaps.append((m + 1, sum(len(sq) * c for sq, c in seqs)))
    return dict(merges=[(a, b) for a, b, _ in merges], counts=[c for _, _, c in merges], base=base,
                vocab=base + [a + b for a, b, _ in merges], snaps=snaps, byte_level=byte_level)


def bpe_encode_word(w, tok, cache=None):
    if cache is not None and w in cache:
        return cache[w]
    rank = tok.setdefault("_rank", {m: i for i, m in enumerate(tok["merges"])})
    sq = [bytes([b]) for b in w.encode("utf-8")] if tok["byte_level"] else list(w)
    if not tok["byte_level"]:
        known = set(tok["base"])
        sq = [s if s in known else "<unk>" for s in sq]
    while len(sq) > 1:
        best = min(((rank.get((x, y), 1e18), k) for k, (x, y) in enumerate(zip(sq, sq[1:]))), default=(1e18, -1))
        if best[0] >= 1e18:
            break
        k = best[1]
        sq = sq[:k] + [sq[k] + sq[k + 1]] + sq[k + 2:]
    if cache is not None:
        cache[w] = sq
    return sq


def bpe_encode(text, tok):
    cache = {}
    out = []
    for w in pretokenize(text):
        out += bpe_encode_word(w, tok, cache)
    return out


# ---------------------------------------------------------------- WordPiece-style scores
def merge_scores(words, k=10):
    seqs = [(list(w), c) for w, c in words.items()]
    unit, pair = collections.Counter(), collections.Counter()
    for sq, c in seqs:
        for s in sq:
            unit[s] += c
        for a, b in zip(sq, sq[1:]):
            pair[(a, b)] += c
    by_count = sorted(pair.items(), key=lambda kv: -kv[1])[:k]
    by_score = sorted(pair.items(), key=lambda kv: -(kv[1] / (unit[kv[0][0]] * unit[kv[0][1]])) if kv[1] >= 20 else 0)[:k]
    return dict(by_count=[(a + b, c) for (a, b), c in by_count],
                by_score=[(a + b, c, c / (unit[a] * unit[b])) for (a, b), c in by_score])


# ---------------------------------------------------------------- Unigram LM tokenizer
def viterbi(w, logp, maxlen):
    n = len(w)
    best = [0.0] + [-1e18] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - maxlen), i):
            s = w[j:i]
            if s in logp and best[j] + logp[s] > best[i]:
                best[i], back[i] = best[j] + logp[s], j
    out, i = [], n
    while i > 0:
        out.append(w[back[i]:i])
        i = back[i]
    return out[::-1], best[n]


def unigram_train(words, size, maxlen=8, init_size=6000, rounds=8):
    """Simplified SentencePiece unigram: seed with frequent substrings, hard-EM (Viterbi counts), prune 20% / round."""
    sub = collections.Counter()
    for w, c in words.items():
        for i in range(len(w)):
            for j in range(i + 1, min(len(w), i + maxlen) + 1):
                sub[w[i:j]] += c
    chars = {ch for w in words for ch in w}
    vocab = dict(sub.most_common(init_size))
    for ch in chars:
        vocab.setdefault(ch, sub[ch])
    tot = sum(vocab.values())
    logp = {s: np.log(c / tot) for s, c in vocab.items()}
    hist = []
    while True:
        cnt = collections.Counter()
        ll = 0.0
        for w, c in words.items():
            seg, s = viterbi(w, logp, maxlen)
            ll += c * s
            for t in seg:
                cnt[t] += c
        tot = sum(cnt.values())
        logp = {s: np.log(cnt[s] / tot) for s in cnt}
        for ch in chars:
            logp.setdefault(ch, np.log(0.5 / tot))
        hist.append((len(logp), ll))
        if len(logp) <= size or (hist and len(hist) > 1 and hist[-2][0] == len(logp)) or len(hist) > 40:
            break
        keep = max(size, int(len(logp) * 0.8))
        ranked = sorted(logp.items(), key=lambda kv: -cnt[kv[0]] * (len(kv[0]) > 1))
        logp = dict(ranked[:keep])
        for ch in chars:
            logp.setdefault(ch, np.log(0.5 / tot))
    return dict(logp=logp, maxlen=maxlen, hist=hist)


def unigram_encode(text, tok):
    out = []
    for w in pretokenize(text):
        seg, _ = viterbi(w, tok["logp"], tok["maxlen"])
        out += seg
    return out


# ---------------------------------------------------------------- experiments
def _words(text):
    return collections.Counter(pretokenize(text))


def exp_units():
    """Characters vs words vs BPE on Shakespeare: vocabulary size, sequence length, unseen-word rate."""
    en = english()
    n = int(len(en) * 0.9)
    tr, te = en[:n], en[n:]
    wtr, wte = pretokenize(tr), pretokenize(te)
    vocab_w = set(wtr)
    unseen = sum(w not in vocab_w for w in wte) / len(wte)
    out = dict(chars=len(en), char_vocab=len(set(en)), words=len(pretokenize(en)), word_vocab=len(set(pretokenize(en))),
               unseen_words=unseen, unseen_examples=[w for w in wte if w not in vocab_w][:12])
    tok = bpe_train(_words(tr), 4000, log_every=250)
    out["bpe_curve"] = [(len(tok["base"]) + m, n_tok) for m, n_tok in tok["snaps"]]
    out["bpe_first"] = tok["merges"][:16]
    out["bpe_counts"] = tok["counts"][:16]
    for k in (1000, 4000):
        t2 = dict(tok, merges=tok["merges"][:k - len(tok["base"])])
        t2.pop("_rank", None)
        enc = bpe_encode(te, t2)
        out[f"bpe{k}_test_tokens"] = len(enc)
        out[f"bpe{k}_unk"] = sum(t == "<unk>" for t in enc)
    out["test_chars"] = len(te)
    out["examples"] = {w: bpe_encode_word(w, dict(tok)) for w in (" the", " king", " unhappiness", " strawberry", " Romeo", " thee", " understanding")}
    return out


def exp_toy_bpe():
    corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    words = collections.Counter({w + "_": c for w, c in corpus.items()})
    steps = []
    seqs = {w: list(w) for w in words}
    for _ in range(6):
        pc = collections.Counter()
        for w, sq in seqs.items():
            for a, b in zip(sq, sq[1:]):
                pc[(a, b)] += words[w]
        (a, b), c = max(pc.items(), key=lambda kv: (kv[1], kv[0]))
        for w, sq in seqs.items():
            out, j = [], 0
            while j < len(sq):
                if j < len(sq) - 1 and sq[j] == a and sq[j + 1] == b:
                    out.append(a + b)
                    j += 2
                else:
                    out.append(sq[j])
                    j += 1
            seqs[w] = out
        steps.append(dict(pair=(a, b), count=c, seqs={w: list(s) for w, s in seqs.items()}))
    return dict(corpus=corpus, steps=steps)


def exp_wp_unigram():
    en = english()
    W = _words(en)
    ms = merge_scores(W)
    uni = unigram_train(collections.Counter(dict(W.most_common(3000))), 1000)
    bpe = bpe_train(W, 1000 - 65)
    words = [" unhappiness", " strawberry", " understanding", " kingdom", " nevertheless"]
    return dict(scores=ms, uni_hist=uni["hist"], uni_size=len(uni["logp"]),
                seg_uni={w: viterbi(w, uni["logp"], uni["maxlen"])[0] for w in words},
                seg_bpe={w: bpe_encode_word(w, dict(bpe)) for w in words})


def exp_bytes():
    en, zh = english(), chinese()
    W = _words(en + zh)
    char = bpe_train(W, 2000)
    byte = bpe_train(W, 2000, byte_level=True)
    samples = {"en": " To be, or not to be, that is the question.", "zh": "床前明月光，疑是地上霜。",
               "emoji": " I love pizza 🍕!", "rare": " Ωμέγα café naïve"}
    res = dict(char_vocab_base=len(char["base"]), byte_vocab_base=len(byte["base"]), zh_chars=len(set(zh)))
    for k, s in samples.items():
        ce, be = bpe_encode(s, char), bpe_encode(s, byte)
        res[k] = dict(text=s, char=ce, byte=[b.decode("utf-8", "replace") if len(b) else "" for b in be], byte_raw=be,
                      n_char=len(ce), n_byte=len(be), unk=sum(t == "<unk>" for t in ce), n_utf8=len(s.encode("utf-8")))
    for name, text in (("en", en[-20000:]), ("zh", zh[-20000:])):
        res[f"{name}_tok_per_char"] = len(bpe_encode(text, byte)) / len(text)
        res[f"{name}_bytes_per_char"] = len(text.encode("utf-8")) / len(text)
    return res


PARALLEL = {
    "English": "The weather is nice today, so let's go for a walk in the park.",
    "中文": "今天天气很好，我们去公园散步吧。",
    "Español": "Hoy hace buen tiempo, así que vamos a dar un paseo por el parque.",
    "Русский": "Сегодня хорошая погода, давай погуляем в парке.",
    "日本語": "今日は天気がいいので、公園を散歩しましょう。",
    "العربية": "الطقس جميل اليوم، فلنذهب للتنزه في الحديقة.",
    "हिन्दी": "आज मौसम अच्छा है, चलो पार्क में टहलने चलते हैं।",
}


def exp_multi():
    en, zh = english(), chinese()
    byte = bpe_train(_words(en + zh), 2000, byte_level=True)
    out = {}
    for k, s in PARALLEL.items():
        out[k] = dict(chars=len(s), bytes=len(s.encode("utf-8")), tokens=len(bpe_encode(s, byte)))
    return out


def exp_numbers():
    """BPE trained on text with numbers written as plain digit strings: how are numbers split?"""
    rng = np.random.default_rng(0)
    nums = [str(int(x)) for x in rng.integers(0, 100000, 20000)]
    PRE_NUM = re.compile(r" ?[0-9]+")
    W = collections.Counter(" " + n for n in nums)
    tok = bpe_train(W, 600)
    test = [" 12345", " 12346", " 99999", " 31415", " 2025", " 2026", " 777", " 1000"]
    return dict(seg={t: bpe_encode_word(t, dict(tok)) for t in test}, vocab=len(tok["vocab"]),
                letters={w: bpe_encode_word(w, dict(bpe_train(_words(english()), 2000))) for w in (" strawberry", " Mississippi", " bookkeeper")})


# ---------------------------------------------------------------- language models on different tokenisations
def train_lm(ids_train, ids_val, V, d=64, steps=2000, batch_size=32, lr=3e-3, seed=0):
    """Small 2-layer Transformer (sclx architecture, context 64 tokens). Returns mean validation loss per token."""
    import jax
    import jax.numpy as jnp
    import optax
    import sclx
    ctx = sclx.CTX
    p = sclx.init(jax.random.PRNGKey(seed), d, 2, V)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.warmup_cosine_decay_schedule(0, lr, 100, steps, lr * 0.1)))
    st = opt.init(p)

    def loss_fn(p, x):
        lg = sclx.forward(p, x[:, :-1])
        return -jnp.take_along_axis(jax.nn.log_softmax(lg), x[:, 1:, None], -1).mean()

    @jax.jit
    def step(p, st, x):
        l, g = jax.value_and_grad(loss_fn)(p, x)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    lossj = jax.jit(loss_fn)
    rng = np.random.default_rng(seed)
    pick = lambda ids, r, b: np.stack([ids[j:j + ctx + 1] for j in r.integers(0, len(ids) - ctx - 1, b)])
    for _ in range(steps):
        p, st, _ = step(p, st, jnp.array(pick(ids_train, rng, batch_size)))
    er = np.random.default_rng(999)
    val = float(np.mean([float(lossj(p, jnp.array(pick(ids_val, er, 64)))) for _ in range(16)]))
    return val, p


def exp_vocab():
    en = english()
    n = int(len(en) * 0.9)
    tr, te = en[:n], en[n:]
    tok = bpe_train(_words(tr), 4000 - 65)
    out = {}
    for V in (65, 256, 1000, 4000):
        if V == 65:
            chars = sorted(set(en))
            enc = lambda s: [chars.index(c) for c in s]
        else:
            t2 = dict(tok, merges=tok["merges"][:V - len(tok["base"])])
            t2.pop("_rank", None)
            vocab = {v: i for i, v in enumerate(t2["base"] + ["<unk>"] + [a + b for a, b in t2["merges"]])}
            enc = lambda s, t2=t2, vocab=vocab: [vocab[x] for x in bpe_encode(s, t2)]
        a, b = np.array(enc(tr), np.int32), np.array(enc(te), np.int32)
        Vt = int(max(a.max(), b.max())) + 1
        loss, _ = train_lm(a, b, Vt)
        out[V] = dict(loss_tok=loss, tok_per_char=len(b) / len(te), bits_per_char=loss * len(b) / len(te) / np.log(2),
                      emb_params=2 * Vt * 64, chars_in_context=64 * len(te) / len(b))
    return out
