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
