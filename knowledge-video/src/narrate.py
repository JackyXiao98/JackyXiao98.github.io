"""Offline Chinese narration for the knowledge videos (sherpa-onnx).

Each scene is synthesized as one paragraph so intonation flows across sentences
(sentence-by-sentence synthesis sounds choppy). Sentence timings for subtitles are
recovered by snapping the expected sentence boundaries to pauses in the audio.

Voices: "kokoro:<sid>" (Kokoro v1.1, 3-57 female zf_*, 58-102 male zm_*), "matcha" (Baker, female),
"melo" (MeloTTS, female).
"""

import re

import numpy as np
import sherpa_onnx


def make_tts(models, voice):
    kind, _, sid = voice.partition(":")
    if kind == "kokoro":
        m = f"{models}/kokoro-multi-lang-v1_1"
        model = sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=f"{m}/model.onnx", voices=f"{m}/voices.bin", tokens=f"{m}/tokens.txt",
                data_dir=f"{m}/espeak-ng-data", dict_dir=f"{m}/dict",
                lexicon=f"{m}/lexicon-us-en.txt,{m}/lexicon-zh.txt"),
            num_threads=8)
        fsts = f"{m}/date-zh.fst,{m}/phone-zh.fst,{m}/number-zh.fst"
    elif kind == "matcha":
        m = f"{models}/matcha-icefall-zh-baker"
        model = sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=f"{m}/model-steps-3.onnx", vocoder=f"{models}/vocos-22khz-univ.onnx",
                lexicon=f"{m}/lexicon.txt", tokens=f"{m}/tokens.txt", dict_dir=f"{m}/dict"),
            num_threads=8)
        fsts = f"{m}/phone.fst,{m}/date.fst,{m}/number.fst"
    elif kind == "melo":
        m = f"{models}/vits-melo-tts-zh_en"
        model = sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=f"{m}/model.onnx", lexicon=f"{m}/lexicon.txt", tokens=f"{m}/tokens.txt", dict_dir=f"{m}/dict"),
            num_threads=8)
        fsts = f"{m}/date.fst,{m}/phone.fst,{m}/number.fst,{m}/new_heteronym.fst"
    else:
        raise ValueError(voice)
    tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model, rule_fsts=fsts, max_num_sentences=20))
    return tts, int(sid or 0)


def _gaps(x, sr, min_ms=90):
    """Silent stretches (start_s, end_s) longer than min_ms."""
    hop = int(sr * 0.01)
    n = len(x) // hop
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1))
    sil = rms < 0.05 * rms.max()
    gaps, i = [], 0
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]:
                j += 1
            if (j - i) * 10 >= min_ms and i > 0 and j < n:
                gaps.append((i * hop / sr, j * hop / sr))
            i = j
        else:
            i += 1
    return gaps


def _edit(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def _hanzi(s):
    return re.sub(r"[^\u4e00-\u9fff]", "", s)


class Narrator:
    def __init__(self, models, voice="matcha", speed=1.0, asr=None):
        self.tts, self.sid = make_tts(models, voice)
        # Optional ASR model dir: used to verify sentence boundaries (pick the pause where the
        # transcript of each sentence's audio best matches its text).
        self.asr = None
        if asr:
            self.asr = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=f"{asr}/model.int8.onnx", tokens=f"{asr}/tokens.txt", num_threads=8)
        self.speed = speed
        self.sr = self.tts.sample_rate

    def scene(self, sentences):
        """Synthesize sentences as one paragraph. Returns (audio, [(start, end), ...])."""
        text = "".join(s.replace("：", "，") for s in sentences)
        a = self.tts.generate(text, sid=self.sid, speed=self.speed)
        x = np.asarray(a.samples, np.float32)
        dur = len(x) / self.sr
        gaps = _gaps(x, self.sr, min_ms=40)
        # Estimate sentence boundaries from each sentence's length when synthesized alone,
        # scaled to the paragraph, then snap each estimate to the nearest pause.
        lens = np.array([len(self.tts.generate(s.replace("：", "，"), sid=self.sid, speed=self.speed).samples)
                         for s in sentences], float)
        est = np.cumsum(lens)[:-1] / lens.sum() * dur
        bounds = []
        for i, e in enumerate(est):
            lo = bounds[-1][1] if bounds else 0.0
            near = [(g0, g1) for g0, g1 in gaps if g0 >= lo and abs((g0 + g1) / 2 - e) < 0.8]
            if not near:
                bounds.append((max(e, lo), max(e, lo)))
            elif self.asr is None:
                bounds.append(min(near, key=lambda g: abs((g[0] + g[1]) / 2 - e)))
            else:
                bounds.append(min(near, key=lambda g: self._mismatch(x, lo, g[0], sentences[i])
                                  + 0.05 * abs((g[0] + g[1]) / 2 - e)))
        starts = [0.0] + [b[1] for b in bounds]
        ends = [b[0] for b in bounds] + [dur]
        return x, list(zip(starts, ends))

    def _mismatch(self, x, t0, t1, sentence):
        """Normalized edit distance between the transcript of x[t0:t1] and the sentence."""
        seg = x[int(t0 * self.sr): int(t1 * self.sr)]
        seg = np.interp(np.arange(0, len(seg), self.sr / 16000), np.arange(len(seg)), seg).astype(np.float32)
        st = self.asr.create_stream()
        st.accept_waveform(16000, seg)
        self.asr.decode_stream(st)
        ref = _hanzi(sentence)
        return _edit(_hanzi(st.result.text), ref) / max(1, len(ref))
