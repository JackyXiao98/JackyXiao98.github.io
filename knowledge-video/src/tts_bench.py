import sys, os, numpy as np, sherpa_onnx, soundfile as sf
M = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
TEXT = ("扩散模型的核心思想只有两步。第一步叫前向过程，把一张真实图片一点一点加上噪声，直到它变成纯粹的随机噪声。"
        "第二步叫反向过程，训练一个神经网络，学会把噪声一步一步去掉。学会以后，我们从纯噪声出发，反复去噪，就能得到一张全新的图片。")

def melo():
    m = f"{M}/vits-melo-tts-zh_en"
    return sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(
        model=f"{m}/model.onnx", lexicon=f"{m}/lexicon.txt", tokens=f"{m}/tokens.txt", dict_dir=f"{m}/dict"), num_threads=8),
        rule_fsts=f"{m}/date.fst,{m}/phone.fst,{m}/number.fst,{m}/new_heteronym.fst")
def matcha():
    m = f"{M}/matcha-icefall-zh-baker"
    return sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
        acoustic_model=f"{m}/model-steps-3.onnx", vocoder=f"{M}/vocos-22khz-univ.onnx", lexicon=f"{m}/lexicon.txt",
        tokens=f"{m}/tokens.txt", dict_dir=f"{m}/dict"), num_threads=8),
        rule_fsts=f"{m}/phone.fst,{m}/date.fst,{m}/number.fst")
def kokoro():
    m = f"{M}/kokoro-multi-lang-v1_1"
    return sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
        model=f"{m}/model.onnx", voices=f"{m}/voices.bin", tokens=f"{m}/tokens.txt", data_dir=f"{m}/espeak-ng-data",
        dict_dir=f"{m}/dict", lexicon=f"{m}/lexicon-us-en.txt,{m}/lexicon-zh.txt"), num_threads=8),
        rule_fsts=f"{m}/date-zh.fst,{m}/phone-zh.fst,{m}/number-zh.fst")

def pauses(x, sr):
    """Count silent gaps > 120 ms between speech (frame RMS below threshold)."""
    hop = int(sr * 0.01); fr = np.lib.stride_tricks.sliding_window_view(x, hop)[::hop]
    rms = np.sqrt((fr ** 2).mean(1)); th = 0.06 * rms.max()
    sil = rms < th; gaps = []; run = 0; started = False
    for s in sil:
        if not s: 
            if started and run >= 12: gaps.append(run * 10)
            run = 0; started = True
        else: run += 1
    return gaps

jobs = [("melo", melo, [0])] + [("matcha", matcha, [0])] + [("kokoro", kokoro, [int(v) for v in sys.argv[3].split(",")])]
for name, cfgf, sids in jobs:
    tts = sherpa_onnx.OfflineTts(cfgf())
    for sid in sids:
        a = tts.generate(TEXT, sid=sid, speed=1.0)
        x = np.asarray(a.samples, np.float32); f = f"{OUT}/{name}_{sid}.wav"; sf.write(f, x, a.sample_rate)
        g = pauses(x, a.sample_rate)
        print(f"{name}_{sid}: {len(x)/a.sample_rate:.1f}s sr={a.sample_rate} rate={len(TEXT)/(len(x)/a.sample_rate):.1f}ch/s pauses={len(g)} {g}", flush=True)
