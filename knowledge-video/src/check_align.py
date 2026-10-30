import sys, json, re, numpy as np, soundfile as sf, sherpa_onnx
m, d = sys.argv[1], sys.argv[2]
rec = sherpa_onnx.OfflineRecognizer.from_paraformer(paraformer=f"{m}/model.int8.onnx", tokens=f"{m}/tokens.txt", num_threads=8)
x, sr = sf.read(f"{d}/narration.wav", dtype="float32")
idx = np.arange(0, len(x), sr / 16000); x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
for s in json.load(open(f"{d}/timeline.json"))["subtitles"]:
    seg = x[int(s["start"] * 16000): int(s["end"] * 16000)]
    st = rec.create_stream(); st.accept_waveform(16000, seg); rec.decode_stream(st)
    ref = re.sub(r"[^一-鿿]", "", s["text"]); hyp = st.result.text.replace(" ", "")
    ok = "OK " if ref[:2] in hyp[:4] and ref[-2:] in hyp[-4:] else "?? "
    print(ok, f'{s["start"]:6.1f}', ref[:14], "|", hyp[:14], "...", hyp[-6:])
