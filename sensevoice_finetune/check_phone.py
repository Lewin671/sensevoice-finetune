"""Does the computer hear what the phone heard, and does loudness matter?

    sensevoice-finetune check-phone <work dir> <stock model dir>

Transcribes the recordings of utterances.jsonl with the stock model as sherpa-onnx runs it and
prints

  - for how many the text is the one the phone logged (utterances that continue an earlier one
    are left out: their logged text covers several files), as saved and with every non-zero
    sample moved one step away from zero, which undoes how Local Voice IME wrote the files
    before 0.9.0;
  - the error rate against the text the user left in the field (sessions of one utterance that
    were not undone; a provisional reference, good enough to compare conditions) as recorded,
    amplified by 10, 20 and 30 dB, and normalized to three peak levels.
"""
import os, sys
import numpy as np
from . import sv


def main():
    work, stock = sys.argv[1], sys.argv[2]
    rows = sv.read_jsonl(os.path.join(work, "utterances.jsonl"))
    recognizer = sv.sherpa_recognizer(os.path.join(stock, "model.int8.onnx"), os.path.join(stock, "tokens.txt"))
    strip = lambda s: s.rstrip("。．.，,、；;：:？?！!… ")

    alone = [r for r in rows if not r["continues"]]
    waves = [sv.read_wav(r["audio"]) for r in alone]
    step = 1 / 32768
    for name, xs in (("as saved", waves), ("one step away from zero", [x + np.sign(x) * step for x in waves])):
        hyps = sv.sherpa_transcribe(recognizer, [x.astype(np.float32) for x in xs])
        same = sum(strip(sv.app_normalize(h)) == strip(r["phone"]) for r, h in zip(alone, hyps))
        print(f"{name:26s} same text as on the phone: {same} of {len(alone)}")

    judged = [r for r in rows if r["of"] == 1 and r["field"] is not None and r["status"] != "undone"]
    waves = [sv.read_wav(r["audio"]) for r in judged]
    conditions = [("as recorded", waves)]
    conditions += [(f"+{db} dB", [np.clip(x * 10 ** (db / 20), -1, 1) for x in waves]) for db in (10, 20, 30)]
    conditions += [(f"peak at {p}", [x * (p / max(np.abs(x).max(), 1e-4)) for x in waves]) for p in (0.1, 0.3, 0.9)]
    print(f"error rate against the field, {len(judged)} utterances:")
    for name, xs in conditions:
        hyps = sv.sherpa_transcribe(recognizer, [x.astype(np.float32) for x in xs])
        counts = [sv.error_counts(r["field"], h) for r, h in zip(judged, hyps)]
        print(f"  {name:14s} {100 * sum(c[0] for c in counts) / max(1, sum(c[1] for c in counts)):5.2f} %")


if __name__ == "__main__":
    main()
