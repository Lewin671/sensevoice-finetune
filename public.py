"""Unpack the public test sets of Local Voice IME's docs/MODELS.md into audio files and manifests, so that a
fine-tuned model can be checked for what it forgot.

    public.py <benchmark dir> <out dir>

<benchmark dir> is the working directory of that repository's scripts/bench (data/<set>.parquet). For every set
this writes two manifests of {"id", "audio", "text"[, "group"]} lines ("group": the speaker or
recording, where the corpus names one):

    <out dir>/test_<set>.jsonl   the 300 utterances every table in that docs/MODELS.md is computed on
                                 (same order and filters as its scripts/bench/bench.py)
    <out dir>/dev_<set>.jsonl    the 100 that follow them, for choosing settings without
                                 touching the ones that are reported

The utterances are those of that benchmark; the numbers are not comparable with its tables. The
audio is resampled here and written as 16 kHz 16-bit WAV, where bench.py hands the recognizer
the decoded samples at their own rate, and score.py counts tokens slightly differently
(sv.score_tokens). Compare models within this pipeline: transcribe the stock model with
transcribe.py too.
"""
import io, json, os, random, sys
from math import gcd
import numpy as np, pyarrow.parquet as pq, soundfile as sf
from scipy.signal import resample_poly

SETS = ["aishell1_test_0", "wenet_test_net_0", "wenet_test_meeting", "ascend_test",
        "librispeech_test_clean", "kespeech_test_0", "cv_zh_test"]
TEST, DEV = 300, 100


def speaker(r):
    """Whoever or whatever recording an utterance belongs to, as far as the corpus says: errors
    of one speaker go together, so score.py resamples these and not single utterances."""
    for key in ("speaker_id", "client_id", "original_speaker_id"):
        if r.get(key) is not None:
            return str(r[key])
    if r.get("sid"):  # WenetSpeech: TEST_NET_Y0000000000_-KTKHdZ2fb8_S00012, a segment of a recording
        return r["sid"].rsplit("_S", 1)[0]
    if r.get("ID") and "_" in r["ID"]:  # KeSpeech: <speaker>_<utterance>
        return r["ID"].split("_", 1)[0]
    return None


def rows(path, name, wanted):
    """Rows in the order bench.py (and bench_mlx.py for KeSpeech / Common Voice) visits them."""
    f = pq.ParquetFile(path)
    out = []
    for i in range(f.num_row_groups):
        out += f.read_row_group(i).to_pylist()
        # bench.py stops reading at six times the 300 it wants; the order after shuffling depends on it
        if len(out) > TEST * 6 and not name.startswith("ascend"):
            break
    random.Random(0).shuffle(out)
    for r in out:
        if name.startswith("ascend") and r["language"] != "mixed":
            continue
        audio = r.get("context") or r.get("audio")
        text = r.get("answer") or r.get("text") or r.get("transcription") or r.get("Text") or r.get("sentence")
        try:
            x, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        except Exception:
            continue
        if x.ndim > 1:
            x = x.mean(1)
        if len(x) / rate < 1.0:
            continue
        yield x, rate, text, speaker(r)
        wanted -= 1
        if wanted == 0:
            return


def main():
    bench, out = sys.argv[1], sys.argv[2]
    for name in SETS:
        os.makedirs(os.path.join(out, name), exist_ok=True)
        manifests = {"test": [], "dev": []}
        for i, (x, rate, text, who) in enumerate(rows(os.path.join(bench, "data", f"{name}.parquet"), name, TEST + DEV)):
            if rate != 16000:
                g = gcd(rate, 16000)
                x = resample_poly(x, 16000 // g, rate // g).astype(np.float32)
            path = os.path.abspath(os.path.join(out, name, f"{i:03d}.wav"))
            sf.write(path, x, 16000, subtype="PCM_16")
            item = {"id": f"{name}-{i:03d}", "audio": path, "text": text}
            if who is not None:
                item["group"] = f"{name}-{who}"
            manifests["test" if i < TEST else "dev"].append(item)
        for split, items in manifests.items():
            with open(os.path.join(out, f"{split}_{name}.jsonl"), "w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(name, {k: len(v) for k, v in manifests.items()})


if __name__ == "__main__":
    main()
