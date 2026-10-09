"""Other people's speech for the fine-tuned model to rehearse, labelled by the model it starts from.

    sensevoice-finetune replay <out dir> <model dir> <corpus.parquet>[:N] [<corpus.parquet>[:N] ...]

A model trained on twenty minutes of one voice drifts away from everything else. Mixing in
utterances of other speakers, with the transcript the *unchanged* model gives them as the
target, holds it in place: wherever it is not being taught something new, it is asked to stay
as it is. No reference transcripts are used, so a corpus needs audio only. The model still hears
that audio: do not read a gain on speech like the rehearsal material as a general one.

Each parquet file has an audio column ("audio" or "context", with embedded bytes) as the Hugging
Face datasets of Local Voice IME's docs/MODELS.md do; N utterances (default 400) of 1-15 s are taken from each at
random. <model dir> holds the stock `model.int8.onnx` and `tokens.txt`. Writes 16 kHz WAV files
and <out dir>/replay.jsonl with {"id", "audio", "text", "seconds"} lines for train.py --replay.

Do not rehearse on the corpora the model is then tested on (public.py): use other files.
"""
import io, json, os, random, sys
from math import gcd
import numpy as np, pyarrow.parquet as pq, soundfile as sf
from scipy.signal import resample_poly
from . import sv


def corpus_name(path, taken):
    """Directory and id prefix of a corpus: the file name, with the directory it is in when two
    corpora have files of the same name (train-00000-of-00001.parquet)."""
    path = os.path.abspath(path)
    name = os.path.splitext(os.path.basename(path))[0]
    if name in taken:
        name = f"{os.path.basename(os.path.dirname(path))}-{name}"
    if name in taken:
        raise ValueError(f"two corpora would both be called {name}: rename one of the files")
    return name


def main():
    out, model_dir = sys.argv[1], sys.argv[2]
    recognizer = sv.sherpa_recognizer(os.path.join(model_dir, "model.int8.onnx"),
                                      os.path.join(model_dir, "tokens.txt"))
    rows, names = [], set()
    for spec in sys.argv[3:]:
        path, _, n = spec.partition(":")
        n = int(n or 400)
        name = corpus_name(path, names)
        names.add(name)
        os.makedirs(os.path.join(out, name), exist_ok=True)
        f = pq.ParquetFile(path)
        column = "audio" if "audio" in f.schema_arrow.names else "context"
        rng = random.Random(0)
        order = list(range(f.num_row_groups))
        rng.shuffle(order)
        kept = 0
        for g in order:
            cells = f.read_row_group(g, columns=[column]).column(column).to_pylist()
            rng.shuffle(cells)
            for cell in cells:
                try:
                    x, rate = sf.read(io.BytesIO(cell["bytes"]), dtype="float32")
                except Exception:
                    continue
                if x.ndim > 1:
                    x = x.mean(1)
                if not 1.0 <= len(x) / rate <= 15.0:
                    continue
                if rate != sv.SAMPLE_RATE:
                    k = gcd(rate, sv.SAMPLE_RATE)
                    x = resample_poly(x, sv.SAMPLE_RATE // k, rate // k).astype(np.float32)
                text = sv.sherpa_transcribe(recognizer, [x])[0]
                if not sv.score_tokens(text):
                    continue
                wav = os.path.abspath(os.path.join(out, name, f"{kept:04d}.wav"))
                sf.write(wav, x, sv.SAMPLE_RATE, subtype="PCM_16")
                rows.append({"id": f"{name}-{kept:04d}", "audio": wav, "text": text,
                             "seconds": round(len(x) / sv.SAMPLE_RATE, 2)})
                kept += 1
                if kept >= n:
                    break
            if kept >= n:
                break
        print(f"{name}: {kept} utterances")
    sv.write_jsonl(os.path.join(out, "replay.jsonl"), rows)


if __name__ == "__main__":
    main()
