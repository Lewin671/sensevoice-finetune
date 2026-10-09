"""Step 2, optional: a second opinion on every utterance from a large recognizer that only a
computer can run, here an mlx-audio speech-to-text checkpoint on Apple silicon.

    hypotheses_mlx.py <work dir> <model dir or hf repo> [name=qwen3]

Writes <work dir>/hyp_<name>.json like hypotheses.py. Needs `mlx-audio` and `soundfile`, in an
environment of its own if they clash with the training packages. Local Voice IME's docs/MODELS.md measured
mlx-community/Qwen3-ASR-1.7B-4bit (1.6 GB); it is about as accurate as the app's large model,
makes different mistakes, and spells English terms properly, which is what a reviewer needs.
"""
import json, os, sys
import mlx.core as mx, soundfile as sf
from mlx_audio.stt.utils import load_model


def main():
    work, repo = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else "qwen3"
    with open(os.path.join(work, "utterances.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    model = load_model(repo)
    out = {}
    for r in rows:
        samples, _ = sf.read(r["audio"], dtype="float32")
        out[r["id"]] = model.generate(mx.array(samples), language=None, max_tokens=256).text
    with open(os.path.join(work, f"hyp_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    print(f"{len(out)} utterances")


if __name__ == "__main__":
    main()
