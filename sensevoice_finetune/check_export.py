"""Does a model exported by export_onnx.py behave like the one it should be? Run on the
original checkpoint, exported with `export-onnx <SenseVoiceSmall dir> - <dir>`, against the
files sherpa-onnx publishes: every transcript must be the same.

    sensevoice-finetune check-export <reference model dir> <exported model dir> <manifest.jsonl> [...]

Both directories hold `model.int8.onnx` and `tokens.txt`; a manifest has {"id", "audio"} lines
(labels.jsonl, replay.jsonl and the manifests of public.py all do). Exit code 1 on a difference.
"""
import filecmp, os, sys
from . import sv


def main():
    reference, exported, manifests = sys.argv[1], sys.argv[2], sys.argv[3:]
    same_tokens = filecmp.cmp(os.path.join(reference, "tokens.txt"), os.path.join(exported, "tokens.txt"), shallow=False)
    print("tokens.txt:", "identical" if same_tokens else "DIFFERENT")
    recognizers = [sv.sherpa_recognizer(os.path.join(d, "model.int8.onnx"), os.path.join(d, "tokens.txt"))
                   for d in (reference, exported)]
    total = different = 0
    for manifest in manifests:
        rows = sv.read_jsonl(manifest)
        waves = [sv.read_wav(r["audio"]) for r in rows]
        a, b = (sv.sherpa_transcribe(r, waves) for r in recognizers)
        for r, x, y in zip(rows, a, b):
            if x != y:
                print(f"{r['id']}\n  reference {x!r}\n  exported  {y!r}")
        total += len(rows)
        different += sum(x != y for x, y in zip(a, b))
    print(f"{total - different} of {total} transcripts identical")
    sys.exit(0 if same_tokens and not different else 1)


if __name__ == "__main__":
    main()
