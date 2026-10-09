"""Transcribe a list of audio files as the app would hear them.

    transcribe.py <manifest.jsonl> <out.json> --onnx <dir with model.int8.onnx and tokens.txt>
    transcribe.py <manifest.jsonl> <out.json> --torch <SenseVoiceSmall dir> [--weights model.pt] [--mix W]

The manifest has one {"id", "audio"} per line (labels.jsonl and what public.py and replay.py
write qualify); the output maps id -> raw transcript, for score.py.

--onnx runs the exported model in sherpa-onnx, configured as in VoiceEngine.kt: this is what
the phone runs, and what reported numbers should come from. --torch runs PyTorch weights (the
original ones without --weights) behind the same front end, which is quicker while comparing
training runs; --mix pulls the weights back towards the original first (mix.py).
"""
import argparse, json, os
import sv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest"), p.add_argument("out")
    p.add_argument("--onnx"), p.add_argument("--torch"), p.add_argument("--weights")
    p.add_argument("--mix", type=float)
    p.add_argument("--device", default="mps")
    p.add_argument("--fold", type=int, help="only the rows of this fold of labels.jsonl")
    args = p.parse_args()
    rows = sv.read_jsonl(args.manifest)
    if args.fold is not None:
        rows = [r for r in rows if r["fold"] == args.fold]
    waves = [sv.read_audio(r["audio"]) for r in rows]
    if args.onnx:
        recognizer = sv.sherpa_recognizer(os.path.join(args.onnx, "model.int8.onnx"),
                                          os.path.join(args.onnx, "tokens.txt"))
        texts = sv.sherpa_transcribe(recognizer, waves)
    else:
        m = sv.SenseVoice(args.torch, device=args.device, weights=args.weights, mix=args.mix)
        texts = m.transcribe(waves)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({r["id"]: t for r, t in zip(rows, texts)}, f, ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
