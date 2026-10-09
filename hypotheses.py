"""Step 2: transcribe every utterance again on the desktop, as the app would.

    hypotheses.py <work dir> <model dir> [name=sensevoice]

<model dir> holds `model.int8.onnx` and `tokens.txt` (voice/models/sense-voice-small-int8).
Writes <work dir>/hyp_<name>.json: id -> raw transcript.

`label.py` compares these with whatever else heard the same audio. Transcripts of other, stronger
recognizers go next to this file in the same form (hyp_<anything>.json); the more independent
opinions, the fewer utterances a person has to review.
"""
import json, os, sys
import sv


def main():
    work, model_dir = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else "sensevoice"
    rows = sv.read_jsonl(os.path.join(work, "utterances.jsonl"))
    recognizer = sv.sherpa_recognizer(os.path.join(model_dir, "model.int8.onnx"),
                                      os.path.join(model_dir, "tokens.txt"))
    texts = sv.sherpa_transcribe(recognizer, [sv.read_wav(r["audio"]) for r in rows])
    with open(os.path.join(work, f"hyp_{name}.json"), "w", encoding="utf-8") as f:
        json.dump({r["id"]: t for r, t in zip(rows, texts)}, f, ensure_ascii=False, indent=0)
    # how well the desktop reproduces the phone, where the phone's text is about this file alone
    strip = lambda s: s.rstrip("。．.，,、；;：:？?！!… ")
    alone = [(r, t) for r, t in zip(rows, texts) if not r["continues"]]
    same = sum(strip(sv.app_normalize(t)) == strip(r["phone"]) for r, t in alone)
    print(f"{len(rows)} utterances; same text as on the phone: {same} of {len(alone)}")


if __name__ == "__main__":
    main()
