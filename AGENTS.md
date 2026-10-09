# Working in this repository

Scripts that fine-tune SenseVoice Small on one person's recordings from
[Local Voice IME](https://github.com/Lewin671/local-voice-ime). [README.md](README.md) is the
recipe, [docs/EVIDENCE.md](docs/EVIDENCE.md) the measurements behind it.

## Privacy comes first

The repository is public; the recordings, what was dictated and the labels are not.

- Never commit audio, transcripts, labels, work directories or anything derived from an export.
  `.gitignore` covers the names the recipe uses; check `git status` before every commit.
- Tests and documentation use invented sentences, never ones taken from real dictation.
- Published numbers may describe the data (counts, error rates), not quote it.

## Layout

- `sensevoice_finetune/`: one module per command of `sensevoice-finetune` (`__main__.py` lists
  them); `sv.py` holds what they share. A new command is a module with a `main()`, a docstring
  that starts with what it does and how it is called, and a line in `__main__.py`.
- `scripts/cross_validate.sh`: the cross-validation loop.
- `tests/`: logic only, no model and no audio; `python -m unittest` from the root, with `numpy`
  and `zhconv` installed.
- `docs/EVIDENCE.md`: every number in it was measured; say with which settings when adding one.

## Conventions

- Everything in the repository is in English.
- Training and evaluation go through the front end in `sv.py`, which is sherpa-onnx's and not
  FunASR's: the model must see in training what the phone will feed it.
- Reported error rates come from the exported, quantized model (`transcribe --onnx`).
- `sensevoice_finetune/export_onnx.py` is adapted from sherpa-onnx (Apache-2.0); keep `NOTICE`
  accurate when it changes.
