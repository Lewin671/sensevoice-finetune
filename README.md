# Fine-tuning SenseVoice Small on your own speech

SenseVoice Small hears most people well and nobody in particular.
[Local Voice IME](https://github.com/Lewin671/local-voice-ime), an Android keyboard that
dictates on the device, can keep what one person actually says to it
([format of the export](https://github.com/Lewin671/local-voice-ime/blob/main/docs/TRAINING_DATA.md));
the scripts here turn such an export into a model that has heard this person before, in the
format [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) loads (`model.int8.onnx` +
`tokens.txt`). Everything runs on a computer; nothing is uploaded anywhere.

This page is the recipe; [docs/EVIDENCE.md](docs/EVIDENCE.md) is the evidence for each choice
in it. The numbers come from
one export: 359 utterances, 21.6 minutes, five days of ordinary use by one speaker (Mandarin
with English terms, some English). They are that speaker's numbers, not a benchmark.

| | Stock model | Fine-tuned | Difference (95 % interval) |
|---|---|---|---|
| The speaker's own utterances, never trained on (5-fold, 326 utterances) | 2.94 % | 1.73 % | −1.22 (−1.72 to −0.74) |
| … of sessions the speaker had to correct (61) | 7.43 % | 4.99 % | −2.44 (−4.32 to −0.77) |
| … with English words in them (73) | 6.10 % | 3.53 % | −2.57 (−4.62 to −0.94) |
| The fifth day, trained on the first four only (81) | 3.77 % | 3.57 % | −0.20 (−1.43 to +0.95) |
| Seven public test sets, other speakers (mean) | 9.46 % | 9.65 % | two sets worse, one better: see [evidence](docs/EVIDENCE.md#the-final-model-on-other-speakers) |

Read the first and the fourth row together. A model that has heard some of what was said on a
day gets a good part of the errors of that day out of the way; a model that has only heard
earlier days gains little on a day with new subjects. The reading offered here is that
fine-tuning on this little data teaches the words and subjects it has seen, more than the
voice; one speaker and one new day cannot prove it. See [Results](docs/EVIDENCE.md#results).

Error rate: wrong, missing or extra CJK characters and Latin words, punctuation and case
ignored. Both columns are the quantized model running in sherpa-onnx, configured as on the phone.

## The model from this export

Two models are published with the
[releases](https://github.com/Lewin671/sensevoice-finetune/releases) of this repository, each
as `model.int8.onnx` and `tokens.txt`, a drop-in replacement for the files of
`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17`:

- `model-20261009` is the one the numbers on this page describe, trained with the scripts as
  they are now.
- `model-20261008` was trained with the first version of the scripts, in which the learning
  rate fell to its floor a third of the way through training and rehearsal was thinner than
  intended (see [History](docs/EVIDENCE.md#history)). In the same tests it is at 1.86 % where the newer one is
  at 1.73 % (difference −0.13, −0.41 to +0.15): the two cannot be told apart.

Both are SenseVoiceSmall by
FunAudioLLM (Alibaba Group), fine-tuned, and stays under the
[FunASR Model Open Source License Agreement 1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE).
They are one person's models: better at that person's words, slightly worse at accented Mandarin
than the original
([measured](docs/EVIDENCE.md#the-final-model-on-other-speakers)), and of use to others mainly as a worked example. The recordings and
their transcripts it was trained on are not published.

Local Voice IME offers `model-20261009` under *Settings → Voice input*, as an
example (0.9.0 offered `model-20261008`; 0.9.1 and later the newer one). From 0.10.0 the app
finds a newer model by itself when asked to: it reads the newest release of this repository,
which must therefore always be a model, tagged `model-<yyyymmdd>` and holding `model.int8.onnx`
and `tokens.txt`. The app accepts catalogued models only, by checksum; a model of your own cannot be
imported yet.

## Recipe

Needs Python 3.11+ and about 4 GB of free memory; an Apple-silicon Mac (the commands default to
its GPU, `--device mps`) or any machine with `--device cuda` / `cpu`. Training takes half an hour
on an M1 Pro and cross-validation five times that, unattended; the review in step 3 takes an
hour or two of attention.

```sh
git clone https://github.com/Lewin671/sensevoice-finetune && cd sensevoice-finetune
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -c "from modelscope import snapshot_download; snapshot_download('iic/SenseVoiceSmall', local_dir='SenseVoiceSmall')"
# the stock model as sherpa-onnx (and the app) runs it, as the reference
mkdir stock && for f in model.int8.onnx tokens.txt; do
    curl -L -o stock/$f https://www.modelscope.cn/models/pengzhendong/sherpa-onnx-sense-voice-zh-en-ja-ko-yue/resolve/master/$f
done
```

Keep the export, the work directory and everything derived from them out of version control
(`.gitignore` covers the names used here): they are your voice and your words.

`sensevoice-finetune` lists the commands, `sensevoice-finetune <command> --help` explains one.

1. **Unpack.** `sensevoice-finetune prepare local-voice-ime-recordings-YYYYMMDD.zip work` writes
   one row per utterance with everything the log says about it.
2. **Listen again.** `sensevoice-finetune hypotheses work stock` transcribes every file on the
   computer. Optionally add opinions of stronger recognizers as `work/hyp_<name>.json`, e.g.
   with `sensevoice-finetune hypotheses-mlx` (`pip install -e ".[mlx]"`); each one makes the next
   step shorter.
3. **Label.** `sensevoice-finetune label work` accepts an utterance when everything that heard
   it wrote the same words (the stock model on the computer, the large model on the phone, any
   extra recognizer, and the field as the user left it), and lists the rest in
   `work/review.txt`. Go through that list and write `reviewed.tsv` (`id`, `A`/`B`/`X`, text;
   tab-separated), then run `sensevoice-finetune label work --manual reviewed.tsv` until
   `review.txt` is empty. See [Labels](docs/EVIDENCE.md#labels) for how to decide; this hour is
   where a small data set is won or lost.
4. **Rehearsal material.** `sensevoice-finetune replay replay stock <corpus.parquet>...` takes
   speech of other people and labels it with the stock model itself
   ([why](docs/EVIDENCE.md#what-it-forgets)). Take about five utterances per one of your own
   (`<corpus.parquet>:600`); `train` says so when the pool is smaller than a pass needs.
5. **Measure before believing.**
   `scripts/cross_validate.sh work SenseVoiceSmall stock cv --replay replay/replay.jsonl` trains
   five models, each without a fifth of the data, exports each as the phone would get it, and
   scores the left-out fifths. It can be interrupted and started again; it refuses to continue
   in a directory that was begun with other labels, options or code.
6. **Train on everything and export.**
   ```sh
   sensevoice-finetune train work SenseVoiceSmall final --replay replay/replay.jsonl
   sensevoice-finetune export-onnx SenseVoiceSmall final/model.pt final/onnx
   ```
7. **Check what it forgot.** `sensevoice-finetune public <benchmark dir> public` unpacks the test
   sets of [Local Voice IME's MODELS.md](https://github.com/Lewin671/local-voice-ime/blob/main/docs/MODELS.md);
   then for each set
   ```sh
   sensevoice-finetune transcribe public/test_<set>.jsonl stock.json --onnx stock
   sensevoice-finetune transcribe public/test_<set>.jsonl tuned.json --onnx final/onnx
   sensevoice-finetune score --manifest public/test_<set>.jsonl --no-digits stock=stock.json tuned=tuned.json
   ```
   The intervals resample speakers where the corpus names them.
8. **On a phone.** In a checkout of Local Voice IME,
   `WAVS=<a few of your recordings> scripts/bench/device-bench.sh tuned final/onnx sensevoice`
   loads the exported model in the app's own runtime and reports speed and memory.

The model that comes out is a derivative of SenseVoice Small and stays under its
[license](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE). It is also a model of
one voice and one vocabulary: treat it like the recordings it was made from.

## Repository

| | |
|---|---|
| [`sensevoice_finetune/`](sensevoice_finetune) | The commands, one module each; `sv.py` is what they share: the front end of sherpa-onnx, the model, the text conventions. |
| [`scripts/cross_validate.sh`](scripts/cross_validate.sh) | Step 5: train, export and score one model per fold. |
| [`tests/`](tests) | The logic that decides what is trained on and how it is scored. |
| [`docs/EVIDENCE.md`](docs/EVIDENCE.md) | What was measured, and what the numbers do not show. |

`python -m unittest` runs the tests; they need `numpy` and `zhconv` only, no model. Three
commands back claims of the evidence: `check-export` (the export is the one sherpa-onnx
publishes), `check-phone` (the computer hears what the phone heard; loudness) and `terms` (how
often each model writes the words on a list of yours).

## License

The code is under the [Apache License 2.0](LICENSE). `sensevoice_finetune/export_onnx.py` is adapted from
`scripts/sense-voice/export-onnx.py` of [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
(Apache-2.0). SenseVoice Small and every model derived from it, including the released one,
are under the [FunASR Model Open Source License Agreement 1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE):
attribute FunAudioLLM / Alibaba Group and keep the model's name.
