# Fine-tuning SenseVoice Small on your own speech

SenseVoice Small hears most people well and nobody in particular.
[Local Voice IME](https://github.com/Lewin671/local-voice-ime), an Android keyboard that
dictates on the device, can keep what one person actually says to it
([format of the export](https://github.com/Lewin671/local-voice-ime/blob/main/docs/TRAINING_DATA.md));
the scripts here turn such an export into a model that has heard this person before, in the
format [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) loads (`model.int8.onnx` +
`tokens.txt`). Everything runs on a computer; nothing is uploaded anywhere.

This page is the recipe, then the evidence for each choice in it. The numbers come from
one export: 359 utterances, 21.6 minutes, five days of ordinary use by one speaker (Mandarin
with English terms, some English). They are that speaker's numbers, not a benchmark.

| | Stock model | Fine-tuned | Difference (95 % interval) |
|---|---|---|---|
| The speaker's own utterances, never trained on (5-fold, 326 utterances) | 2.94 % | 1.86 % | −1.09 (−1.55 to −0.65) |
| … of sessions the speaker had to correct (61) | 7.43 % | 5.63 % | −1.80 (−3.37 to −0.38) |
| … with English words in them (73) | 6.10 % | 3.72 % | −2.38 (−4.04 to −1.02) |
| The fifth day, trained on the first four only (81) | 3.77 % | 3.57 % | −0.20 (−1.18 to +0.75) |
| Seven public test sets, other speakers (mean) | 9.53 % | 9.65 % | one set worse, one better: see [below](#the-final-model-on-other-speakers) |

Read the first and the fourth row together. A model that has heard some of what was said on a
day gets a third of the errors of that day out of the way; a model that has only heard earlier
days gains little on a day with new subjects. Fine-tuning on this little data teaches the
words and subjects it has seen, more than the voice. See [Results](#results).

Error rate: wrong, missing or extra CJK characters and Latin words, punctuation and case
ignored. Both columns are the quantized model running in sherpa-onnx, configured as on the phone.

## The model from this export

The model these numbers describe is published with the
[releases](https://github.com/Lewin671/sensevoice-finetune/releases) of this repository:
`model.int8.onnx` and `tokens.txt`, a drop-in replacement for the files of
`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17`. It is SenseVoiceSmall by
FunAudioLLM (Alibaba Group), fine-tuned, and stays under the
[FunASR Model Open Source License Agreement 1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE).
It is one person's model: better at that person's words, slightly worse at accented Mandarin
than the original (below), and of use to others mainly as a worked example. The recordings and
their transcripts it was trained on are not published.

Local Voice IME 0.9.0 and later offer this model under *Settings → Voice input*, as an
example. The app accepts catalogued models only, by checksum; a model of your own cannot be
imported yet.

## Recipe

Needs Python 3.11+ and about 4 GB of free memory; an Apple-silicon Mac (the scripts default to
its GPU, `--device mps`) or any machine with `--device cuda` / `cpu`. Training takes half an hour
on an M1 Pro and cross-validation five times that, unattended; the review in step 3 takes an
hour or two of attention.

```sh
python3 -m venv venv
./venv/bin/pip install torch torchaudio funasr modelscope onnx onnxruntime sherpa-onnx \
    kaldi-native-fbank sentencepiece numpy scipy soundfile pyarrow zhconv
./venv/bin/python -c "from modelscope import snapshot_download; snapshot_download('iic/SenseVoiceSmall', local_dir='SenseVoiceSmall')"
# the stock model as sherpa-onnx (and the app) runs it, as the reference
mkdir stock && for f in model.int8.onnx tokens.txt; do
    curl -L -o stock/$f https://www.modelscope.cn/models/pengzhendong/sherpa-onnx-sense-voice-zh-en-ja-ko-yue/resolve/master/$f
done
F=.  S=stock  W=work  P=./venv/bin/python
```

Keep the export, the work directory and everything derived from them out of version control
(`.gitignore` covers the names used here): they are your voice and your words.

1. **Unpack.** `$P $F/prepare.py local-voice-ime-recordings-YYYYMMDD.zip $W` writes one row per
   utterance with everything the log says about it.
2. **Listen again.** `$P $F/hypotheses.py $W $S` transcribes every file on the computer.
   Optionally add opinions of stronger recognizers as `$W/hyp_<name>.json`, e.g. with
   `hypotheses_mlx.py`; each one makes the next step shorter.
3. **Label.** `$P $F/label.py $W` accepts an utterance when everything that heard it wrote the
   same words (the stock model on the computer, the large model on the phone, any extra
   recognizer, and the field as the user left it), and lists the rest in `$W/review.txt`. Go
   through that list and write `reviewed.tsv` (`id`, `A`/`B`/`X`, text; tab-separated), then run
   `label.py $W --manual reviewed.tsv` until `review.txt` is empty. See [Labels](#labels) for
   how to decide; this hour is where a small data set is won or lost.
4. **Rehearsal material.** `$P $F/replay.py replay $S <corpus.parquet>...` takes speech of
   other people and labels it with the stock model itself ([why](#what-it-forgets)).
5. **Measure before believing.** `PYTHON=$P $F/cross_validate.sh $W SenseVoiceSmall $S cv
   --replay replay/replay.jsonl` trains five models, each without a fifth of
   the data, exports each as the phone would get it, and scores the left-out fifths.
6. **Train on everything and export.**
   ```sh
   $P $F/train.py $W SenseVoiceSmall final --replay replay/replay.jsonl
   $P $F/export_onnx.py SenseVoiceSmall final/model.pt final/onnx
   ```
7. **Check what it forgot.** `$P $F/public.py <benchmark dir> public` unpacks the test sets of
   [Local Voice IME's MODELS.md](https://github.com/Lewin671/local-voice-ime/blob/main/docs/MODELS.md); then for each set
   `$P $F/transcribe.py public/test_<set>.jsonl out.json --onnx final/onnx` and
   `$P $F/score.py --manifest public/test_<set>.jsonl --no-digits stock=... tuned=out.json`.
8. **On a phone.** In a checkout of Local Voice IME,
   `WAVS=<a few of your recordings> scripts/bench/device-bench.sh tuned final/onnx sensevoice`
   loads the exported model in the app's own runtime and reports speed and memory.

`python3 -m unittest` tests the logic that decides
what is trained on and how it is scored; it needs no model.

The model that comes out is a derivative of SenseVoice Small and stays under its
[license](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE). It is also a model of
one voice and one vocabulary: treat it like the recordings it was made from.

## Evidence

Everything below was measured on the export described at the top, with the scripts as
first committed here.

### The computer hears what the phone heard, nearly

Training only helps if the features in training are the ones on the phone. sherpa-onnx computes
them differently from FunASR, the toolkit SenseVoice was trained with: no dither, `snip_edges`,
and no three frames of left padding before frames are stacked. `sv.py` implements sherpa-onnx's
version, and everything here (training, cross-validation, the public sets) goes through it.

- **The export script is right:** the original checkpoint, exported by `export_onnx.py`,
  transcribes 1021 of 1021 utterances (352 of the speaker's, 669 public) exactly like the file the
  app downloads; its graph differs from that file in one shape computation and two metadata
  strings, and `tokens.txt` is identical byte for byte.
- **The stock model on the computer reproduces the phone's text for 299 of 337 utterances**
  (the ones whose logged text is about that file alone). Part of the rest is a rounding
  difference in how the app saved audio at the time: it handed the recognizer `pcm / 32768` and
  wrote the file as `trunc(sample * 32767)`, so every non-zero sample in the file was one step
  closer to zero than what the recognizer got (fixed in the app since; files written by later versions are exact). Undoing that brings the match to 312 of 337. That one step
  out of 32768 changes a transcript at all says something about the stock model on this audio:
  its decisions are close calls. The unquantized model differs from the quantized one on 105
  of 359 utterances, mostly in punctuation, while the two have the same error rate.
- **Loudness is not the problem.** The recordings are quiet (median peak −30 dBFS, as
  `VOICE_RECOGNITION` delivers them without gain control). Amplifying them by 10, 20 or 30 dB,
  or normalizing each to a fixed peak, changes the stock model's error rate by less than a
  tenth of a point (185 utterances). There is no free win in the app's audio path here.

### Labels

What the log offers, and why none of it can be used blindly:

| Session | Share of utterances | What it is worth |
|---|---|---|
| accepted (field left as dictated) | 44 % | Mostly right. But people leave mistakes they can live with: in 71 of the 158 utterances the recognizers disagreed about the words, and a third opinion or the neighbouring sessions showed the accepted text to be wrong often enough to matter (a brand name written as three unrelated characters; a two-word English phrase standing in for a Chinese noun; a one-syllable particle swapped). |
| corrected (field changed) | 20 % | The most valuable, and the dirtiest. The last state of the field is sometimes the fix, sometimes a rewording of what was meant rather than what was said, sometimes a half-finished edit caught mid-keystroke (a deleted character, a single Latin letter of pinyin), and for sessions with several utterances it does not say which file a word belongs to. |
| unconfirmed (field not found again) | 33 % | No evidence either way: the app the user dictated into cleared the field. |
| undone | 3 % | Usually wrong, usually said again in the next session, which then tells what the undone one was. |

So `label.py` trusts agreement, not status: 192 of the 359 utterances were transcribed with
the same words by the stock model, the phone's large model (FireRedASR2) and Qwen3-ASR 1.7B,
and by the user's field where there was one. The other 167 were reviewed one by one against all
of those plus the sessions around them: 134 could be settled (grade A), 26 are a best reading
(grade B: trained on, not tested on), 7 were dropped (a single unclear syllable, someone else
talking, a term the user deleted on purpose). Rules that came out of the review:

- **Label what was said, in the form the model writes.** Punctuation, digits and casing follow
  the stock model's habits; the words follow the audio. A filler the user deleted from the
  field stays in the label; a phrase the user rewrote is labelled as spoken.
- **A correction counts where it explains the recognizers.** The user's word wins when it
  sounds like what the recognizers wrote; when four recognizers agree against it and it does
  not sound alike, the user was rewording.
- **Half-finished edits are evidence, not labels.** A character deleted and one Latin letter
  typed is the first key of the pinyin of the word that was meant.
- **A retry labels the attempt before it.** An undone or garbled session followed seconds
  later by a clean one of the same length is the same sentence.
- **Homophones go to the dictionary**, not to whichever recognizer wrote them.

The test never contains a sentence that was trained on: folds keep a session together, and
with it every utterance whose words are within 30 % edit distance of another's (retries, a
phrase said three times with one number changed).

### What to train

One fold held out, eight to ten passes over the rest, no rehearsal yet; error rate on the
held-out fold (64 utterances, stock 3.41 % with unquantized weights):

| What moves | Parameters | Held-out error after 2 / 4 / 10 passes | Memory (Apple GPU) |
|---|---|---|---|
| Everything, lr 2e-5 | 234 M | 1.98 / – / – (stopped: the machine swapped) | 7 GB |
| Last 20 of the 70 encoder blocks, lr 2e-5 | 76 M | 2.97 / 2.75 / 2.64 | 3 GB |
| Last 35 blocks, lr 2e-5 | 123 M | 2.75 / 2.86 / 2.75 | 5 GB |
| **Rank-16 updates of every linear layer (LoRA), lr 2e-4** | 22 M | 2.09 / 1.54 / 1.65 | 4 GB |

What the model has to learn is mostly how this person and this microphone sound, and that is
decided in the lower blocks: training only the upper ones gets a third of the gain. Low-rank
updates of all blocks get as far as moving everything, in a fifth of the memory, and are folded
back into ordinary weights before export, so the phone sees a model of the same size and speed.

### What it forgets

A model trained on twenty minutes of one voice gets worse at everything else, and "everything
else" includes what this speaker will say next month. Rehearsal (`replay.py`) counters that:
in every pass the model also sees utterances of other speakers, 1100 of them here from three
corpora that are *not* among the test sets (WenetSpeech test_net second shard, LibriSpeech
dev-clean, ASCEND validation), with the stock model's own transcript as the target. No
reference transcripts are involved, so this cannot teach the model those corpora; it only asks
it to stay as it is wherever it is not being taught.

Fold 0 again, unquantized weights, the 100 utterances per public set that follow the 300 of the
tables in Local Voice IME's MODELS.md (so that those 300 stay untouched for the final check). Change of the error
rate in points; **bold** where the 95 % interval excludes zero:

| | Held-out | AISHELL-1 | Wenet net | Wenet meeting | ASCEND | LibriSpeech | KeSpeech | Common Voice |
|---|---|---|---|---|---|---|---|---|
| Stock | 3.41 | 3.23 | 10.4 | 8.54 | 15.7 | 2.99 | 9.57 | 13.8 |
| No rehearsal | 1.65 | +0.09 | **+2.19** | **+0.84** | +0.00 | **+0.83** | **+7.33** | **+2.24** |
| No rehearsal, weights 60 % of the way | 1.54 | −0.09 | **+0.79** | +0.14 | −0.55 | +0.31 | **+3.19** | +0.65 |
| Rehearsal 1 : 1 | 1.65 | +0.09 | +0.33 | −0.17 | −0.88 | +0.18 | **+2.07** | +0.51 |
| Rehearsal 3 : 1 | 1.98 | −0.17 | +0.11 | −0.17 | −0.77 | +0.18 | +0.00 | −0.22 |

Without rehearsal the model nearly doubles its errors on accented Mandarin (KeSpeech) and gets
measurably worse on five of seven sets. One rehearsed utterance per own utterance removes all
of that except on KeSpeech; three remove that too and start to cost on the speaker's own
speech (three more errors in 908 tokens, which is within noise, but in the expected
direction). The recipe uses **2 : 1**. Pulling the weights part of the way back to the
original (`mix.py`) helps a model trained without rehearsal and is not needed with it.

### Which utterances to repeat

The 167 utterances whose label needed review are the ones at least one recognizer got wrong:
they hold nearly everything there is to learn. Using each of them three times per pass instead
of once (`--hard 3`, each time with other augmentation) was compared on all five folds,
unquantized weights: 2.30 % → 1.86 % held-out error (−0.45, interval −0.78 to −0.16; stock
3.31 %). It is the default. It also means a quarter more training steps, which were not
separated from the effect of the selection, and it is the one choice made by looking at all
five folds, so the cross-validation below is slightly flattered by it; the comparison was
between two settings, not a search.

### Results

`cross_validate.sh` with the defaults (rank-16 updates, learning rate 2e-4, 8 passes, reviewed
utterances three times, rehearsal 2 : 1): five models, each exported and quantized like the
final one and run by sherpa-onnx on the fifth it never saw. Grade A only, 4688 tokens.

| | Utterances | Stock | Fine-tuned | Difference (95 % interval) |
|---|---|---|---|---|
| All | 326 | 2.94 % | 1.86 % | −1.09 (−1.55 to −0.65) |
| Sessions the speaker corrected | 61 | 7.43 % | 5.63 % | −1.80 (−3.37 to −0.38) |
| With Latin letters | 73 | 6.10 % | 3.72 % | −2.38 (−4.04 to −1.02) |
| Without | 253 | 2.03 % | 1.32 % | −0.71 (−1.16 to −0.29) |
| Grades A and B | 352 | 3.55 % | 2.05 % | −1.51 (−2.13 to −0.92) |

Utterances with at least one error: 88 → 56. 36 that the stock model got wrong came out
right, 4 that it got right came out wrong, 52 are wrong either way. Per fold: 2.86 → 1.32,
2.31 → 1.26, 2.72 → 2.07, 3.17 → 2.54, 3.63 → 2.08. For comparison, the large model on the
phone (FireRedASR2) is at 2.71 % and Qwen3-ASR 1.7B at 3.16 % on the same utterances, though
both had a say in the labels, which favours them.

**What it learns is what recurs.** Twenty terms of this speaker (product names, place names,
English jargon) occur 80 times in held-out utterances. The stock model writes 53 of them
correctly, the fine-tuned ones 71. Terms the stock model never got right and that occur three
or four times are learned from the other occurrences (3 of 4, 3 of 3); of those that occur
twice, so that training saw one example, about half are. Trained on everything, the final
model gets nearly all of them, which says nothing about new ones.

**A new day is harder than a held-out fifth.** The export covers five days. Trained on the
first four (260 utterances) and tested on the fifth (81 utterances, 981 tokens, a third of
them English sentences that no earlier day has): 3.77 % → 3.57 %, −0.20 (−1.18 to +0.75).
On the same 81 utterances the cross-validation models, which saw other utterances of that day,
are at 2.85 % (−0.92, −2.06 to 0.00). The test is small and one day is one day, but the
direction is clear: most of the gain is in subjects and words the model has met, and a day
that brings new ones gets little of it. Expect the first row of the table for what you keep
talking about and the fourth for what you start talking about, and train again as the
recordings grow; twenty minutes is very little.

### The final model on other speakers

Trained on all 352 utterances (23.7 minutes with the joined sentences), exported, quantized;
the 300 utterances per set of Local Voice IME's MODELS.md, none of which took part in any decision above.
Utterances in which either model wrote a digit are left out, as there.

| | AISHELL-1 | Wenet net | Wenet meeting | ASCEND | LibriSpeech | KeSpeech | Common Voice | Mean |
|---|---|---|---|---|---|---|---|---|
| Stock | 2.85 | 9.69 | 9.52 | 14.80 | 3.36 | 12.09 | 14.39 | 9.53 |
| Fine-tuned | 2.79 | 10.15 | 9.84 | 13.40 | 3.62 | 13.42 | 14.34 | 9.65 |
| Difference | −0.05 | +0.46 | +0.32 | **−1.40** | +0.26 | **+1.33** | −0.05 | |

Accented Mandarin (KeSpeech) is measurably worse, by a ninth. Code-switched speech (ASCEND) is
better, with a caveat: the rehearsal material includes ASCEND's validation split, with the
stock model's transcripts rather than the references, but the same kind of speech. Three
other sets lean worse without the interval excluding zero. Without `--hard` the same table
has smaller drifts (Wenet net +0.17, KeSpeech +1.10): repeating one's own utterances costs a
little of everything else. A keyboard that one person dictates to can afford that; a model
for several people should rehearse more (`--replay-ratio 3`).

On the speaker's own utterances the final model makes 0.34 % errors, which only says that it
has learned them.

### On a phone

`device-bench.sh` on a Pixel 3 (Snapdragon 845), app runtime, four threads: the fine-tuned
model loads and transcribes like the stock one (same file size, load time within 10 %, peak
memory 378 against 371 MB, real-time factor 0.14 to 0.15 against 0.14 for the first candidate;
the final model was checked back to back with the stock one on a throttled phone, where both
were ten times slower and equal). The weights changed; the graph did not.

### What these numbers do not show

- **One speaker, five days, 4688 test tokens.** The intervals are honest about the sample
  and silent about next month.
- **The labels were reviewed from text, not by ear.** Whoever reviewed had the transcripts of
  four recognizers, the user's field and the neighbouring sessions, and did not listen. Where
  the recognizers agree on a mistake, the label has it too, and the stock model is one of the
  four, so its error rate is if anything understated. The 26 grade-B labels are guesses and
  are kept out of the test for that reason.
- **The phone hears slightly different samples** than the files hold (see above).
- **Selection.** The number of passes, the learning rate and the rehearsal ratio were chosen on
  fold 0, `--hard` on all five. The fifth-day test was run once, after everything was fixed, and
  the public test sets were not used to choose anything.
- **Nothing was measured in the app.** The model was loaded by the app's runtime on a phone,
  not dictated to through the keyboard: the app cannot load it yet.

## License

The scripts are under the [Apache License 2.0](LICENSE). `export_onnx.py` is adapted from
`scripts/sense-voice/export-onnx.py` of [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
(Apache-2.0). SenseVoice Small and every model derived from it, including the released one,
are under the [FunASR Model Open Source License Agreement 1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE):
attribute FunAudioLLM / Alibaba Group and keep the model's name.
