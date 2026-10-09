"""Shared pieces of the SenseVoice fine-tuning scripts: the audio front end, the model, scoring.

The front end here is the one sherpa-onnx applies on the phone (80 mel bins, Hamming window,
no dither, `snip_edges`, low frame rate 7/6 without padding, global mean/variance), not the one
FunASR trains with, which pads three frames on the left. Fine-tuning and evaluation therefore
see the features the app will feed the exported model; `check_phone.py` measures how closely.
"""
import json, os, re, wave
import numpy as np

SAMPLE_RATE = 16000
LFR_WINDOW, LFR_SHIFT = 7, 6
BLANK = 0
# ids in the 16-entry prompt embedding (SenseVoiceSmall.lid_dict / textnorm_dict)
LANG_AUTO, WITH_ITN = 0, 14
# what the app asks for (VoiceEngine.kt): language "auto", inverse text normalization on
PROMPT = (LANG_AUTO, WITH_ITN)


def read_wav(path):
    """16 kHz mono 16-bit PCM -> float32 in [-1, 1)."""
    with wave.open(path) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit PCM")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def read_audio(path):
    """Any mono or stereo audio file soundfile can read -> 16 kHz float32."""
    if path.endswith(".wav"):
        try:
            return read_wav(path)
        except ValueError:
            pass
    import soundfile
    from math import gcd
    from scipy.signal import resample_poly

    x, rate = soundfile.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if rate != SAMPLE_RATE:
        g = gcd(rate, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, rate // g).astype(np.float32)
    return x


def load_cmvn(path):
    """(neg_mean, inv_stddev) from a Kaldi nnet `am.mvn`, each of length 80 * LFR_WINDOW."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("<LearnRateCoef>"):
                rows.append(np.array(line.split()[3:-1], dtype=np.float32))
    neg_mean, inv_stddev = rows
    return neg_mean, inv_stddev


def features(samples, cmvn):
    """float32 samples in [-1, 1) -> (T, 560) features, or None if the audio is too short."""
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = True
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = 0
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(SAMPLE_RATE, (samples * 32768.0).tolist())
    fbank.input_finished()
    n = fbank.num_frames_ready
    if n < LFR_WINDOW:
        return None
    f = np.stack([fbank.get_frame(i) for i in range(n)])
    t = (n - LFR_WINDOW) // LFR_SHIFT + 1
    idx = np.arange(t)[:, None] * LFR_SHIFT + np.arange(LFR_WINDOW)[None, :]
    f = f[idx].reshape(t, -1)
    neg_mean, inv_stddev = cmvn
    return ((f + neg_mean) * inv_stddev).astype(np.float32)


class SenseVoice:
    """The FunASR SenseVoiceSmall checkpoint, with the forward pass sherpa-onnx's export uses."""

    def __init__(self, model_dir, device="cpu", weights=None, mix=None):
        import torch, sentencepiece
        from funasr import AutoModel

        model, kwargs = AutoModel.build_model(model=model_dir, device="cpu", disable_update=True)
        if weights:
            state = torch.load(weights, map_location="cpu")
            if mix is not None:  # original + mix * (fine-tuned - original), see mix.py
                base = model.state_dict()
                state = {k: base[k] + mix * (v - base[k]) if v.is_floating_point() else v
                         for k, v in state.items()}
            model.load_state_dict(state)
        self.torch = torch
        self.device = device
        self.model = model.to(device).eval()
        self.model_dir = model_dir
        self.cmvn = load_cmvn(os.path.join(model_dir, "am.mvn"))
        self.sp = sentencepiece.SentencePieceProcessor(
            model_file=os.path.join(model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")
        )

    def logits(self, feats, lengths, prompt=PROMPT):
        """(N, T, 560) features -> (N, T + 4, vocab) logits; the first four frames are the
        language, emotion, event and text-normalization tokens."""
        torch, m = self.torch, self.model
        n = feats.size(0)
        ids = torch.tensor([[prompt[0], 1, 2, prompt[1]]], device=feats.device).repeat(n, 1)
        x = torch.cat((m.embed(ids), feats), dim=1)
        out, out_lengths = m.encoder(x, lengths + 4)[:2]
        return m.ctc.ctc_lo(out), out_lengths

    def batch(self, feats_list):
        torch = self.torch
        lengths = torch.tensor([len(f) for f in feats_list], dtype=torch.int32)
        x = torch.zeros(len(feats_list), int(lengths.max()), feats_list[0].shape[1])
        for i, f in enumerate(feats_list):
            x[i, : len(f)] = torch.from_numpy(f)
        return x.to(self.device), lengths.to(self.device)

    def greedy(self, logits, lengths):
        """Greedy CTC decoding as sherpa-onnx does it: -> [(rich token ids, text token ids)]."""
        out = []
        best = logits.argmax(-1).cpu().numpy()
        for row, n in zip(best, lengths.cpu().numpy()):
            row = row[:n]
            rich, body = row[:4].tolist(), row[4:]
            keep = np.ones(len(body), dtype=bool)
            keep[1:] = body[1:] != body[:-1]
            out.append((rich, [int(t) for t in body[keep] if t != BLANK]))
        return out

    def text(self, ids):
        return "".join(self.sp.id_to_piece(i) for i in ids).replace("▁", " ").strip()

    def transcribe(self, samples_list, batch_size=16):
        """Raw transcripts (as sherpa-onnx returns them) of a list of sample arrays."""
        torch = self.torch
        feats = [features(s, self.cmvn) for s in samples_list]
        order = sorted((i for i, f in enumerate(feats) if f is not None), key=lambda i: len(feats[i]))
        texts = [""] * len(feats)
        with torch.no_grad():
            for k in range(0, len(order), batch_size):
                part = order[k : k + batch_size]
                x, lengths = self.batch([feats[i] for i in part])
                logits, out_lengths = self.logits(x, lengths)
                for i, (_, ids) in zip(part, self.greedy(logits, out_lengths)):
                    texts[i] = self.text(ids)
        return texts


def sherpa_recognizer(model, tokens, threads=4):
    """The recognizer as the app configures it (VoiceEngine.kt)."""
    import sherpa_onnx

    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model, tokens=tokens, num_threads=threads, language="auto", use_itn=True
    )


def sherpa_transcribe(recognizer, samples_list):
    out = []
    for s in samples_list:
        st = recognizer.create_stream()
        st.accept_waveform(SAMPLE_RATE, s)
        recognizer.decode_stream(st)
        out.append(st.result.text)
    return out


# --- text ---------------------------------------------------------------------------------------

_CJK = "一-鿿㐀-䶿"
_WIDE = _CJK + "，。？！、；：“”‘’（）《》"
_HALF_TO_FULL = {",": "，", "?": "？", "!": "！", ";": "；", ":": "："}


def app_normalize(raw):
    """Port of Local Voice IME's VoiceText.normalize: what the app does to every raw transcript."""
    s = raw.strip()
    out = []
    for i, c in enumerate(s):
        prev = out[-1] if out else None
        if c == " ":
            nxt = s[i + 1] if i + 1 < len(s) else None
            prev_wide = prev is not None and re.match(f"[{_WIDE}]", prev)
            next_wide = nxt is not None and re.match(f"[{_WIDE}]", nxt)
            if prev is None or prev == " " or prev_wide or next_wide:
                continue
            out.append(c)
        elif c in _HALF_TO_FULL and prev is not None and re.match(f"[{_CJK}]", prev):
            out.append(_HALF_TO_FULL[c])
        else:
            out.append(c)
    return "".join(out).strip()


def score_tokens(text):
    """Tokens an error rate is counted on: every CJK character, every Latin word or number;
    case ignored. Punctuation separates words, where Local Voice IME's scripts/bench/bench.py
    deletes it first ("Wi-Fi" is two tokens here and one there), so rates on the same
    transcripts differ slightly between the two."""
    import zhconv

    s = zhconv.convert(text, "zh-cn").lower()
    s = re.sub(r"<[^>]*>", "", s)
    return re.findall(rf"[{_CJK}]|[a-z0-9']+", s)


def edit_distance(ref, hyp):
    """Levenshtein distance between two token lists."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def error_counts(ref_text, hyp_text):
    """(errors, reference tokens) for one utterance."""
    ref, hyp = score_tokens(ref_text), score_tokens(hyp_text)
    return edit_distance(ref, hyp), len(ref)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
