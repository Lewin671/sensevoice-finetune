"""Fine-tune SenseVoice Small on the labelled utterances.

    sensevoice-finetune train <work dir> <SenseVoiceSmall dir> <out dir> [options]

<SenseVoiceSmall dir> is the FunASR checkpoint (ModelScope iic/SenseVoiceSmall: model.pt,
am.mvn, config.yaml, the sentencepiece model). With --holdout K the utterances of fold K are
left out and transcribed after the epochs named by --eval-epochs, into
<out dir>/holdout_e<epoch>[_w<weight>].json; that is how every choice below was made
(docs/EVIDENCE.md). Without it, everything is trained on and the result is saved for export.

What is trained on, per epoch:
  - every labelled utterance (grades A and B) once, each time at another speed and loudness and
    with a few time/frequency stripes masked; those whose label needed review, which are
    the ones the recognizers got wrong, --hard times (3);
  - the parts of a sentence that went on after a pause, joined, as the app transcribes them;
  - now and then two unrelated utterances back to back, so that long input stays familiar;
  - with --replay, utterances of other people's speech (see replay.py), --replay-ratio (2) for
    each of the above, labelled by the unchanged model itself: what it could do before, it should still do.

The model is trained as the app uses it: language "auto", inverse text normalization on, and the
front end of sherpa-onnx (sv.py). The loss is CTC on the text plus cross-entropy on the four
leading tokens (language, emotion, event, normalization), which are held at what the unchanged
model says, as nothing here labels them.

The weights are saved as <out dir>/model.pt (always without --holdout, with --save otherwise).
`mix.py` pulls them back towards the original model, which is what keeps a fine-tuned model
from forgetting what it never saw here; `transcribe.py` and `export_onnx.py` take either.

--train says how much of the model moves: "lora:R" (default), "top:N" or "all"; see trainable().
Memory on an Apple-silicon GPU: about 7 GB for "all", 3-5 GB for the others; --max-cells
(padded frames per batch) is small to keep it there.
"""
import argparse, json, math, os, random, time
import numpy as np
from . import sv

SPEEDS = {0.9: (10, 9), 0.95: (20, 19), 1.0: (1, 1), 1.05: (20, 21), 1.1: (10, 11)}
RICH_WEIGHT = 1.0


def latin(c):
    return c.isascii() and c.isalnum()


def join_texts(parts, continued):
    """Transcript of utterances heard as one. `continued`: the later parts carry a sentence on
    (so the earlier ones lose their closing full stop), otherwise they are separate sentences."""
    out = parts[0]
    for nxt in parts[1:]:
        if continued:
            out = out.rstrip()
            if out.endswith(("。", ".")):
                out = out[:-1]
            if nxt[:1].isupper() and nxt[1:2].islower() and latin(out[-1:]):
                nxt = nxt[0].lower() + nxt[1:]
        out += (" " if latin(out[-1:]) or (out[-1:] in ".,?!" and latin(nxt[:1])) else "") + nxt
    return out


def build_items(labels, holdout):
    """Training and held-out items; an item is {"ids": [...], "audio": [...], "text": str}."""
    by_id = {r["id"]: r for r in labels}
    train, held = [], []
    # --grades may have removed a part of a sentence: then there is no whole to train on
    joinable = lambda r: r["joins"] and all(i in by_id for i in r["joins"])
    for r in labels:
        item = {"ids": [r["id"]], "audio": [r["audio"]], "text": r["text"], "seconds": r["seconds"],
                "hard": r.get("source") == "reviewed"}
        if r["fold"] == holdout:
            if r["grade"] == "A":
                held.append(item)
            continue
        train.append(item)
        if joinable(r):
            parts = [by_id[i] for i in r["joins"]] + [r]
            train.append({"ids": [p["id"] for p in parts], "audio": [p["audio"] for p in parts],
                          "text": join_texts([p["text"] for p in parts], True),
                          "seconds": sum(p["seconds"] for p in parts)})
    return train, held


def schedule(progress, warmup=0.1, floor=0.1):
    """Factor on the learning rate at `progress` (0..1) through training: linear warm-up over
    the first tenth, then a cosine down to a tenth. It follows the batches actually trained
    on, whatever the repeats and the rehearsal add to an epoch."""
    if progress < warmup:
        return max(progress / warmup, 0.01)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / (1 - warmup))))


class Augment:
    def __init__(self, rng, enabled=True):
        self.rng, self.enabled = rng, enabled

    def wave(self, x):
        if not self.enabled:
            return x
        from scipy.signal import resample_poly

        up, down = SPEEDS[self.rng.choice(list(SPEEDS))]
        if up != down:
            x = resample_poly(x, up, down).astype(np.float32)
        gain = 10 ** (self.rng.uniform(-6, 12) / 20)
        return np.clip(x * gain, -1.0, 1.0)

    def feats(self, f):
        """Mask one band of mel bins (in each of the stacked frames) and up to two short spans."""
        if not self.enabled:
            return f
        f = f.copy()
        width = self.rng.randint(0, 12)
        start = self.rng.randint(0, 80 - width)
        for k in range(sv.LFR_WINDOW):
            f[:, k * 80 + start : k * 80 + start + width] = 0.0
        for _ in range(2):
            span = self.rng.randint(0, min(3, max(len(f) // 10, 0)))
            at = self.rng.randint(0, len(f) - span)
            f[at : at + span] = 0.0
        return f


def batches(lengths, max_cells, max_items, rng):
    """Indices grouped by similar length; a batch has at most max_cells padded frames."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i] + rng.random())
    out, cur = [], []
    for i in order:
        if cur and (lengths[i] * (len(cur) + 1) > max_cells or len(cur) >= max_items):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


class LoRA:
    """Low-rank updates of the encoder's linear layers: W x becomes W x + (B A) x * scale, with
    only A and B trained (Hu et al. 2021). `merged()` folds them into ordinary weights."""

    def __init__(self, model, rank, alpha=None):
        import torch

        self.torch, self.model, self.scale = torch, model, (alpha or rank) / rank
        self.items = []  # (name of the weight, A, B)
        self.hooks = []
        for name, module in model.encoder.named_modules():
            if isinstance(module, torch.nn.Linear) and ".embed" not in name:
                a = torch.nn.Parameter(torch.empty(rank, module.in_features, device=module.weight.device))
                b = torch.nn.Parameter(torch.zeros(module.out_features, rank, device=module.weight.device))
                torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                self.items.append((f"encoder.{name}.weight", a, b))
                self.hooks.append(module.register_forward_hook(self._hook(a, b)))

    def _hook(self, a, b):
        def add(module, inputs, output):
            return output + (inputs[0] @ a.t() @ b.t()) * self.scale
        return add

    def parameters(self):
        return [q for _, a, b in self.items for q in (a, b)]

    def merged(self):
        """State of the model with the updates folded in, on the CPU."""
        state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        for name, a, b in self.items:
            state[name] += (b.detach().cpu() @ a.detach().cpu()) * self.scale
        return state


def trainable(model, spec):
    """Mark what is trained and return (parameters, function giving the weights to save).

    "all"     every parameter. Needs about 7 GB.
    "top:N"   the last N of the encoder's 70 blocks and the output layer.
    "lora:R"  rank-R updates of every linear layer of the encoder, and the output layer in full
              at a tenth of the learning rate.
    """
    names = [n for n, _ in model.named_parameters()]
    plain = lambda: {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if spec.startswith("lora:"):
        for q in model.parameters():
            q.requires_grad = False
        lora = LoRA(model, int(spec[5:]))
        for q in model.ctc.parameters():
            q.requires_grad = True
        # the output layer is ordinary weights and takes the small steps ordinary weights need
        return [{"params": lora.parameters()}, {"params": list(model.ctc.parameters()), "scale": 0.1}], lora.merged
    if spec == "all":
        chosen = set(names)
    elif spec.startswith("top:"):
        n = int(spec[4:])
        blocks = (["encoder.encoders0.0."] + [f"encoder.encoders.{i}." for i in range(49)]
                  + [f"encoder.tp_encoders.{i}." for i in range(20)])[-n:]
        chosen = {x for x in names if x.startswith(tuple(blocks)) or x.startswith(("ctc.", "encoder.tp_norm"))}
        if n > 20:
            chosen |= {x for x in names if x.startswith("encoder.after_norm")}
    else:
        raise SystemExit(f"--train {spec}?")
    for n, q in model.named_parameters():
        q.requires_grad = n in chosen
    return [{"params": [q for q in model.parameters() if q.requires_grad]}], plain


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("work"), p.add_argument("model_dir"), p.add_argument("out")
    p.add_argument("--holdout", type=int, default=-1)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4, help="for lora; use about 2e-5 for top:N and all")
    p.add_argument("--train", default="lora:16")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--pairs", type=float, default=0.15, help="share of utterances also used back to back with another")
    p.add_argument("--hard", type=int, default=3, help="times per epoch an utterance is used whose label needed review")
    p.add_argument("--replay"), p.add_argument("--replay-ratio", type=float, default=2.0)
    p.add_argument("--eval-epochs", default="")
    p.add_argument("--save", action="store_true")
    p.add_argument("--extra-eval", nargs="*", default=[], help="name=manifest.jsonl sets transcribed along with the holdout")
    p.add_argument("--grades", default="AB")
    p.add_argument("--max-cells", type=int, default=480)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    labels = [r for r in sv.read_jsonl(os.path.join(args.work, "labels.jsonl")) if r["grade"] in args.grades or r["fold"] == args.holdout]
    train, held = build_items(labels, args.holdout)
    replay = sv.read_jsonl(args.replay) if args.replay else []
    extra = {}
    for spec in args.extra_eval:
        name, path = spec.split("=", 1)
        extra[name] = sv.read_jsonl(path)
    eval_epochs = {int(e) for e in args.eval_epochs.split(",") if e} | {args.epochs}

    m = sv.SenseVoice(args.model_dir, device=args.device)
    model = m.model
    waves = {}

    def wave(path):
        if path not in waves:
            waves[path] = sv.read_wav(path)
        return waves[path]

    def rich_tokens(items):
        """What the unchanged model puts in the four leading positions, per item."""
        feats = [sv.features(np.concatenate([wave(a) for a in it["audio"]]), m.cmvn) for it in items]
        out = [None] * len(items)
        order = sorted(range(len(items)), key=lambda i: len(feats[i]))
        with torch.no_grad():
            for k in range(0, len(order), 16):
                part = order[k : k + 16]
                x, n = m.batch([feats[i] for i in part])
                logits, _ = m.logits(x, n)
                for i, row in zip(part, logits[:, :4].argmax(-1).cpu().tolist()):
                    out[i] = row
        return out

    for it, rich in zip(train, rich_tokens(train)):
        it["rich"] = rich
    for it, rich in zip(replay, rich_tokens([{"audio": [r["audio"]]} for r in replay]) if replay else []):
        it["rich"], it["audio"] = rich, [it["audio"]]
    print(f"train {len(train)} items ({sum(i['seconds'] for i in train) / 60:.1f} min), "
          f"held out {len(held)}, replay pool {len(replay)}", flush=True)

    groups, weights = trainable(model, args.train)
    params = [q for g in groups for q in g["params"]]
    print(f"training {sum(q.numel() for q in params) / 1e6:.1f} M parameters ({args.train})", flush=True)
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0)
    aug = Augment(rng, not args.no_augment)

    def epoch_items():
        items = list(train)
        # the utterances the recognizers disagreed about are the ones with something to learn
        items += [it for it in train if it.get("hard")] * (args.hard - 1)
        for it in train:
            if len(it["ids"]) == 1 and rng.random() < args.pairs:
                other = rng.choice(train)
                if len(other["ids"]) == 1 and other is not it and it["seconds"] + other["seconds"] < 20:
                    items.append({"audio": it["audio"] + other["audio"], "rich": it["rich"],
                                  "text": join_texts([it["text"], other["text"]], False)})
        if replay:
            # per utterance of one's own in this epoch, repeats and pairs included
            wanted = round(args.replay_ratio * len(items))
            if wanted > len(replay) and not short:
                short.append(True)
                print(f"the replay pool has {len(replay)} utterances where --replay-ratio asks for "
                      f"{wanted} per epoch: all are used, {len(replay) / len(items):.2f} per own one",
                      flush=True)
            items += rng.sample(replay, min(len(replay), wanted))
        return items

    short = []  # said once

    lr_at = schedule

    def transcribe(items):
        model.eval()
        texts = m.transcribe([np.concatenate([wave(a) for a in it["audio"]]) for it in items])
        return {it["ids"][0]: t for it, t in zip(items, texts)}

    def evaluate(epoch):
        line = f"  eval e{epoch}:"
        for name, items in [("holdout", held)] + list(extra.items()):
            if not items:
                continue
            if name != "holdout":
                items = [{"ids": [r["id"]], "audio": [r["audio"]], "text": r["text"]} for r in items]
            hyp = transcribe(items)
            with open(os.path.join(args.out, f"{name}_e{epoch}.json"), "w", encoding="utf-8") as f:
                json.dump(hyp, f, ensure_ascii=False, indent=0)
            counts = [sv.error_counts(it["text"], hyp[it["ids"][0]]) for it in items]
            line += f"  {name} {100 * sum(c[0] for c in counts) / max(1, sum(c[1] for c in counts)):.2f} %"
        print(line, flush=True)

    if 0 in eval_epochs:
        evaluate(0)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        items = epoch_items()
        feats, targets = [], []
        for it in items:
            x = np.concatenate([wave(a) for a in it["audio"]])
            f = sv.features(aug.wave(x), m.cmvn)
            ids = m.sp.encode(it["text"])
            if f is None or len(ids) == 0 or len(f) < len(ids):
                f, ids = None, None
            feats.append(None if f is None else aug.feats(f))
            targets.append(ids)
        keep = [i for i in range(len(items)) if feats[i] is not None]
        loss_sum = n_batches = 0
        plan = list(batches([len(feats[i]) for i in keep], args.max_cells, 24, rng))
        for b, part in enumerate(plan):
            idx = [keep[i] for i in part]
            x, n = m.batch([feats[i] for i in idx])
            logits, out_n = m.logits(x, n)
            logp = logits.float().log_softmax(-1).cpu()
            ys = [targets[i] for i in idx]
            flat = torch.tensor([t for y in ys for t in y], dtype=torch.long)
            ctc = torch.nn.functional.ctc_loss(
                logp[:, 4:].transpose(0, 1), flat, (out_n.cpu() - 4).long(),
                torch.tensor([len(y) for y in ys]), blank=sv.BLANK, reduction="sum", zero_infinity=True)
            rich = torch.tensor([items[i]["rich"] for i in idx])
            ce = torch.nn.functional.nll_loss(logp[:, :4].reshape(-1, logp.size(-1)), rich.reshape(-1), reduction="sum")
            loss = (ctc + RICH_WEIGHT * ce) / len(idx)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at((epoch - 1 + b / len(plan)) / args.epochs) * g.get("scale", 1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            if args.device == "mps":  # the allocator otherwise keeps a buffer for every batch shape it has seen
                torch.mps.empty_cache()
            loss_sum += float(loss)
            n_batches += 1
        print(f"epoch {epoch}: loss {loss_sum / max(1, n_batches):.3f}  {n_batches} batches  "
              f"{time.time() - t0:.0f} s", flush=True)
        if epoch in eval_epochs:
            evaluate(epoch)

    if args.holdout < 0 or args.save:
        torch.save(weights(), os.path.join(args.out, "model.pt"))
        with open(os.path.join(args.out, "train.json"), "w", encoding="utf-8") as f:
            json.dump({**vars(args), "items": len(train)}, f, indent=1)
        print("saved", os.path.join(args.out, "model.pt"))


if __name__ == "__main__":
    main()
