"""Error rates of saved transcripts on the labelled utterances.

    score.py <work dir> <name>=<transcripts.json> [<name>=<transcripts.json> ...] [--grade AB]
    score.py --manifest <test.jsonl> [--no-digits] <name>=<transcripts.json> ...

Each JSON maps utterance id -> transcript (transcribe.py writes them); several files joined by
commas count as one system. The second form scores
any manifest of {"id", "text"} lines, e.g. the public sets of public.py; --no-digits leaves out
the utterances in which a system wrote a digit, as Local Voice IME's docs/MODELS.md does, because those references
spell numbers out. The error rate counts every CJK character and every
Latin word or number, ignoring case and punctuation (sv.score_tokens), over the
verified utterances (grade A) unless --grade says otherwise. The first system is the reference
point: for every other one the difference to it is printed with a 95 % interval from a
bootstrap that resamples groups (sessions and near-identical sentences; in a manifest the
"group" of a row if it has one, e.g. its speaker), so an interval that
excludes 0 is a difference that holds beyond these particular utterances.

Subsets: "corrected" are the utterances of sessions whose text the user changed, where the
label rests on the user more than on any recognizer; "latin" contain English words or letters.
"""
import argparse, json, os, random, re
import sv


def subsets(rows, status):
    yield "all", rows
    if not status:
        return
    yield "corrected", [r for r in rows if status[r["id"]] == "corrected"]
    yield "latin", [r for r in rows if re.search("[A-Za-z]", r["text"])]
    yield "no latin", [r for r in rows if not re.search("[A-Za-z]", r["text"])]


def rate(counts):
    errors = sum(c[0] for c in counts)
    total = sum(c[1] for c in counts)
    return 100.0 * errors / max(total, 1)


def bootstrap(rows, a, b, rounds=2000, seed=0):
    """95 % interval of rate(b) - rate(a), resampling groups."""
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r["id"])
    names = sorted(by_group)
    rng = random.Random(seed)
    diffs = []
    for _ in range(rounds):
        ids = [i for g in rng.choices(names, k=len(names)) for i in by_group[g]]
        diffs.append(rate([b[i] for i in ids]) - rate([a[i] for i in ids]))
    diffs.sort()
    return diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds) - 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("args", nargs="+", metavar="[work dir] name=transcripts.json")
    p.add_argument("--manifest")
    p.add_argument("--no-digits", action="store_true")
    p.add_argument("--grade", default="A")
    p.add_argument("--errors", help="write the utterances a system got wrong to this file")
    p.add_argument("--common", action="store_true",
                   help="score only the utterances every system transcribed, e.g. one fold; "
                        "without it a missing transcript is an error of the run, not of the model")
    args = p.parse_args()

    if args.manifest:
        # a manifest may name the speaker or recording of an utterance as its "group"
        labels = [{"group": r["id"], "grade": "-", **r} for r in sv.read_jsonl(args.manifest)]
        status, specs = {}, args.args
    else:
        work, specs = args.args[0], args.args[1:]
        labels = [r for r in sv.read_jsonl(os.path.join(work, "labels.jsonl")) if r["grade"] in args.grade]
        status = {r["id"]: r["status"] for r in sv.read_jsonl(os.path.join(work, "utterances.jsonl"))}
    systems = {}
    for spec in specs:
        name, paths = spec.split("=", 1)
        systems[name] = {}
        for path in paths.split(","):  # e.g. one file per fold
            with open(path, encoding="utf-8") as f:
                systems[name].update(json.load(f))
    missing = {name: [r["id"] for r in labels if r["id"] not in h] for name, h in systems.items()}
    if any(missing.values()):
        for name, ids in missing.items():
            if ids:
                print(f"{name}: no transcript for {len(ids)} of {len(labels)} utterances, e.g. {ids[0]}")
        if not args.common:
            raise SystemExit("transcripts are missing; pass --common to score the rest on purpose")
        labels = [r for r in labels if all(r["id"] in h for h in systems.values())]
    if args.no_digits:
        labels = [r for r in labels if sv.score_tokens(r["text"])
                  and not any(re.search(r"\d", h[r["id"]]) for h in systems.values())]
    counts = {name: {r["id"]: sv.error_counts(r["text"], h[r["id"]]) for r in labels}
              for name, h in systems.items()}
    first = next(iter(systems))
    for title, rows in subsets(labels, status):
        if not rows:
            continue
        tokens = sum(counts[first][r["id"]][1] for r in rows)
        print(f"{title}: {len(rows)} utterances, {tokens} tokens")
        for name in systems:
            c = counts[name]
            wrong = sum(c[r["id"]][0] > 0 for r in rows)
            line = f"  {name:22s} {rate([c[r['id']] for r in rows]):6.2f} %   utterances with an error: {wrong:3d}"
            if name != first:
                lo, hi = bootstrap(rows, counts[first], c)
                d = rate([c[r["id"]] for r in rows]) - rate([counts[first][r["id"]] for r in rows])
                line += f"   vs {first}: {d:+.2f} ({lo:+.2f}..{hi:+.2f})"
            print(line)
    if args.errors:
        with open(args.errors, "w", encoding="utf-8") as f:
            for r in labels:
                bad = {n: h[r["id"]] for n, h in systems.items() if counts[n][r["id"]][0] > 0}
                if bad:
                    f.write(f"## {r['id']}  {r['grade']}\n  {'label':22s} {r['text']}\n")
                    for n, t in bad.items():
                        f.write(f"  {n:22s} {t}\n")


if __name__ == "__main__":
    main()
