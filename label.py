"""Step 3: decide what was said in each utterance, and how far to trust that.

    label.py <work dir> [--manual reviewed.tsv] [--folds 5] [--seed 0]

Reads <work dir>/utterances.jsonl and every <work dir>/hyp_*.json (id -> transcript;
hyp_sensevoice.json, from hypotheses.py, is required and supplies the punctuation, digits and
casing the model is used to). Writes

    labels.jsonl   one row per usable utterance: id, session, audio, seconds, text, grade,
                   group, fold, joins (ids of the earlier utterances of a sentence that went on)
    review.txt     every utterance the sources disagree about and nobody has reviewed yet

Grades:

    A  verified: all recognizers heard the same words and, where the user left the session's
       text alone or corrected it, that text has them too; or a reviewer said so.
    B  probable: a reviewer's best reading. Trained on, kept out of the headline test.

An utterance the sources disagree about gets no label until someone settles it in the reviewed
file (tab-separated: id, grade A/B/X, text; X drops it). The log cannot settle it by itself:
"accepted" sessions contain mistakes the user did not bother with, "corrected" ones contain
rewording and half-finished edits, and neither says which audio file a word belongs to. That
review is where most of the value of a small data set is.

Folds keep a session, and utterances that say nearly the same thing (a retry after a wrong
result), on one side of every split, so that no test sentence was trained on.
"""
import argparse, glob, json, os, random
import sv


def agreed(row, hyps):
    """The standard model's transcript if every source has the same words, else None."""
    words = [sv.score_tokens(h[row["id"]]) for h in hyps.values()]
    if row["of"] == 1 and row["field"] is not None and row["status"] != "undone":
        words.append(sv.score_tokens(row["field"]))
    if row["refined"] is not None:
        words.append(sv.score_tokens(row["refined"]))
    if len(words) < 2 or not words[0] or any(w != words[0] for w in words):
        return None
    return hyps["sensevoice"][row["id"]]


def read_manual(path):
    out = {}
    if not path:
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cells = line.rstrip("\n").split("\t")
            uid, grade, text = cells[0], cells[1], cells[2] if len(cells) > 2 else ""
            if grade not in ("A", "B", "X") or (grade != "X" and not text.strip()):
                raise ValueError(f"{path}: bad line for {uid}")
            if uid in out:
                raise ValueError(f"{path}: {uid} appears twice")
            out[uid] = (grade, text.strip())
    return out


def similar(a, b, threshold=0.3):
    """Two transcripts that say nearly the same thing."""
    longest = max(len(a), len(b))
    return longest > 0 and sv.edit_distance(a, b) <= threshold * longest


def groups(rows):
    """Union of sessions and near-identical transcripts -> group id per row."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    words = [sv.score_tokens(r["text"]) for r in rows]
    for i in range(len(rows)):
        for j in range(i):
            if rows[i]["session"] == rows[j]["session"] or similar(words[i], words[j]):
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(rows))]


def assign_folds(rows, k, seed):
    """Spread groups over k folds, balancing the seconds of audio."""
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    order = sorted(by_group)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: -sum(r["seconds"] for r in by_group[g]))
    load = [0.0] * k
    for g in order:
        fold = load.index(min(load))
        for r in by_group[g]:
            r["fold"] = fold
        load[fold] += sum(r["seconds"] for r in by_group[g])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("work")
    p.add_argument("--manual")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = sv.read_jsonl(os.path.join(args.work, "utterances.jsonl"))
    hyps = {}
    for path in sorted(glob.glob(os.path.join(args.work, "hyp_*.json"))):
        with open(path, encoding="utf-8") as f:
            hyps[os.path.basename(path)[4:-5]] = json.load(f)
    if "sensevoice" not in hyps:
        raise SystemExit("hyp_sensevoice.json is missing; run hypotheses.py first")
    hyps = {"sensevoice": hyps.pop("sensevoice"), **hyps}
    manual = read_manual(args.manual)
    unknown = set(manual) - {r["id"] for r in rows}
    if unknown:
        raise SystemExit(f"reviewed file names utterances that are not in the export: {sorted(unknown)[:5]}")

    labels, review, counts = [], [], {"A": 0, "B": 0, "dropped": 0, "unreviewed": 0}
    for r in rows:
        if r["id"] in manual:
            grade, text = manual[r["id"]]
            source = "reviewed"
        else:
            text = agreed(r, hyps)
            grade, source = ("A", "agreed") if text else (None, None)
        if grade == "X":
            counts["dropped"] += 1
            continue
        if grade is None:
            counts["unreviewed"] += 1
            review.append(f"## {r['id']}  {r['status']}  {r['seconds']} s  {r['index'] + 1}/{r['of']}"
                          + ("  continues" if r["continues"] else ""))
            review += [f"  {name:12s} {h[r['id']]}" for name, h in hyps.items()]
            review.append(f"  {'large model':12s} {r['refined']}")
            if r["field"] is not None:
                review.append(f"  {'dictated':12s} {r['dictated']!r}")
                review.append(f"  {'field':12s} {r['field']!r}")
            continue
        counts[grade] += 1
        labels.append({"id": r["id"], "session": r["session"], "audio": r["audio"],
                       "seconds": r["seconds"], "text": text, "grade": grade, "source": source,
                       "index": r["index"], "continues": r["continues"]})

    # a sentence that went on after a pause: the app transcribes the audio of all its parts
    # again as one, so those parts are also a training example together
    by_id = {r["id"]: r for r in labels}
    for r in labels:
        r["joins"] = []
        if r.pop("continues"):
            number = int(r["id"][-2:])
            before = by_id.get(f"{r['session']}-{number - 1:02d}")
            if before is not None:
                r["joins"] = before["joins"] + [before["id"]]
        r.pop("index")

    for r, g in zip(labels, groups(labels)):
        r["group"] = g
    assign_folds(labels, args.folds, args.seed)
    sv.write_jsonl(os.path.join(args.work, "labels.jsonl"), labels)
    with open(os.path.join(args.work, "review.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(review) + ("\n" if review else ""))
    minutes = sum(r["seconds"] for r in labels) / 60
    print(f"{len(labels)} labelled ({minutes:.1f} min): {counts}")
    for k in range(args.folds):
        part = [r for r in labels if r["fold"] == k]
        print(f"  fold {k}: {len(part)} utterances, {sum(r['seconds'] for r in part) / 60:.1f} min, "
              f"{sum(r['grade'] == 'A' for r in part)} verified")


if __name__ == "__main__":
    main()
