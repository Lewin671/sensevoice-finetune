"""How often does each system write the words that matter to this speaker?

    sensevoice-finetune terms <work dir> <terms.txt> <name>=<transcripts.json>[,<more.json>] [...]

<terms.txt> has one term per line (a name, a place, a piece of jargon; "#" starts a comment).
For every term: in how many labelled utterances (grade A) it occurs, and in how many of those
each system wrote it. Case and spaces are ignored. Only utterances every system transcribed
are counted, so held-out transcripts of a cross-validation can be compared with the stock model.
"""
import json, os, re, sys
from . import sv


def main():
    work, terms_file, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
    squash = lambda s: re.sub(r"\s", "", s.lower())
    with open(terms_file, encoding="utf-8") as f:
        terms = [squash(line) for line in f if line.strip() and not line.startswith("#")]
    systems = {}
    for spec in specs:
        name, paths = spec.split("=", 1)
        systems[name] = {}
        for path in paths.split(","):
            with open(path, encoding="utf-8") as f:
                systems[name].update(json.load(f))
    labels = [r for r in sv.read_jsonl(os.path.join(work, "labels.jsonl"))
              if r["grade"] == "A" and all(r["id"] in h for h in systems.values())]
    print(f"{'term':16s} {'said':>5s}" + "".join(f" {name:>10s}" for name in systems))
    total = [0] * (len(systems) + 1)
    for term in terms:
        ids = [r["id"] for r in labels if term in squash(r["text"])]
        row = [len(ids)] + [sum(term in squash(h[i]) for i in ids) for h in systems.values()]
        total = [a + b for a, b in zip(total, row)]
        print(f"{term:16s} {row[0]:5d}" + "".join(f" {n:10d}" for n in row[1:]))
    print(f"{'all':16s} {total[0]:5d}" + "".join(f" {n:10d}" for n in total[1:]))


if __name__ == "__main__":
    main()
