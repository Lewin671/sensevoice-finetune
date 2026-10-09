"""Turn an export of Local Voice IME's recordings (its docs/TRAINING_DATA.md) into one row per
utterance, with everything the log says about it.

    sensevoice-finetune prepare <export.zip or extracted directory> <work dir>

Writes <work dir>/utterances.jsonl. Each row:

    id, session, audio (absolute path), seconds, index (within the session), of (utterances in
    the session), continues, phone (the standard model's text on the phone; for `continues` it
    covers the earlier audio as well), refined (the large model's text, or null),
    status: accepted | corrected | unconfirmed | undone
    dictated, field (what dictation wrote / what stood there when the field was last read)

Nothing is filtered here; `label.py` decides what each utterance is worth.
"""
import collections, os, sys, zipfile
from . import sv


MAX_FILES, MAX_BYTES = 200_000, 20 << 30  # an export of years of dictation is far below both


def extract(archive, dest):
    """Unpack an export into `dest`. Refuses an archive with entries that are not plain relative
    paths or that would be written through a symbolic link already in `dest`, and one that is
    larger than any export could be; each file is written to the path that was checked."""
    os.makedirs(dest, exist_ok=True)
    root = os.path.realpath(dest)
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        if len(infos) > MAX_FILES or sum(i.file_size for i in infos) > MAX_BYTES:
            raise ValueError(f"{archive} is larger than an export can be")
        targets = []
        for info in infos:
            parts = info.filename.replace("\\", "/").split("/")
            if info.is_dir():
                parts = parts[:-1]
            if info.filename.startswith("/") or not parts or any(p in ("", ".", "..") or ":" in p for p in parts):
                raise ValueError(f"unsafe path in archive: {info.filename}")
            target = os.path.join(root, *parts)
            if os.path.realpath(target) != target:
                raise ValueError(f"{info.filename} would be written through a link in {dest}")
            targets.append(target)
        for info, target in zip(infos, targets):
            if info.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            left = info.file_size
            with z.open(info) as src, open(target, "wb") as out:
                while chunk := src.read(min(1 << 20, left + 1)):
                    left -= len(chunk)
                    if left < 0:
                        raise ValueError(f"{info.filename} is longer than the archive declares")
                    out.write(chunk)
    return root


def sessions(records):
    """session -> {"utterances": [...], "fields": [...], "undone": bool}, in log order."""
    out = collections.OrderedDict()
    refined = {}
    for r in records:
        if r.get("v") != 1:
            raise ValueError(f"unknown record version: {r.get('v')}")
        if r["type"] == "refined":
            refined[r["id"]] = r["text"]
            continue
        s = out.setdefault(r["session"], {"utterances": [], "fields": [], "undone": False})
        if r["type"] == "utterance":
            s["utterances"].append(r)
        elif r["type"] == "field":
            s["fields"].append(r)
        elif r["type"] == "undone":
            s["undone"] = True
    return out, refined


def status(session):
    if session["undone"]:
        return "undone"
    if not session["fields"]:
        return "unconfirmed"
    last = session["fields"][-1]
    return "accepted" if last["text"] == last["dictated"] else "corrected"


def main():
    source, work = sys.argv[1], sys.argv[2]
    os.makedirs(work, exist_ok=True)
    root = source if os.path.isdir(source) else extract(source, os.path.join(work, "export"))
    by_session, refined = sessions(sv.read_jsonl(os.path.join(root, "log.jsonl")))
    rows, counts = [], collections.Counter()
    for name, s in by_session.items():
        st = status(s)
        last = s["fields"][-1] if s["fields"] else None
        for i, u in enumerate(s["utterances"]):
            path = os.path.realpath(os.path.join(root, u["audio"]))
            if not path.startswith(os.path.realpath(root) + os.sep) or not os.path.isfile(path):
                counts["audio missing"] += 1
                continue
            rows.append({
                "id": u["id"], "session": name, "audio": path, "seconds": u["seconds"],
                "index": i, "of": len(s["utterances"]), "continues": u["continues"],
                "phone": u["text"], "refined": refined.get(u["id"]), "status": st,
                "dictated": last["dictated"] if last else None,
                "field": last["text"] if last else None,
            })
            counts[st] += 1
    sv.write_jsonl(os.path.join(work, "utterances.jsonl"), rows)
    minutes = sum(r["seconds"] for r in rows) / 60
    print(f"{len(rows)} utterances, {len(by_session)} sessions, {minutes:.1f} min: {dict(counts)}")


if __name__ == "__main__":
    main()
