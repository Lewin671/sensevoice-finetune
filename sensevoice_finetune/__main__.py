"""Fine-tune SenseVoice Small on an export of one's own recordings.

    sensevoice-finetune <command> [arguments]
    sensevoice-finetune <command> --help

README.md has the recipe these commands are the steps of; each is a module of this package and
can also be run as `python -m sensevoice_finetune.<module>`.
"""
import importlib, sys

COMMANDS = {
    "From an export to labels": {
        "prepare": "unpack an export into one row per utterance",
        "hypotheses": "transcribe every utterance again with the stock model",
        "hypotheses-mlx": "a second opinion from a large recognizer (Apple silicon, optional)",
        "label": "accept what all recognizers agree on, list the rest for review",
    },
    "Training": {
        "replay": "other people's speech for the model to rehearse",
        "train": "fine-tune on the labelled utterances",
        "mix": "pull fine-tuned weights back towards the original",
        "export-onnx": "write model.int8.onnx and tokens.txt for sherpa-onnx",
    },
    "Measuring": {
        "transcribe": "transcribe a manifest with an exported model or PyTorch weights",
        "score": "error rates of saved transcripts, with intervals",
        "terms": "how often each system writes the speaker's own terms",
        "public": "public test sets, to see what a model forgot",
        "check-export": "does an exported model transcribe like the published one?",
        "check-phone": "does the computer hear what the phone heard?",
    },
}


def usage():
    lines = [__doc__.strip()]
    for title, commands in COMMANDS.items():
        lines += ["", f"{title}:"] + [f"    {name:16s}{what}" for name, what in commands.items()]
    return "\n".join(lines)


def main():
    name, *rest = sys.argv[1:] or ["--help"]
    if name in ("-h", "--help"):
        print(usage())
        return
    if not any(name in commands for commands in COMMANDS.values()):
        raise SystemExit(f"no such command: {name}\n\n{usage()}")
    module = importlib.import_module(f".{name.replace('-', '_')}", __package__)
    # the modules that take options print them themselves
    if (not rest or rest[0] in ("-h", "--help")) and not hasattr(module, "argparse"):
        print(module.__doc__.strip())
        return
    sys.argv = [f"sensevoice-finetune {name}", *rest]
    module.main()


if __name__ == "__main__":
    main()
