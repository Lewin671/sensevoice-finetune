"""Pull fine-tuned weights back towards the original model.

    mix.py <SenseVoiceSmall dir> <fine-tuned model.pt> <weight> <out model.pt>

Writes original + weight * (fine-tuned - original): weight 1 is the fine-tuned model, 0 the
original. Somewhere in between a model keeps most of what fine-tuning taught it and loses less
of what it knew (weight-space ensembling, Wortsman et al. 2022, "Robust fine-tuning of
zero-shot models"); README.md has the numbers.
"""
import os, sys
import torch


def mix(base, tuned, weight):
    return {k: base[k] + weight * (v - base[k]) if v.is_floating_point() else v for k, v in tuned.items()}


def main():
    model_dir, tuned_path, weight, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    base = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    base = base.get("state_dict", base)
    tuned = torch.load(tuned_path, map_location="cpu")
    missing = set(tuned) - set(base)
    if missing:
        raise SystemExit(f"not the same model: {sorted(missing)[:3]}")
    torch.save(mix(base, tuned, weight), out)


if __name__ == "__main__":
    main()
