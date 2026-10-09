#!/usr/bin/env bash
# How much does fine-tuning help on speech it was not trained on? Trains one model per fold of
# labels.jsonl with that fold left out, exports each exactly as the final model would be
# exported, lets sherpa-onnx transcribe the left-out fold with it, and scores all folds together
# against the stock model.
#
#   ./cross_validate.sh <work dir> <SenseVoiceSmall dir> <stock model dir> <out dir> [train.py options]
#
#   MIX=<weight>    pull each model back towards the original before exporting (mix.py); default 1
#   FOLDS="0 1 2"   only these folds (default: all five)
#   PYTHON=<path>   the interpreter with the packages of README.md (default: python3)
#
# A fold that is already transcribed is skipped, so an interrupted run can be started again;
# <out dir>/fingerprint remembers the labels, the scripts, the options and the files they name,
# and a run with anything else in the same directory is refused rather than mixed in.
# The weights and the unquantized graph of a fold are deleted once it is transcribed (2 GB each).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
work=$1 model=$2 stock=$3 out=$4
shift 4
python=${PYTHON:-python3}
mix=${MIX:-1}
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$out"

# everything a fold's result depends on, files named by the options included
fingerprint=$("$python" - "$work/labels.jsonl" "$model/model.pt" "$mix" "$here" "$@" <<'PY'
import hashlib, os, sys
h = hashlib.sha256()
def add(path):
    h.update(path.encode())
    if os.path.getsize(path) > 64 << 20:  # the checkpoint: its size and date will do
        h.update(f"{os.path.getsize(path)} {os.path.getmtime(path)}".encode())
    else:
        with open(path, "rb") as f:
            h.update(f.read())
labels, checkpoint, mix, here, *options = sys.argv[1:]
add(labels), add(checkpoint)
for name in sorted(os.listdir(here)):
    if name.endswith((".py", ".sh")):
        add(os.path.join(here, name))
h.update(repr((mix, options)).encode())
for o in options:
    if os.path.isfile(o):
        add(o)
print(h.hexdigest())
PY
)
if [[ -f $out/fingerprint && $(cat "$out/fingerprint") != "$fingerprint" ]]; then
    echo "$out holds folds of another run (other labels, scripts or options): use a new directory" >&2
    exit 1
fi
echo "$fingerprint" > "$out/fingerprint"

files=
for fold in ${FOLDS:-0 1 2 3 4}; do
    dir=$out/fold$fold
    if [[ ! -f $dir/holdout.json ]]; then
        mkdir -p "$dir"
        rm -f "$dir/model.pt"  # of a run that was interrupted: never export what this run did not train
        set +e
        "$python" -u "$here/train.py" "$work" "$model" "$dir" --holdout "$fold" --save "$@" 2>&1 |
            tee "$dir.log" | grep -E "^(train|training|the replay|epoch|  eval|saved)"
        status=${PIPESTATUS[0]}
        set -e
        if [[ $status != 0 ]]; then
            echo "training fold $fold failed, see $dir.log" >&2
            exit "$status"
        fi
        "$python" "$here/export_onnx.py" "$model" "$dir/model.pt" "$dir/onnx" "$mix" >"$dir.export.log" 2>&1
        "$python" "$here/transcribe.py" "$work/labels.jsonl" "$dir/holdout.tmp.json" --fold "$fold" --onnx "$dir/onnx"
        rm -f "$dir/model.pt" "$dir/onnx/model.onnx" "$dir/onnx/model.int8.onnx"
        mv "$dir/holdout.tmp.json" "$dir/holdout.json"
    fi
    files+=${files:+,}$dir/holdout.json
done

[[ -f $work/hyp_sensevoice.json ]] || "$python" "$here/hypotheses.py" "$work" "$stock"
# with FOLDS only those folds were transcribed, on purpose
"$python" "$here/score.py" ${FOLDS:+--common} "$work" "stock=$work/hyp_sensevoice.json" "fine-tuned=$files"
