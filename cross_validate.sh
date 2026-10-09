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
# A fold that is already transcribed is skipped, so an interrupted run can be started again.
# The weights and the unquantized graph of a fold are deleted once it is transcribed (2 GB each).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
work=$1 model=$2 stock=$3 out=$4
shift 4
python=${PYTHON:-python3}
mix=${MIX:-1}
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$out"

files=
for fold in ${FOLDS:-0 1 2 3 4}; do
    dir=$out/fold$fold
    if [[ ! -f $dir/holdout.json ]]; then
        "$python" -u "$here/train.py" "$work" "$model" "$dir" --holdout "$fold" --save "$@" 2>&1 |
            tee "$dir.log" | grep -E "^(train|training|epoch|  eval|saved)" || true
        "$python" "$here/export_onnx.py" "$model" "$dir/model.pt" "$dir/onnx" "$mix" >/dev/null 2>&1
        "$python" "$here/transcribe.py" "$work/labels.jsonl" "$dir/holdout.json" --fold "$fold" --onnx "$dir/onnx"
        rm -f "$dir/model.pt" "$dir/onnx/model.onnx" "$dir/onnx/model.int8.onnx"
    fi
    files+=${files:+,}$dir/holdout.json
done

[[ -f $work/hyp_sensevoice.json ]] || "$python" "$here/hypotheses.py" "$work" "$stock"
"$python" "$here/score.py" "$work" "stock=$work/hyp_sensevoice.json" "fine-tuned=$files"
