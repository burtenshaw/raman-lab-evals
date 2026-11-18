#!/usr/bin/env bash
# Launch one HF Job per model. Each job runs the original task and the 8 adapted tasks with K rollouts
# and uploads the raw Harbor job folders to RESULTS_REPO under runs/<RUN_DATE>/<model slug>/.
#
#   RESULTS_REPO=<you>/raman-lab-runs ./scripts/launch.sh
set -euo pipefail
cd "$(dirname "$0")"

RESULTS_REPO=${RESULTS_REPO:?set RESULTS_REPO to a dataset repo you can write to}
NAMESPACE=${NAMESPACE:-openenv}       # org that pays for the job and its sandboxes
RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
K=${K:-3}

MODELS=(
  "openai/gpt-oss-120b:deepinfra"
  "Qwen/Qwen3.8-27B:deepinfra"
  "deepseek-ai/DeepSeek-V4.1-Flash:deepinfra"
  "moonshotai/Kimi-K3:deepinfra"
  "MiniMaxAI/MiniMax-M3.1-Flash:together"
  "google/gemma-4-31B-it:deepinfra"
)

SCRIPT_B64=$(python3 - <<'EOF'
import base64, pathlib
tpl = pathlib.Path("hf_job_models.py").read_text()
src = tpl.replace("__BUILD_DATASET__", pathlib.Path("build_dataset.py").read_text())
print(base64.b64encode(src.encode()).decode())
EOF
)

START=${START:-0}                     # index of the first model to launch (to resume a partial launch)
for i in "${!MODELS[@]}"; do
  [ "$i" -lt "$START" ] && continue
  model=${MODELS[$i]}
  slug=$(basename "${model%%:*}" | tr '[:upper:]' '[:lower:]')
  oracle=$([ "$i" -eq 0 ] && echo 1 || echo 0)   # run the oracle once, in the first job
  hf jobs run --detach --namespace "$NAMESPACE" --flavor cpu-basic --timeout 3h --secrets HF_TOKEN \
    -e MODEL="$model" -e ROLLOUTS="$K" -e ORACLE="$oracle" -e N_CONCURRENT=5 \
    -e RUN_TAG="$RUN_DATE/$slug" -e RESULTS_REPO="$RESULTS_REPO" \
    --name "raman-models-${slug//./-}" \
    ghcr.io/astral-sh/uv:python3.12-bookworm -- \
    /bin/sh -lc "echo $SCRIPT_B64 | base64 -d | uv run -"
done
