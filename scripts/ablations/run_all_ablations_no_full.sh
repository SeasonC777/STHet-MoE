#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/ablations/run_${RUN_ID}"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1

SCRIPTS=(
    "scripts/ablations/train_nanhai_no_moe.sh"
    "scripts/ablations/train_nanhai_no_graph_conv.sh"
    "scripts/ablations/train_nanhai_no_multiscale_tcn.sh"
    "scripts/ablations/train_nanhai_no_tcn.sh"
    "scripts/ablations/train_nanhai_wodg_static.sh"
    "scripts/ablations/train_bohai_no_moe.sh"
    "scripts/ablations/train_bohai_no_graph_conv.sh"
    "scripts/ablations/train_bohai_no_multiscale_tcn.sh"
    "scripts/ablations/train_bohai_no_tcn.sh"
    "scripts/ablations/train_bohai_wodg_static.sh"
)

echo "Running ablation scripts without full baselines"
echo "Log directory: ${LOG_DIR}"
echo

for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "Missing script: ${script}" >&2
        exit 1
    fi
done

for script in "${SCRIPTS[@]}"; do
    name="$(basename "$script" .sh)"
    log_file="${LOG_DIR}/${name}.log"

    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${script}"
    echo "Log: ${log_file}"
    echo "============================================================"

    bash "$script" 2>&1 | tee "$log_file"

    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE ${script}"
    echo "============================================================"
    echo
done

echo "All ablation scripts completed."
echo "Logs saved under: ${LOG_DIR}"
