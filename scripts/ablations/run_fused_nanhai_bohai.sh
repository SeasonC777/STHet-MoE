#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/ablations/fused_${RUN_ID}"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1

SCRIPTS=(
    "scripts/ablations/train_nanhai_fused.sh"
    "scripts/ablations/train_bohai_fused.sh"
)

echo "Running fused dynamic+static graph experiments"
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

echo "Fused experiments completed."
echo "Logs saved under: ${LOG_DIR}"
