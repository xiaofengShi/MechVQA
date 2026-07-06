#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/vqa_eval.example.json}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

cd "$(dirname "$0")/.."

if [[ -n "${MAX_SAMPLES}" ]]; then
  python -m mechvqa_eval.evaluate_vqa --config "${CONFIG}" --phase phase1 --max-samples "${MAX_SAMPLES}"
else
  python -m mechvqa_eval.evaluate_vqa --config "${CONFIG}" --phase phase1
fi
