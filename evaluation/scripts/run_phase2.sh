#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/vqa_eval.example.json}"

cd "$(dirname "$0")/.."
python -m mechvqa_eval.evaluate_vqa --config "${CONFIG}" --phase phase2
