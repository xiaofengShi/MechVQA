#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  MODEL_NAME_OR_PATH=/path/or/hf-id/to/qwen3-vl \
  DATASET_DIR=/path/to/llamafactory/data/mechvqa_sft \
  OUTPUT_DIR=/path/to/output \
  bash training/run_sft.sh training/qwen3_vl_sft_4b_full_finetune.yaml

Required environment variables:
  MODEL_NAME_OR_PATH     Base Qwen3-VL model path or Hugging Face id.
  DATASET_DIR            Directory containing dataset_info.json and JSONL files.
  OUTPUT_DIR             Checkpoint output directory.

Optional environment variables:
  LLAMAFACTORY_DIR       Defaults to training/LLaMA-Factory.
  MEDIA_DIR              Image/media root. Defaults to DATASET_DIR.
  DEEPSPEED_CONFIG       Defaults to $LLAMAFACTORY_DIR/examples/deepspeed/ds_z3_config.json.
  DRY_RUN=1              Render the runtime config without launching training.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CONFIG_TEMPLATE="${1:-training/qwen3_vl_sft_4b_full_finetune.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-${SCRIPT_DIR}/LLaMA-Factory}"

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to your base model path or Hugging Face id.}"
: "${DATASET_DIR:?Set DATASET_DIR to the prepared LLaMA Factory dataset directory.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the checkpoint output directory.}"

MEDIA_DIR="${MEDIA_DIR:-${DATASET_DIR}}"

if [[ ! -f "${CONFIG_TEMPLATE}" ]]; then
  echo "Config template not found: ${CONFIG_TEMPLATE}" >&2
  exit 1
fi
if [[ ! -d "${LLAMAFACTORY_DIR}" ]]; then
  echo "LLaMA Factory directory not found: ${LLAMAFACTORY_DIR}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_DIR}/dataset_info.json" ]]; then
  echo "dataset_info.json not found under DATASET_DIR: ${DATASET_DIR}" >&2
  exit 1
fi

if grep -q "__DEEPSPEED_CONFIG__" "${CONFIG_TEMPLATE}"; then
  DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${LLAMAFACTORY_DIR}/examples/deepspeed/ds_z3_config.json}"
  if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
    echo "DeepSpeed config not found: ${DEEPSPEED_CONFIG}" >&2
    exit 1
  fi
else
  DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
fi
export MODEL_NAME_OR_PATH DATASET_DIR MEDIA_DIR OUTPUT_DIR DEEPSPEED_CONFIG

mkdir -p "${OUTPUT_DIR}"
RUNTIME_CONFIG="${OUTPUT_DIR}/$(basename "${CONFIG_TEMPLATE%.yaml}").runtime.yaml"

python - "${CONFIG_TEMPLATE}" "${RUNTIME_CONFIG}" <<'PY'
import os
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text(encoding="utf-8")
replacements = {
    "__MODEL_NAME_OR_PATH__": os.environ["MODEL_NAME_OR_PATH"],
    "__DATASET_DIR__": os.environ["DATASET_DIR"],
    "__MEDIA_DIR__": os.environ["MEDIA_DIR"],
    "__OUTPUT_DIR__": os.environ["OUTPUT_DIR"],
    "__DEEPSPEED_CONFIG__": os.environ["DEEPSPEED_CONFIG"],
}
for old, new in replacements.items():
    text = text.replace(old, new)
dst.write_text(text, encoding="utf-8")
print(dst)
PY

echo "Runtime config: ${RUNTIME_CONFIG}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

cd "${LLAMAFACTORY_DIR}"
llamafactory-cli train "${RUNTIME_CONFIG}"
