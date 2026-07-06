# MechVQA SFT Training

This directory contains the public LLaMA Factory training recipe used for
MechVQA-style supervised fine-tuning. The `LLaMA-Factory/` subdirectory is a
vendored snapshot of the LLaMA Factory codebase used in our local training run,
with one public MechVQA 4B training config kept under its `examples/` directory.

## Files

- `LLaMA-Factory/`: vendored LLaMA Factory training code snapshot. Its
  `examples/train_full/` directory includes one MechVQA Qwen3-VL 4B training
  config for the public example data.
- `dataset_info.json`: minimal standalone LLaMA Factory dataset registry for
  `mechvqa_sft_examples_zh_en_20`.
- `prepare_llamafactory_dataset.py`: validates train/val JSONL files and writes
  a self-contained LLaMA Factory `dataset_info.json`.
- `LLaMA-Factory/data/mechvqa_sft_examples/`: 20 public SFT examples sampled
  from the training set, with 10 Chinese and 10 English records.
- `qwen3_vl_sft_4b_full_finetune.yaml`: 4B full fine-tuning recipe.
- `run_sft.sh`: renders the selected standalone config with environment
  variables and launches the vendored `LLaMA-Factory`.

## Data Schema

Training data uses LLaMA Factory `sharegpt` formatting:

```json
{
  "messages": [
    {"role": "user", "content": "<image>...question..."},
    {"role": "assistant", "content": "...answer..."}
  ],
  "images": ["images/example.png"],
  "metadata": {
    "difficulty": "Hard",
    "capability": "Reasoning",
    "subcategory": "Structure Understanding",
    "language": "英文"
  }
}
```

`images` may contain absolute paths or paths relative to `MEDIA_DIR`. For
relative paths, place images under the media directory and pass that directory
to `run_sft.sh` through `MEDIA_DIR`.

## Example Training Data

The vendored LLaMA Factory snapshot includes a small public example dataset:

```text
training/LLaMA-Factory/data/mechvqa_sft_examples/
  mechvqa_sft_examples_zh_en_20.jsonl
  images/
```

It contains 20 records sampled from the SFT training data: 10 Chinese and 10
English examples. Each language covers all 10 VQA subcategories, all three
capabilities, and all three difficulty levels. The dataset is registered in
`training/LLaMA-Factory/data/dataset_info.json` as:

```yaml
dataset: mechvqa_sft_examples_zh_en_20
```

## Run With The Vendored Snapshot

Install the vendored LLaMA Factory snapshot, then run from the root of this
repository:

```bash
cd training/LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ../..

export MODEL_NAME_OR_PATH=/path/or/hf-id/to/qwen3-vl-4b-instruct
export DATASET_DIR=$PWD/training/LLaMA-Factory/data
export MEDIA_DIR=$PWD/training/LLaMA-Factory/data
export OUTPUT_DIR=/path/to/outputs/mechvqa_qwen3_vl_4b_full

bash training/run_sft.sh training/qwen3_vl_sft_4b_full_finetune.yaml
```

To inspect the rendered config without launching training:

```bash
DRY_RUN=1 bash training/run_sft.sh training/qwen3_vl_sft_4b_full_finetune.yaml
```

You can also run the copied MechVQA config directly from the vendored
LLaMA Factory root:

```bash
cd training/LLaMA-Factory
llamafactory-cli train examples/train_full/qwen3_vl_full_sft_mechvqa_examples_4b.yaml
```

The vendored LLaMA Factory `data/dataset_info.json` intentionally contains only
the public example dataset registration.

## Recipe Notes

The full fine-tuning config mirrors the training settings used in our local
LLaMA Factory runs:

- Qwen3-VL chat template.
- `image_max_pixels: 262144`.
- vision tower and multimodal projector frozen.
- language model trainable.
- bf16 training.
- DeepSpeed ZeRO-3 for full fine-tuning.
- cosine scheduler, 3 epochs, warmup ratio 0.1.
