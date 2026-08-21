# MechVQA: Benchmarking and Enhancing Multimodal LLMs on Comprehensive Mechanical Drawing Understanding

[![Conference](https://img.shields.io/badge/ICML-2026-4B8BB2.svg)](https://icml.cc/virtual/2026/poster/66437)
[![Paper](https://img.shields.io/badge/arXiv-2605.30794-b31b1b.svg)](https://arxiv.org/abs/2605.30794)
[![HuggingFace](https://img.shields.io/badge/🤗%20HF-Models%20%26%20Paper-ffbd21.svg)](https://huggingface.co/collections/XiaofengAlg/mechvqa)
[![ModelScope](https://img.shields.io/badge/ModelScope-Collection-6b31e3.svg)](https://modelscope.cn/collections/xiaofengalg/MechVQA)
[![Cite](https://img.shields.io/badge/Cite-CITATION.cff-0f766e.svg)](CITATION.cff)


> **Official code repository** for the ICML 2026 paper *"MechVQA: Benchmarking and Enhancing Multimodal LLMs on Comprehensive Mechanical Drawing Understanding"*.

> If you use MechVQA, MechVL, or the released training and evaluation assets,
> please cite the [paper](https://arxiv.org/abs/2605.30794). Machine-readable
> metadata are available in [CITATION.cff](CITATION.cff).

> 🚧 **Status:** This repository is under **active development**. Inference code, the RL training framework, the public evaluation benchmark, the evaluation pipeline, the VQA-free data-generation pipeline, a compact SFT recipe, and a validated public VQA-only SFT train/validation release are available now. Model checkpoints and training data are released through ModelScope/HuggingFace links; additional internal training data are not included in this repository.

---

## 📖 Introduction

Mechanical engineering drawings encode semantics through a compact, standardized graphical language — orthographic multi-view projections, dense dimensioning, section views, symbolic notations, and structured text. General Multimodal Large Language Models (MLLMs) remain **brittle** on them: high annotation density and weak domain priors, combined with unreliable spatial-relation reasoning under strict projection rules, make decisive cues easy to miss.

**MechVQA** bridges this gap with two contributions:

- 📊 **MechVQA benchmark** — the first comprehensive mechanical-drawing understanding dataset, built via a semi-automated construction and quality-control pipeline: **3.3K high-density drawings** with **21K question–answer pairs**, **10 fine-grained tasks** across three capability levels — **Recognition**, **Reasoning**, and **Judging**.
- 🤖 **MechVL model** — a strong domain-specialized baseline built via a **multi-stage training paradigm** (SFT → two-stage self-play RL), reaching a **Total score of 84.85** and outperforming the strongest closed-source MLLMs on MechVQA.

## 🏆 Highlights

| Model | Recognition | Reasoning | Judging | **Total** |
|---|:---:|:---:|:---:|:---:|
| GPT-5 | 69.77 | 84.99 | 71.02 | 75.44 |
| Gemini-3-Pro-Preview | 76.74 | 87.74 | 77.28 | 77.28 |
| GLM-4.6V (best closed-source) | 88.37 | 86.68 | 78.91 | 78.91 |
| MechVL-4B-SFT (Ours) | 88.37 | 85.20 | 76.36 | 76.36 |
| **MechVL-4B-RL (Ours)** | **88.37** | **90.70** | **84.85** | **84.85** |

- **MechVL-4B-RL** achieves the best Total score (84.85), surpassing the strongest closed-source model (GLM-4.6V, 78.91) and all open-source MLLMs.
- On the **hard** subset, MechVL-4B-RL reaches **75%**, beating the best closed-source model (Qwen3-VL-Plus, 66%) by **+9 points**.
- Ablations confirm **DAPO > GRPO > GSPO**, the value of **two-stage self-play RL** (81.95 → 84.85), and the necessity of all three reward terms.

> See [§6 of the paper](https://arxiv.org/abs/2605.30794) for full tables and the 10 subtask definitions.

## 📰 Release Status

| Component | Status |
|---|---|
| Inference scripts (SFT & RL, dual-mode) | ✅ Ready |
| Self-contained example samples (10 QA + drawings) | ✅ Ready |
| RL training framework (`EasyR1/`, sanitized) | ✅ Ready |
| RL format prompt & reward functions | ✅ Ready |
| MechVL-4B-SFT / -RL checkpoints | ✅ Released (ModelScope full weights; HF mirroring in progress) |
| Public MechVQA evaluation benchmark (1,185 QA + drawings) | ✅ Ready |
| Evaluation script & metrics | ✅ Ready |
| VQA-free data-generation pipeline | ✅ Ready |
| SFT recipe (LLaMA Factory 4B config + 20 examples) | ✅ Ready |
| Public VQA-only SFT train/val release (13,515 QA + 3,371 images) | ✅ Released ([Hugging Face](https://huggingface.co/datasets/XiaofengAlg/MechVQA) · [ModelScope](https://modelscope.cn/datasets/xiaofengalg/MechVQA)) |
| Additional internal training data | Not included |

## 🗂️ Repository Structure

```
MechVQA/
├── ckpt/                # MechVL checkpoints (SFT & RL) — download separately, gitignored
├── scripts/
│   ├── batch_infer.py   # Inference entry: SFT/RL dual-mode (toggle MODE at top)
│   ├── modelscope_to_hf_dataset.py  # ModelScope-to-HF dataset mirror utility
│   └── README.md        # Inference usage (environment, params, outputs)
├── data/                # Built-in example samples (10 QA + 10 drawings)
├── benchmark_data/      # Public evaluation benchmark JSONL + packaged drawings
├── evaluation/          # OpenAI-compatible VQA evaluation pipeline
├── data_generation/     # Extract + VQA-free generation + QC + split scripts
├── training/            # LLaMA Factory SFT recipe, 4B config, public examples
├── prompts/
│   └── mech_r1.jinja    # RL format prompt (<think>/<answer> schema)
├── EasyR1/              # RL training framework (verl-based; GRPO/GSPO/DAPO/CISPO)
│   ├── verl/            # Core RL framework
│   ├── examples/        # mech_qwen3_vl_4b_*.sh training scripts + reward_function/ + format_prompt/
│   ├── scripts/         # Utilities (ray cluster, judge server, dataset download, model merge)
│   └── tests/ docs/
└── paper/               # Paper PDF
```

## 🔧 Environment

**Inference** (tested):

| Dependency | Version |
|---|---|
| Python | 3.10+ |
| vLLM | 0.11.0 (native `qwen3_vl` support) |
| transformers | 4.57.1 |
| torch | 2.8.0+ (CUDA 12.x) |
| Pillow / jinja2 / tqdm | — |

```bash
pip install "vllm>=0.11" "transformers>=4.57.1" pillow jinja2 tqdm
```

**Training** (EasyR1 / RL): see [`EasyR1/requirements.txt`](./EasyR1/requirements.txt) or use the provided [`EasyR1/Dockerfile`](./EasyR1/Dockerfile).

**Evaluation**: see [`evaluation/requirements.txt`](./evaluation/requirements.txt).

**Data generation**: see [`data_generation/requirements.txt`](./data_generation/requirements.txt).

**SFT recipe**: see [`training/README.md`](./training/README.md).

## 🧠 Model Checkpoints

Place checkpoints under `ckpt/` (gitignored due to size):

```
ckpt/
├── MechVQA_SFT/    # MechVL-4B-SFT  (Qwen3-VL-4B-Instruct, full-param SFT)
└── MechVQA_RL/     # MechVL-4B-RL   (DAPO two-stage self-play on top of SFT)
```

> 🤗 **HuggingFace** (checkpoints + paper): [MechVQA Collection](https://huggingface.co/collections/XiaofengAlg/mechvqa)

> 🟣 **ModelScope** (full weights): [MechVQA Collection](https://modelscope.cn/collections/xiaofengalg/MechVQA)

**Download:**

| Model | HuggingFace | ModelScope (recommended, full weights) |
|---|---|---|
| MechVL-4B-SFT | [XiaofengAlg/MechVL-4B-SFT](https://huggingface.co/XiaofengAlg/MechVL-4B-SFT) | [xiaofengalg/MechVL-4B-SFT](https://modelscope.cn/models/xiaofengalg/MechVL-4B-SFT) |
| MechVL-4B-RL | [XiaofengAlg/MechVL-4B-RL](https://huggingface.co/XiaofengAlg/MechVL-4B-RL) | [xiaofengalg/MechVL-4B-RL](https://modelscope.cn/models/xiaofengalg/MechVL-4B-RL) |

> ModelScope repos contain the full checkpoints (all weights). HuggingFace repos currently hold configs + model card; the large weight files are being mirrored (HF mirror upload is bandwidth-limited) — use ModelScope for immediate access to the weights.

## 🚀 Quick Start: Inference

`scripts/batch_infer.py` runs vLLM inference for **both** SFT and RL models — toggle `MODE` at the top of the file:

| `MODE` | Model | Prompt | Output |
|---|---|---|---|
| `sft` | `ckpt/MechVQA_SFT` | system prompt + image + question | free-form answer |
| `rl`  | `ckpt/MechVQA_RL`  | `prompts/mech_r1.jinja` rendered (no system) | `<think>...</think><answer>...</answer>`, `<answer>` extracted |

From the **repository root**:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/batch_infer.py
```

- Runs on the built-in `data/test_samples.jsonl` (10 QA) by default — fully self-contained, no external data needed.
- Outputs `outputs/{sft,rl}_infer_test.jsonl` with **GT vs prediction side-by-side**.
- Supports **resumable** runs (skips already-succeeded samples), bad-image tolerance, incremental JSONL writing.
- See [`scripts/README.md`](./scripts/README.md) for all configuration knobs.

## 📦 Public Benchmark Data

The public evaluation benchmark is packaged under [`benchmark_data/`](./benchmark_data/):

```text
benchmark_data/
├── images/
└── vqa_benchmark/
    └── mechvqa_benchmark.jsonl
```

- `mechvqa_benchmark.jsonl` contains 1,185 QA records and 562 packaged drawing images.
- Image paths in each JSONL record are relative to `benchmark_data/`.
- Each record follows the public message schema:

```json
{
  "messages": [
    {"role": "user", "content": "question text"},
    {"role": "assistant", "content": "reference answer"}
  ],
  "images": ["images/example.png"],
  "metadata": {
    "capability": "Reasoning",
    "subcategory": "Assembly Relationship",
    "difficulty": "Hard",
    "language": "中文"
  },
  "qualityscore": 1.0
}
```

Taxonomy labels in `capability`, `subcategory`, and `difficulty` use the English labels reported in the paper.

## 📦 Public SFT Training Data

The validated, VQA-only SFT train/validation release is hosted on both
Hugging Face and ModelScope:

| Platform | Dataset |
|---|---|
| Hugging Face | [XiaofengAlg/MechVQA](https://huggingface.co/datasets/XiaofengAlg/MechVQA) |
| ModelScope | [xiaofengalg/MechVQA](https://modelscope.cn/datasets/xiaofengalg/MechVQA) |

The release contains 12,749 training records, 766 validation records, and
3,371 content-addressed images. It also includes the SHA-256 checksum manifest,
release manifest, provenance records, and validation audit. The public package
is scoped to VQA SFT data; additional internal training artifacts are not part
of this release.

## 🏋️ Training (MechVL)

MechVL is trained in a **multi-stage paradigm**:

1. **SFT stage** — Initialize from `Qwen3-VL-Instruct-4B`, full-parameter SFT on the LLM module (vision encoder & projection frozen) over the MechVQA training split. Produces `MechVL-4B-SFT` (the reference policy π_ref).
2. **RL stage — DAPO two-stage self-play**:
   - **Stage 2a (full)**: DAPO on the full training split.
   - **Stage 2b (targeted)**: DAPO on a re-sampled subset with an increased proportion of underperforming subtasks.
   - **Reward** = Accuracy (LLM-as-a-Judge, semantic equivalence in [0,1]) + Format (binary, well-formed `<think>/<answer>`) + Quality (Logic / Professionalism / Conciseness, LLM-judge).

The RL pipeline is built on the included `EasyR1/` (a verl-based framework). Example training scripts: `EasyR1/examples/mech_qwen3_vl_4b_*.sh` (covering GRPO / GSPO / DAPO / CISPO and the round-2 reward variants). Reward functions live in `EasyR1/examples/reward_function/mech*.py`.

The public SFT recipe is included under [`training/`](./training/). It contains a vendored LLaMA Factory snapshot, one Qwen3-VL 4B full-parameter SFT config, and a compact 20-record example dataset for format verification:

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

See [`training/README.md`](./training/README.md) for the dataset registry and
dry-run command. For the complete public VQA-only train/validation package,
download [XiaofengAlg/MechVQA from Hugging Face](https://huggingface.co/datasets/XiaofengAlg/MechVQA)
or [xiaofengalg/MechVQA from ModelScope](https://modelscope.cn/datasets/xiaofengalg/MechVQA).

## 📊 Evaluation

MechVQA evaluates MLLMs across **10 fine-grained tasks** grouped into three capability levels (Recognition / Reasoning / Judging), reported as per-level means and an overall **Total** score. See [§3 and §6 of the paper](https://arxiv.org/abs/2605.30794) for the task taxonomy, metrics, and full results.

The open-source evaluator is included under [`evaluation/`](./evaluation/). It runs target-model inference, judges responses with an OpenAI-compatible judge model, and reports aggregate and metadata-level metrics.

```bash
cd evaluation
pip install -r requirements.txt
cp configs/vqa_eval.example.json configs/vqa_eval.local.json
# Edit input_file, image_root, model names, API keys, and base URLs.
MAX_SAMPLES=5 bash scripts/run_all.sh configs/vqa_eval.local.json
```

See [`evaluation/README.md`](./evaluation/README.md) for the full two-phase workflow.

## 🏭 Data Generation

The public data-generation pipeline is included under [`data_generation/`](./data_generation/). It covers the open-source portion of the paper pipeline: drawing metadata extraction, VQA-free question generation from extracted metadata, model-based question checking, multi-answer generation with semantic voting, message-format conversion, difficulty assignment, and dataset splitting.

```bash
pip install -r data_generation/requirements.txt
python data_generation/extract_pipeline.py --help
python data_generation/generate_vqa_free_query.py --help
```

Internal/manual routes such as expert source curation, human metadata correction, template GT construction, 2D/3D candidate pairing, and CAD expert edits are intentionally not included.

## 📝 Citation

If you find **MechVQA** or **MechVL** useful in your research, please ⭐ star this repository and cite our paper:

```bibtex
@misc{kou2026mechvqabenchmarkingenhancingmultimodal,
      title={MechVQA: Benchmarking and Enhancing Multimodal LLMs on Comprehensive Mechanical Drawing Understanding},
      author={Qian Kou and Xiaofeng Shi and Yulin Li and Xiaosong Qiu and Xinyang Wang and Hua Zhou and Cao Dongxing},
      year={2026},
      eprint={2605.30794},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.30794}
}
```

<details>
<summary>Plain-text citation</summary>

Qian Kou, Xiaofeng Shi, Yulin Li, Xiaosong Qiu, Xinyang Wang, Hua Zhou, and Cao Dongxing. 2026. *MechVQA: Benchmarking and Enhancing Multimodal LLMs on Comprehensive Mechanical Drawing Understanding.* arXiv:2605.30794. (Accepted to ICML 2026.)

</details>

<details>
<summary>APA 7th</summary>

Kou, Q., Shi, X., Li, Y., Qiu, X., Wang, X., Zhou, H., & Cao, D. (2026). *MechVQA: Benchmarking and enhancing multimodal LLMs on comprehensive mechanical drawing understanding.* arXiv. https://arxiv.org/abs/2605.30794

</details>

## 📄 License

This project is released under the **Apache 2.0 License**. The `EasyR1/` RL framework retains its own license (see [`EasyR1/LICENSE`](./EasyR1/LICENSE)).

## 🙏 Acknowledgements

MechVQA/MechVL are built on top of [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) and the [EasyR1/verl](https://github.com/volcengine/verl) RL framework. We thank their contributors.

## ✉️ Contact

- **Xiaofeng Shi** — <xfshi@baai.ac.cn>
- **Qian Kou** — <kouqian@baai.ac.cn>

Beijing Academy of Artificial Intelligence (BAAI) · Institute of Information Engineering, CAS · Beijing University of Technology.


## Star History

<a href="https://www.star-history.com/?repos=xiaofengShi%2FMechVQA&type=date&legend=bottom-right">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=xiaofengShi/MechVQA&type=date&theme=dark&legend=bottom-right&sealed_token=KkradenYsJrXA9Uyh_-J283fivVInUhGxPwGE-0CzTChMJhv4E2GtMOg7Z5mWZd9THmN22-l__LTTkaG1vfx3gmiEYEkbdeoAfp9XKEkRTN-OGsh7ePJHg" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=xiaofengShi/MechVQA&type=date&legend=bottom-right&sealed_token=KkradenYsJrXA9Uyh_-J283fivVInUhGxPwGE-0CzTChMJhv4E2GtMOg7Z5mWZd9THmN22-l__LTTkaG1vfx3gmiEYEkbdeoAfp9XKEkRTN-OGsh7ePJHg" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=xiaofengShi/MechVQA&type=date&legend=bottom-right&sealed_token=KkradenYsJrXA9Uyh_-J283fivVInUhGxPwGE-0CzTChMJhv4E2GtMOg7Z5mWZd9THmN22-l__LTTkaG1vfx3gmiEYEkbdeoAfp9XKEkRTN-OGsh7ePJHg" />
 </picture>
</a>
