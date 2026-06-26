# MechVQA: Benchmarking and Enhancing Multimodal LLMs on Comprehensive Mechanical Drawing Understanding

[![Paper](https://img.shields.io/badge/arXiv-2605.30794-b31b1b.svg)](https://arxiv.org/abs/2605.30794)
[![Conference](https://img.shields.io/badge/ICML-2026-4B8BB2.svg)](https://icml.cc/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HF-Models%20%26%20Paper-ffbd21.svg)](https://huggingface.co/collections/MonteXiaofeng/mechvqa)
[![ModelScope](https://img.shields.io/badge/ModelScope-Collection-6b31e3.svg)](https://modelscope.cn/collections/xiaofengalg/MechVQA)


> **Official code repository** for the ICML 2026 paper *"MechVQA: Benchmarking and Enhancing Multimodal LLMs on Comprehensive Mechanical Drawing Understanding"*.

> 🚧 **Status:** This repository is under **active development**. Inference code and the RL training framework are ready now. We are progressively open-sourcing the model checkpoints, the full **MechVQA** benchmark, the complete SFT recipe, and the evaluation pipeline. Please watch/star the repo for updates.

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
| MechVL-4B-SFT / -RL checkpoints | 🚧 Releasing soon (HF / ModelScope) |
| Full MechVQA benchmark data (21K QA) | 🚧 Releasing soon |
| Complete SFT training recipe | 🚧 Releasing soon |
| Evaluation script & metrics | 🚧 Releasing soon |

## 🗂️ Repository Structure

```
MechVQA/
├── ckpt/                # MechVL checkpoints (SFT & RL) — download separately, gitignored
├── scripts/
│   ├── batch_infer.py   # Inference entry: SFT/RL dual-mode (toggle MODE at top)
│   └── README.md        # Inference usage (environment, params, outputs)
├── data/                # Built-in example samples (10 QA + 10 drawings)
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

## 🧠 Model Checkpoints

Place checkpoints under `ckpt/` (gitignored due to size):

```
ckpt/
├── MechVQA_SFT/    # MechVL-4B-SFT  (Qwen3-VL-4B-Instruct, full-param SFT)
└── MechVQA_RL/     # MechVL-4B-RL   (DAPO two-stage self-play on top of SFT)
```

> 🤗 **HuggingFace** (checkpoints + paper): [MechVQA Collection](https://huggingface.co/collections/MonteXiaofeng/mechvqa)

> 🟣 **ModelScope** (full weights): [MechVQA Collection](https://modelscope.cn/collections/xiaofengalg/MechVQA)

**Download:**

| Model | HuggingFace | ModelScope (recommended, full weights) |
|---|---|---|
| MechVL-4B-SFT | [MonteXiaofeng/MechVL-4B-SFT](https://huggingface.co/MonteXiaofeng/MechVL-4B-SFT) | [xiaofengalg/MechVL-4B-SFT](https://modelscope.cn/models/xiaofengalg/MechVL-4B-SFT) |
| MechVL-4B-RL | [MonteXiaofeng/MechVL-4B-RL](https://huggingface.co/MonteXiaofeng/MechVL-4B-RL) | [xiaofengalg/MechVL-4B-RL](https://modelscope.cn/models/xiaofengalg/MechVL-4B-RL) |

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

## 🏋️ Training (MechVL)

MechVL is trained in a **multi-stage paradigm**:

1. **SFT stage** — Initialize from `Qwen3-VL-Instruct-4B`, full-parameter SFT on the LLM module (vision encoder & projection frozen) over the MechVQA training split. Produces `MechVL-4B-SFT` (the reference policy π_ref).
2. **RL stage — DAPO two-stage self-play**:
   - **Stage 2a (full)**: DAPO on the full training split.
   - **Stage 2b (targeted)**: DAPO on a re-sampled subset with an increased proportion of underperforming subtasks.
   - **Reward** = Accuracy (LLM-as-a-Judge, semantic equivalence in [0,1]) + Format (binary, well-formed `<think>/<answer>`) + Quality (Logic / Professionalism / Conciseness, LLM-judge).

The RL pipeline is built on the included `EasyR1/` (a verl-based framework). Example training scripts: `EasyR1/examples/mech_qwen3_vl_4b_*.sh` (covering GRPO / GSPO / DAPO / CISPO and the round-2 reward variants). Reward functions live in `EasyR1/examples/reward_function/mech*.py`.

> 🚧 The complete, reproducible SFT recipe and turn-key training configs are being prepared.

## 📊 Evaluation

MechVQA evaluates MLLMs across **10 fine-grained tasks** grouped into three capability levels (Recognition / Reasoning / Judging), reported as per-level means and an overall **Total** score. See [§3 and §6 of the paper](https://arxiv.org/abs/2605.30794) for the task taxonomy, metrics, and full results.

> 🚧 The standalone evaluation script and the full benchmark download will be released soon. Meanwhile, `scripts/batch_infer.py` can produce model predictions over any MechVQA-format JSONL for manual inspection.

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
