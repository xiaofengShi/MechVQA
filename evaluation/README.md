# MechVQA Evaluation

This folder contains the cleaned open-source evaluation code for MechVQA benchmarks.
It is derived from the internal `datasets2/eval_vqa` workflow, but removes hard-coded
absolute paths, private endpoints, API keys, logs, caches, and experiment result files.

## What Is Included

- `mechvqa_eval/evaluate_vqa.py`: two-stage VQA evaluation pipeline.
- `mechvqa_eval/providers.py`: OpenAI-compatible request wrapper with image support.
- `mechvqa_eval/summarize.py`: metadata breakdown by subcategory, difficulty, and language.
- `configs/vqa_eval.example.json`: config template for target and judge models.
- `scripts/run_phase1.sh`: generate target-model responses.
- `scripts/run_phase2.sh`: judge saved responses and produce stats.
- `scripts/run_all.sh`: run both phases.

## Input Format

The evaluator expects JSONL records with fields compatible with the project benchmark:

```json
{
  "messages": [
    {"role": "user", "content": "question text"},
    {"role": "assistant", "content": "reference answer"}
  ],
  "images": ["images/00/example.png"],
  "metadata": {
    "explanation": "optional explanation",
    "capability": "Reasoning",
    "subcategory": "Assembly Relationship",
    "difficulty": "Hard",
    "language": "中文"
  },
  "qualityscore": 1.0
}
```

The released benchmark uses relative image paths resolved against `image_root`.
The assistant message is the reference answer used by the evaluator. Legacy
`metadata.correct_answer`, `check_results.question`, and
`check_results.correct_answer` are accepted as fallbacks by the evaluator, but
they are not present in the public JSONL.

## Quick Start

Install dependencies:

```bash
cd evaluation
pip install -r requirements.txt
```

Prepare a config:

```bash
cp configs/vqa_eval.example.json configs/vqa_eval.local.json
```

Edit `configs/vqa_eval.local.json`:

- `input_file`: benchmark JSONL path.
- `image_root`: directory used to resolve relative image paths in the benchmark.
- `target_models`: model(s) being evaluated.
- `judge_models`: model(s) used for automatic judging.
- `api_key_env`: environment variable name containing the API key.
- `base_url`: OpenAI-compatible API endpoint.

Set API keys:

```bash
export VQA_TARGET_API_KEY=your-target-model-key
export OPENAI_API_KEY=your-judge-model-key
```

Run phase 1 only:

```bash
bash scripts/run_phase1.sh configs/vqa_eval.local.json
```

Run phase 2 only:

```bash
bash scripts/run_phase2.sh configs/vqa_eval.local.json
```

Run both phases:

```bash
bash scripts/run_all.sh configs/vqa_eval.local.json
```

Smoke test with a small subset:

```bash
MAX_SAMPLES=5 bash scripts/run_all.sh configs/vqa_eval.local.json
```

## Outputs

Phase 1 writes a response JSONL:

```json
{
  "record_idx": 0,
  "original_record": {},
  "responses": {
    "example-model": {
      "model_response": "<think>...</think><answer>...</answer>",
      "model_answer": "...",
      "response_error": null
    }
  }
}
```

Phase 2 writes:

- `evaluated_output`: per-record judge results and voted score.
- `stats_output`: aggregate score and full-score rate per target model.

Optional metadata breakdown:

```bash
python -m mechvqa_eval.summarize \
  --evaluated-file outputs/evaluated_example-model.jsonl \
  --target-model example-model \
  --output outputs/breakdown_example-model.json
```

## Notes For Release

- Keep benchmark data in `../benchmark_data`; generated outputs and logs should stay out of version control.
- Do not put API keys or private endpoints in config files committed to the repository.
- The request layer assumes OpenAI-compatible chat completion APIs.
