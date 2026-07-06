# MechVQA Data Generation Pipeline

This directory contains the public data-generation pipeline that we currently
plan to open source:

1. Extract structured metadata from mechanical drawing images.
2. Generate VQA-free question prompts from the extracted metadata.
3. Check generated questions with a vision model.
4. Generate answers and keep questions whose answers pass consistency voting.
5. Convert verified QA pairs into message-format JSONL.
6. Assign difficulty labels.
7. Split the dataset into train/validation/test partitions.

Other internal generation routes are outside this public package for now.

## Scripts

- `extract_pipeline.py`: OCR + vision-language extraction of structured drawing metadata.
- `generate_vqa_free_query.py`: VQA-free question-prompt generation from extract metadata.
- `check_generated_questions.py`: question fact-check prompt construction, model calls, and result parsing.
- `generate_fixed_questions.py`: keep validated questions and rebuild answer-generation prompts.
- `generate_answers_for_vqa.py`: answer generation with repeated sampling and semantic voting.
- `format_verified_data.py`: convert verified QA pairs into `messages` / `images` / `metadata` JSONL.
- `original_vqa_assign_difficulty.py`: assign and apply difficulty labels.
- `split_dataset.py`: grouped and stratified train/val/test split with image/text embeddings.
- `model_client.py`: OpenAI-compatible request helper configured by environment variables.
- `taxonomy.py`: shared paper-aligned taxonomy normalization for capability, difficulty, and subcategory labels.

All public pipeline outputs use English taxonomy labels for `capability`
(`Recognition`, `Reasoning`, `Judging`), `difficulty` (`Easy`, `Medium`,
`Hard`), and the paper subcategories such as `Identification & Counting`,
`Item Localization`, and `Structure Understanding`. Legacy Chinese or older
internal labels are accepted as input aliases and normalized before writing new
outputs.

## Alignment With The Paper Pipeline

The public scripts cover the open-source portion of the MechVQA data pipeline:
metadata extraction, Source I VQA-free generation, quality control, answer
voting, difficulty assignment, and dataset splitting. Internal or manual
construction routes are intentionally excluded.

| Paper pipeline step | Public script coverage | Notes |
| --- | --- | --- |
| Expert filtering of raw drawing sources | Not included | This is an offline source-curation step. |
| OCR and raw metadata extraction | `extract_pipeline.py` | Uses `OCR_URL` plus an OpenAI-compatible vision model. Multi-run extraction voting is enabled by default. |
| Expert metadata verification | Not included | The paper's human verification process is offline. Public users can start from their own verified extract JSONL. |
| Source I VQA-free question generation from drawing + metadata | `generate_vqa_free_query.py generate` and `generate_vqa_free_query.py run` | The script first builds metadata-conditioned prompts, then runs a configurable generator model. Model names are supplied by `--model` or `DEFAULT_MODEL`; no private model list is hard-coded. |
| Question grounding, answerability, and subtask-consistency checking | `check_generated_questions.py` and `generate_fixed_questions.py` | Builds validator prompts, runs a configurable validator model, parses `verdict`, and keeps validated questions. |
| Multi-answer generation and semantic majority voting | `generate_answers_for_vqa.py` | Generates repeated candidate answers, uses a judge model for semantic consistency/voting, and drops questions without a clear consistent answer. |
| Final SFT/message JSONL schema | `format_verified_data.py` | Converts verified `qa_pairs` into `messages`, `images`, and `metadata`. |
| Difficulty assignment | `original_vqa_assign_difficulty.py` | Assigns difficulty from question, answer, image, capability, and subcategory, then applies it to `metadata.difficulty`. |
| Drawing-level 8:1:1 split with similarity-aware grouping | `split_dataset.py` | Keeps all QA pairs from the same drawing in one split and supports fused image/text embedding clusters plus metadata stratification. |
| Source II template non-GT generation | Not included | Explicitly outside this public package. |
| Source III template GT, 2D/3D pairing, and CAD expert edits | Not included | Explicitly outside this public package. |
| Final manual/expert audit | Not included | Manual QC is not part of the public code release. |

## Environment

```bash
export OPENAI_API_KEY=your-api-key
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
export DEFAULT_MODEL=your-vision-model
export JUDGE_MODEL=your-judge-model
export OCR_URL=https://your-ocr-service/extract
```

`JUDGE_MODEL` is optional for answer voting; when unset, the answer model is
also used as the judge. `OCR_URL` is required only for the extract stage.

Install dependencies:

```bash
pip install -r data_generation/requirements.txt
```

## 1. Extract Metadata

Run one image:

```bash
python data_generation/extract_pipeline.py single \
  --image /path/to/drawing.png \
  --output-json outputs/extract/page_001.json
```

Run a directory:

```bash
python data_generation/extract_pipeline.py dataset \
  --dataset-dir /path/to/drawing_images \
  --output-jsonl outputs/extract/params.jsonl \
  --max-workers 4
```

Retry failed records:

```bash
python data_generation/extract_pipeline.py retry \
  --input-jsonl outputs/extract/params.jsonl \
  --output-jsonl outputs/extract/params_retry.jsonl \
  --max-workers 4
```

By default, extraction uses multi-run voting. Add `--no-voting` for one model
call per image.

## 2. Generate VQA-Free Questions

Generate question prompts from the unified extract metadata:

```bash
# Chinese questions
python data_generation/generate_vqa_free_query.py generate \
  --metadata-jsonl outputs/extract/params.jsonl \
  --output-jsonl outputs/vqa_free/question_prompts.zh.jsonl \
  --sample-num 10 \
  --language 中文

# English questions
python data_generation/generate_vqa_free_query.py generate \
  --metadata-jsonl outputs/extract/params.jsonl \
  --output-jsonl outputs/vqa_free/question_prompts.en.jsonl \
  --sample-num 10 \
  --language 英文
```

Run the generator model and write responses back to the prompt records:

```bash
python data_generation/generate_vqa_free_query.py run \
  --input-jsonl outputs/vqa_free/question_prompts.zh.jsonl \
  --output-jsonl outputs/vqa_free/question_prompts_output.zh.jsonl \
  --workers 8
```

The `run` command fills each record's `response` field with the model response
containing the generated `问题列表`. It also writes `error` for failed records and
supports `--retry-errors` for a later retry pass.

## 3. Check Generated Questions

Build question-check prompts after generation responses have been written back:

```bash
python data_generation/check_generated_questions.py from-free-output \
  --input-jsonl outputs/vqa_free/question_prompts_output.zh.jsonl \
  --output-jsonl outputs/vqa_free/question_check_prompts.zh.jsonl
```

Run the question checks:

```bash
python data_generation/check_generated_questions.py run \
  --input-jsonl outputs/vqa_free/question_check_prompts.zh.jsonl \
  --output-jsonl outputs/vqa_free/question_check_outputs.zh.jsonl \
  --workers 8
```

Parse the check outputs:

```bash
python data_generation/check_generated_questions.py analyze \
  --input-jsonl outputs/vqa_free/question_check_outputs.zh.jsonl \
  --output-jsonl outputs/vqa_free/question_check_analyzed.zh.jsonl
```

Keep validated questions and rebuild answer prompts:

```bash
python data_generation/generate_fixed_questions.py \
  --input-jsonl outputs/vqa_free/question_check_analyzed.zh.jsonl \
  --output-jsonl outputs/vqa_free/validated_questions.zh.jsonl \
  --language 中文
```

## 4. Generate And Vote Answers

```bash
python data_generation/generate_answers_for_vqa.py \
  --input-jsonl outputs/vqa_free/validated_questions.zh.jsonl \
  --output-jsonl outputs/vqa_free/answers.zh.jsonl \
  --language 中文 \
  --num-workers 8
```

This script samples two answers first, compares them semantically, and samples a
third answer only when the first two disagree. Questions without a consistent
answer are dropped from `qa_pairs`.

## 5. Format Message JSONL

```bash
python data_generation/format_verified_data.py \
  --input-jsonl outputs/vqa_free/answers.zh.jsonl \
  --output-jsonl outputs/vqa_free/formatted.zh.jsonl \
  --language 中文 \
  --data-source vqa_free
```

For an English run, use the `.en.jsonl` files from the generation step and pass
`--language 英文` to `generate_fixed_questions.py`, `generate_answers_for_vqa.py`,
and `format_verified_data.py`. Output metadata stores language as `中文` or
`英文`.

## 6. Assign Difficulty

Call a model to assign difficulty labels:

```bash
python data_generation/original_vqa_assign_difficulty.py assign \
  --input-jsonl outputs/vqa_free/formatted.jsonl \
  --output-jsonl outputs/vqa_free/with_difficulty_raw.jsonl \
  --max-workers 8
```

Apply `metadata.assign_diff.difficulty` into `metadata.difficulty`:

```bash
python data_generation/original_vqa_assign_difficulty.py apply \
  --input-jsonl outputs/vqa_free/with_difficulty_raw.jsonl \
  --output-jsonl outputs/vqa_free/with_difficulty.jsonl
```

## 7. Split Dataset

The split script keeps QA items from the same image together and stratifies by
metadata fields and embedding clusters.

```bash
python data_generation/split_dataset.py \
  --input-jsonl outputs/vqa_free/with_difficulty.jsonl \
  --output-dir outputs/vqa_free/split \
  --model-name-or-path /path/to/chinese-clip-model \
  --inner-stratify-field capability \
  --inner-stratify-field difficulty
```

Outputs are written under the selected `--output-dir`.
