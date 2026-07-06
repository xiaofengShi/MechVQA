# Final VQA Benchmark Open-Release Check

Official dataset:

- `vqa_benchmark/mechvqa_benchmark.jsonl`

## Release Status

- Record count: 1,185
- Unique referenced images: 562
- Image references: 1,185
- Missing referenced images: 0
- Image paths: all relative to `benchmark_data/`
- JSONL schema: public fields only
- Excluded-source text references: 0
- Internal absolute workspace paths in benchmark data: 0
- Generated reasoning markers in public JSONL: 0
- Taxonomy labels for `metadata.difficulty`, `metadata.capability`, and
  `metadata.subcategory`: paper-aligned English labels only.

## Public Fields Kept

- `messages[0].content`: the prompt shown to evaluated models.
- `messages[1].content`: reference answer used by evaluation.
- `images`: packaged relative image paths.
- `metadata.explanation`: optional explanation, empty for 9 records.
- `metadata.difficulty`
- `metadata.capability`
- `metadata.subcategory`
- `metadata.language`
- `qualityscore`: normalized adjusted quality score in `[0, 1]`.

## Fields Removed From Public JSONL

- `check_results`
- `_lineno`
- `manual_repair`
- `metadata.question_type`
- `metadata.correct_answer`
- `metadata.original_q`
- `metadata.original_a`
- `metadata.acceptable_answers`
- Top-level `acceptable_answers`
- Top-level `explanation`
- Generated reasoning-prefix text in public reference answers

These fields were useful for internal auditing and repair, but they are not used
by the evaluation scripts and are not part of the public benchmark schema.

## Distribution

Language:

- Chinese: 850
- English: 335

Difficulty:

- Easy: 580
- Medium: 419
- Hard: 186

Capability:

- Recognition: 642
- Reasoning: 289
- Judging: 254

Subcategory:

- Dimension & Annotation: 445
- Geometric Calculation: 207
- Anomaly Detection: 192
- Item Localization: 113
- Consistency Judgment: 60
- Text & Table: 43
- Identification & Counting: 41
- Projection & Multi-view: 33
- Structure Understanding: 29
- Assembly Relationship: 22

Quality score:

- 1.0: 1,117
- 0.75: 48
- 0.5: 3
- 0.25: 10
- 0.0: 7

## Final Checklist

- [x] Keep only one official benchmark JSONL in `vqa_benchmark/`.
- [x] Rewrite all image references to packaged relative paths.
- [x] Remove unused image files from the open package.
- [x] Remove internal model-check and repair fields from the public JSONL.
- [x] Remove chain-of-thought style provenance fields from the public JSONL.
- [x] Remove generated answer-format markers from public reference answers.
- [x] Remove stale acceptable-answer fields that are not used by evaluation.
- [x] Verify that referenced images exist.
