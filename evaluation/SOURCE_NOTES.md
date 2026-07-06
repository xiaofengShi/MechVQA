# Evaluation Code Source Notes

This open-source evaluation folder is a cleaned extraction of the evaluation
workflow used for MechVQA.

## Included In The Open-Source Folder

- A two-stage VQA evaluation pipeline:
  - phase 1: call target models and save responses
  - phase 2: call judge models, parse scores, and save aggregate stats
- OpenAI-compatible endpoint wrapper with local image-to-data-URL support.
- Metadata summary helper.
- Shell launchers and example config.

## Deliberately Excluded

- API keys and private endpoints.
- Hard-coded private machine paths.
- Internal logs and evaluation outputs.
- Checkpoints, generated response JSONL files, and stats JSON files.
- Experiment-specific model lists and commented historical runs.

## Release Checklist

- Put benchmark JSONL files under a release data location or document how to download them.
- Keep local configs such as `configs/vqa_eval.local.json` out of git.
- Confirm all model endpoints in public docs are user-provided, not internal services.
