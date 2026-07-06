import argparse
from collections import defaultdict
from typing import Any, Dict

from .io_utils import load_jsonl, write_json


def build_breakdown(records, target_model: str, field: str) -> Dict[str, Any]:
    stats = defaultdict(lambda: {"count": 0, "sum_score": 0.0, "full_score_count": 0, "error_count": 0})
    for record in records:
        metadata = record.get("original_record", record).get("metadata", {})
        key = metadata.get(field, "unknown")
        result = record.get("evaluation_result", {}).get(target_model, {})
        score = float(result.get("voted_score", 0))
        stats[key]["count"] += 1
        stats[key]["sum_score"] += score
        stats[key]["full_score_count"] += int(score == 1.0)
        stats[key]["error_count"] += int(bool(result.get("error")))

    output = {}
    for key, value in stats.items():
        count = value["count"] or 1
        output[key] = {
            "count": value["count"],
            "avg_score": value["sum_score"] / count,
            "full_score_rate": value["full_score_count"] / count,
            "error_count": value["error_count"],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize evaluated VQA records by metadata fields.")
    parser.add_argument("--evaluated-file", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = load_jsonl(args.evaluated_file)
    summary = {
        "target_model": args.target_model,
        "by_subcategory": build_breakdown(records, args.target_model, "subcategory"),
        "by_difficulty": build_breakdown(records, args.target_model, "difficulty"),
        "by_language": build_breakdown(records, args.target_model, "language"),
    }
    write_json(summary, args.output)
    print(f"summary saved to {args.output}")


if __name__ == "__main__":
    main()

