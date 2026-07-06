import argparse
import copy
import os
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Dict, Any, Iterable, Optional

try:
    from .language import json_language_label, normalize_language, question_language_instruction
    from .taxonomy import (
        normalize_capability,
        CAPABILITIES,
        DIFFICULTIES,
        SUBCATEGORIES,
    )
except ImportError:
    from language import json_language_label, normalize_language, question_language_instruction
    from taxonomy import (
        normalize_capability,
        CAPABILITIES,
        DIFFICULTIES,
        SUBCATEGORIES,
    )


# Capability labels follow the English taxonomy used in the paper.
CAPABILITY_DIMENSIONS = {
    "Recognition": {
        "description":
        "从图中准确提取显性信息，OCR识别、尺寸读取、符号识别等；答案尽量简练",
        "subcategories": [
            "Identification & Counting",
            "Dimension & Annotation",
            "Text & Table",
            "Item Localization",
        ]
    },
    "Reasoning": {
        "description": "从显性几何/标注信息中推断隐含的关系",
        "subcategories": [
            "Structure Understanding",
            "Geometric Calculation",
            "Assembly Relationship",
            "Projection & Multi-view",
        ]
    },
    "Judging": {
        "description": "对图纸的正确性作出判断",
        "subcategories": [
            "Anomaly Detection",
            "Consistency Judgment",
        ]
    }
}

# Difficulty labels follow the English taxonomy used in the paper.
DIFFICULTY_LEVELS = {
    "Easy": {
        "description":
        "基础识别和直接读取，OCR和简单定位",
        "question_characteristics": [
            "直接从图中读取单一显性信息", "识别明确标注的尺寸、符号或文字", "简单的命名、定位或计数", "无需推理、计算或专业知识",
            "答案可以一句话说清楚"
        ],
        "answer_requirements": ["答案简洁明了，1-2句话", "直接陈述事实，无需解释", "使用基础术语即可"],
        "example_patterns": [
            "图中标注的XX尺寸是多少？", "该零件的名称是什么？", "主视图中使用了什么线型？", "图中有几个螺栓孔？",
            "左侧视图中的符号Φ表示什么？", "XX部位的粗糙度要求是什么？"
        ]
    },
    "Medium": {
        "description":
        "需要理解和简单推理",
        "question_characteristics": [
            "需要综合2-3个信息点", "理解不同视图之间的对应关系", "进行简单的尺寸计算或几何推导", "理解零件的结构特征或装配关系",
            "需要基本的机械制图知识", "答案需要一定的逻辑解释"
        ],
        "answer_requirements": ["答案包含分析过程，3-5句话", "需要说明推理依据", "使用专业术语准确表述"],
        "example_patterns": [
            "计算图中未直接标注的XX尺寸", "分析两个零件之间的装配顺序或配合关系", "零件采用了什么典型结构？",
            "根据剖视图，推断该零件的内部结构", "该设计中采用了什么工艺特征（如倒角、圆角、退刀槽）？"
        ]
    },
    "Hard": {
        "description":
        "需要深度分析和综合推理",
        "question_characteristics": [
            "需要综合多个视图和多处分散的信息", "复杂的几何推理或空间想象", "涉及多步计算、尺寸链分析",
            "需要专业的机械工程知识（材料、工艺、标准）", "需要考虑多种设计约束或功能要求", "答案需要完整的分析框架和论证"
        ],
        "answer_requirements":
        ["答案包含完整分析，5句话以上", "需要多角度论证和专业知识支撑", "可能涉及标准规范引用", "需要严谨的逻辑推导"],
        "example_patterns":
        ["计算该装配的尺寸链", "推导该零件未标注尺寸的尺寸链", "分析零件或装配图设计中的薄弱位置"]
    }
}

def _progress(iterable, **kwargs):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def load_jsonl_files(
    paths: Iterable[str],
    encoding: str = "utf-8",
    ignore_errors: bool = False,
) -> List[Dict[str, Any]]:
    """
    将多个 .jsonl 文件中的行合并，返回所有 JSON 对象的列表（只拼接每行的 JSON 对象）。
    """
    results: List[Dict[str, Any]] = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding=encoding) as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    if ignore_errors:
                        continue
                    raise
                # 如果顶层 note 字段为 "舍弃"（忽略首尾空白），则跳过该条记录
                try:
                    note_val = obj.get("note") if isinstance(obj, dict) else None
                except Exception:
                    note_val = None
                if isinstance(note_val, str) and note_val.strip() == "舍弃":
                    continue
                results.append(obj)
    return results


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    output = Path(path)
    if output.parent:
        output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_extract_metadata(params: Dict[str, Any]) -> str:
    if not params:
        return "{}"
    return json.dumps(params, ensure_ascii=False, indent=2)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _count_views(detail: Dict[str, Any]) -> int:
    explicit_count = _as_int(detail.get("视图数量"))
    if explicit_count:
        return explicit_count

    total = 0
    main_views = detail.get("主要视图", {})
    if isinstance(main_views, dict):
        total += sum(_as_int(v) for v in main_views.values())

    special_views = detail.get("特殊视图", {})
    if isinstance(special_views, dict):
        total += sum(_as_int(v) for v in special_views.values())

    return total


def summarize_extract_params(params: Dict[str, Any]) -> Dict[str, Any]:
    details = params.get("详细内容", [])
    if not isinstance(details, list):
        details = []

    drawing_count = max(1, len(details))
    total_main_views = 0
    total_special_views = 0
    total_views = 0
    drawing_types = []
    part_names = []
    detail_descriptions = []
    has_tables = False
    has_technical_requirements = False
    has_multi_view_layout = False
    assembly_part_count = 0

    for detail in details:
        if not isinstance(detail, dict):
            continue

        drawing_type = detail.get("图纸类别")
        if drawing_type:
            drawing_types.append(str(drawing_type))

        name = detail.get("零件名称") or detail.get("装配图名称")
        if name:
            part_names.append(str(name))

        layout = str(detail.get("布局关系", ""))
        has_multi_view_layout = has_multi_view_layout or layout == "多视图"

        main_views = detail.get("主要视图", {})
        if isinstance(main_views, dict):
            total_main_views += sum(_as_int(v) for v in main_views.values())

        special_views = detail.get("特殊视图", {})
        if isinstance(special_views, dict):
            total_special_views += sum(_as_int(v) for v in special_views.values())

        total_views += _count_views(detail)
        has_tables = (
            has_tables
            or bool(detail.get("主要参数表"))
            or bool(detail.get("零件明细表"))
            or bool(detail.get("标题栏"))
        )
        has_technical_requirements = has_technical_requirements or bool(detail.get("技术要求"))

        sub_parts = detail.get("子零件列表")
        if isinstance(sub_parts, list):
            assembly_part_count += len(sub_parts)

        special_names = []
        if isinstance(special_views, dict):
            special_names = [k for k, v in special_views.items() if _as_int(v) > 0]
        desc_name = name or "未命名对象"
        detail_descriptions.append(
            f"{drawing_type or '未知图纸类别'}:{desc_name}; "
            f"布局={layout or '未标注'}; "
            f"视图数={detail.get('视图数量', '未知')}; "
            f"特殊视图={','.join(special_names) if special_names else '无'}"
        )

    if drawing_count >= 3 or total_views >= 6 or assembly_part_count >= 8 or (has_tables and has_technical_requirements):
        complexity = "复杂"
    elif drawing_count == 2 or total_views >= 3 or assembly_part_count > 0 or has_tables or has_technical_requirements:
        complexity = "中等"
    else:
        complexity = "简单"

    combination_type = str(params.get("组合关系", "单张图"))
    if drawing_count > 1 or "多张" in combination_type:
        prompt_structure_type = "multiple_independent_drawings"
    elif total_views > 1 or has_multi_view_layout:
        prompt_structure_type = "single_drawing_multi_view"
    else:
        prompt_structure_type = "single_drawing_single_view"

    return {
        "combination_type": combination_type,
        "drawing_count": drawing_count,
        "prompt_structure_type": prompt_structure_type,
        "drawing_types": sorted(set(drawing_types)),
        "part_names": part_names[:8],
        "total_main_views": total_main_views,
        "total_special_views": total_special_views,
        "total_views": total_views,
        "has_tables": has_tables,
        "has_technical_requirements": has_technical_requirements,
        "assembly_part_count": assembly_part_count,
        "complexity": complexity,
        "structure_description": " | ".join(detail_descriptions),
    }


def build_metadata_guidance(summary: Dict[str, Any]) -> str:
    names = "、".join(summary["part_names"]) if summary["part_names"] else "未显式命名"
    drawing_types = "、".join(summary["drawing_types"]) if summary["drawing_types"] else "未显式标注"
    prompt_structure_type = summary["prompt_structure_type"]

    if prompt_structure_type == "multiple_independent_drawings":
        structure_note = "Metadata 显示该图片包含多张独立图纸。问题需要明确目标子图、图纸或零件。"
    elif prompt_structure_type == "single_drawing_multi_view":
        structure_note = "Metadata 显示该图片是一张多视图图纸。可生成单视图问题，也可生成需要综合多个视图的问题。"
    else:
        structure_note = "Metadata 显示该图片主要是一张单视图图纸。问题应围绕该图纸的显性信息生成。"

    table_note = (
        "可优先生成参数表、标题栏、技术要求相关问题。"
        if summary["has_tables"] or summary["has_technical_requirements"]
        else "不要强行生成表格或技术要求问题。"
    )

    return f"""## Metadata 派生图纸结构 ##
- 组合关系: {summary["combination_type"]}
- Prompt 结构类型: {prompt_structure_type}
- 独立图纸数量: {summary["drawing_count"]}
- 图纸类别: {drawing_types}
- 零件/对象名称: {names}
- 主要视图数量: {summary["total_main_views"]}
- 特殊视图数量: {summary["total_special_views"]}
- 复杂度估计: {summary["complexity"]}
- 结构描述: {summary["structure_description"]}
- {structure_note}
- {table_note}"""


def generate_prompts(
    metadata_paths: List[str],
    output_path: str,
    sample_num: int = 10,
    capability_distribution: dict = None,
    language: str = "chinese",
) -> None:
    """
    从统一 extract metadata JSONL 生成问题生成 prompt，保存到新的 jsonl 文件供批量调用模型使用。
    
    Args:
        metadata_paths: 统一 extract 产出的 metadata JSONL 文件路径列表
        output_path: 输出文件路径
        sample_num: 每张图期望生成的问题数量
        capability_distribution: 能力维度分布
        language: 语言设置
    """
    language = normalize_language(language)

    if capability_distribution is None:
        capability_distribution = {"Recognition": 0.33, "Reasoning": 0.34, "Judging": 0.33}
    else:
        capability_distribution = {
            normalize_capability(cap): prob
            for cap, prob in capability_distribution.items()
        }

    all_records = load_jsonl_files(metadata_paths)
    valid_records = []
    for record in all_records:
        params = record.get("params")
        if isinstance(params, dict) and "error" not in params:
            valid_records.append(record)

    if not valid_records:
        raise RuntimeError("No valid extract metadata loaded. Check --metadata-jsonl inputs.")

    print(f"\n总共加载了 {len(valid_records)} 条有效 metadata 记录")
    
    results = []
    
    for record in _progress(valid_records, desc="生成 prompts"):
        extract_params = record["params"]
        metadata_summary = summarize_extract_params(extract_params)

        image_path = record.get("image_path") or record.get("original_image_path", "")
        original_image_path = record.get("original_image_path") or image_path

        if not original_image_path:
            continue
        
        adjusted_sample_num = sample_num
        if metadata_summary["prompt_structure_type"] == "multiple_independent_drawings":
            adjusted_sample_num = min(sample_num * metadata_summary["drawing_count"], sample_num * 2)

        # 根据分布决定每个能力维度生成多少问题
        capability_counts = {}
        for cap, prob in capability_distribution.items():
            capability_counts[cap] = max(1, int(adjusted_sample_num * prob))

        # 调整使总数等于adjusted_sample_num
        diff = adjusted_sample_num - sum(capability_counts.values())
        if diff > 0:
            for _ in range(diff):
                cap = random.choice(list(capability_counts.keys()))
                capability_counts[cap] += 1
        elif diff < 0:
            for _ in range(abs(diff)):
                caps_with_extra = [c for c in capability_counts.keys() if capability_counts[c] > 1]
                if caps_with_extra:
                    cap = random.choice(caps_with_extra)
                    capability_counts[cap] -= 1

        # 构建能力维度详细说明
        capability_details = ""
        for cap, info in CAPABILITY_DIMENSIONS.items():
            subcats = "\n    - ".join(info["subcategories"])
            capability_details += f"\n{cap}({info['description']}):\n    - {subcats}\n"

        capability_details += "\n对于 Recognition 类问题，答案应简练，不要引入过多冗余信息。\n对于 Reasoning 和 Judging 类问题，应提供合理的分析和解释。\n"
        difficulty_details = "\n".join(
            [f"- {level}: {desc}" for level, desc in DIFFICULTY_LEVELS.items()])

        language_instruction = question_language_instruction(language)
        language_label = json_language_label(language)

        metadata_guidance = build_metadata_guidance(metadata_summary)

        prompt = f"""请仔细分析这张机械图纸，生成高质量的专业问题。

{metadata_guidance}

## 统一抽取 Metadata ##
以下 metadata 来自统一 extract 与人工核查流程，是生成问题时必须参考的结构化输入。
请优先依据 metadata 中的图纸类别、视图组成、技术要求、标题栏和参数表生成问题。
如果 metadata 与图像明显冲突，以图像内容为最终依据，并避免基于冲突字段生成问题。

```json
{format_extract_metadata(extract_params)}
```

## 重要说明 ##
1. **根据图纸实际内容生成问题**，不要强行生成图纸中不存在的内容
2. **忽略图片周边的文字说明**，只关注图纸本身
3. **如果 metadata 显示有多张独立图纸，需明确指出问题针对哪个子图/图纸/零件**
4. **如果图纸内容简单**，可以适当减少问题数量或调整能力维度分布
5. **优先保证问题质量**，不要为了凑数生成低质量或无法回答的问题

## 能力维度参考 ##
以下是可选的能力维度和子类别，请根据图纸实际情况选择合适的维度。
能力维度必须从 {list(CAPABILITIES)} 中选择，子类别必须从 {list(SUBCATEGORIES)} 中选择。
{capability_details}

**期望分布**(可根据图纸内容灵活调整):
{json.dumps(capability_counts, ensure_ascii=False)}

## 难度分级参考 ##
{difficulty_details}

问题难度应该符合图纸复杂度。如果图纸简单，以 Easy 和 Medium 难度为主。
难度字段必须从 {list(DIFFICULTIES)} 中选择。

## 语言要求 ##
{language_instruction}

## 生成要求 ##
1. **首先结合 metadata 评估图纸的复杂度**(独立图纸数量、视图数量、尺寸密度、装配关系等)
2. **根据图纸内容决定生成问题的数量和类型**
   - 简单图纸(单视图、少量尺寸): 生成2-5个问题，以 "Recognition" 为主
   - 中等图纸(2-3视图、中等尺寸): 生成5-8个问题，以 "Recognition" + "Reasoning" 为主
   - 复杂图纸(多视图、装配图): 生成8-10个问题，全维度覆盖
   - 多独立子图: 为每个子图生成适量问题
3. 问题应该专业、客观、具有实际工程意义
4. 问题必须基于图纸内容，尊重事实
5. 问题表述清晰、具体、专业
6. **多独立图纸情况下，问题需明确指出目标子图/图纸/零件**

## 返回格式 ##
请严格按照以下JSON格式返回，不要添加任何其他内容:
 其中“子类别”必须从上方列出的英文子类别中选择，不能输出中文子类别名。
 “能力维度”和“难度”也必须输出英文 taxonomy 标签。
{{
  "图纸复杂度评估": "{metadata_summary['complexity']}",
  "实际生成数量": {adjusted_sample_num},
  "问题列表": [
    ["问题1", "能力维度1", "子类别1", "难度1", "语言1"],
    ["问题2", "能力维度2", "子类别2", "难度2", "语言2"],
    ...
  ]
}}

示例(**多子图场景，仅为JSON格式说明，实际问题需根据图纸内容灵活生成**):
{{
  "图纸复杂度评估": "复杂",
  "实际生成数量": 6,
  "问题列表": [
    ["问题文本示例1", "Recognition", "Dimension & Annotation", "Easy", "{language_label}"],
    ["问题文本示例2", "Recognition", "Text & Table", "Easy", "{language_label}"],
    ["问题文本示例3", "Reasoning", "Structure Understanding", "Medium", "{language_label}"],
    ["问题文本示例4", "Reasoning", "Geometric Calculation", "Medium", "{language_label}"],
    ["问题文本示例5", "Judging", "Consistency Judgment", "Medium", "{language_label}"]
  ]
}}

**重要**: 以上仅为格式说明，实际问题应：
1. 根据图纸实际内容生成，不必拘泥于示例类型
2. 单图、多视图、多子图都应灵活处理
3. 问题类型、数量、难度应匹配图纸复杂度
4. 不要因为示例是多子图场景就强行生成位置定位类问题
"""

        # 保存生成的 prompt 数据
        result_item = {
            "id": record.get("id", ""),
            "prompt": prompt,
            "image_path": image_path,
            "original_image_path": original_image_path,
            "params": extract_params,
            "capability": "",
            "subcategory": "",
            "language": language_label,
            "metadata": {
                "language": language_label,
                "extract_metadata_summary": metadata_summary,
                "extract_metadata_source_image_path": record.get("image_path", ""),
                "adjusted_sample_num": adjusted_sample_num,
                "capability_counts": capability_counts,
            } 
        }
        
        results.append(result_item)
    
    # 保存到新的 jsonl 文件
    output = Path(output_path)
    if output.parent:
        output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("\n" + "=" * 80)
    print(f"生成完成！")
    print(f"总有效记录数:   {len(valid_records)}")
    print(f"生成 prompt 数: {len(results)}")
    print(f"结果已保存到:   {output_path}")


RequestFn = Callable[[str, str, Optional[List[str]], float, int], str]


def import_model_client() -> RequestFn:
    try:
        from .model_client import call_openai_compatible
    except ImportError:
        from model_client import call_openai_compatible

    return call_openai_compatible


def image_paths_for_record(record: Dict[str, Any]) -> Optional[List[str]]:
    image_path = record.get("image_path") or record.get("original_image_path")
    return [str(image_path)] if image_path else None


def run_one_generation(
    record: Dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    request_fn: Optional[RequestFn] = None,
) -> Dict[str, Any]:
    result = copy.deepcopy(record)
    prompt = str(result.get("prompt") or "")
    if not prompt:
        result["response"] = ""
        result["error"] = "Missing prompt."
        return result

    call_model = request_fn or import_model_client()
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            response = call_model(
                prompt,
                model=model,
                image_paths=image_paths_for_record(result),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result["response"] = response or ""
            result["error"] = None
            result["generation_model"] = model
            return result
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max_retries:
                break

    result["response"] = ""
    result["error"] = last_error
    result["generation_model"] = model
    return result


def run_generation_prompts(
    input_jsonl: str,
    output_jsonl: str,
    model: str,
    workers: int = 8,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    max_retries: int = 2,
    overwrite: bool = False,
    retry_errors: bool = False,
    request_fn: Optional[RequestFn] = None,
) -> None:
    input_records = read_jsonl(input_jsonl)
    output_exists = os.path.exists(output_jsonl)
    if output_exists and not overwrite:
        output_records = read_jsonl(output_jsonl)
        records = output_records if len(output_records) == len(input_records) else copy.deepcopy(input_records)
    else:
        records = copy.deepcopy(input_records)

    pending = []
    for idx, record in enumerate(records):
        has_response = bool(record.get("response"))
        has_error = bool(record.get("error"))
        if overwrite or not has_response and (retry_errors or not has_error):
            pending.append(idx)

    print(f"Input records: {len(input_records)}")
    print(f"Pending generations: {len(pending)}")
    print(f"Model: {model}; workers: {workers}")
    if not pending:
        write_jsonl(records, output_jsonl)
        print(f"Output is up to date: {output_jsonl}")
        return

    completed_since_save = 0

    def save_checkpoint(force: bool = False) -> None:
        nonlocal completed_since_save
        if force or completed_since_save >= 20:
            write_jsonl(records, output_jsonl)
            completed_since_save = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_one_generation,
                records[idx],
                model,
                temperature,
                max_tokens,
                max_retries,
                request_fn,
            ): idx
            for idx in pending
        }
        for future in _progress(as_completed(futures), total=len(futures), desc="Run VQA-free generation"):
            idx = futures[future]
            records[idx] = future.result()
            completed_since_save += 1
            save_checkpoint()

    save_checkpoint(force=True)
    errors = sum(1 for record in records if record.get("error"))
    print(f"Output: {output_jsonl}")
    print(f"Errors: {errors}")


def parse_distribution(value: Optional[str]) -> Optional[Dict[str, float]]:
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Distribution must be a JSON object.")
    return {normalize_capability(k): float(v) for k, v in parsed.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate VQA-free question prompts from extract metadata.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate question prompts from extract metadata JSONL files.")
    generate.add_argument("--metadata-jsonl", action="append", required=True, help="Unified extract metadata JSONL from scripts/extract_pipeline.py. Can be repeated.")
    generate.add_argument("--output-jsonl", required=True, help="Output prompt JSONL path.")
    generate.add_argument("--sample-num", type=int, default=10, help="Target question count per drawing.")
    generate.add_argument(
        "--language",
        default="中文",
        choices=["中文", "英文", "chinese", "english", "zh", "en"],
        help="Question language. Supports 中文/英文 and aliases chinese/english.",
    )
    generate.add_argument("--capability-distribution", default=None, help="JSON object such as '{\"Recognition\":0.33,\"Reasoning\":0.34,\"Judging\":0.33}'.")

    run = subparsers.add_parser("run", help="Run VQA-free question prompts with an OpenAI-compatible vision model.")
    run.add_argument("--input-jsonl", required=True, help="Prompt JSONL produced by the generate command.")
    run.add_argument("--output-jsonl", required=True, help="Output JSONL with model responses.")
    run.add_argument("--model", default=os.environ.get("DEFAULT_MODEL"), required=os.environ.get("DEFAULT_MODEL") is None)
    run.add_argument("--workers", type=int, default=8, help="Parallel worker count.")
    run.add_argument("--temperature", type=float, default=0.2)
    run.add_argument("--max-tokens", type=int, default=4096)
    run.add_argument("--max-retries", type=int, default=2)
    run.add_argument("--overwrite", action="store_true", help="Regenerate every record even if response already exists.")
    run.add_argument("--retry-errors", action="store_true", help="Retry records whose previous output has error set.")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "generate":
        generate_prompts(
            metadata_paths=args.metadata_jsonl,
            output_path=args.output_jsonl,
            sample_num=args.sample_num,
            capability_distribution=parse_distribution(args.capability_distribution),
            language=args.language,
        )
        return

    if args.command == "run":
        run_generation_prompts(
            input_jsonl=args.input_jsonl,
            output_jsonl=args.output_jsonl,
            model=args.model,
            workers=args.workers,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            overwrite=args.overwrite,
            retry_errors=args.retry_errors,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
