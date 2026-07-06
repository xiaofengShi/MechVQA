"""Extract structured mechanical drawing parameters from image files.

Pipeline:
1. Send each drawing image to an OCR service to get plain text and markdown tables.
2. Send the drawing image plus OCR context to a vision-language model.
3. Parse the model response into a normalized JSON object.
4. Optionally run the extraction multiple times and vote on inconsistent fields.

Required runtime configuration:
- OPENAI_API_KEY for the OpenAI-compatible model endpoint.
- OPENAI_BASE_URL if the endpoint is not the default OpenAI endpoint.
- DEFAULT_MODEL or --model for the model name.
- OCR_URL or --ocr-url for the OCR extraction service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL")
DEFAULT_OCR_URL = os.environ.get("OCR_URL")
DEFAULT_OCR_OUTPUT_DIR = "outputs/ocr"

STANDARD_PARTS_LIST = [
    "螺栓类",
    "螺母类",
    "螺钉类",
    "键类",
    "垫圈类",
    "挡圈类",
    "销类",
    "法兰类",
    "轴承类",
]

NON_STANDARD_PARTS_LIST = [
    "弹簧类",
    "泵与阀类",
    "齿轮类",
    "轴类",
    "连杆类",
    "结构件类",
    "轴套类",
    "轮盘与盖类",
    "叉架类",
    "箱壳类",
    "密封件类",
    "其他类",
]

PART_OUTPUT_FORMAT = {
    "图纸类别": "零件图",
    "零件名称": "沉头螺钉",
    "是否标准件": "标准件",
    "零件类别": "螺钉类",
    "布局关系": "多视图",
    "组合关系": "单张图",
    "视图数量": 3,
    "主要视图": {"主视图": 1, "侧视图": 1, "俯视图": 0},
    "特殊视图": {
        "轴测图": 0,
        "放大图": 0,
        "剖视图": 0,
        "爆炸图": 0,
        "局部视图": 0,
        "向视图": 0,
        "断裂图": 0,
    },
    "技术要求": r"""# 技术要求
1.热处理后齿面硬度为  $241\sim 286HBW$  
2. 未注倒角为  $C2$  。  
3. 齿轮内在质量按  ${MQ}$  级  $\left( {{GB}/{T3480.5} -  - {2008}}\right)$  执行。  
4.齿根圆滑过渡，棱角倒钝。  
5. 未注尺寸公差按GB/T 1804--2000--m。  
6. 未注几何公差按GB/T 1184--1996--K。""",
    "主要参数表": [2],
    "标题栏": [1],
}

ASSEMBLY_OUTPUT_FORMAT = {
    "图纸类别": "装配图",
    "装配图名称": "柴油机",
    "布局关系": "多视图",
    "组合关系": "单张图",
    "视图数量": 3,
    "主要视图": {"主视图": 1, "侧视图": 1, "俯视图": 0},
    "特殊视图": {
        "轴测图": 0,
        "放大图": 0,
        "剖视图": 0,
        "爆炸图": 0,
        "局部视图": 0,
        "向视图": 0,
        "断裂图": 0,
    },
    "零件明细表": [1, 2],
    "标题栏": [3],
    "子零件列表": [
        {
            "序号": 1,
            "零件名称": "活塞",
            "是否标准件": "非标准件",
            "零件类别": "结构件类",
            "数量": 1,
            "材料": "铸铁",
        },
        {
            "序号": 2,
            "零件名称": "连杆螺栓",
            "是否标准件": "标准件",
            "零件类别": "螺栓类",
            "数量": 4,
            "材料": "45号钢",
        },
        {"...": "..."},
    ],
}

OUTPUT_FORMAT = {
    "组合关系": "多张图/单张图",
    "详细内容": [
        PART_OUTPUT_FORMAT,
        ASSEMBLY_OUTPUT_FORMAT,
        {"...": "..."},
    ]
}


# ================== JSON 解析（容错） ==================
def _extract_json_block(s: str) -> str:
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")
    stack = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            stack += 1
        elif s[i] == "}":
            stack -= 1
            if stack == 0:
                return s[start : i + 1]
    raise ValueError("Unbalanced JSON braces in model response.")


def safe_json_loads(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        block = _extract_json_block(s)
        return json.loads(block)


def _resolve_ocr_url(ocr_url: Optional[str]) -> str:
    resolved = ocr_url or DEFAULT_OCR_URL
    if not resolved:
        raise ValueError("OCR URL is empty. Set OCR_URL or pass --ocr-url.")
    return resolved


def _progress(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def run_ocr(
    drawing_path: str,
    ocr_url: Optional[str] = None,
    ocr_output_dir: str = DEFAULT_OCR_OUTPUT_DIR,
    ocr_timeout: int = 300,
) -> Tuple[str, List[str]]:
    """Run OCR service and return plain text plus markdown tables."""
    output_dir = Path(ocr_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import requests

    response = requests.post(
        _resolve_ocr_url(ocr_url),
        json={"input_path": drawing_path, "output_dir": str(output_dir)},
        timeout=ocr_timeout,
    )
    response.raise_for_status()
    payload = response.json()

    text = payload.get("processed_text") or ""
    tables = payload.get("processed_tables") or []
    if not isinstance(tables, list):
        tables = []
    return text, [str(table) for table in tables]


def build_extract_prompt(drawing_path: str, ocr_text: str, md_tables: Sequence[str]) -> str:
    tables_block = "\n\n".join([f"【表{i + 1}】\n{tbl}" for i, tbl in enumerate(md_tables)]) or "（无表格）"
    pure_text_block = ocr_text if ocr_text.strip() else "（无正文）"

    return f"""
    你是资深机械制图工程师与信息抽取专家。请基于给定的图纸（模型已同时接收图像）、
    以及以下 OCR 结果，完成结构化抽取。务必遵守输出规范与保守原则。

    【输入】
    - 图纸路径：{drawing_path}
    - OCR 纯文本：
    {pure_text_block}

    - OCR 表格（Markdown；按出现顺序编号）：
    {tables_block}

    【任务】
    首先判断图纸是否为单张或多张组合；无论结果如何，均需将输出包装为 "详细内容"列表。
    对于每个列表元素（代表一张独立的图纸），字段要求如下：
    1) 判断该图是“零件图”还是“装配图”。
    - 识别提示：若存在含“序号/名称/数量（或类似）”的明细表，通常为装配图；仅有单一零件特征与标题栏则通常为零件图。
    2) 提取主名称：
    优先级① 零件明细表的标题列/表头 → ② 标题栏/图签 → ③ 文件名中最可能的中文/英文名称。
    如果是多图组合
    如果不能从上述途径明确提取名称，则在 "装配图名称"/"零件名称" 字段填 "N/A"。
    3) 分类：
    - 若为零件图：判断是否标准件；参考下述标准件/非标准件词表进行模糊匹配（无需完全一致），给出“是否标准件”与“零件类别”。
    - 若为装配图：从零件明细表逐行抽取子零件（序号/名称/数量/材料等）。
        * “零件名称”与“序号”“数量”“材料”必须逐字来自表格相应列。
        * “是否标准件”：若“零件名称”匹配标准件任一二级项 → "标准件"，否则 → "非标准件"。
        * “零件类别”：填对应二级类别名。
    4) 视觉属性：
    - 视图数量：整数，图中所有独立视图总数。
        - 独立视图的定义：必须与其他视图在画面上有明确空白分隔，彼此不相连；或有单独的边框/标题/比例；
        - 不是独立视图的情况（全部排除）：
            - 在同一轮廓内出现的局部剖示/端部剖示（仅在某端涂剖面线，且与主体轮廓相连、没有独立分隔/标题）；
            - 同一视图内的尺寸链延伸段、倒角细化、粗糙度/形位公差框、局部阴影；
    - 主要视图：请输出一个包含以下键的字典 {{ "主视图", "侧视图", "俯视图"}}，一般都至少有一个主视图，每个键的取值为整数（出现次数，无则填 0）。
        注意：侧视图或俯视图一定与主视图高或宽相同，否则可能是局部视图等特殊视图。
    - 特殊视图：请输出一个包含以下键的字典 {{ "轴测图", "放大图", "剖视图", "爆炸图", "局部视图", "向视图", "断裂图" }}，每个键的取值为整数（出现次数，无则填 0）。
        注意：轴测图即三维图，放大图通常有比例注明，剖视图有剖切线或剖面符号，爆炸图零件间有间隙，局部视图只有部分特征，向视图有箭头指示，断裂图则通常有断裂线或断面符号。
    - 布局关系 ∈ {"单视图","多视图"}
    5) 技术要求：
    - 从“OCR 纯文本”中提取对应的以“技术要求”或“要求”开头的段落，保留编号与换行；若没有则设为空字符串 ""。
    - 如果“OCR 纯文本”中没有“技术要求”相关文字，或无法明确区分技术要求对应哪张图纸，则从图中提取对应的技术要求；若还没有则设为空字符串 ""。
    6) 表格索引：
     - 说明：所有索引字段（"主要参数表"、"零件明细表"、"标题栏表"）一律从上文“OCR 表格（Markdown；按出现顺序编号）”中选择并返回**编号数组**；若未识别到对应表格，填 []。不得从纯文本中选择。
    - 零件图：
        • "主要参数表"：最可能承载尺寸/参数的表（如尺寸、上/下偏差、孔径、长度、齿数等，可能有多张）
        • "标题栏"：包含图名/图号/比例/签名/日期/阶段标记/材料/重量等图签信息的表
    - 装配图：
        • "零件明细表"：含“序号/名称/数量（或同义列）”的清单表（可能有多张）
        • "标题栏"：同上

    【列名与同义参考】
    - 名称列常见：{"零件名称","名称","部件名称","名称及规格","零件名称及规格"}
    - 数量列常见：{"数量","数目","件","PCS"}
    - 材料列常见：{"材料","材质","牌号"}
    - 明细表常含：{"序号","零件名称","数量","材料"} 的任意组合

    【标准件/非标准件参考词表】（用于匹配与归类；不要求完全匹配）
    - 标准件：{STANDARD_PARTS_LIST}
    - 非标准件：{NON_STANDARD_PARTS_LIST}

    【输出】
    仅输出一个 JSON，且符合以下 Schema（不要输出任何解释）：
    {OUTPUT_FORMAT}

    【保守原则】
    - 不要臆造未在图/表中出现的子零件或材料；无法确认时填 "N/A"。
    - “零件明细表/主要参数表”字段只可填上文 OCR 表格的编号数组（例如 [2,3]），不要粘贴表格内容。
    - 所有数量均为整数；缺失但明显为单件时可填 1；材料缺失填 "N/A"。
    - 严禁输出 JSON 之外的任何文字、代码块标记或说明。
    """


def _replace_table_indices_with_content(parameters: dict, md_tables: Sequence[str]) -> dict:
    """Replace model-emitted 1-based table indices with markdown table contents."""

    def _lookup(indices: Any) -> List[str]:
        if not isinstance(indices, list):
            return []
        return [md_tables[idx - 1] for idx in indices if isinstance(idx, int) and 1 <= idx <= len(md_tables)]

    def _apply(detail: Dict[str, Any]) -> None:
        drawing_type = detail.get("图纸类别")
        if drawing_type == "零件图":
            detail["主要参数表"] = _lookup(detail.get("主要参数表"))
        elif drawing_type == "装配图":
            detail["零件明细表"] = _lookup(detail.get("零件明细表"))
        detail["标题栏"] = _lookup(detail.get("标题栏"))

    details = parameters.get("详细内容")
    if isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict):
                _apply(detail)
    elif isinstance(parameters, dict):
        _apply(parameters)

    return parameters


def extract_parameters(
    drawing_path: str,
    model: Optional[str] = DEFAULT_MODEL,
    ocr_url: Optional[str] = None,
    ocr_output_dir: str = DEFAULT_OCR_OUTPUT_DIR,
    ocr_timeout: int = 300,
    temperature: float = 0.7,
) -> dict:
    """Extract structured drawing parameters from one image."""
    if not Path(drawing_path).exists():
        raise FileNotFoundError(f"Image file does not exist: {drawing_path}")

    ocr_text, md_tables = run_ocr(
        drawing_path=drawing_path,
        ocr_url=ocr_url,
        ocr_output_dir=ocr_output_dir,
        ocr_timeout=ocr_timeout,
    )
    prompt = build_extract_prompt(drawing_path, ocr_text, md_tables)

    try:
        from .model_client import call_openai_compatible
    except ImportError:
        from model_client import call_openai_compatible

    response = call_openai_compatible(
        prompt=prompt,
        model=model,
        image_paths=[drawing_path],
        temperature=temperature,
    )
    parameters = safe_json_loads(response)
    return _replace_table_indices_with_content(parameters, md_tables)


def _hash_value(value: Any) -> str:
    return hashlib.md5(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _compare_params(params1: dict, params2: dict, path: str = "") -> List[str]:
    inconsistent_fields: List[str] = []
    for key in set(params1.keys()) | set(params2.keys()):
        current_path = f"{path}.{key}" if path else key
        if key not in params1 or key not in params2:
            inconsistent_fields.append(current_path)
            continue

        val1, val2 = params1[key], params2[key]
        if isinstance(val1, dict) and isinstance(val2, dict):
            inconsistent_fields.extend(_compare_params(val1, val2, current_path))
        elif isinstance(val1, list) and isinstance(val2, list):
            if len(val1) != len(val2):
                inconsistent_fields.append(current_path)
                continue
            for idx, (item1, item2) in enumerate(zip(val1, val2)):
                item_path = f"{current_path}[{idx}]"
                if isinstance(item1, dict) and isinstance(item2, dict):
                    inconsistent_fields.extend(_compare_params(item1, item2, item_path))
                elif _hash_value(item1) != _hash_value(item2):
                    inconsistent_fields.append(item_path)
        elif _hash_value(val1) != _hash_value(val2):
            inconsistent_fields.append(current_path)

    return inconsistent_fields


def _get_field_by_path(params: dict, path: str) -> Any:
    current: Any = params
    parts = [part for part in re.split(r"\.|\[|\]", path) if part]
    for part in parts:
        if part.isdigit():
            if not isinstance(current, list):
                return None
            current = current[int(part)]
        else:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        if current is None:
            return None
    return current


def _set_field_by_path(params: dict, path: str, value: Any) -> None:
    current: Any = params
    parts = [part for part in re.split(r"\.|\[|\]", path) if part]
    for i, part in enumerate(parts[:-1]):
        if part.isdigit():
            current = current[int(part)]
            continue
        if part not in current:
            current[part] = [] if parts[i + 1].isdigit() else {}
        current = current[part]

    last_part = parts[-1]
    if last_part.isdigit():
        current[int(last_part)] = value
    else:
        current[last_part] = value


def _vote_on_technical_requirements(requirements_list: Sequence[str], judge_model: Optional[str] = None) -> str:
    valid_requirements = [req for req in requirements_list if req and req.strip()]
    if not valid_requirements:
        return ""
    if len(valid_requirements) == 1:
        return valid_requirements[0]

    requirements_block = "\n\n".join(f"【版本{i + 1}】\n{req}" for i, req in enumerate(valid_requirements))
    judge_prompt = f"""你是机械制图领域的专家。现有同一张机械图纸的技术要求经过多次识别产生了不同版本。
请选取最准确、最完整、最规范的技术要求条目。

【判断要求】
1. 识别哪些条目是真实存在的技术要求。
2. 排除明显识别错误、幻觉或不合理条目。
3. 如果多个版本表述略有差异，选择更规范、更完整的版本。
4. 保留原有编号格式。

【各版本技术要求】
{requirements_block}

    【输出要求】
    只输出最终技术要求文本。如果都没有真实技术要求，则输出空字符串。
"""
    try:
        try:
            from .model_client import call_openai_compatible
        except ImportError:
            from model_client import call_openai_compatible

        return call_openai_compatible(judge_prompt, model=judge_model).strip()
    except Exception:
        return max(valid_requirements, key=lambda x: valid_requirements.count(x))


def _vote_on_fields(params_list: Sequence[dict], inconsistent_fields: Sequence[str], judge_model: Optional[str] = None) -> dict:
    result = json.loads(json.dumps(params_list[0], ensure_ascii=False))

    for field_path in inconsistent_fields:
        values = [_get_field_by_path(params, field_path) for params in params_list]
        values = [value for value in values if value is not None]
        if not values:
            continue

        if "技术要求" in field_path and all(isinstance(value, str) for value in values):
            _set_field_by_path(result, field_path, _vote_on_technical_requirements(values, judge_model=judge_model))
            continue

        value_counts: Dict[str, Tuple[Any, int]] = {}
        for value in values:
            value_hash = _hash_value(value)
            previous_value, previous_count = value_counts.get(value_hash, (value, 0))
            value_counts[value_hash] = (previous_value, previous_count + 1)

        best_value = max(value_counts.values(), key=lambda item: item[1])[0]
        _set_field_by_path(result, field_path, best_value)

    return result


def extract_parameters_with_voting(
    drawing_path: str,
    model: Optional[str] = DEFAULT_MODEL,
    ocr_url: Optional[str] = None,
    ocr_output_dir: str = DEFAULT_OCR_OUTPUT_DIR,
    ocr_timeout: int = 300,
    judge_model: Optional[str] = None,
) -> dict:
    params1 = extract_parameters(drawing_path, model=model, ocr_url=ocr_url, ocr_output_dir=ocr_output_dir, ocr_timeout=ocr_timeout)
    params2 = extract_parameters(drawing_path, model=model, ocr_url=ocr_url, ocr_output_dir=ocr_output_dir, ocr_timeout=ocr_timeout)
    inconsistent_fields = _compare_params(params1, params2)
    if not inconsistent_fields:
        return params1

    params3 = extract_parameters(drawing_path, model=model, ocr_url=ocr_url, ocr_output_dir=ocr_output_dir, ocr_timeout=ocr_timeout)
    return _vote_on_fields([params1, params2, params3], inconsistent_fields, judge_model=judge_model)


def iter_image_files(dataset_dir: str, extensions: Sequence[str]) -> List[str]:
    suffixes = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    files = [
        str(path)
        for path in Path(dataset_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No image files with extensions {sorted(suffixes)} found in {dataset_dir}")
    return files


def process_single_image(
    image_path: str,
    model: Optional[str] = DEFAULT_MODEL,
    ocr_url: Optional[str] = None,
    ocr_output_dir: str = DEFAULT_OCR_OUTPUT_DIR,
    ocr_timeout: int = 300,
    use_voting: bool = True,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        if use_voting:
            params = extract_parameters_with_voting(
                image_path,
                model=model,
                ocr_url=ocr_url,
                ocr_output_dir=ocr_output_dir,
                ocr_timeout=ocr_timeout,
                judge_model=judge_model,
            )
        else:
            params = extract_parameters(
                image_path,
                model=model,
                ocr_url=ocr_url,
                ocr_output_dir=ocr_output_dir,
                ocr_timeout=ocr_timeout,
            )
    except Exception as exc:
        params = {"error": f"pipeline_error: {exc}"}

    return {
        "image_path": image_path,
        "model": model,
        "use_voting": use_voting,
        "params": params,
    }


def process_dataset(
    dataset_dir: str,
    output_path: str,
    model: Optional[str] = DEFAULT_MODEL,
    max_workers: int = 4,
    ocr_url: Optional[str] = None,
    ocr_output_dir: str = DEFAULT_OCR_OUTPUT_DIR,
    ocr_timeout: int = 300,
    use_voting: bool = True,
    judge_model: Optional[str] = None,
    extensions: Sequence[str] = (".png", ".jpg", ".jpeg"),
) -> None:
    image_files = iter_image_files(dataset_dir, extensions)
    results: List[Optional[Dict[str, Any]]] = [None] * len(image_files)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_image,
                image_path,
                model,
                ocr_url,
                ocr_output_dir,
                ocr_timeout,
                use_voting,
                judge_model,
            ): idx
            for idx, image_path in enumerate(image_files)
        }
        for future in _progress(as_completed(futures), total=len(futures), desc="Extracting drawings"):
            results[futures[future]] = future.result()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_no} in {path} is not valid JSON: {exc}") from exc
    return records


def reprocess_failed_records(
    input_jsonl: str,
    output_jsonl: Optional[str] = None,
    model: Optional[str] = DEFAULT_MODEL,
    max_workers: int = 4,
    ocr_url: Optional[str] = None,
    ocr_output_dir: str = DEFAULT_OCR_OUTPUT_DIR,
    ocr_timeout: int = 300,
    use_voting: bool = True,
    judge_model: Optional[str] = None,
) -> None:
    records = load_jsonl(input_jsonl)
    target_indices = [
        idx
        for idx, record in enumerate(records)
        if isinstance(record.get("params"), dict) and (not record["params"] or "error" in record["params"])
    ]

    if not target_indices:
        print("No failed records found.")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_image,
                str(records[idx]["image_path"]),
                model,
                ocr_url,
                ocr_output_dir,
                ocr_timeout,
                use_voting,
                judge_model,
            ): idx
            for idx in target_indices
        }
        for future in _progress(as_completed(futures), total=len(futures), desc="Retrying failed records"):
            records[futures[future]] = future.result()

    output = Path(output_jsonl or input_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_extensions(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        required=DEFAULT_MODEL is None,
        help="Vision-language model name. Defaults to DEFAULT_MODEL env when set.",
    )
    parser.add_argument(
        "--ocr-url",
        default=None,
        required=DEFAULT_OCR_URL is None,
        help="OCR service URL. Defaults to OCR_URL env when set.",
    )
    parser.add_argument("--ocr-output-dir", default=DEFAULT_OCR_OUTPUT_DIR, help="Directory used by the OCR service for intermediate outputs.")
    parser.add_argument("--ocr-timeout", type=int, default=300, help="OCR request timeout in seconds.")
    parser.add_argument("--no-voting", action="store_true", help="Disable multi-run voting and run one extraction per image.")
    parser.add_argument("--judge-model", default=None, help="Optional text model used to vote on inconsistent technical requirements.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract structured parameters from mechanical drawing images.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Extract one drawing image.")
    add_common_args(single)
    single.add_argument("--image", required=True, help="Input drawing image path.")
    single.add_argument("--output-json", default=None, help="Optional output JSON path.")

    dataset = subparsers.add_parser("dataset", help="Extract every image in a directory into JSONL.")
    add_common_args(dataset)
    dataset.add_argument("--dataset-dir", required=True, help="Directory containing drawing images.")
    dataset.add_argument("--output-jsonl", required=True, help="Output JSONL path.")
    dataset.add_argument("--max-workers", type=int, default=4, help="Parallel worker count.")
    dataset.add_argument("--extensions", default=".png,.jpg,.jpeg", help="Comma-separated image extensions.")

    retry = subparsers.add_parser("retry", help="Reprocess records whose params contain an error.")
    add_common_args(retry)
    retry.add_argument("--input-jsonl", required=True, help="Input JSONL produced by the dataset command.")
    retry.add_argument("--output-jsonl", default=None, help="Output JSONL path. Defaults to in-place rewrite.")
    retry.add_argument("--max-workers", type=int, default=4, help="Parallel worker count.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    use_voting = not args.no_voting

    if args.command == "single":
        result = process_single_image(
            args.image,
            model=args.model,
            ocr_url=args.ocr_url,
            ocr_output_dir=args.ocr_output_dir,
            ocr_timeout=args.ocr_timeout,
            use_voting=use_voting,
            judge_model=args.judge_model,
        )
        if args.output_json:
            output = Path(args.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "dataset":
        process_dataset(
            dataset_dir=args.dataset_dir,
            output_path=args.output_jsonl,
            model=args.model,
            max_workers=args.max_workers,
            ocr_url=args.ocr_url,
            ocr_output_dir=args.ocr_output_dir,
            ocr_timeout=args.ocr_timeout,
            use_voting=use_voting,
            judge_model=args.judge_model,
            extensions=parse_extensions(args.extensions),
        )
        return

    if args.command == "retry":
        reprocess_failed_records(
            input_jsonl=args.input_jsonl,
            output_jsonl=args.output_jsonl,
            model=args.model,
            max_workers=args.max_workers,
            ocr_url=args.ocr_url,
            ocr_output_dir=args.ocr_output_dir,
            ocr_timeout=args.ocr_timeout,
            use_voting=use_voting,
            judge_model=args.judge_model,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
