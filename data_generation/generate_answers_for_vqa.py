import argparse
import os
import json
import traceback
import threading
from tqdm import tqdm
from typing import List, Dict, Any, Iterable, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .language import answer_language_instruction, normalize_language
    from .model_client import call_openai_compatible
    from .taxonomy import normalize_capability, normalize_subcategory
except ImportError:
    from language import answer_language_instruction, normalize_language
    from model_client import call_openai_compatible
    from taxonomy import normalize_capability, normalize_subcategory

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL")
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL")


def _resolve_model(model: Optional[str] = None) -> str:
    resolved = model or DEFAULT_MODEL
    if not resolved:
        raise RuntimeError("Missing model. Pass --model or set DEFAULT_MODEL.")
    return resolved


def _resolve_judge_model(judge_model: Optional[str], model: str) -> str:
    return judge_model or DEFAULT_JUDGE_MODEL or model


def _image_paths(image_path: str) -> Optional[List[str]]:
    return [image_path] if image_path else None


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

def _extract_json_block(s: str) -> str:
    """从字符串中提取 JSON 块"""
    start = s.find("{")
    if start == -1:
        raise ValueError("未在回复中找到 JSON 对象")
    stack = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            stack += 1
        elif s[i] == "}":
            stack -= 1
            if stack == 0:
                return s[start : i + 1]
    raise ValueError("JSON 括号不平衡")


def safe_json_loads(s: str) -> dict:
    """安全地解析 JSON，如果直接解析失败则尝试提取 JSON 块"""
    try:
        return json.loads(s)
    except Exception:
        block = _extract_json_block(s)
        return json.loads(block)



def generate_answer_with_temperature(
    question: str,
    capability: str,
    image_path: str,
    model: str,
    params=None,
    language: str = "中文",
    request_timeout: int = 240,
) -> str:
    """根据能力维度使用不同温度生成答案"""

    capability = normalize_capability(capability)
    language = normalize_language(language)

    # 配置不同能力的推理参数
    config_map = {
        "Recognition": {"temperature": 0.8, "top_p": 0.95},
        "Reasoning": {"temperature": 0.8, "top_p": 0.95},
        "Judging": {"temperature": 0.8, "top_p": 0.95},
    }

    config = config_map.get(capability, {"temperature": 0.8, "top_p": 0.95})
    metadata_block = json.dumps(params, ensure_ascii=False, indent=2) if params else "{}"

    if language == "英文":
        prompt = f"""Please carefully read this mechanical drawing and answer the question.

Question type: {capability}
Question: {question}

## Unified extract metadata
The following metadata is verified structured information extracted from the drawing. Use it to understand views, dimensions, title blocks, technical requirements, and parameter tables. If it conflicts with the image, rely on the image.

```json
{metadata_block}
```

Requirements:
1. Answer accurately based on the drawing.
2. Keep the answer professional and precise.
3. For dimension or numeric questions, give exact values.
4. For reasoning questions, provide clear logic.
5. {answer_language_instruction(language)}
"""
    else:
        prompt = f"""请仔细阅读这张机械图纸，并回答以下问题。

问题类型: {capability}
问题: {question}

## 统一 extract metadata
以下 metadata 是已抽取并核查过的图纸结构化信息。请优先依据它理解图纸中的视图、尺寸、标题栏、技术要求和参数表；若与图像冲突，以图像为准。

```json
{metadata_block}
```

回答要求:
1. 基于图纸内容，准确回答
2. 保持专业性和准确性
3. 如果是尺寸或数值问题，给出精确答案
4. 如果是分析推理问题，给出清晰的逻辑
5. {answer_language_instruction(language)}
"""

    try:
        answer = call_openai_compatible(
            prompt,
            model=model,
            image_paths=_image_paths(image_path),
            temperature=config["temperature"],
            thinking=True,
            timeout=request_timeout,
        )
        return answer.strip()
    except Exception as e:
        print(f"生成答案失败: {str(e)}")
        return ""


def compare_answers_semantic(
    question: str,
    capability: str,
    answer1: str,
    answer2: str,
    judge_model: str,
    language: str = "中文",
    request_timeout: int = 240,
) -> dict:
    """比较两个答案的语义一致性"""

    capability = normalize_capability(capability)
    language = normalize_language(language)

    # 根据问题类型调整比较标准
    if language == "英文":
        if capability == "Recognition":
            comparison_focus = "For recognition questions, focus on whether key information such as dimensions, symbols, and names is exactly consistent."
        elif capability == "Reasoning":
            comparison_focus = "For reasoning questions, focus on whether the reasoning logic and final conclusion are consistent. Different wording is acceptable."
        elif capability == "Judging":
            comparison_focus = "For judging questions, focus on whether the judgment, evidence, and error localization are consistent."
        else:
            comparison_focus = "For other questions, focus on whether the core content and conclusion are consistent."
        prompt = f"""Compare the semantic consistency of the following two answers.

Question: {question}
Question type: {capability}

Answer 1: {answer1}

Answer 2: {answer2}

Comparison focus:
{comparison_focus}

Special rules:
- For numeric or dimension questions: values must match exactly to be considered consistent.
- For reasoning questions: the core conclusion and main logic must be consistent; wording may differ.

Return JSON only:
{{
    "is_consistent": true/false,
    "consistency_score": 0.0-1.0,
    "analysis": "Detailed consistency analysis in English",
    "key_differences": ["difference 1", "difference 2"]
}}

Decision rules:
- consistency_score >= 0.85: is_consistent = true
- For recognition questions, exact key information consistency means is_consistent = true
- For reasoning questions, consistent conclusions and reasonable logic mean is_consistent = true
"""
    elif capability == "Recognition":
        comparison_focus = "对于识别类问题，重点比较关键信息（如尺寸数值、符号、名称）是否完全一致"
    elif capability == "Reasoning":
        comparison_focus = "对于推理类问题，重点比较推理逻辑和最终结论是否一致，允许表述方式不同"
    elif capability == "Judging":
        comparison_focus = "对于判断类问题，重点比较判断结论、依据和错误定位是否一致"
    else:
        comparison_focus = "对于其他类问题，重点比较核心内容和结论是否一致"

    if language == "中文":
        prompt = f"""请比较以下两个答案的语义一致性。

问题: {question}
问题类型: {capability}

答案1: {answer1}

答案2: {answer2}

比较要求:
{comparison_focus}

特殊规则:
- 对于数值/尺寸问题: 数值必须完全相同才视为一致
- 对于分析推理问题: 核心结论一致即可，表述可以不同

请以JSON格式返回比较结果（不要添加其他内容）：
{{
    "is_consistent": true/false,
    "consistency_score": 0.0-1.0,
    "analysis": "详细分析两个答案的一致性",
    "key_differences": ["差异点1", "差异点2"]
}}

判断标准:
- consistency_score >= 0.85: is_consistent = true
- 对于识别类，关键信息完全一致: is_consistent = true
- 对于推理类，结论一致且逻辑合理: is_consistent = true
"""

    try:
        response = call_openai_compatible(
            prompt,
            model=judge_model,
            temperature=0.3,
            timeout=request_timeout,
        )
        # 使用 safe_json_loads 替代 json.loads
        result = safe_json_loads(response)
        return result
    except Exception as e:
        print(f"答案比较失败: {str(e)}")
        return {
            "is_consistent": False,
            "consistency_score": 0.0,
            "analysis": "比较失败",
            "key_differences": []
        }


def split_think_and_answer(text: str) -> tuple:
    """
    分离答案中的think部分和answer部分
    
    Args:
        text: 完整的回答文本
        
    Returns:
        (think_part, answer_part) 元组
    """
    if not text:
        return "", ""
    
    # 查找 </think> 标签
    think_end_marker = "</think>"
    
    if think_end_marker in text:
        # 找到标签位置
        end_pos = text.find(think_end_marker)
        think_part = text[:end_pos].strip()
        answer_part = text[end_pos + len(think_end_marker):].strip()
        return think_part, answer_part
    else:
        # 没有think标签，整个文本作为answer
        return "", text.strip()


def vote_and_merge_answers(
    question: str,
    capability: str,
    answers: List[str],
    judge_model: str,
    language: str = "中文",
    request_timeout: int = 240,
) -> dict:
    """对多个答案进行投票和合并（分别处理think和answer部分）"""

    capability = normalize_capability(capability)
    language = normalize_language(language)

    if len(answers) < 2:
        # 单个答案时，分离think和answer
        if answers:
            think_part, answer_part = split_think_and_answer(answers[0])
            return {
                "final_answer": answer_part,
                "final_answer_think": think_part,
                "vote_result": {},
                "merged": False,
                "confidence": "low"
            }
        return {
            "final_answer": "",
            "final_answer_think": "",
            "vote_result": {},
            "merged": False,
            "confidence": "low"
        }
    
    # 分离每个答案的think和answer部分
    separated_answers = []
    answer_parts = []
    
    for ans in answers:
        think_part, answer_part = split_think_and_answer(ans)
        separated_answers.append({
            "think": think_part,
            "answer": answer_part
        })
        answer_parts.append(answer_part)
    
    # 构建答案文本（仅使用answer部分进行投票）
    answer_label = "Answer" if language == "英文" else "答案"
    answers_text = "\n\n".join([f"{answer_label}{i+1}: {ans}" for i, ans in enumerate(answer_parts)])

    if language == "英文":
        prompt = f"""Vote on and merge the following answers.

Question: {question}
Question type: {capability}

{answers_text}

Tasks:
1. Analyze the core content of each answer.
2. Group semantically equivalent answers together.
3. For recognition questions, focus on whether key information such as values and names is consistent.
4. For reasoning questions, focus on whether conclusions and main evidence are consistent.
5. Count support for each group.
6. Select the highest-vote answer group.
7. Merge and optimize that group into the final answer.

Return JSON only:
{{
    "answer_groups": [
        {{
            "group_id": 1,
            "answer_indices": [0, 2],
            "core_content": "Core content summary for this group",
            "vote_count": 2
        }}
    ],
    "winner_group": 1,
    "final_answer": "Merged final answer in English",
    "confidence": "high/medium/low",
    "analysis": "Voting and merging analysis in English"
}}

Merging rules:
- Preserve accurate key information from all answers.
- For conflicting content, prefer the more logical or better-supported statement.
- Keep the final answer complete and professional.
- For recognition questions, ensure numeric values and names are accurate.
- {answer_language_instruction(language)}
"""
    else:
        prompt = f"""请对以下多个答案进行投票和合并。

问题: {question}
问题类型: {capability}

{answers_text}

任务:
1. 分析每个答案的核心内容
2. 识别语义相同但表述不同的答案（归为同一组）
3. 对于识别类问题，关注关键信息（数值、名称等）是否一致
4. 对于推理类问题，关注结论和主要论据是否一致
5. 统计各组的支持度
6. 选出得票最高的答案组
7. 对该组的答案进行合并优化，生成最终答案

请以JSON格式返回结果（不要添加其他内容）：
{{
    "answer_groups": [
        {{
            "group_id": 1,
            "answer_indices": [0, 2],
            "core_content": "该组答案的核心内容概述",
            "vote_count": 2
        }}
    ],
    "winner_group": 1,
    "final_answer": "合并优化后的最终答案（使用中文）",
    "confidence": "high/medium/low",
    "analysis": "投票和合并的详细分析"
}}

合并原则:
- 保留所有答案中准确的关键信息
- 对于矛盾的内容，选择逻辑更合理或支持度更高的
- 保持答案的完整性和专业性
- 对于识别类，确保数值/名称的准确性
- {answer_language_instruction(language)}
"""

    try:
        response = call_openai_compatible(
            prompt,
            model=judge_model,
            temperature=0.3,
            timeout=request_timeout,
        )
        result = safe_json_loads(response)
        
        merged_final_answer = result.get("final_answer", answer_parts[0])
        merged_think = ""
        
        # 获取获胜组的答案索引
        winner_group_id = result.get("winner_group", 1)
        answer_groups = result.get("answer_groups", [])
        winner_indices = []
        
        for group in answer_groups:
            if group.get("group_id") == winner_group_id:
                winner_indices = group.get("answer_indices", [])
                break
        
        # 如果没有找到获胜组，使用所有索引
        if not winner_indices:
            winner_indices = list(range(len(answers)))
        
        # 收集获胜组的think部分（只收集非空的think）
        winner_think_parts = []
        for idx in winner_indices:
            if idx < len(separated_answers):
                think_content = separated_answers[idx]["think"]
                if think_content and think_content.strip():
                    winner_think_parts.append(think_content)
        
        # 合并think部分
        if len(winner_think_parts) == 0:
            merged_think = ""
        elif len(winner_think_parts) == 1:
            merged_think = winner_think_parts[0]
        else:
            # 调用模型合并多个think部分
            think_label = "Reasoning process" if language == "英文" else "推理过程"
            think_texts = "\n\n".join([f"{think_label}{i+1}:\n{t}" for i, t in enumerate(winner_think_parts)])
            if language == "英文":
                think_merge_prompt = f"""Merge the following reasoning processes into one complete and coherent reasoning process.

Question: {question}
Question type: {capability}

{think_texts}

Merging requirements:
1. Preserve all key reasoning steps and analysis.
2. Merge repeated content.
3. Organize the reasoning in a logical order.
4. Keep the wording professional and accurate.
5. {answer_language_instruction(language)}

Output only the merged reasoning process. Do not add extra explanations or tags."""
            else:
                think_merge_prompt = f"""请将以下多个推理过程进行合并，生成一个完整、连贯的推理过程。

问题: {question}
问题类型: {capability}

{think_texts}

合并要求:
1. 保留所有关键的推理步骤和分析
2. 合并重复的内容
3. 按照合理的逻辑顺序组织
4. 保持专业性和准确性
5. {answer_language_instruction(language)}

请直接输出合并后的推理过程（不要添加任何额外说明或标记）："""
            
            try:
                merged_think_response = call_openai_compatible(
                    think_merge_prompt, 
                    model=judge_model,
                    temperature=0.3,
                    timeout=request_timeout,
                )
                merged_think = merged_think_response.strip() if merged_think_response else ""
            except Exception as e:
                print(f"  ⚠️ 合并think部分失败: {traceback.format_exc()}")
                # 失败时使用第一个think部分
                merged_think = winner_think_parts[0] if winner_think_parts else ""
        
        # 将合并结果添加到result中
        result["final_answer_think"] = merged_think
        result["winner_think_parts"] = winner_think_parts  # 记录获胜组的原始think部分

        return {
            "final_answer": merged_final_answer,
            "final_answer_think": merged_think,
            "vote_result": result,
            "merged": len(answer_groups) > 1 or len(winner_think_parts) > 1,
            "confidence": result.get("confidence", "medium"),
        }
    except Exception as e:
        print(f"投票合并失败: {traceback.format_exc()}")
        # 失败时尝试返回第一个答案的信息
        first_think = separated_answers[0]["think"] if separated_answers else ""
        first_answer = answer_parts[0] if answer_parts else (answers[0] if answers else "")
        return {
            "final_answer": first_answer,
            "final_answer_think": first_think,
            "vote_result": {"error": str(e)},
            "merged": False,
            "confidence": "low",
        }

def verify_and_generate_answer_for_question(
    question_data,
    image_path,
    model: str,
    judge_model: str,
    language: str = "中文",
    request_timeout: int = 240,
):
    """为单个问题生成答案并验证（多轮推理）"""

    question = question_data["question"]
    capability = normalize_capability(question_data["capability"])
    question_data["capability"] = capability
    language = normalize_language(language)
    params = question_data.get("params", {})

    print(f"\n  问题: {question[:60]}...")
    print(f"  类型: {capability}")

    # 第1次推理
    print(f"  第1次推理...")
    answer1 = generate_answer_with_temperature(
        question, capability, image_path, model, params=params, language=language, request_timeout=request_timeout
    )

    if not answer1:
        print(f"  ✗ 第1次推理失败")
        return None

    # 第2次推理
    print(f"  第2次推理...")
    answer2 = generate_answer_with_temperature(
        question, capability, image_path, model, params=params, language=language, request_timeout=request_timeout
    )

    if not answer2:
        print(f"  ✗ 第2次推理失败")
        return None

    # 比较前两次答案
    print(f"  比较前两次答案...")
    comparison = compare_answers_semantic(
        question, capability, answer1, answer2, judge_model, language=language, request_timeout=request_timeout
    )

    is_consistent = comparison.get("is_consistent", False)
    score = comparison.get("consistency_score", 0)
    print(f"  一致性: {is_consistent} (得分: {score:.2f})")

    # 如果前两次一致，直接通过
    if is_consistent:
        print(f"  ✓ 前两次一致，通过")
        vote_result = vote_and_merge_answers(
            question, capability, [answer1, answer2], judge_model, language=language, request_timeout=request_timeout
        )

        result = question_data.copy()
        result["answer"] = vote_result["final_answer"]
        result["verification_status"] = "pass"
        result["verification_rounds"] = 2
        result["consistency_score"] = score
        result["all_answers"] = [answer1, answer2]
        result["vote_result"] = vote_result

        return result

    # 第3次推理
    print(f"  前两次不一致，进行第3次推理...")
    answer3 = generate_answer_with_temperature(
        question, capability, image_path, model, params=params, language=language, request_timeout=request_timeout
    )

    if not answer3:
        print(f"  ✗ 第3次推理失败")
        return None

    # 对3个答案进行投票
    print(f"  对3个答案进行投票...")
    vote_result = vote_and_merge_answers(
        question, capability, [answer1, answer2, answer3], judge_model, language=language, request_timeout=request_timeout
    )

    # 检查是否有足够的一致性
    answer_groups = vote_result.get("vote_result", {}).get("answer_groups", [])
    if not answer_groups:
        print(f"  ✗ 投票结果无效")
        return None

    max_votes = max([g.get("vote_count", 0) for g in answer_groups])

    if max_votes < 2:
        print(f"  ✗ 无法达成一致 (最高票数: {max_votes}/3)")
        return None

    print(f"  ✓ 投票通过 (得票: {max_votes}/3)")

    result = question_data.copy()
    result["answer"] = vote_result["final_answer"]
    result["verification_status"] = "pass"
    result["verification_rounds"] = 3
    result["max_votes"] = max_votes
    result["all_answers"] = [answer1, answer2, answer3]
    result["vote_result"] = vote_result

    return result


def generate_and_verify_questions_for_one(
    data,
    model: str,
    judge_model: str,
    request_timeout: int = 240,
    language: str = "中文",
):
    """为单个记录生成答案并验证（处理新的数据格式）"""

    image_path = data.get("original_image_path", "")
    
    print(f"\n{'='*80}")
    print(f"处理图片: {image_path}")
    print(f"ID: {data.get('id', 'N/A')}")
    print(f"{'='*80}")

    # 从数据中提取问题信息
    question = data.get("question") or data.get("prompt", "")
    capability = normalize_capability(data.get("capability", ""))
    subcategory = normalize_subcategory(data.get("subcategory", ""))
    difficulty = data.get("difficulty", "")
    params = data.get("params", {})
    metadata = data.get("metadata", {})
    record_language = normalize_language(data.get("language") or metadata.get("language") or language)
    
    # 提取问题
    actual_question = question

    # 构造QA对象
    qa_data = {
        "question": actual_question,
        "capability": capability,
        "subcategory": subcategory,
        "difficulty": difficulty,
        "id": data.get("id", ""),
        "original_image_path": data.get("original_image_path", ""),
        "fix_result": data.get("fix_result", {}),
        "params": params,
        "metadata": metadata,
        "language": record_language,
    }

    print(f"\n  问题: {actual_question[:60]}...")
    print(f"  类型: {capability}")
    print(f"  子类: {subcategory}")

    # Step 1: 多轮推理验证
    print(f"\n=== Step 1: 多轮推理验证 ===")
    verified_qa_pairs = []

    result = verify_and_generate_answer_for_question(
        qa_data,
        image_path,
        model,
        judge_model,
        language=record_language,
        request_timeout=request_timeout,
    )
    if result is not None:
        verified_qa_pairs.append(result)

    # 统计信息
    stats = {
        "total_questions": 1,
        "verified_qa": len(verified_qa_pairs),
        "rejected_qa": 1 - len(verified_qa_pairs),
        "pass_rate": len(verified_qa_pairs),
        "capability": {},
        "subcategory": {}
    }

    for qa in verified_qa_pairs:
        stats["capability"][qa["capability"]] = stats["capability"].get(qa["capability"], 0) + 1
        stats["subcategory"][qa["subcategory"]] = stats["subcategory"].get(qa["subcategory"], 0) + 1

    print(f"\n=== 处理完成 ===")
    print(f"原始问题数: {stats['total_questions']}")
    print(f"通过验证: {stats['verified_qa']}")
    print(f"被拒绝: {stats['rejected_qa']}")
    print(f"通过率: {stats['pass_rate']:.1%}")

    return {
        "image_path": image_path,
        "params": data.get("params", {}),
        "metadata": data.get("metadata", {}),
        "language": record_language,
        "qa_pairs": verified_qa_pairs,
        "statistics": stats,
        "error": None
    }


def main_generate_with_verification(paths, output_path, model, judge_model, num_workers, request_timeout, language):
    """生成问题并进行多轮推理验证的主函数"""

    language = normalize_language(language)
    print(f"\n{'='*80}")
    print(f"开始生成问题并进行验证")
    print(f"输出文件: {output_path}")
    print(f"答案生成模型: {model}")
    print(f"一致性裁判模型: {judge_model}")
    print(f"图片级并行: {num_workers}")
    print(f"单次模型请求超时: {request_timeout}s")
    print(f"默认输出语言: {language}")
    print(f"{'='*80}\n")

    # 加载数据
    datas_list = load_jsonl_files(paths)
    print(f"加载了 {len(datas_list)} 条记录")

    # 统计信息
    total_questions = 0
    total_verified = 0
    total_rejected = 0

    # 使用线程池处理
    lock = threading.Lock()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(
                    generate_and_verify_questions_for_one,
                    data,
                    model,
                    judge_model,
                    request_timeout,
                    language,
                ): idx
                for idx, data in enumerate(datas_list)
            }

            # 使用tqdm显示进度
            with tqdm(total=len(datas_list), desc="Processing images") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]

                    try:
                        result = future.result()

                        # 线程安全地写入文件
                        with lock:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f.flush()

                            # 统计
                            stats = result.get("statistics", {})
                            total_questions += stats.get("total_questions", 0)
                            total_verified += stats.get("verified_qa", 0)
                            total_rejected += stats.get("rejected_qa", 0)

                    except Exception as e:
                        print(f"\n✗ 处理记录 {idx + 1} 时出错: {str(e)}")
                        traceback.print_exc()

                    pbar.update(1)

    print(f"\n{'='*80}")
    print(f"全部处理完成！")
    print(f"{'='*80}")
    print(f"总问题数: {total_questions}")
    print(f"通过验证: {total_verified} ({total_verified/total_questions*100:.1f}%)" if total_questions > 0 else "通过验证: 0")
    print(f"被拒绝: {total_rejected} ({total_rejected/total_questions*100:.1f}%)" if total_questions > 0 else "被拒绝: 0")
    print(f"结果已保存到: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and verify answers for VQA questions with an OpenAI-compatible model."
    )
    parser.add_argument(
        "--input-jsonl",
        action="append",
        required=True,
        help="Input JSONL path. Can be repeated.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Answer-generation model. Defaults to DEFAULT_MODEL env when set.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Model used for answer consistency judging. Defaults to JUDGE_MODEL env, then --model.",
    )
    parser.add_argument("--num-workers", type=int, default=10, help="Number of parallel image workers.")
    parser.add_argument("--request-timeout", type=int, default=240, help="Timeout in seconds for each model request.")
    parser.add_argument(
        "--language",
        default="中文",
        choices=["中文", "英文", "chinese", "english", "zh", "en"],
        help="Fallback answer language when input records do not have language.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    model = _resolve_model(args.model)
    judge_model = _resolve_judge_model(args.judge_model, model)
    main_generate_with_verification(
        paths=args.input_jsonl,
        output_path=args.output_jsonl,
        model=model,
        judge_model=judge_model,
        num_workers=args.num_workers,
        request_timeout=args.request_timeout,
        language=args.language,
    )


if __name__ == "__main__":
    main()
