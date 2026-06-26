#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MechVQA 批量推理验证脚本（支持 SFT / RL 双模式）

用 vLLM 对 ckpt/MechVQA_SFT 或 ckpt/MechVQA_RL 做多模态 batch 推理（每条：中文问题 + 1 张机械图纸）。
适配自 EasyR1/data_mechvlm/qwen_vl_vllm_batch_infer.py（原面向 Qwen3-VL-235B-Thinking）。

两种模式（顶部 MODE 切换）：
  - sft: 直接问答。system 提示 + 图片 + question，整段输出即答案。
  - rl : thinking 模型。用 prompts/mech_r1.jinja 渲染 question 作为 user content
         （复现 RL 训练 format_prompt，字面 \\n 原样保留，无独立 system），
         输出 <think>...</think><answer>...</answer>，提取 <answer> 作为答案。

与原 235B 脚本差异：TP=1 / 适中上下文 / 去掉 enable_thinking / 内置扁平 jsonl / GT 与预测并排。

运行（在项目根目录执行）：
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 \\
      python scripts/batch_infer.py
"""

# ---- 清理被污染的 PYTHONPATH（系统指向 python3.10 的旧包会干扰 import）----
import os
import sys
sys.path = [p for p in sys.path if p and "/python3.10/" not in p and "llm_lib" not in p]

# ---- Triton/vLLM kernel 编译缓存目录（默认 ~/.triton 在 HOME=/root 不可写会报错）----
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_PROJECT_ROOT, ".triton_cache"))

import json
import re
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm  # noqa: F401  (vllm use_tqdm 依赖)
from vllm import LLM, SamplingParams


# =========================
# 模式与配置区
# =========================
MODE = "rl"   # "sft" 或 "rl"，切换测试目标

INPUT_JSONL = "data/test_samples.jsonl"

if MODE == "sft":
    MODEL_PATH = "ckpt/MechVQA_SFT"
    OUTPUT_JSONL = "outputs/sft_infer_test.jsonl"
    PROMPT_TEMPLATE = None          # 不套模板，直接 system + 图片 + question
    SYSTEM_PROMPT = "你是一名机械制图专家。请仔细阅读图纸并回答问题。"
    ANSWER_TAG = None               # 整段输出即答案
    MAX_MODEL_LEN = 8192
    MAX_NEW_TOKENS = 1024
    TEMPERATURE = 0.1
    TOP_P = 0.9
    TOP_K = 20
elif MODE == "rl":
    MODEL_PATH = "ckpt/MechVQA_RL"
    OUTPUT_JSONL = "outputs/rl_infer_test.jsonl"
    PROMPT_TEMPLATE = "prompts/mech_r1.jinja"   # 复现 RL 训练 format_prompt
    SYSTEM_PROMPT = None            # RL 用 format_prompt 包装，无独立 system
    ANSWER_TAG = "answer"           # 提取 <answer>...</answer>
    MAX_MODEL_LEN = 16384           # thinking 输出较长
    MAX_NEW_TOKENS = 4096
    TEMPERATURE = 0.6               # thinking 模型常用稍高温度
    TOP_P = 0.95
    TOP_K = 20
else:
    raise ValueError(f"未知 MODE={MODE}，应为 'sft' 或 'rl'")

# vLLM 引擎参数（4B 单卡 A800-80GB 绰绰有余）
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_SEQS = 64

# 一次喂入 llm.chat 的样本数
BATCH_SIZE = 10

# 图片超过该像素数则等比缩放
MAX_IMAGE_TOTAL_PIXELS = 4_000_000
Image.MAX_IMAGE_PIXELS = None  # 关闭 PIL DecompressionBombWarning


# =========================
# 时间统计
# =========================
class TimeTracker:
    """跟踪推理进度、已耗时与 ETA。"""

    def __init__(self, total_pending: int):
        self.total_pending = total_pending
        self.start_time = None
        self.processed = 0
        self.success = 0

    def start(self):
        self.start_time = time.time()

    def update(self, processed: int, success: int):
        self.processed += processed
        self.success += success

    def get_elapsed(self) -> float:
        return 0.0 if self.start_time is None else time.time() - self.start_time

    def get_eta(self) -> float:
        elapsed = self.get_elapsed()
        if self.processed == 0:
            return 0.0
        avg = elapsed / self.processed
        return avg * (self.total_pending - self.processed)

    @staticmethod
    def format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{int(seconds // 60)}m {seconds % 60:.0f}s"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"

    def status(self) -> str:
        elapsed = self.get_elapsed()
        eta = self.get_eta()
        pct = (self.processed / self.total_pending * 100) if self.total_pending > 0 else 0
        return (f"Processed: {self.processed}/{self.total_pending} ({pct:.1f}%) | "
                f"Success: {self.success} | "
                f"Elapsed: {self.format_time(elapsed)} | ETA: {self.format_time(eta)}")


# =========================
# 工具函数
# =========================
_TEMPLATE_CACHE = None


def render_prompt(question: str) -> str:
    """sft: 原样返回 question；rl: 用 jinja 模板渲染（复现 EasyR1 format_prompt）。

    EasyR1 (verl/utils/dataset.py): Template(open(path).read().strip()).render(content=q)
    模板里的字面 \\n 不会被 Jinja2 转义，原样保留（与训练一致）。
    """
    global _TEMPLATE_CACHE
    if not PROMPT_TEMPLATE:
        return question
    if _TEMPLATE_CACHE is None:
        from jinja2 import Template
        text = Path(PROMPT_TEMPLATE).read_text(encoding="utf-8").strip()
        _TEMPLATE_CACHE = Template(text)
    return _TEMPLATE_CACHE.render(content=question)


def extract_answer(text: str) -> str:
    """sft: 返回整段文本；rl: 从 <answer>...</answer> 提取（失败返回空）。"""
    if not ANSWER_TAG:
        return text.strip()
    m = re.search(rf"<{ANSWER_TAG}>(.*?)</{ANSWER_TAG}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def is_valid_pred(pred: str) -> bool:
    return isinstance(pred, str) and bool(pred.strip())


def load_samples(path: Path):
    """读取项目内置的扁平测试数据，每行一个样本 dict。"""
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] JSON 解析失败，跳过: {repr(e)}")
                continue
            samples.append({
                "id": obj.get("id", ""),
                "image_path": obj.get("image_path", ""),
                "question": obj.get("question", ""),
                "gt_answer": obj.get("gt_answer", ""),
                "capability": obj.get("capability", ""),
                "subcategory": obj.get("subcategory", ""),
                "difficulty": obj.get("difficulty", ""),
            })
    return samples


def build_conversations_safe(samples):
    """安全构造 conversations：坏图/缺字段只跳过。返回 (convs, good, bad)。"""
    conversations, good, bad = [], [], []
    for s in samples:
        if not s.get("question") or not s.get("image_path"):
            print(f"[WARN] 缺 question 或 image_path，跳过 id={s.get('id')}")
            bad.append(s)
            continue
        try:
            img = Image.open(s["image_path"]).convert("RGB")
            w, h = img.size
            if w * h > MAX_IMAGE_TOTAL_PIXELS:
                scale = (MAX_IMAGE_TOTAL_PIXELS / (w * h)) ** 0.5
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                print(f"[RESIZE] {s['image_path']}: {w}x{h} -> {new_w}x{new_h}")
                img = img.resize((new_w, new_h), Image.LANCZOS)
        except Exception as e:
            print(f"[ERROR] 打开图片失败 id={s.get('id')} path={s.get('image_path')}: {repr(e)}")
            bad.append(s)
            continue
        user_text = render_prompt(s["question"])
        messages = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": [
            {"type": "image_pil", "image_pil": img},
            {"type": "text", "text": user_text},
        ]})
        conversations.append(messages)
        good.append(s)
    return conversations, good, bad


def make_sample_key(sample: dict) -> str:
    return str(sample.get("id"))


# =========================
# 主流程
# =========================
def main():
    input_path = Path(INPUT_JSONL)
    output_path = Path(OUTPUT_JSONL)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] MODE={MODE} | MODEL={MODEL_PATH} | PROMPT_TEMPLATE={PROMPT_TEMPLATE} | ANSWER_TAG={ANSWER_TAG}")

    # 1. 加载内置测试数据
    samples = load_samples(input_path)
    if not samples:
        raise ValueError(f"未从 {input_path} 取到样本")
    total = len(samples)
    print(f"[INFO] 取到 {total} 条样本（来自内置 {input_path}）")

    # 2. 断点续跑：读已有输出，构建已成功的 key -> obj
    existing_success = {}
    if output_path.exists():
        print(f"[INFO] 发现已有输出 {output_path}，加载...")
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not is_valid_pred(obj.get("pred_answer", "")):
                    continue
                existing_success[make_sample_key(obj)] = obj
        print(f"[INFO] 已有成功结果 {len(existing_success)} 条")

    pending = [s for s in samples if make_sample_key(s) not in existing_success]
    print(f"[INFO] 待推理 {len(pending)} 条")

    # 3. 先把已成功结果覆盖写回
    with output_path.open("w", encoding="utf-8") as f:
        for obj in existing_success.values():
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if not pending:
        print("[INFO] 全部已完成，无需推理。")
        return

    # 4. 加载模型
    print(f"[INFO] 加载模型: {MODEL_PATH}")
    load_start = time.time()
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        limit_mm_per_prompt={"image": 1},
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
    )
    print(f"[INFO] 模型加载耗时 {TimeTracker.format_time(time.time() - load_start)}")

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_NEW_TOKENS,
    )
    print(f"[INFO] 采样参数: {sampling_params}")

    # 5. 批量推理，每批完成立即追加写入
    num_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[INFO] 将运行 {num_batches} 批 (BATCH_SIZE={BATCH_SIZE})")

    tracker = TimeTracker(len(pending))
    tracker.start()
    total_success = 0
    total_truncated = 0

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(pending))
        batch_samples = pending[start:end]

        print(f"\n{'=' * 60}")
        print(f"[BATCH {batch_idx + 1}/{num_batches}] 处理 {len(batch_samples)} 条")
        print(f"[TIME] {tracker.status()}")

        batch_convs, good, bad = build_conversations_safe(batch_samples)
        if bad:
            print(f"[WARN] {len(bad)} 条坏图跳过")
        if not batch_convs:
            print("[ERROR] 本批全部失败，跳过")
            tracker.update(len(batch_samples), 0)
            continue

        try:
            outputs = llm.chat(
                messages=batch_convs,
                sampling_params=sampling_params,
                use_tqdm=True,
            )
        except Exception as e:
            print(f"[ERROR] llm.chat 失败: {repr(e)}")
            tracker.update(len(batch_samples), 0)
            continue

        success_in = 0
        trunc_in = 0
        with output_path.open("a", encoding="utf-8") as f:
            for s, o in zip(good, outputs):
                try:
                    out0 = o.outputs[0]
                    text = out0.text
                    finish = out0.finish_reason
                except Exception as e2:
                    print(f"[ERROR] 解析输出失败 id={s.get('id')}: {repr(e2)}")
                    continue
                pred = extract_answer(text)
                ok = is_valid_pred(pred)
                out_obj = {
                    "id": s["id"],
                    "image_path": s["image_path"],
                    "question": s["question"],
                    "gt_answer": s["gt_answer"],
                    "pred_answer": pred,
                    "extract_ok": ok,
                    "capability": s["capability"],
                    "subcategory": s["subcategory"],
                    "difficulty": s["difficulty"],
                    "finish_reason": finish,
                }
                if ANSWER_TAG:
                    out_obj["raw_response"] = text  # 保留完整 <think>/<answer>，便于排查
                f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                if ok:
                    success_in += 1
                else:
                    trunc_in += 1

        total_success += success_in
        total_truncated += trunc_in
        tracker.update(len(batch_samples), success_in)
        print(f"[BATCH {batch_idx + 1}/{num_batches}] 成功 {success_in} | 提取失败 {trunc_in} | 坏图 {len(bad)}")
        print(f"[TIME] {tracker.status()}")

    # 6. 汇总
    print(f"\n{'=' * 60}\n[SUMMARY] 推理完成 (MODE={MODE})\n{'=' * 60}")
    print(f"  总样本:      {total}")
    print(f"  已有成功:    {len(existing_success)}")
    print(f"  本次推理:    {len(pending)}")
    print(f"  新增成功:    {total_success}")
    print(f"  提取失败:    {total_truncated}")
    print(f"  当前总计:    {len(existing_success) + total_success}")
    print(f"  总耗时:      {TimeTracker.format_time(tracker.get_elapsed())}")
    print(f"{'=' * 60}")

    # 7. 抽样打印 GT vs Pred（便于人工核对）
    print("\n[抽样式例] 前 3 条 GT vs Pred：")
    with output_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            o = json.loads(line)
            print(f"\n--- {o['id']} [{o.get('capability')}] ---")
            print(f"Q:    {o['question']}")
            print(f"GT:   {o['gt_answer'][:150]}")
            print(f"PRED: {o['pred_answer'][:150]}")


if __name__ == "__main__":
    main()
