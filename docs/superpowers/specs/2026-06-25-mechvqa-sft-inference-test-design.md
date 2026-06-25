# MechVQA SFT 模型推理验证 —— 设计文档

- **日期**：2026-06-25
- **状态**：已实现并验证（SFT 10/10、RL 10/10 全部成功，答案正确）
- **参考脚本**：`/share/project/shixiaofeng/code/EasyR1/data_mechvlm/qwen_vl_vllm_batch_infer.py`
- **测试数据**：`/share/project/shixiaofeng/code/MechVLM/vqa_data/mechnical_data_vote_based.jsonl`

## 1. 目标

把参考脚本（原本面向 Qwen3-VL-**235B-Thinking**）适配为面向本项目 **Qwen3-VL-4B-Instruct** 微调模型的批量推理脚本，整理到 `MechVQA/scripts/`，在本机跑 **10 条**样本验证 `ckpt/MechVQA_SFT` 能否正确加载与生成。

成功标准：① 模型成功加载；② 10 条样本均生成非空中文回答；③ 抽样人工核对，回答与图纸 GT 基本合理。

## 2. 技术栈（已实测可用）

| 项 | 值 | 备注 |
|---|---|---|
| 解释器 | `/share/project/kouqian/miniconda3/envs/anyrag/bin/python` | python 3.12.11 |
| vLLM | 0.11.0 | 已注册 `Qwen3VLForConditionalGeneration` |
| transformers | 4.57.1 | 与模型 `config.json` 要求一致 |
| torch | 2.8.0+cu128 | CUDA 可用，2 GPU 可见 |
| ⚠️ PYTHONPATH 坑 | 启动需 `env -u PYTHONPATH` | 系统 `PYTHONPATH` 指向 python3.10 的 `/usr/local/lib/python3.10/dist-packages`，会污染 import；脚本内会再做 `sys.path` 清理兜底 |
| 模型 | `/share/project/shixiaofeng/code/MechVQA/ckpt/MechVQA_SFT` | 2 个 safetensors 完整，约 8.9GB |
| GPU | 2 × A800-80GB（160GB，当前空闲） | 4B bf16≈8GB → **TP=1 单卡** |

## 3. 推理流程图

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 320" font-family="sans-serif" font-size="13">
  <defs>
    <marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#555"/></marker>
  </defs>
  <!-- boxes -->
  <g>
    <rect x="10"  y="120" width="150" height="80" rx="8" fill="#eef5ff" stroke="#3b7dd8"/>
    <text x="85"  y="150" text-anchor="middle" font-weight="bold">加载数据</text>
    <text x="85"  y="172" text-anchor="middle">mechnical_data</text>
    <text x="85"  y="188" text-anchor="middle">_vote_based.jsonl</text>

    <rect x="190" y="120" width="150" height="80" rx="8" fill="#eef5ff" stroke="#3b7dd8"/>
    <text x="265" y="150" text-anchor="middle" font-weight="bold">展平 + 采样</text>
    <text x="265" y="172" text-anchor="middle">qa_pairs[] → 扁平</text>
    <text x="265" y="188" text-anchor="middle">取前10张图各1条</text>

    <rect x="370" y="120" width="150" height="80" rx="8" fill="#fff4e6" stroke="#e88a1a"/>
    <text x="445" y="150" text-anchor="middle" font-weight="bold">构建对话</text>
    <text x="445" y="172" text-anchor="middle">system + 图片(PIL)</text>
    <text x="445" y="188" text-anchor="middle">+ question</text>

    <rect x="550" y="120" width="150" height="80" rx="8" fill="#eafff0" stroke="#2e9e5b"/>
    <text x="625" y="150" text-anchor="middle" font-weight="bold">vLLM 推理</text>
    <text x="625" y="172" text-anchor="middle">TP=1 · 单卡</text>
    <text x="625" y="188" text-anchor="middle">enable_thinking=False</text>

    <rect x="730" y="120" width="180" height="80" rx="8" fill="#f3eaff" stroke="#7a3ec8"/>
    <text x="820" y="150" text-anchor="middle" font-weight="bold">解析 + 写入</text>
    <text x="820" y="172" text-anchor="middle">gt/pred 并排 → jsonl</text>
    <text x="820" y="188" text-anchor="middle">增量写 · 断点续跑</text>
  </g>
  <!-- arrows -->
  <line x1="160" y1="160" x2="186" y2="160" stroke="#555" stroke-width="1.6" marker-end="url(#arr)"/>
  <line x1="340" y1="160" x2="366" y2="160" stroke="#555" stroke-width="1.6" marker-end="url(#arr)"/>
  <line x1="520" y1="160" x2="546" y2="160" stroke="#555" stroke-width="1.6" marker-end="url(#arr)"/>
  <line x1="700" y1="160" x2="726" y2="160" stroke="#555" stroke-width="1.6" marker-end="url(#arr)"/>
  <!-- label -->
  <text x="490" y="60" text-anchor="middle" font-size="16" font-weight="bold" fill="#222">MechVQA-SFT 批量推理 Pipeline（10 条验证）</text>
  <text x="85"  y="240" text-anchor="middle" fill="#666">8720 QA · 图100%可访问</text>
  <text x="265" y="240" text-anchor="middle" fill="#666">确定性 · 可复现</text>
  <text x="445" y="240" text-anchor="middle" fill="#666">超像素自动缩放</text>
  <text x="625" y="240" text-anchor="middle" fill="#666">MAX_MODEL_LEN=8192</text>
  <text x="820" y="240" text-anchor="middle" fill="#666">outputs/*.jsonl</text>
</svg>
```

## 4. 脚本设计：`scripts/batch_infer.py`

**保留**参考脚本的优良结构：断点续跑（读已有输出跳过已成功 key）、坏图跳过不抛异常、JSONL 增量写入、`TimeTracker` 时间统计、PIL 像素上限自动缩放、按 `BATCH_SIZE` 分批。

**适配 4B-Instruct 的 5 处关键改动**：

| # | 维度 | 原脚本（235B-Thinking） | 适配后（4B-Instruct） |
|---|---|---|---|
| 1 | 引擎参数 | TP=8 / MAX_MODEL_LEN=51200 / MAX_NEW_TOKENS=40960 | **TP=1 / MAX_MODEL_LEN=8192 / MAX_NEW_TOKENS=1024** |
| 2 | thinking | `chat_template_kwargs={"enable_thinking": True}` | **删除该 kwarg**（chat_template 无 thinking 逻辑） |
| 3 | 有效性判定 | 要求含 `</think>` 且其后是有效 JSON | **改为**：response 为非空字符串即记为成功 |
| 4 | 数据加载 | 扁平 `{id, prompt, image_path}` | **展平**嵌套 `{image_path, qa_pairs[]}`，每条产出 `(image_path, question, answer, capability, difficulty)` |
| 5 | 输出字段 | verdict JSON | **保存** `id/image_path/question/gt_answer/pred_answer/capability/difficulty`（GT 与预测并排，便于人工核对） |

### 4.1 关键参数（脚本顶部配置区，均可改）

```python
MODEL_PATH = "/share/project/shixiaofeng/code/MechVQA/ckpt/MechVQA_SFT"
INPUT_JSONL = "/share/project/shixiaofeng/code/MechVLM/vqa_data/mechnical_data_vote_based.jsonl"
OUTPUT_JSONL = "outputs/sft_infer_10.jsonl"

TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 64

# 采样：低温度保证可复现，便于核对
TEMPERATURE = 0.1
TOP_P = 0.9
TOP_K = 20
MAX_NEW_TOKENS = 1024

BATCH_SIZE = 10          # 10 条一次性喂入
NUM_TEST_SAMPLES = 10    # 取前 10 张图各 1 条 qa
MAX_IMAGE_TOTAL_PIXELS = 4_000_000

SYSTEM_PROMPT = "你是一名机械制图专家。请仔细阅读图纸并回答问题。"
```

### 4.2 prompt 构造（直接问题 + 简短系统提示）

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": [
        {"type": "image_pil", "image_pil": img},   # vLLM 0.11 支持 PIL 直传
        {"type": "text", "text": question},
    ]},
]
```

> 说明：本数据集是直接中文问答（`question`→`answer`），SFT 训练时的确切 prompt 模板未知；先用通用「系统提示 + 问题」跑通，观察 10 条输出后若答非所问，再去匹配训练 prompt。

### 4.3 采样策略（确定性、可复现）

展平全部 8720 条 → 按文件顺序取**前 10 张不同图**，每图各取**第 1 条 qa**，凑 10 条。保证：每条来自不同图、确定性、覆盖不同图。`id` 用 `"<image_idx>_q0"`。

### 4.4 PYTHONPATH 清理兜底

脚本入口处移除指向 `/usr/local/lib/python3.10/dist-packages` 与 `.local/share/llm_lib` 的污染路径，保证即使忘了 `env -u PYTHONPATH` 也能正确 import：

```python
import sys
sys.path = [p for p in sys.path
            if p and "/python3.10/" not in p and "llm_lib" not in p]
```

## 5. 运行命令

```bash
# 推荐：用 anyrag 解释器，清空污染 PYTHONPATH
env -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES=0 \
  /share/project/kouqian/miniconda3/envs/anyrag/bin/python \
  scripts/batch_infer.py
```

## 6. 项目结构

```
MechVQA/
  ckpt/                          # 已有
  docs/superpowers/specs/        # 本设计文档
  scripts/
    batch_infer.py               # 主推理脚本（适配后）
    README.md                    # 使用说明：环境要求 / 参数 / 运行 / 输出说明
  outputs/                       # 推理结果 .jsonl（建议 gitignore）
```

## 7. 验证标准（完成定义）

1. 模型加载无报错，打印加载耗时。
2. 10 条样本全部生成非空中文 `pred_answer`。
3. 终端 SUMMARY 打印：总数 / 成功 / 截断 / 耗时。
4. `outputs/sft_infer_10.jsonl` 含 10 行，每行 `question / gt_answer / pred_answer` 并排，人工抽样核对回答合理。

## 8. 风险与应对

| 风险 | 应对 |
|---|---|
| `config.json` 中 `use_cache=false` 影响 vLLM | vLLM 自管 KV cache，忽略该 HF 训练字段，实测无影响 |
| prompt 模板不匹配训练格式，答非所问 | 先跑 10 条观察；不匹配再去 EasyR1/MechVLM 找数据处理脚本对齐 |
| `image_pil` 传入方式 vLLM 0.11 变动 | 已确认 `LLM.chat` 支持 PIL 直传；若失败回退 `{"type":"image","image":path}` |
| 非本环境人员误用系统 python | README 写明必须用 anyrag 解释器 + `env -u PYTHONPATH` |

## 9. 范围外（YAGNI）

- 全量 8720 条推理与自动评测指标（本轮仅 10 条验证）
- 多卡 TP=2（4B 无需）

## 10. RL 扩展（后续追加，已实现）

RL 模型补齐缺失 shard 后纳入同一脚本，顶部 `MODE="rl"` 切换：

- **Prompt**：`prompts/mech_r1.jinja`（从 EasyR1 整理），用 `Template(text.strip()).render(content=question)` 渲染，字面 `\n` 原样保留，复现训练 format_prompt；无独立 system。
- **输出**：thinking 模型生成 `<think>...</think><answer>...</answer>`，正则提取 `<answer>` 为 `pred_answer`，完整输出存 `raw_response`。
- **参数**：MAX_MODEL_LEN=16384、MAX_NEW_TOKENS=4096、TEMPERATURE=0.6（thinking 常用稍高温度）。
- **健壮性**：失败样本（未输出 `<answer>`，如 temperature 随机走偏）也写入 jsonl（`extract_ok:false` + `raw_response`），便于排查。
- **结果**：10/10 成功，数值答案全部正确；answer 文本带冗余「符合要求」元评论，系 RL reward 训练所致。

## 11. 验证结果汇总

| 模型 | 成功 | 答案 | 备注 |
|---|---|---|---|
| SFT | 10/10 | 抽样全正确 | 直接问答，输出简洁 |
| RL | 10/10 | 数值全正确 | thinking，answer 含冗余元评论 |
