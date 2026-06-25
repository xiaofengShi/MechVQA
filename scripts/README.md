# MechVQA 推理脚本

基于 vLLM 对 `ckpt/` 下的 SFT / RL 模型（Qwen3-VL-4B-Instruct 微调）做多模态批量推理验证。**一个脚本支持两种模式**，顶部 `MODE` 切换。

## 两种模式

| 模式 | 模型 | Prompt 构造 | 输出格式 | 答案提取 |
|---|---|---|---|---|
| `sft` | `ckpt/MechVQA_SFT` | system 提示 + 图片 + question | 自由文本 | 整段输出即答案 |
| `rl` | `ckpt/MechVQA_RL` | 用 `prompts/mech_r1.jinja` 渲染 question（复现 RL 训练 format_prompt，无 system） | `<think>...</think><answer>...</answer>` | 正则提取 `<answer>...</answer>` |

> RL 是 thinking 模型：训练时用 EasyR1 的 `mech_r1.jinja` 作为 format_prompt（`Template(text.strip()).render(content=question)`），其字面 `\n` 不被 Jinja2 转义，推理时原样复现。

## 数据（自包含）

```
data/
  test_samples.jsonl    # 10 条扁平样本（id/image_path/question/gt_answer/...）
  images/               # 10 张机械图纸（约295KB），image_path 为相对路径
prompts/
  mech_r1.jinja         # RL 训练用 format_prompt（从 EasyR1 整理）
```

- 样本来源于 vote-based VQA 数据集前 10 张图各第 1 条 qa。
- `image_path` 使用相对项目根的路径，**需在项目根目录运行脚本**。
- 换/增样本：编辑 `data/test_samples.jsonl` 并把图片放进 `data/images/`。

## 环境

| 依赖 | 要求 |
|---|---|
| Python 解释器 | `/share/project/kouqian/miniconda3/envs/anyrag/bin/python`（3.12） |
| vLLM | 0.11.0（原生支持 `qwen3_vl`） |
| transformers | 4.57.1（与模型 `config.json` 一致） |
| GPU | 单张 A800-80GB（4B bf16 约 8.8GB，TP=1） |

> ⚠️ **两个环境坑**（脚本均已内置兜底，但运行命令仍建议显式处理）：
> - `PYTHONPATH` 污染：系统指向 python3.10 旧包 → 必须 `env -u PYTHONPATH`（脚本入口也清理 `sys.path`）。
> - Triton 缓存不可写：`HOME=/root` 不可写 → 脚本入口把 `TRITON_CACHE_DIR` 设到项目内 `.triton_cache`。

> 模型权重 `ckpt/` 未随仓库提供（约14G，已 gitignore），需另行放置到 `ckpt/MechVQA_SFT/` 与 `ckpt/MechVQA_RL/`。

## 运行

在**项目根目录**执行（改 `MODE` 切换 SFT/RL）：

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 \
  /share/project/kouqian/miniconda3/envs/anyrag/bin/python \
  scripts/batch_infer.py
```

- 支持断点续跑：中断后重跑读取已有输出，跳过已成功样本。
- 失败样本（如 RL 未输出 `<answer>`）也写入 jsonl（`extract_ok: false` + `raw_response`），便于排查。

## 配置区关键参数（`scripts/batch_infer.py` 顶部）

按 `MODE` 自动设置；通用项：

| 参数 | sft | rl | 说明 |
|---|---|---|---|
| `MODEL_PATH` | `ckpt/MechVQA_SFT` | `ckpt/MechVQA_RL` | 权重路径 |
| `PROMPT_TEMPLATE` | None | `prompts/mech_r1.jinja` | None=直接问；否则 jinja 渲染 |
| `SYSTEM_PROMPT` | 机械制图专家 | None | RL 用 format_prompt，无 system |
| `ANSWER_TAG` | None | `answer` | None=整段输出；否则提取 `<tag>` |
| `MAX_MODEL_LEN` | 8192 | 16384 | RL thinking 输出较长 |
| `MAX_NEW_TOKENS` | 1024 | 4096 | |
| `TEMPERATURE` | 0.1 | 0.6 | thinking 模型常用稍高温度 |
| `TENSOR_PARALLEL_SIZE` | 1 | 1 | 4B 单卡 |

## 输出格式（`outputs/{sft,rl}_infer_test.jsonl`）

每行一个 JSON（RL 额外含 `raw_response`、`extract_ok`）：

```json
{
  "id": "img00000_q0",
  "image_path": "data/images/sample_00000.gif",
  "question": "图纸中标注的零件总长度是多少？",
  "gt_answer": "根据图纸内容...5.5英寸...",
  "pred_answer": "<提取后的答案>",
  "extract_ok": true,
  "capability": "识别", "subcategory": "...", "difficulty": "简单",
  "finish_reason": "stop",
  "raw_response": "<完整 think+answer，仅 RL 模式>"
}
```

## 验证结果（10 条）

| 模型 | 成功 | 答案正确性 |
|---|---|---|
| SFT | 10/10 | 抽样核对全部正确（零件长度/名称/锥度） |
| RL | 10/10 | 数值答案全部正确（answer 文本带冗余「符合要求」元评论，系 RL reward 训练所致，核心数值清晰） |

## 与参考脚本的差异

参考脚本 `EasyR1/data_mechvlm/qwen_vl_vllm_batch_infer.py` 面向 **Qwen3-VL-235B-Thinking**，本脚本针对 **4B-Instruct（SFT/RL）** 适配：

| 项 | 参考脚本（235B-Thinking） | 本脚本 |
|---|---|---|
| 引擎参数 | TP=8 / 51200 / 40960 | **TP=1 / 8192~16384 / 1024~4096** |
| thinking | `enable_thinking=True` | RL 走 `<think>/<answer>` 标签提取；SFT 直接问答 |
| 有效性判定 | 要求 `</think>`+JSON | 非空（SFT）/ 含 `<answer>`（RL） |
| 数据 | 外部 jsonl | **项目内置扁平 jsonl**，自包含 |
| 输出 | verdict JSON | **GT 与预测并排** |

保留：断点续跑、坏图跳过、JSONL 增量写入、时间统计、PIL 像素缩放、批处理。
