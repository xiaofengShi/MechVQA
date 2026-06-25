# MechVQA 推理脚本

基于 vLLM 对 `ckpt/MechVQA_SFT`（Qwen3-VL-4B-Instruct 微调）做多模态批量推理验证。

## 数据（自包含）

本脚本**不依赖任何外部数据**，测试数据随仓库内置：

```
data/
  test_samples.jsonl    # 10 条扁平样本（id/image_path/question/gt_answer/...）
  images/               # 10 张机械图纸（约 295KB），image_path 为相对路径
```

- 样本来源于 vote-based VQA 数据集（`mechnical_data_vote_based.jsonl`）的前 10 张图各第 1 条 qa。
- `image_path` 使用相对项目根的路径（如 `data/images/sample_00000.gif`），**需在项目根目录运行脚本**。
- 想换/增测试样本：直接编辑 `data/test_samples.jsonl` 并把图片放进 `data/images/` 即可。

## 环境

| 依赖 | 要求 |
|---|---|
| Python 解释器 | `/share/project/kouqian/miniconda3/envs/anyrag/bin/python`（3.12） |
| vLLM | 0.11.0（已原生支持 `qwen3_vl`） |
| transformers | 4.57.1（与模型 `config.json` 一致） |
| GPU | 单张 A800-80GB（4B bf16 约 8GB，TP=1） |

> ⚠️ **PYTHONPATH 坑**：系统 `PYTHONPATH` 指向 python3.10 的旧包，会污染 import。运行时**必须** `env -u PYTHONPATH`（脚本入口也做了 `sys.path` 清理兜底）。

> 注：模型权重 `ckpt/` 未随仓库提供（约 14G，已 gitignore），需另行放置到 `ckpt/MechVQA_SFT/`。

## 运行

在**项目根目录**执行：

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 \
  /share/project/kouqian/miniconda3/envs/anyrag/bin/python \
  scripts/batch_infer.py
```

- 单条命令即可，参数在脚本顶部「配置区」修改。
- 支持断点续跑：中断后重跑会读取已有输出，跳过已成功的样本。

## 配置区关键参数（`scripts/batch_infer.py` 顶部）

| 参数 | 默认 | 说明 |
|---|---|---|
| `MODEL_PATH` | `ckpt/MechVQA_SFT` | SFT 权重路径（相对项目根） |
| `INPUT_JSONL` | `data/test_samples.jsonl` | 内置扁平测试数据 |
| `OUTPUT_JSONL` | `outputs/sft_infer_test.jsonl` | 推理结果（GT 与预测并排） |
| `TENSOR_PARALLEL_SIZE` | 1 | 4B 单卡即可 |
| `MAX_MODEL_LEN` | 8192 | 上下文长度 |
| `MAX_NEW_TOKENS` | 1024 | 直接问答足够 |
| `TEMPERATURE` | 0.1 | 低温度，便于核对 |
| `BATCH_SIZE` | 10 | 一次喂入 `llm.chat` 的样本数 |
| `SYSTEM_PROMPT` | 机械制图专家 | 简短系统提示 |

## 输出格式（`outputs/sft_infer_test.jsonl`）

每行一个 JSON：

```json
{
  "id": "img00000_q0",
  "image_path": "data/images/sample_00000.gif",
  "question": "图纸中标注的零件总长度是多少？",
  "gt_answer": "根据图纸内容，零件的总长度标注为5.5英寸...",
  "pred_answer": "<模型生成的回答>",
  "capability": "识别",
  "subcategory": "零件定位与尺寸识别",
  "difficulty": "简单",
  "finish_reason": "stop"
}
```

> `finish_reason == "length"` 表示触达 `MAX_NEW_TOKENS` 截断，可调大该值。

## 与参考脚本的差异

参考脚本 `EasyR1/data_mechvlm/qwen_vl_vllm_batch_infer.py` 面向 **Qwen3-VL-235B-Thinking**，本脚本针对 **4B-Instruct** 做了如下适配：

| 项 | 参考脚本（235B-Thinking） | 本脚本（4B-Instruct） |
|---|---|---|
| 引擎参数 | TP=8 / 51200 / 40960 | **TP=1 / 8192 / 1024** |
| thinking | `enable_thinking=True` | **去掉**（非 thinking 模型） |
| 有效性判定 | 要求 `</think>` + JSON | **非空文本即可** |
| 数据 | 外部 jsonl，`{id, prompt, image_path}` | **项目内置扁平 jsonl**，自包含 |
| 输出 | verdict JSON | **GT 与预测并排** |

保留了：断点续跑、坏图跳过、JSONL 增量写入、时间统计、PIL 像素上限缩放、批处理。
