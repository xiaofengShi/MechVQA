#!/bin/bash
# ---------------------------------------------------------------
# [Author]       : shixiaofeng
# [Descriptions] : <JUDGE_HOST> 8卡80G 部署4个 gpt-oss-120b judge 实例
# [Usage]        : bash scripts/setup_vllm_judge_gpt_oss_120b.sh
# ---------------------------------------------------------------

MODEL_PATH=${MODEL_PATH:-./ckpt/gpt-oss-120b}
MODEL_NAME="gpt-oss-120b"
TP_SIZE=2
MAX_MODEL_LEN=16384
GPU_MEM=0.85

echo "============================================"
echo "启动 4 个 vLLM judge 实例"
echo "模型: $MODEL_PATH"
echo "GPU: 8卡80G, 每实例 2卡, 共 4 实例"
echo "端口: 8086, 8087, 8088, 8089"
echo "============================================"

# 实例1: GPU 0-1, 端口 8086
CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve $MODEL_PATH \
  --served-model-name $MODEL_NAME \
  --port 8086 \
  --tensor-parallel-size $TP_SIZE \
  --gpu-memory-utilization $GPU_MEM \
  --max-model-len $MAX_MODEL_LEN \
  --trust-remote-code \
  --dtype auto \
  > logs/judge_8086.log 2>&1 &
echo "实例1 启动: GPU 0-1, port 8086, PID $!"

# 实例2: GPU 2-3, 端口 8087
CUDA_VISIBLE_DEVICES=2,3 nohup vllm serve $MODEL_PATH \
  --served-model-name $MODEL_NAME \
  --port 8087 \
  --tensor-parallel-size $TP_SIZE \
  --gpu-memory-utilization $GPU_MEM \
  --max-model-len $MAX_MODEL_LEN \
  --trust-remote-code \
  --dtype auto \
  > logs/judge_8087.log 2>&1 &
echo "实例2 启动: GPU 2-3, port 8087, PID $!"

# 实例3: GPU 4-5, 端口 8088
CUDA_VISIBLE_DEVICES=4,5 nohup vllm serve $MODEL_PATH \
  --served-model-name $MODEL_NAME \
  --port 8088 \
  --tensor-parallel-size $TP_SIZE \
  --gpu-memory-utilization $GPU_MEM \
  --max-model-len $MAX_MODEL_LEN \
  --trust-remote-code \
  --dtype auto \
  > logs/judge_8088.log 2>&1 &
echo "实例3 启动: GPU 4-5, port 8088, PID $!"

# 实例4: GPU 6-7, 端口 8089
CUDA_VISIBLE_DEVICES=6,7 nohup vllm serve $MODEL_PATH \
  --served-model-name $MODEL_NAME \
  --port 8089 \
  --tensor-parallel-size $TP_SIZE \
  --gpu-memory-utilization $GPU_MEM \
  --max-model-len $MAX_MODEL_LEN \
  --trust-remote-code \
  --dtype auto \
  > logs/judge_8089.log 2>&1 &
echo "实例4 启动: GPU 6-7, port 8089, PID $!"

echo ""
echo "============================================"
echo "所有实例已启动, 日志在 logs/judge_*.log"
echo "检查状态: tail -f logs/judge_8086.log"
echo "健康检查: curl http://localhost:8086/health"
echo "============================================"

wait
