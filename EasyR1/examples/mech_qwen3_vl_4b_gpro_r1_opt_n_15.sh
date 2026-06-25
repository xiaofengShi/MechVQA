#!/bin/bash
# ---------------------------------------------------------------
# [Author]       : shixiaofeng
# [Descriptions] :
set -x

EXP_NAME=${1:-qwen3_vl_4b_mech_sft_reasoning_cooling_and_gpro_roolout_n_15}

MODEL_PATH="${MODEL_PATH:-./ckpt/MechVQA_SFT}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./outputs/checkpoints/${EXP_NAME}}"
WANDB_DIR=wandb/$EXP_NAME

mkdir -p $WANDB_DIR $CHECKPOINT_PATH

export HF_ENDPOINT="https://hf-mirror.com"
export HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
export WANDB_MODE="offline"
export WANDB_PROJECT="MechVLM"
export WANDB_DIR=$WANDB_DIR

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=data/train.json \
    data.val_files=data/test.json \
    data.format_prompt=./examples/format_prompt/mech_r1.jinja \
    worker.actor.model.model_path=$MODEL_PATH \
    worker.rollout.gpu_memory_utilization=0.85 \
    worker.rollout.n=15 \
    worker.reward.reward_function="./examples/reward_function/mech_opt.py:compute_score" \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=$CHECKPOINT_PATH \
    trainer.val_freq=10 \
    trainer.save_freq=10 \
    trainer.save_limit=5 \
    trainer.total_epochs=10