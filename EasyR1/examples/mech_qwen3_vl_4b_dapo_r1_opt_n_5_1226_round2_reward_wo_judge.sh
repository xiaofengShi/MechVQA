#!/bin/bash
# ---------------------------------------------------------------
# [Author]       : shixiaofeng
# [Descriptions] :
set -x

EXP_NAME=${1:-mechvlmv2-vqa-4b_dapo_rollout_10_data_1226_round2_rolln10_ep10_reward_wo_judge}

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
    data.train_files=data/train_2.json \
    data.val_files=data/val.json \
    data.image_dir=${IMAGE_DIR:-./data/images} \
    data.max_prompt_length=1024 \
    data.max_pixels=262144 \
    data.format_prompt=./examples/format_prompt/mech_r1.jinja \
    worker.actor.model.model_path=$MODEL_PATH \
    worker.rollout.gpu_memory_utilization=0.85 \
    worker.rollout.n=10 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    worker.reward.reward_function="./examples/reward_function/mech_diverse_wo_judge.py:compute_score" \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=$CHECKPOINT_PATH \
    trainer.val_freq=10 \
    trainer.save_freq=10 \
    trainer.save_limit=5 \
    trainer.total_epochs=10