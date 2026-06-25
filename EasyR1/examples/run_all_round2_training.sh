#!/bin/bash
# 依次运行所有 mech_qwen3_vl_4b_dapo_r1_opt_n_5_1226_round2 相关训练脚本

set -e  # 遇到错误时退出

# cd 到项目根目录（examples 的上一级），确保 verl 能正确引入
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "开始运行所有 round2 训练脚本"
echo "工作目录: $(pwd)"
echo "=========================================="

# 脚本列表（相对于项目根目录的路径）
SCRIPTS=(
    "examples/mech_qwen3_vl_4b_dapo_r1_opt_n_5_1226_round2_reward_01.sh"
    "examples/mech_qwen3_vl_4b_dapo_r1_opt_n_5_1226_round2_reward_F1.sh"
    "examples/mech_qwen3_vl_4b_dapo_r1_opt_n_5_1226_round2_reward_w_judge.sh"
    "examples/mech_qwen3_vl_4b_dapo_r1_opt_n_5_1226_round2_reward_wo_judge.sh"
)

# 依次运行每个脚本
for script in "${SCRIPTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "开始运行: $script"
    echo "时间: $(date)"
    echo "=========================================="

    if [ -f "$script" ]; then
        bash "$script"
        echo ""
        echo "完成: $script"
        echo "时间: $(date)"
    else
        echo "警告: 脚本不存在 - $script"
    fi
done

echo ""
echo "=========================================="
echo "所有训练脚本运行完成"
echo "时间: $(date)"
echo "=========================================="
