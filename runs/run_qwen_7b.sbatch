#!/bin/bash
#SBATCH --job-name=qwen2.5_7b
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

mkdir -p logs
. .venv/bin/activate

# model configuration
MODEL="Qwen/Qwen2.5-7B-Instruct"
STAGES=("basic" "zero-shot" "few-shot" "cot")
TEMPERATURES=(0 0.6 1)

# run grid search
for stage in "${STAGES[@]}"; do
    for temp in "${TEMPERATURES[@]}"; do
        echo "Running experiment with:"
        echo "Model: $MODEL"
        echo "Stage: $stage"
        echo "Temperature: $temp"

        python run_llm.py \
            --model "$MODEL" \
            --stage "$stage" \
            --temperature "$temp" \
            --num_runs 2 \
            --seed 42

        # add small delay between runs
        sleep 5
    done
done

echo "8B model experiments completed!"