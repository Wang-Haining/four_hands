#!/bin/bash
#SBATCH --job-name=llama3.1_70b
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

mkdir -p logs
. .venv/bin/activate

# model configuration
MODEL="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
STAGES=("basic" "zero-shot" "few-shot" "cot")

# run grid search
for stage in "${STAGES[@]}"; do
    echo "Running experiment with:"
    echo "Model: $MODEL"
    echo "Stage: $stage"

    python run_llm.py \
        --model "$MODEL" \
        --stage "$stage" \
        --temperature 0.6 \
        --num_runs 50 \
        --seed 42

    # add small delay between runs
    sleep 5
done

echo "Meta-Llama-3.1-70B-Instruct-AWQ-INT4 experiments completed!"