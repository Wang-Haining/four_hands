#!/bin/bash
#SBATCH --job-name=gemma2_7b
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

mkdir -p logs
. .venv/bin/activate

# model configuration
MODEL="google/gemma-2-9b-it"
STAGES=("basic" "zero-shot" "few-shot" "cot")

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
            --temperature 0.6 \
            --num_runs 50 \
            --seed 42

        # add small delay between runs
        sleep 5
    done
done

echo "gemma-2-9b-it experiments completed!"