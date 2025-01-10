#!/bin/bash
#SBATCH --job-name=llama_70b_auth
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
MODEL="meta-llama/Llama-3.1-70B-Instruct"
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
            --num_runs 30 \
            --seed 42

        # add small delay between runs
        sleep 5
    done
done

echo "70B model experiments completed!"