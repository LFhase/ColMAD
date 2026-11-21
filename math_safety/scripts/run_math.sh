#!/bin/bash
# Table 3, math: GSM8K and AIME-2024 with Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct.
# Models run in-process with vLLM, agent 1 on GPU 0 and agent 2 on GPU 1.
#   bash scripts/run_math.sh            # both datasets
#   bash scripts/run_math.sh gsm8k
set -e
cd "$(dirname "$0")/.."
export VLLM_WORKER_MULTIPROC_METHOD=spawn
QWEN=Qwen/Qwen2.5-7B-Instruct
LLAMA=meta-llama/Llama-3.1-8B-Instruct
DATASETS=${1:-"gsm8k aime2024"}
DEBATE="--num_agents 2 --num_query 500 --round 8 --temp 0.7 --add_self_response"

for DATASET in $DATASETS; do
    # Single agent: self-consistency over 16 samples; budget K uses the first K samples.
    for M in $QWEN $LLAMA; do
        python src/inference_self.py --model_1 $M --gpu_1 0 --num_query 500 --round 1 --temp 0.7 \
            --self_consistency --num_samples 16 --dataset $DATASET
    done
    # Two agents, 8 rounds with early stopping; budget K uses the first K/2 rounds.
    for PAIR in "$QWEN $QWEN" "$QWEN $LLAMA"; do
        read -r M1 M2 <<< "$PAIR"
        python src/inference_multi.py        $DEBATE --model_1 $M1 --gpu_1 0 --model_2 $M2 --gpu_2 1 --dataset $DATASET  # CosMAD
        python src/inference_multi_colmad.py $DEBATE --model_1 $M1 --gpu_1 0 --model_2 $M2 --gpu_2 1 --dataset $DATASET  # ColMAD
    done
    # Scores: results/$DATASET/*/accuracy_summary_sample_K.json
    # (single agent: the model's accuracy; two agents: total_best, the majority answer of the last round).
    python -m src.evaluation_self  --dataset $DATASET
    python -m src.evaluation_multi --dataset $DATASET
done
