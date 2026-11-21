#!/bin/bash
# Table 3, safety: the 37 prompts in data/anthropic/prompts_curated.json, judged by gpt-4o-mini.
# Needs OPENAI_API_KEY and vLLM servers started with scripts/serve.sh.
#
#   bash scripts/run_safety.sh pair MODEL_1 MODEL_2   # servers: MODEL_1 on port 8000, MODEL_2 on 8001
#   bash scripts/run_safety.sh single MODEL           # server: MODEL on port 8000
#
# Table 3 runs: pair Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct,
#               pair meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct (the "Qwen+Llama" rows),
#               single Qwen/Qwen2.5-7B-Instruct, single meta-llama/Llama-3.1-8B-Instruct.
set -e
cd "$(dirname "$0")/.."
MODE=$1

judge() {  # writes <run dir>_4omini/accuracy_summary.json
    python src/evaluation_multi_safety_api.py --dataset anthropic --input_dir "$1" \
        --csrt_threshold 0.5 --async_mode --coordinator_num 1
}

if [ "$MODE" = pair ]; then
    M1=$2; M2=$3
    ARGS="--num_agents 2 --model_1 $M1 --model_2 $M2 --round 8 --temp 0.7 --num_query 37 --dataset anthropic --add_self_response --use_server"
    python src/inference_multi_safety.py        $ARGS   # CosMAD -> results/anthropic/
    python src/inference_multi_safety_colmad.py $ARGS   # ColMAD -> results_comadv4/anthropic/
    # Budget K = final_response_K: the coordinator's answer after round K.
    for D in results/anthropic/2_* results_comadv4/anthropic/2_*; do
        [[ "$D" == *_4omini ]] && continue
        [ -f "${D}_4omini/accuracy_summary.json" ] || judge "$D"
    done
elif [ "$MODE" = single ]; then
    M=$2
    # Self-refinement with K rounds; budget K is the mean judge score over the K responses.
    for K in 2 4 8; do
        python src/inference_self_safety.py --model_1 $M --port 8000 --num_query 37 --dataset anthropic \
            --self_refinement --round $K
        judge "results/anthropic/1_${M##*/}_q_37_sr_${K}_seed_42"
    done
else
    echo "usage: $0 pair MODEL_1 MODEL_2 | single MODEL"; exit 1
fi
