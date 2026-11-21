#!/bin/bash
# Start a vLLM OpenAI-compatible server for one agent (agent i listens on port 8000 + i - 1).
#   bash scripts/serve.sh MODEL PORT GPU
# The safety scripts send requests with the same MODEL string, so pass the same id to both.
set -e
MODEL=$1; PORT=$2; GPU=$3
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" --max-model-len 32768
