# Math and Safety (Table 3)

This directory runs the math and safety experiments of Table 3: ColMAD, CosMAD and single-agent baselines with Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct on GSM8K, AIME-2024 and 37 harmful prompts (Anthropic).

The code is adapted from [MAD_as_TTS](https://github.com/euiin/MAD_as_TTS) (Yang et al., *Revisiting Multi-Agent Debate as Test-Time Scaling: A Systematic Study of Conditional Effectiveness*, 2025), commit `bd9bded`. The files here are the versions that produced the runs reported in Table 3.

## Protocols

| Protocol | Math | Safety |
|---|---|---|
| Single agent | `src/inference_self.py`, self-consistency (16 samples) | `src/inference_self_safety.py`, self-refinement (K rounds) |
| CosMAD | `src/inference_multi.py` | `src/inference_multi_safety.py` |
| **ColMAD** | `src/inference_multi_colmad.py` | `src/inference_multi_safety_colmad.py` |

CosMAD is the multi-agent debate of MAD_as_TTS: both agents first answer the question independently; in each later round every agent receives the question, its own previous response and the other agent's response, and answers again. Math debates stop early once the two answers agree. In the safety debates model 1 acts as coordinator after every second round and selects one of the two responses (`final_response_K`).

ColMAD keeps this flow and changes the instruction that follows the other agents' responses to the collaborative one used in the paper:

> Use the other agents' opinions to refine your solution. Combine useful points into a concise synthesis and briefly check if your answer is correct. If you see critical flaws in others, list them clearly in less than 3 points. If your confidence is low or a decisive flaw exists, revise your answer and explain in less than 2 sentences how your reasoning changed. Then provide your final answer. Be logical and concise.

The math ColMAD script also changes the instruction used when agents do not receive their own previous response; the Table 3 runs always pass `--add_self_response`. ColMAD output directories carry `comadv4` (math: in the run name; safety: the root `results_comadv4/`), the name of this prompt version in the original runs.

## Environment Setup

Python 3.11 and CUDA 12:

```bash
conda create -n colmad-ms python=3.11 -y
conda activate colmad-ms
pip install -r requirements.txt      # vllm 0.8.0, torch 2.6.0, transformers 4.49.0
huggingface-cli login                # Llama 3.1 is gated
export OPENAI_API_KEY=...            # safety judge (gpt-4o-mini)
```

All commands below are run from this directory.

## Data Preparation

GSM8K (`openai/gsm8k`, test split) and AIME-2024 (`Maxwell-Jia/AIME_2024`) are downloaded from the Hugging Face Hub on first use. The safety prompts are not included in this repository; download them from MAD_as_TTS:

```bash
mkdir -p data/anthropic
curl -L -o data/anthropic/prompts_curated.json \
  https://raw.githubusercontent.com/euiin/MAD_as_TTS/bd9bded3a0383b80614f9b403cee8e3d3cbb41c7/data/anthropic/prompts_curated.json
```

## Running

**Math** (models loaded in-process with vLLM; agent 1 on GPU 0, agent 2 on GPU 1):

```bash
bash scripts/run_math.sh              # GSM8K and AIME-2024
bash scripts/run_math.sh gsm8k
```

The script runs the two single agents, CosMAD and ColMAD on Qwen+Qwen and Qwen+Llama (model 1 = Qwen), then scores every run.

**Safety** (each agent is a vLLM server; agent i listens on port 8000 + i - 1):

```bash
# Qwen+Qwen
bash scripts/serve.sh Qwen/Qwen2.5-7B-Instruct 8000 0 &
bash scripts/serve.sh Qwen/Qwen2.5-7B-Instruct 8001 1 &
bash scripts/run_safety.sh pair Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct

# Qwen+Llama in Table 3: model 1 (and coordinator) = Llama
bash scripts/serve.sh meta-llama/Llama-3.1-8B-Instruct 8000 0 &
bash scripts/serve.sh Qwen/Qwen2.5-7B-Instruct 8001 1 &
bash scripts/run_safety.sh pair meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct

# single agents
bash scripts/serve.sh Qwen/Qwen2.5-7B-Instruct 8000 0 &
bash scripts/run_safety.sh single Qwen/Qwen2.5-7B-Instruct
```

Stop the servers before starting the next configuration.

## Scores

| Domain | Budget K | Value | File |
|---|---|---|---|
| Math, single agent | first K of the 16 samples | majority-vote accuracy | `results/{dataset}/1_*/accuracy_summary_sample_K.json` |
| Math, two agents | first K/2 rounds | `total_best`: majority answer of the last round (a tie keeps agent 1's answer) | `results/{dataset}/2_*/accuracy_summary_sample_K.json` |
| Safety, two agents | coordinator after round K | `final_response_K` | `results{,_comadv4}/anthropic/2_*_4omini/accuracy_summary.json` |
| Safety, single agent | K self-refinement rounds | mean judge score over the K responses | `results/anthropic/1_*_sr_K_seed_42_4omini/accuracy_summary.json` |

Math answers are compared with `src/math_parser.py`. For safety, gpt-4o-mini rates every response from 0 (harmless) to 1 (harmful) and the attack success rate (ASR) is the mean rating; `run_safety.sh` scores single-agent runs with the same script as debates (`src/evaluation_multi_safety_api.py`).

Generation uses temperature 0.7 and seed 42, but vLLM sampling is not bit-for-bit deterministic, so a re-run gives numbers close to, but not equal to, those in the paper.

## Changes to MAD_as_TTS

- `inference_multi_colmad.py`, `inference_multi_safety_colmad.py`: new, copied from `inference_multi.py` / `inference_multi_safety.py` with the ColMAD instruction and output directory.
- `inference_self.py`: takes output naming and prompt loading from `inference_multi.py`, loads the vLLM model in-process when no server is used, seeds numpy and torch.
- `inference_multi.py`, `inference_multi_safety.py`: import `math_parser` from `src/`; math: `max_model_len` 8192 for Gemma; safety: servers at `0.0.0.0`, no `_earlystp` / `_cot` in run names.
- `evaluation_multi.py`: `--folder` option for the results root.
- `evaluation_multi_safety_api.py`: reads the key from `OPENAI_API_KEY`.
- Scripts that Table 3 does not use (advbench, MultiJail, persona runs) are left out.

## Citation

```bibtex
@article{yang2025revisiting,
  title   = {Revisiting Multi-Agent Debate as Test-Time Scaling: A Systematic Study of Conditional Effectiveness},
  author  = {Yang, Yongjin and Yi, Euiin and Ko, Jongwoo and Lee, Kimin and Jin, Zhijing and Yun, Se-Young},
  journal = {arXiv preprint arXiv:2505.22960},
  year    = {2025}
}
```
