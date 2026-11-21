# (NeurIPS'26) Towards Scalable Oversight with Collaborative Multi-Agent Debate in Error Detection

<p align="center">
  <strong>ColMAD · NeurIPS 2026</strong><br>
  <a href="https://arxiv.org/abs/2510.20963">
    <img src="https://img.shields.io/badge/arXiv-ColMAD-b31b1b?logo=arxiv&style=flat-square" />
  </a>
  <a href="https://github.com/psunlpgroup/ReaLMistake">
    <img src="https://img.shields.io/badge/Benchmark-ReaLMistake-181717?logo=github&style=flat-square" />
  </a>
</p>

This repository contains the official implementation of **ColMAD** (Towards Scalable Oversight with Collaborative Multi-Agent Debate in Error Detection, **NeurIPS 2026**).

Detecting errors in the responses of large language models (LLMs) is central to scalable oversight, yet LLMs are unreliable at diagnosing their own mistakes without external feedback. Multi-agent debate (MAD) is a natural source of such feedback: several LLMs examine a response and cross-check each other. Prior MAD protocols, however, frame debate as a zero-sum game in which each debater tries to win rather than to find the truth. This leads to *debate hacking*: debaters mislead the judge by misreading the task or making overconfident claims, and the debate can end up below a single agent.

ColMAD reframes MAD as a non-zero-sum game. The two debaters still defend opposite answers, but they are asked to criticize each other in a supportive way, to surface decisive errors on either side and to adopt valid points of the other debater, so that the judge decides from more complete evidence.

The repository includes the single-agent baseline, ColMAD, and the baselines compared in the paper (competitive debate, multi-persona debate with a moderator, society-of-minds debate and voting), together with the evaluation used for the paper's tables.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Running the Protocols](#running-the-protocols)
- [Evaluation and Reproduction](#evaluation-and-reproduction)
- [Extending the Repository](#extending-the-repository)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9 or later. The code only needs `openai`, `numpy` and `scikit-learn`.

All model calls go to an OpenAI-compatible endpoint, OpenRouter by default:

```bash
cp .env.example .env              # fill in OPENROUTER_API_KEY
set -a; source .env; set +a
```

To use another endpoint, set `LLM_BASE_URL` and, if its key is in a different variable, `LLM_API_KEY_ENV`, for example:

```bash
export LLM_BASE_URL=https://api.openai.com/v1 LLM_API_KEY_ENV=OPENAI_API_KEY
```

## Data Preparation

We evaluate on [ReaLMistake](https://github.com/psunlpgroup/ReaLMistake) (Kamoi et al., COLM 2024): binary error detection on responses of GPT-4 and Llama 2 70B for three tasks, math word problem generation (Math), fine-grained fact verification (Fact) and answerability classification (Ans).

Download `data.zip` from the ReaLMistake repository and unzip it in the repository root (password `open-realmistake`):

```text
data/
├── math_word_problem_generation/{gpt-4-0613,Llama-2-70b-chat-hf}.jsonl
├── finegrained_fact_verification/{gpt-4-0613,Llama-2-70b-chat-hf}.jsonl
└── answerability_classification/{gpt-4-0613,Llama-2-70b-chat-hf}.jsonl
```

Another location can be set with `REALMISTAKE_DATA`. The ReaLMistake authors ask that examples of the dataset are not published online in plain text. The result files written by this code contain the benchmark inputs, so please keep them out of public repositories (`outputs/` is git-ignored).

## Running the Protocols

| Protocol | Script | What happens | Calls per sample |
|---|---|---|---|
| Single agent | `colmad/run_single.py` | one critique per model; round 0 of every protocol below | 1 |
| **ColMAD** | `colmad/run_colmad.py` | two debaters with collaborative prompts, rounds 1-2, then a judge | 5 |
| CopMAD | `colmad/run_copmad.py` | two debaters with competitive prompts, rounds 0-2, then a judge | 7 |
| MP | `colmad/run_mp.py` | competitive debate with a moderator after every round, who may stop it | 9 |
| CosMAD | `colmad/run_cosmad.py` | society of minds: both agents revise simultaneously until they agree; no judge | at most 4 |
| Voting | `colmad/evaluate.py` | agreeing labels are kept, disagreements are decided by a coin flip | 0 |

In ColMAD, CopMAD and MP, model 1 (`-j1`) is Debater Alice and the judge (or moderator), and model 2 (`-j2`) is Debater Bob. Alice defends her single-agent label and Bob the other label.

```bash
# round 0: single-agent critiques of both models (temperature 0)
python -m colmad.run_single -m mini
python -m colmad.run_single -m l70b

# debates
python -m colmad.run_colmad -j1 mini -j2 l70b
python -m colmad.run_copmad -j1 mini -j2 l70b
python -m colmad.run_mp     -j1 mini -j2 l70b
python -m colmad.run_cosmad -j1 mini -j2 l70b
```

Short model names: `mini` (gpt-4o-mini-2024-07-18), `l70b` (Llama-3.1-70B-Instruct), `r1` (DeepSeek-R1), `ms26` (Mistral-Small-2603), `qwen` (Qwen2.5-72B-Instruct). Any other name is passed to the endpoint unchanged.

Common options:

- `--tasks math,fact,ans` selects tasks (default: all three);
- `-bl llama2` uses the Llama 2 70B split (160 samples per task; default: the GPT-4 split, 140 samples per task);
- `-r` sets the number of debate rounds (default 2), `-t` / `-s` the temperature and seed;
- `-o` sets the output directory (default `outputs/`); existing result files are skipped unless `--overwrite` is given.

Outputs:

```text
outputs/single/{base_llm}/{model}/{task}-temp0.json
outputs/{protocol}/{base_llm}/{model_1}__{model_2}/{task}-r2-temp0.json
```

Each file is a JSON list with one record per benchmark sample: the benchmark fields, every prompt and response of the run (`critic_prompt_{a,b}_r{k}`, `critic_response_{a,b}_r{k}`, `judge_prompt`, `judge_response`; for MP `judge_prompt_r{k}`, `judge_response_r{k}`), and the protocol's answer in `prediction`.

## Evaluation and Reproduction

```bash
python -m colmad.evaluate all -j1 mini -j2 l70b      # Voting and every protocol whose files exist
python -m colmad.evaluate colmad -j1 mini -j2 l70b
python -m colmad.evaluate single -m mini
```

This prints the F1 / F2 of each task and their average. The evaluation follows the conventions of the paper:

- `error` is the positive class and F2 is the primary metric; the average is the mean of the three per-task values;
- gold labels are the benchmark labels with six corrections (`colmad/labels.py`);
- judge verdicts are decoded with the two options printed in the stored judge prompt;
- agreement gate: when the two single-agent labels agree, that label is the answer; it is on for ColMAD and CosMAD and off for CopMAD and MP (`--gate on|off` overrides it);
- Voting draws its coin flips from one `numpy.random.RandomState(2333)` stream over Math, Fact and Ans in that order.

To run every protocol on the six model pairs of Table 1 (GPT-4 split, temperature 0, two rounds):

```bash
bash scripts/run_table1.sh
```

The prompts and the debate flow are those of the runs reported in the paper: replaying the stored transcripts of those runs through this code reproduces every stored prompt byte for byte, and `colmad/evaluate.py` on those transcripts reproduces the Table 1 cells. For ColMAD on gpt-4o-mini:Llama-3.1-70B, gpt-4o-mini:Mistral-Small-2603 and Llama-3.1-70B:Mistral-Small-2603, the judge's quote check was run with `--quote_check lowercase_problem`; `scripts/run_table1.sh` sets it for these pairs.

Model outputs from API endpoints are not deterministic even at temperature 0, and providers change their serving stacks, so a re-run gives numbers close to, but not equal to, those in the paper. Mistral-7B-Instruct-v0.3 and QwQ-32B are no longer served by the providers used for the paper.

## Extending the Repository

### Adding a Model

Add a short name to `MODEL_ALIASES` and its endpoint id to `OPENROUTER_IDS` in `colmad/llm.py`. Reasoning models that need a larger completion budget go in `REASONING_MODELS`. Then run `colmad.run_single` for the model before using it in a debate, since round 0 of every protocol is read from the single-agent outputs.

### Adding a Protocol

A protocol is a `run(single_a, single_b, j1, j2, ...)` function that takes the two single-agent result lists and returns one record per sample with a `prediction`. The shared pieces are in `colmad/debate.py`: `compose_problem` (the problem shown to every agent), `fill_debater` / `fill_judge` (template filling) and `extract_transcript` (the transcript the next speaker and the judge see, with quote verification). All model calls go through `colmad.llm.chat_batch`. To score the new protocol, add it to `PROTOCOLS` and `GATE_DEFAULT` in `colmad/evaluate.py`.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── colmad/
│   ├── llm.py              # batched calls to an OpenAI-compatible endpoint, model names
│   ├── data.py             # benchmark loading, result-file paths
│   ├── labels.py           # label parsing, gold-label corrections
│   ├── metrics.py          # F1 / F2 / MCC
│   ├── debate.py           # prompt filling and transcripts shared by the debate protocols
│   ├── prompts/            # prompt templates of every protocol
│   ├── run_single.py       # single-agent baseline (round 0)
│   ├── run_colmad.py       # ColMAD
│   ├── run_copmad.py       # CopMAD
│   ├── run_mp.py           # MP
│   ├── run_cosmad.py       # CosMAD
│   └── evaluate.py         # scores and Voting
└── scripts/
    └── run_table1.sh       # all protocols on the six pairs of Table 1
```

## Citation

If you use this repository, please cite our paper and the ReaLMistake benchmark.

```bibtex
@inproceedings{chen2026colmad,
  title     = {Towards Scalable Oversight with Collaborative Multi-Agent Debate in Error Detection},
  author    = {Chen, Yongqiang and Niu, Gang and Cheng, James and Han, Bo and Sugiyama, Masashi},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026}
}
```

```bibtex
@inproceedings{kamoi2024realmistake,
  title     = {Evaluating {LLMs} at Detecting Errors in {LLM} Responses},
  author    = {Kamoi, Ryo and Das, Sarkar Snigdha Sarathi and Lou, Renze and Ahn, Jihyun Janice and Zhao, Yilun and Lu, Xiaoxin and Zhang, Nan and Zhang, Yusen and Zhang, Ranran Haoran and Vummanthala, Sujeeth Reddy and Dave, Salika and Qin, Shaobo and Cohan, Arman and Yin, Wenpeng and Zhang, Rui},
  booktitle = {First Conference on Language Modeling},
  year      = {2024}
}
```

## License

The code is released under the MIT License (`LICENSE`). The single-agent prompt and the label parser are adapted from the ReaLMistake code (MIT). The ReaLMistake data is distributed by its authors under CC BY-SA 4.0 and is not included in this repository.
