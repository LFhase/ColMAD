"""Benchmark loading and result-file paths."""

from __future__ import annotations

import json
import os

TASKS = ["math_word_problem_generation", "finegrained_fact_verification", "answerability_classification"]
TASK_SHORT = {"math_word_problem_generation": "math", "finegrained_fact_verification": "fact",
              "answerability_classification": "ans"}
#: ReaLMistake splits: responses written by GPT-4 (140 per task) or Llama 2 70B (160 per task).
BASE_LLMS = {"gpt4": "gpt-4-0613", "llama2": "Llama-2-70b-chat-hf"}
NUM_SAMPLES = {"gpt-4-0613": 140, "Llama-2-70b-chat-hf": 160}

DATA_DIR = os.environ.get("REALMISTAKE_DATA", "data")


def base_llm_name(name):
    return BASE_LLMS.get(name, name)


def tasks_arg(value):
    """``all`` or a comma-separated list of task names / short names (math, fact, ans)."""
    if value == "all":
        return list(TASKS)
    short = {v: k for k, v in TASK_SHORT.items()}
    return [short.get(t, t) for t in value.split(",")]


def load_benchmark(task, base_llm):
    """The ReaLMistake records of one task and split, in benchmark order."""
    path = os.path.join(DATA_DIR, task, f"{base_llm}.jsonl")
    with open(path) as fh:
        return [json.loads(line) for line in fh][:NUM_SAMPLES[base_llm]]


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(obj, path, overwrite=False):
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=4, ensure_ascii=False)
    print(f"saved {path}")


def _run_tag(temperature, seed):
    tag = f"temp{temperature:g}"
    return tag + (f"-seed{seed}" if seed else "")


def single_path(out_dir, base_llm, model, task, temperature=0, seed=0):
    return os.path.join(out_dir, "single", base_llm, model, f"{task}-{_run_tag(temperature, seed)}.json")


def debate_path(out_dir, protocol, base_llm, j1, j2, task, rounds=2, temperature=0, seed=0):
    return os.path.join(out_dir, protocol, base_llm, f"{j1}__{j2}",
                        f"{task}-r{rounds}-{_run_tag(temperature, seed)}.json")


def check_aligned(*record_lists):
    """Every list must hold the same samples in the same order."""
    n = len(record_lists[0])
    for recs in record_lists:
        if len(recs) != n:
            raise ValueError(f"length mismatch: {[len(r) for r in record_lists]}")
    for i in range(n):
        ids = {recs[i]["metadata"]["id"] for recs in record_lists}
        if len(ids) != 1:
            raise ValueError(f"sample {i} is misaligned: {ids}")
