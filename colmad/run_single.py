"""Single-agent error detection. Its outputs are round 0 of every debate protocol.

    python -m colmad.run_single -m mini
    python -m colmad.run_single -m l70b -t 1 -s 3      # seeded temperature-1 sample
"""

from __future__ import annotations

import argparse
import copy
import os

from . import llm
from .data import base_llm_name, load_benchmark, single_path, tasks_arg, write_json
from .debate import compose_problem
from .labels import error_label


def run(task, model, base_llm, temperature=0.0, seed=0):
    dataset = load_benchmark(task, base_llm)
    results = copy.deepcopy(dataset)
    prompts = [{"idx": i, "content": compose_problem(d)} for i, d in enumerate(dataset)]
    for r in llm.chat_batch(prompts, model, temperature=temperature, seed=seed):
        results[r["idx"]]["critic_response"] = r["response"]
        results[r["idx"]]["prediction"] = error_label(r["response"])
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-bl", "--base_llm", default="gpt4", help="gpt4 (default) or llama2")
    ap.add_argument("-t", "--temperature", type=float, default=0)
    ap.add_argument("-s", "--seed", type=int, default=0)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("-o", "--out_dir", default="outputs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    model, base_llm = llm.model_name(args.model), base_llm_name(args.base_llm)
    for task in tasks_arg(args.tasks):
        out = single_path(args.out_dir, base_llm, model, task, args.temperature, args.seed)
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out} exists")
            continue
        print(f"=== single  {task}  model={model}")
        results = run(task, model, base_llm, args.temperature, args.seed)
        write_json(results, out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
