"""CosMAD: society-of-minds debate without a judge (Du et al., 2024).

Round 0 is the two agents' single-agent critiques. In each round both agents update at the
same time: each sees its own previous solution and the other agent's solution
(``prompts/cosmad.py``). A sample stops once the two labels agree. The answer is agent 1's last
response.

    python -m colmad.run_cosmad -j1 mini -j2 l70b
"""

from __future__ import annotations

import argparse
import copy
import os

from . import llm
from .data import base_llm_name, check_aligned, debate_path, read_json, single_path, tasks_arg, write_json
from .debate import compose_problem
from .labels import error_label
from .prompts.cosmad import compose_madtts_prompt


def strip_think(text):
    if "<think>" in text.lower() and "</think>" in text.lower():
        return text.split("<think>")[-1].split("</think>")[-1].strip()
    return text


def run(single_a, single_b, j1, j2, rounds=2, temperature=0.0, seed=0, early_stop=True):
    check_aligned(single_a, single_b)
    n = len(single_a)
    results = copy.deepcopy(single_a)
    problem, last_a, last_b, active = {}, {}, {}, []
    for idx in range(n):
        problem[idx] = compose_problem(single_a[idx])
        a0, b0 = strip_think(single_a[idx]["critic_response"]), strip_think(single_b[idx]["critic_response"])
        results[idx]["critic_response_a_r0"] = a0
        results[idx]["critic_response_b_r0"] = b0
        last_a[idx], last_b[idx] = a0, b0
        if error_label(a0) == error_label(b0):
            results[idx]["consensus_round"] = 0
        else:
            active.append(idx)
    if not early_stop:
        active = list(range(n))

    for r in range(1, rounds + 1):
        if not active:
            break
        print(f"  round {r}: {len(active)} samples")
        pa = [{"idx": i, "content": compose_madtts_prompt(problem[i], last_a[i], last_b[i], 2, r)} for i in active]
        pb = [{"idx": i, "content": compose_madtts_prompt(problem[i], last_b[i], last_a[i], 1, r)} for i in active]
        na = {x["idx"]: x["response"] for x in llm.chat_batch(pa, j1, temperature=temperature, seed=seed)}
        nb = {x["idx"]: x["response"] for x in llm.chat_batch(pb, j2, temperature=temperature, seed=seed)}
        sa = {p["idx"]: p["content"] for p in pa}
        sb = {p["idx"]: p["content"] for p in pb}
        still = []
        for idx in active:
            results[idx][f"critic_prompt_a_r{r}"] = sa[idx]
            results[idx][f"critic_prompt_b_r{r}"] = sb[idx]
            results[idx][f"critic_response_a_r{r}"] = na[idx]
            results[idx][f"critic_response_b_r{r}"] = nb[idx]
            last_a[idx], last_b[idx] = na[idx], nb[idx]
            if error_label(na[idx]) == error_label(nb[idx]):
                results[idx]["consensus_round"] = r
            else:
                still.append(idx)
        if early_stop:
            active = still

    for idx in range(n):
        last = max(r for r in range(rounds + 1) if f"critic_response_a_r{r}" in results[idx])
        results[idx]["judge_response"] = results[idx][f"critic_response_a_r{last}"]
        results[idx]["prediction"] = error_label(results[idx]["judge_response"])
        results[idx].setdefault("consensus_round", None)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j1", "--model_1", required=True, help="Agent 1 (its last response is the answer)")
    ap.add_argument("-j2", "--model_2", required=True, help="Agent 2")
    ap.add_argument("-bl", "--base_llm", default="gpt4", help="gpt4 (default) or llama2")
    ap.add_argument("-r", "--rounds", type=int, default=2)
    ap.add_argument("-t", "--temperature", type=float, default=0)
    ap.add_argument("-s", "--seed", type=int, default=0,
                    help="with -t > 0, round 0 is read from the seed-matched single-agent samples")
    ap.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("-o", "--out_dir", default="outputs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    j1, j2, base_llm = llm.model_name(args.model_1), llm.model_name(args.model_2), base_llm_name(args.base_llm)
    seeded = args.temperature > 0 and args.seed > 0
    t0, s0 = (args.temperature, args.seed) if seeded else (0, 0)
    for task in tasks_arg(args.tasks):
        print(f"=== CosMAD  {task}  {j1} : {j2}")
        out = debate_path(args.out_dir, "cosmad", base_llm, j1, j2, task, args.rounds, args.temperature, args.seed)
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out} exists")
            continue
        a = read_json(single_path(args.out_dir, base_llm, j1, task, t0, s0))
        b = read_json(single_path(args.out_dir, base_llm, j2, task, t0, s0))
        results = run(a, b, j1, j2, args.rounds, args.temperature, args.seed, args.early_stop)
        write_json(results, out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
