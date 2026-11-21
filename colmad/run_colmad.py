"""ColMAD: collaborative multi-agent debate with a judge.

Round 0 is the two agents' single-agent critiques (``colmad.run_single``). In rounds 1..R,
Debater Alice (model 1) and Debater Bob (model 2) speak in turn; Alice defends her round-0
label and Bob the other label, with the collaborative template of ``prompts/colmad.py``.
Model 1 then judges the transcript, in which every quote has been checked against the problem.

    python -m colmad.run_colmad -j1 mini -j2 l70b
"""

from __future__ import annotations

import argparse
import copy
import os

from . import llm
from .data import base_llm_name, check_aligned, debate_path, read_json, single_path, tasks_arg, write_json
from .debate import Turn, compose_problem, extract_transcript, fill_debater, fill_judge
from .labels import error_label, flip, judge_label
from .prompts.colmad import DEBATE_COLLAB_ERRORDETECTION_PROMPT_JUDGE as JUDGE
from .prompts.colmad import DEBATE_COLLAB_ERRORDETECTION_PROMPT_SECOND as DEBATER


def strip_think(text):
    if "<think>" in text.lower() and "</think>" in text.lower():
        return text.split("<think>")[-1].split("</think>")[-1].strip()
    return text


def run(single_a, single_b, j1, j2, rounds=2, temperature=0.0, seed=0, lowercase_problem_only=False):
    check_aligned(single_a, single_b)
    n = len(single_a)
    results = copy.deepcopy(single_a)
    problem, conv, defend, oppose, prompts = [], [], [], [], []
    for idx in range(n):
        problem.append(compose_problem(single_a[idx]))
        a0 = strip_think(single_a[idx]["critic_response"])
        b0 = strip_think(single_b[idx]["critic_response"])
        results[idx]["critic_response_a_r0"] = a0
        results[idx]["critic_response_b_r0"] = b0
        conv.append([Turn("user", ""), Turn("bot", f"<argument>{a0}</argument>"),
                     Turn("user", ""), Turn("bot", f"<argument>{b0}</argument>")])
        defend.append(error_label(single_a[idx]["critic_response"]))
        oppose.append(flip(defend[idx]))
        prompts.append({"idx": idx, "content": fill_debater(DEBATER, problem[idx], extract_transcript(conv[idx]),
                                                            "Alice", defend[idx], oppose[idx])})

    for r in range(1, rounds + 1):
        print(f"  round {r}")
        responses = llm.chat_batch(prompts, j1, temperature=temperature, seed=seed)
        prompts = []
        for resp in responses:
            idx, out = resp["idx"], resp["response"]
            results[idx][f"critic_response_a_r{r}"] = out
            results[idx][f"critic_prompt_a_r{r}"] = fill_debater(DEBATER, problem[idx], extract_transcript(conv[idx]),
                                                                 "Alice", defend[idx], oppose[idx])
            conv[idx] += [Turn("user", results[idx][f"critic_prompt_a_r{r}"]), Turn("bot", out)]
            prompts.append({"idx": idx, "content": fill_debater(DEBATER, problem[idx], extract_transcript(conv[idx]),
                                                                "Bob", oppose[idx], defend[idx])})
        responses = llm.chat_batch(prompts, j2, temperature=temperature, seed=seed)
        prompts = []
        for resp in responses:
            idx, out = resp["idx"], resp["response"]
            results[idx][f"critic_response_b_r{r}"] = out
            results[idx][f"critic_prompt_b_r{r}"] = fill_debater(DEBATER, problem[idx], extract_transcript(conv[idx]),
                                                                 "Bob", oppose[idx], defend[idx])
            conv[idx] += [Turn("user", results[idx][f"critic_prompt_b_r{r}"]), Turn("bot", out)]
            prompts.append({"idx": idx, "content": fill_debater(DEBATER, problem[idx], extract_transcript(conv[idx]),
                                                                "Alice", defend[idx], oppose[idx])})

    prompts = []
    for idx in range(n):
        transcript = extract_transcript(conv[idx], verify_quotes=True, problem=problem[idx],
                                        lowercase_problem_only=lowercase_problem_only)
        results[idx]["judge_prompt"] = fill_judge(JUDGE, problem[idx], transcript, defend[idx], oppose[idx])
        prompts.append({"idx": idx, "content": results[idx]["judge_prompt"]})
    print("  judge")
    for resp in llm.chat_batch(prompts, j1, temperature=temperature, seed=seed):
        idx = resp["idx"]
        results[idx]["judge_response"] = resp["response"]
        results[idx]["prediction"] = judge_label(resp["response"], defend[idx], oppose[idx])[0]
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j1", "--model_1", required=True, help="Debater Alice and the judge")
    ap.add_argument("-j2", "--model_2", required=True, help="Debater Bob")
    ap.add_argument("-bl", "--base_llm", default="gpt4", help="gpt4 (default) or llama2")
    ap.add_argument("-r", "--rounds", type=int, default=2)
    ap.add_argument("-t", "--temperature", type=float, default=0)
    ap.add_argument("-s", "--seed", type=int, default=0)
    ap.add_argument("--quote_check", choices=["insensitive", "lowercase_problem"], default="insensitive",
                    help="how the judge's quote check compares a quote with the problem: case-insensitive "
                         "(default), or the problem lowercased and the quote as written")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("-o", "--out_dir", default="outputs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    j1, j2, base_llm = llm.model_name(args.model_1), llm.model_name(args.model_2), base_llm_name(args.base_llm)
    for task in tasks_arg(args.tasks):
        print(f"=== ColMAD  {task}  {j1} : {j2}")
        out = debate_path(args.out_dir, "colmad", base_llm, j1, j2, task, args.rounds, args.temperature, args.seed)
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out} exists")
            continue
        # Round 0 is always the temperature-0 single-agent critique.
        a = read_json(single_path(args.out_dir, base_llm, j1, task))
        b = read_json(single_path(args.out_dir, base_llm, j2, task))
        results = run(a, b, j1, j2, args.rounds, args.temperature, args.seed,
                      lowercase_problem_only=args.quote_check == "lowercase_problem")
        write_json(results, out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
