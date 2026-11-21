"""CopMAD: competitive debate (Khan et al., 2024; Kenton et al., 2024) with a judge.

Round 0 is a real debate turn: Alice (model 1) argues for her single-agent label with the
opening template and Bob (model 2) for the other label. Rounds 1..R follow with the templates of
``prompts/copmad.py``; debaters are told to persuade the judge. Model 1 then judges the
transcript, with every quote checked against the problem.

    python -m colmad.run_copmad -j1 mini -j2 l70b
"""

from __future__ import annotations

import argparse
import copy
import os

from . import llm
from .data import base_llm_name, check_aligned, debate_path, read_json, single_path, tasks_arg, write_json
from .debate import Turn, compose_problem, extract_transcript, fill_debater, fill_judge
from .labels import error_label, flip, judge_label
from .prompts.copmad import (
    DEBATE_ERRORDETECTION_PROMPT_JUDGE as JUDGE,
    DEBATE_ERRORDETECTION_PROMPT_OPENING as OPENING,
    DEBATE_ERRORDETECTION_PROMPT_SECOND as SECOND,
    DEBATE_ERRORDETECTION_PROMPT_THIRD as THIRD,
)


def template(num_round, total_rounds=2):
    if num_round == 0:
        return OPENING
    if num_round == total_rounds - 1:
        return THIRD
    return SECOND


def run(single_a, single_b, j1, j2, rounds=2, temperature=0.0, seed=0):
    check_aligned(single_a, single_b)
    n = len(single_a)
    results = copy.deepcopy(single_a)
    problem, conv, defend, oppose, prompts = [], [], [], [], []
    for idx in range(n):
        problem.append(compose_problem(single_a[idx]))
        results[idx]["critic_response_a_r0"] = single_a[idx]["critic_response"]
        results[idx]["critic_response_b_r0"] = single_b[idx]["critic_response"]
        conv.append([])
        defend.append(error_label(single_a[idx]["critic_response"]))
        oppose.append(flip(defend[idx]))
        prompts.append({"idx": idx, "content": fill_debater(template(0), problem[idx], extract_transcript(conv[idx]),
                                                            "Alice", defend[idx], oppose[idx])})

    for r in range(0, rounds + 1):
        print(f"  round {r}")
        responses = llm.chat_batch(prompts, j1, temperature=temperature, seed=seed)
        prompts = []
        # The positions are re-read from Alice's round-0 turn. As in the runs reported in the
        # paper, position i takes the label of the i-th response in completion order; the judge
        # prompt records the options actually used, and evaluate.py decodes verdicts from it.
        for i, resp in enumerate(responses):
            idx, out = resp["idx"], resp["response"]
            results[idx][f"critic_response_a_r{r}"] = out
            results[idx][f"critic_prompt_a_r{r}"] = fill_debater(template(r, total_rounds=rounds), problem[idx],
                                                                 extract_transcript(conv[idx]), "Alice",
                                                                 defend[idx], oppose[idx])
            conv[idx] += [Turn("user", results[idx][f"critic_prompt_a_r{r}"]), Turn("bot", out)]
            if r == 0:
                defend[i] = error_label(out)
                oppose[i] = flip(defend[i])
            prompts.append({"idx": idx, "content": fill_debater(template(r), problem[idx], extract_transcript(conv[idx]),
                                                                "Bob", oppose[idx], defend[idx])})
        responses = llm.chat_batch(prompts, j2, temperature=temperature, seed=seed)
        prompts = []
        for resp in responses:
            idx, out = resp["idx"], resp["response"]
            results[idx][f"critic_response_b_r{r}"] = out
            results[idx][f"critic_prompt_b_r{r}"] = fill_debater(template(r), problem[idx], extract_transcript(conv[idx]),
                                                                 "Bob", oppose[idx], defend[idx])
            conv[idx] += [Turn("user", results[idx][f"critic_prompt_b_r{r}"]), Turn("bot", out)]
            # Alice's next turn uses the template of the round just finished.
            prompts.append({"idx": idx, "content": fill_debater(template(r), problem[idx], extract_transcript(conv[idx]),
                                                                "Alice", defend[idx], oppose[idx])})

    prompts = []
    for idx in range(n):
        transcript = extract_transcript(conv[idx], verify_quotes=True, problem=problem[idx])
        results[idx]["judge_prompt"] = fill_judge(JUDGE, problem[idx], transcript, defend[idx], oppose[idx],
                                                  info_asymmetry=True)
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
    ap.add_argument("--tasks", default="all")
    ap.add_argument("-o", "--out_dir", default="outputs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    j1, j2, base_llm = llm.model_name(args.model_1), llm.model_name(args.model_2), base_llm_name(args.base_llm)
    for task in tasks_arg(args.tasks):
        print(f"=== CopMAD  {task}  {j1} : {j2}")
        out = debate_path(args.out_dir, "copmad", base_llm, j1, j2, task, args.rounds, args.temperature, args.seed)
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out} exists")
            continue
        a = read_json(single_path(args.out_dir, base_llm, j1, task))
        b = read_json(single_path(args.out_dir, base_llm, j2, task))
        results = run(a, b, j1, j2, args.rounds, args.temperature, args.seed)
        write_json(results, out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
