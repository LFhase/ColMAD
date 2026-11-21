"""MP: debate with a moderator that may end the debate after any round (Liang et al., 2024).

Rounds 0..R as in CopMAD (opening template in round 0, ``SECOND`` afterwards, prompts of
``prompts/mp.py``). After every round model 1 acts as moderator and either picks an answer or
replies "No clear conclusion"; after the last round it must pick one. The verdict is the first
round with a conclusion.

    python -m colmad.run_mp -j1 mini -j2 l70b
"""

from __future__ import annotations

import argparse
import copy
import os

from . import llm
from .data import base_llm_name, check_aligned, debate_path, read_json, single_path, tasks_arg, write_json
from .debate import Turn, compose_problem, extract_transcript, fill_debater, fill_judge
from .labels import error_label, flip, judge_label
from .prompts.mp import (
    DEBATE_ERRORDETECTION_PROMPT_JUDGE as JUDGE,
    DEBATE_ERRORDETECTION_PROMPT_JUDGE_FINAL as JUDGE_FINAL,
    DEBATE_ERRORDETECTION_PROMPT_OPENING as OPENING,
    DEBATE_ERRORDETECTION_PROMPT_SECOND as SECOND,
)

NO_CONCLUSION = "no clear conclusion"


def template(num_round):
    return OPENING if num_round == 0 else SECOND


def run(single_a, single_b, j1, j2, rounds=2, temperature=0.0):
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

    def transcript(idx, **kw):
        return extract_transcript(conv[idx], moderator=True, **kw)

    for r in range(0, rounds + 1):
        print(f"  round {r}")
        # In rounds >= 1 ``prompts`` holds the moderator prompts of the previous round, which
        # are what model 1 receives for Alice's turn (as in the runs reported in the paper).
        responses = llm.chat_batch(prompts, j1, temperature=temperature)
        prompts = []
        for i, resp in enumerate(responses):
            idx, out = resp["idx"], resp["response"]
            results[idx][f"critic_response_a_r{r}"] = out
            results[idx][f"critic_prompt_a_r{r}"] = fill_debater(template(r), problem[idx], transcript(idx),
                                                                 "Alice", defend[idx], oppose[idx])
            conv[idx] += [Turn("user", results[idx][f"critic_prompt_a_r{r}"]), Turn("bot", out)]
            if r == 0:
                # Position i takes the label of the i-th response in completion order (see run_copmad.py).
                defend[i] = error_label(out)
                oppose[i] = flip(defend[i])
            prompts.append({"idx": idx, "content": fill_debater(template(r), problem[idx], transcript(idx),
                                                                "Bob", oppose[idx], defend[idx])})
        responses = llm.chat_batch(prompts, j2, temperature=temperature)
        prompts = []
        for resp in responses:
            idx, out = resp["idx"], resp["response"]
            results[idx][f"critic_response_b_r{r}"] = out
            results[idx][f"critic_prompt_b_r{r}"] = fill_debater(template(r), problem[idx], transcript(idx),
                                                                 "Bob", oppose[idx], defend[idx])
            conv[idx] += [Turn("user", results[idx][f"critic_prompt_b_r{r}"]), Turn("bot", out)]

        prompts = []
        for idx in range(n):
            judge = JUDGE if r < rounds else JUDGE_FINAL
            results[idx][f"judge_prompt_r{r}"] = fill_judge(
                judge, problem[idx], transcript(idx, verify_quotes=True, problem=problem[idx]),
                defend[idx], oppose[idx], info_asymmetry=True)
            prompts.append({"idx": idx, "content": results[idx][f"judge_prompt_r{r}"]})
        print(f"  moderator, round {r}")
        for resp in llm.chat_batch(prompts, j1, temperature=temperature):
            idx, out = resp["idx"], resp["response"]
            results[idx][f"judge_response_r{r}"] = out
            results[idx][f"prediction_r{r}"] = judge_label(out, defend[idx], oppose[idx])[0]
            conv[idx] += [Turn("user", results[idx][f"judge_prompt_r{r}"]), Turn("bot", out)]

    for idx in range(n):
        for r in range(0, rounds + 1):
            if NO_CONCLUSION not in results[idx][f"judge_response_r{r}"].lower():
                results[idx]["judge_response"] = results[idx][f"judge_response_r{r}"]
                results[idx]["prediction"] = results[idx][f"prediction_r{r}"]
                results[idx]["consensus_round"] = r
                break
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j1", "--model_1", required=True, help="Debater Alice and the moderator")
    ap.add_argument("-j2", "--model_2", required=True, help="Debater Bob")
    ap.add_argument("-bl", "--base_llm", default="gpt4", help="gpt4 (default) or llama2")
    ap.add_argument("-r", "--rounds", type=int, default=2)
    ap.add_argument("-t", "--temperature", type=float, default=0)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("-o", "--out_dir", default="outputs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    j1, j2, base_llm = llm.model_name(args.model_1), llm.model_name(args.model_2), base_llm_name(args.base_llm)
    for task in tasks_arg(args.tasks):
        print(f"=== MP  {task}  {j1} : {j2}")
        out = debate_path(args.out_dir, "mp", base_llm, j1, j2, task, args.rounds, args.temperature)
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out} exists")
            continue
        a = read_json(single_path(args.out_dir, base_llm, j1, task))
        b = read_json(single_path(args.out_dir, base_llm, j2, task))
        results = run(a, b, j1, j2, args.rounds, args.temperature)
        write_json(results, out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
