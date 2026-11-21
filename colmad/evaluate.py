"""Score result files: per-task F1 / F2 and the three-task average.

    python -m colmad.evaluate single -m mini
    python -m colmad.evaluate colmad -j1 mini -j2 l70b
    python -m colmad.evaluate voting -j1 mini -j2 l70b
    python -m colmad.evaluate all -j1 mini -j2 l70b      # every protocol whose files exist

Conventions (those of the paper):
* gold labels: the benchmark labels with the six corrections in ``labels.LABEL_CORRECTIONS``;
* judge verdicts (ColMAD, CopMAD, MP) are decoded with the two options printed in the stored
  judge prompt; CosMAD has no judge and its answer is agent 1's last response;
* the agreement gate (the two round-0 labels agree -> that label is the answer) is on for
  ColMAD and CosMAD and off for CopMAD and MP; ``--gate on|off`` overrides it;
* Voting: agreeing round-0 labels are kept, disagreements are a coin flip drawn from one
  ``numpy.random.RandomState(2333)`` stream over Math, Fact, Ans in that order.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np

from . import llm
from .data import TASK_SHORT, TASKS, base_llm_name, check_aligned, debate_path, read_json, single_path
from .labels import error_label, flip, gold_labels, judge_label
from .metrics import average, score

PROTOCOLS = ["colmad", "copmad", "cosmad", "mp"]
GATE_DEFAULT = {"colmad": True, "cosmad": True, "copmad": False, "mp": False}
VOTING_SEED = 2333
OPTIONS = re.compile(r"<1 (no_error|error)\|2 (no_error|error)>")


def truth(records, task, base_llm):
    gold = gold_labels(task, base_llm)
    return [gold[r["metadata"]["id"]] for r in records]


def _judge_prompt(item):
    if "judge_prompt" in item:
        return item["judge_prompt"]
    r = item.get("consensus_round")
    return item.get(f"judge_prompt_r{r}") if r is not None else None


def verdicts(protocol, records, p1):
    """Ungated protocol answers and the number of judge replies parsed by the fallback rule."""
    out, fallbacks = [], 0
    for item, a in zip(records, p1):
        if protocol == "cosmad":
            out.append(error_label(item["judge_response"]) if "judge_response" in item else item.get("prediction", a))
            continue
        if "judge_response" not in item:
            # MP: no round reached a conclusion; the record keeps its round-0 prediction.
            out.append(item.get("prediction", a))
            continue
        options = OPTIONS.findall(_judge_prompt(item) or "")
        label, used = judge_label(item["judge_response"], *(options[-1] if options else (a, flip(a))))
        out.append(label)
        fallbacks += used
    return out, fallbacks


def score_debate(protocol, run, single_a, single_b, task, base_llm, gate=None, same_model=False):
    n = len(single_a)
    run = run[:n]
    check_aligned(run, single_a, single_b)
    p1 = [error_label(x["critic_response"]) for x in single_a]
    p2 = [error_label(x["critic_response"]) for x in single_b]
    raw, fallbacks = verdicts(protocol, run, p1)
    gate = GATE_DEFAULT[protocol] if gate is None else gate
    pred = [a if (gate and a == b and not same_model) else v for a, b, v in zip(p1, p2, raw)]
    m = score(truth(run, task, base_llm), pred)
    m["fallbacks"] = fallbacks
    return m


def score_single(records, task, base_llm):
    return score(truth(records, task, base_llm), [error_label(x["critic_response"]) for x in records])


def score_voting(singles_a, singles_b, tasks, base_llm):
    """``singles_*``: per-task record lists, in the order of ``tasks``."""
    rng = np.random.RandomState(VOTING_SEED)
    out = {}
    for task, a, b in zip(tasks, singles_a, singles_b):
        check_aligned(a, b)
        pa = [error_label(x["critic_response"]) for x in a]
        pb = [error_label(x["critic_response"]) for x in b]
        pred = [y if x == y else rng.choice(["error", "no_error"]) for x, y in zip(pa, pb)]
        out[task] = score(truth(a, task, base_llm), pred)
    return out


def print_row(name, per_task):
    cells = [f"{per_task[t]['f1']:6.2f} {per_task[t]['f2']:6.2f}" for t in TASKS]
    avg = average([per_task[t] for t in TASKS])
    print(f"{name:<24}" + "  ".join(cells) + f"   {avg['f1']:6.2f} {avg['f2']:6.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("protocol", choices=["single", "voting", "all"] + PROTOCOLS)
    ap.add_argument("-m", "--model", help="single: the model")
    ap.add_argument("-j1", "--model_1")
    ap.add_argument("-j2", "--model_2")
    ap.add_argument("-bl", "--base_llm", default="gpt4")
    ap.add_argument("-r", "--rounds", type=int, default=2)
    ap.add_argument("-t", "--temperature", type=float, default=0)
    ap.add_argument("-s", "--seed", type=int, default=0)
    ap.add_argument("--gate", choices=["default", "on", "off"], default="default")
    ap.add_argument("-o", "--out_dir", default="outputs")
    args = ap.parse_args()

    base_llm = base_llm_name(args.base_llm)
    gate = {"default": None, "on": True, "off": False}[args.gate]
    print(f"{'':<24}" + "  ".join(f"{TASK_SHORT[t]:>6} F1/F2" for t in TASKS) + "     Avg F1/F2")

    def singles(model, t=0, s=0):
        return [read_json(single_path(args.out_dir, base_llm, model, task, t, s)) for task in TASKS]

    if args.protocol == "single":
        model = llm.model_name(args.model)
        print_row(model, {task: score_single(x, task, base_llm) for task, x in zip(TASKS, singles(model))})
        return

    j1, j2 = llm.model_name(args.model_1), llm.model_name(args.model_2)
    if args.protocol in ("voting", "all"):
        print_row("Voting", score_voting(singles(j1), singles(j2), TASKS, base_llm))
    for protocol in (PROTOCOLS if args.protocol == "all" else [args.protocol] if args.protocol != "voting" else []):
        seeded = protocol == "cosmad" and args.temperature > 0 and args.seed > 0
        t0, s0 = (args.temperature, args.seed) if seeded else (0, 0)
        per_task = {}
        for task in TASKS:
            path = debate_path(args.out_dir, protocol, base_llm, j1, j2, task, args.rounds, args.temperature, args.seed)
            if not os.path.exists(path):
                break
            per_task[task] = score_debate(protocol, read_json(path),
                                          read_json(single_path(args.out_dir, base_llm, j1, task, t0, s0)),
                                          read_json(single_path(args.out_dir, base_llm, j2, task, t0, s0)),
                                          task, base_llm, gate, same_model=(j1 == j2))
        if len(per_task) == len(TASKS):
            print_row(protocol, per_task)
        elif args.protocol != "all":
            raise FileNotFoundError(path)


if __name__ == "__main__":
    main()
