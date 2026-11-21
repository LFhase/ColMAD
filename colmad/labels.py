"""Parsing ``error`` / ``no_error`` out of model responses, and the gold labels."""

from __future__ import annotations

from .data import load_benchmark

#: Six benchmark labels corrected after manual re-inspection (GPT-4 split).
LABEL_CORRECTIONS = {
    "math_problem_generation_19_gpt-4-0613": "error",
    "math_problem_generation_36_gpt-4-0613": "no_error",
    "math_problem_generation_45_gpt-4-0613": "error",
    "finegrained_fact_verification_wice_test04259_gpt-4-0613": "no_error",
    "finegrained_fact_verification_wice_test00874_gpt-4-0613": "no_error",
    "finegrained_fact_verification_wice_test02547_gpt-4-0613": "no_error",
}


def flip(label):
    return "error" if label == "no_error" else "no_error"


def _check_error_label(text):
    if "contains no error" in text:
        return "no_error"
    if "response is correct" in text:
        return "no_error"
    if "response is valid" in text:
        return "no_error"
    if "no_error" in text:
        return "no_error"
    if "there is an error" in text:
        return "error"
    if "contains an error" in text:
        return "error"
    if "response is not correct" in text:
        return "error"
    if "response is not valid" in text:
        return "error"
    return None


def error_label(response):
    """Label of a single-agent or debater response.

    Looks for the conclusion sentence in the last line, then anywhere in the response; if
    neither is found, ``error`` when the response contains "not", else ``no_error``.
    """
    response = response.lower()
    label = _check_error_label(response.split("\n")[-1])
    if label is None:
        label = _check_error_label(response)
    if label is None:
        return "error" if "not" in response else "no_error"
    return label


def judge_label(response, option_1, option_2):
    """Label chosen by a judge answering ``Answer: <1 option_1|2 option_2>``.

    Returns ``(label, used_fallback)``. The last line is read first; if it names neither option,
    the judge's explicit "Alice/Bob is correct" statements are used, then :func:`error_label`.
    """
    last = response.strip().split("\n")[-1].lower()
    if "answer" in last:
        label, skip = None, False
        if option_1.lower() in last or "1" in last:
            label = option_1
            skip = option_1.lower() == "no_error"
        if not skip and (option_2.lower() in last or "2" in last):
            label = option_2
        if label is not None:
            return label, False
    lower = response.lower()
    if "alice is correct" in last or "debater alice's argument is more convincing" in lower:
        return option_1, True
    if "bob is correct" in last or "debater bob's argument is more convincing" in lower:
        return option_2, True
    return error_label(last), True


_GOLD = {}


def gold_labels(task, base_llm):
    """``{sample id: label}``: benchmark labels with :data:`LABEL_CORRECTIONS` applied."""
    if (task, base_llm) not in _GOLD:
        _GOLD[(task, base_llm)] = {
            d["metadata"]["id"]: LABEL_CORRECTIONS.get(d["metadata"]["id"], d["error_label"])
            for d in load_benchmark(task, base_llm)
        }
    return _GOLD[(task, base_llm)]
