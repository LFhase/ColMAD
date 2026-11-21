"""Prompt assembly shared by the debate protocols.

These functions are kept character-for-character equivalent to the code that produced the
paper's runs, so that a re-run sends exactly the same prompts.
"""

from __future__ import annotations

from dataclasses import dataclass

from .prompts.single import BASELINE_ERRORDETECTION_PROMPT_1


@dataclass
class Turn:
    role: str      # "user" (prompt) or "bot" (response)
    content: str


def compose_problem(record):
    """The single-agent prompt for one benchmark record; debaters see it as the problem."""
    return (BASELINE_ERRORDETECTION_PROMPT_1
            .replace("{input}", record["input"])
            .replace("{response}", record["llm_response"]))


def fill_debater(template, problem, transcript, debater_name, answer_defending, answer_opposing,
                 word_limit="150"):
    return (template
            .replace("{debater_name}", debater_name)
            .replace("{answer_defending}", answer_defending)
            .replace("{answer_opposing}", answer_opposing)
            .replace("{word_limit}", word_limit)
            .replace("{transcript}", transcript)
            .replace("{problem}", problem))


def fill_judge(template, problem, transcript, answer_1, answer_2, names=("Alice", "Bob"),
               word_limit="150", info_asymmetry=False):
    prompt = (template
              .replace("{name_1}", names[0])
              .replace("{name_2}", names[1])
              .replace("{answer_1}", answer_1)
              .replace("{answer_2}", answer_2)
              .replace("{word_limit}", word_limit)
              .replace("{transcript}", transcript)
              .replace("{problem}", problem))
    if info_asymmetry:
        # Intended to cut the problem block out of the judge prompt. The CopMAD and MP judge
        # templates do not contain the marker sentence, so find() returns -1 and the prompt
        # becomes prompt[:-1] + prompt[end:], as in the runs reported in the paper.
        begin = prompt.find("Here is the model input and the model response:")
        end = prompt.find("\n</problem>\n\n") + len("\n</problem>\n\n\n")
        prompt = prompt[:begin] + prompt[end:]
    return prompt


def _verify_quotes(text, problem, lowercase_problem_only=False):
    """Tag each ``<quote>`` as ``<v_quote>`` if it occurs in the problem (case-insensitive),
    else ``<u_quote>``. With ``lowercase_problem_only`` only the problem is lowercased, so a
    quote that contains a capital letter is always unverified."""
    begin = text.find("<quote>")
    end = text.find("</quote>")
    while begin < end and begin > 0 and end > 0:
        quote = text[begin:end].replace("<quote>", "").replace("</quote>", "")
        if not lowercase_problem_only:
            quote = quote.lower()
        tag = "v_quote" if quote in problem.lower() else "u_quote"
        text = text[:begin] + f"<{tag}>" + text[begin + len("<quote>"):end] + f"</{tag}>" + text[end + len("</quote>"):]
        begin = text.find("<quote>", end)
        end = text.find("</quote>", end + 1)
    return text


def _argument(text):
    begin = text.find("<argument>") + len("<argument>")
    end = text.find("</argument>")
    if end < 0:
        end = len(text) + 1
    return text[begin:end].lstrip().rstrip()


def extract_transcript(conv, names=("Alice", "Bob"), verify_quotes=False, problem="", moderator=False,
                       lowercase_problem_only=False):
    """Render a conversation (alternating prompt/response turns) as the debate transcript.

    Only the ``<argument>`` part of each debater response is shown. Without a moderator a round
    has two speakers (Alice, Bob); with ``moderator=True`` (MP) it has three (Alice, Bob, Judge).
    """
    per_round = 3 if moderator else 2
    out = ""
    for i in range(len(conv) // 2):
        if i % per_round == 0:
            out += f"Round {i // per_round + 1}:\n\n"
        content = conv[i * 2 + 1].content
        if moderator and (i + 1) % 3 == 0:
            out += f"Judge: {content}\n"
            continue
        arg = _argument(content)
        if verify_quotes:
            arg = _verify_quotes(arg, problem, lowercase_problem_only)
        out += f"Debater {names[i % 2]}: {arg}\n\n"
    return out
