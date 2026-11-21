"""CosMAD (society-of-minds debate) prompt, following the vanilla MAD prompt of
"Revisiting Multi-Agent Debate as Test-Time Scaling" (each agent sees its own previous
solution and the other agent's solution)."""

# The conclusion sentence that BASELINE_ERRORDETECTION_PROMPT_1 asks for.
ANSWER_FORMAT = ('\nConclude your response with "Therefore, the model response contains an error."'
                 ' or "Therefore, the model response contains no error."')


def compose_madtts_prompt(question, own_prev, other_prev, other_agent_no, debate_round):
    other_opinions = f"Agent {other_agent_no}'s solution: \n```{other_prev}```"
    debate_context = (f"\n\nLet's revisit the original question:\n{question}\n"
                      f"You are now in round {debate_round} of the debate.")
    shared_opinions = ("\n\nHere is your previous solution to the question:"
                       f"\nYour previous solution: \n```{own_prev}```"
                       f"\n\nThese are the solutions from the other agents:\n{other_opinions}")
    opinion_instruction = (
        "\n\nListen carefully to other agents' solutions and update your solution if necessary. Then, "
        "provide a reasoning on how your thought process has changed since your previous solution "
        "and then provide your final answer.")
    return debate_context + shared_opinions + opinion_instruction + ANSWER_FORMAT
