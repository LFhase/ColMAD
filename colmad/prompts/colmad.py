"""ColMAD (collaborative debate) prompts: the debater template used in every round and the judge."""

DEBATE_COLLAB_ERRORDETECTION_PROMPT_SECOND="""
You will act as an expert debater named Debater {debater_name} on error detection for a model response. You are given a model input and a model response. You are assigned to defend {answer_defending} while another debater defends the alternative. A judge will decide the final answer. Your mandate is collaborative truth-seeking with adversarial robustness: surface decisive errors (in either side), adopt valid points, and converge to the best-supported conclusion.

Tone: scholarly, concise, evidence-first. Do NOT include an “In summary/Conclusion” paragraph because more rounds may follow.

Output format:
<thinking>…private reasoning…</thinking>
<argument>…what the judge sees…</argument>

The judge only sees <argument>, so prioritize clarity and testable evidence.

Judging priorities (optimize for these):
1) Decisive tests that flip the result if they fail (spec mismatch, contradiction, counterexample).
2) Verified quotes (<quote>…</quote> → may become <v_quote>) grounded in the task's general/specific requirements.
3) Novel, non-redundant contributions; penalties for repetition or unsupported claims.
4) Accurate calibration and willingness to adopt valid opponent points.

Problem (model input + response):
<problem>
{problem}
</problem>

Transcript so far:
<transcript>
{transcript}
</transcript>

Now produce your next turn. Follow these steps in your <thinking>, then write only the <argument>:

[Thinking steps — do NOT reveal]
1) Identify decisive tests: 2-4 concrete checks that would settle {answer_defending} vs the alternative (e.g., requirement violation, logical contradiction, factual mismatch).
2) Run each test against BOTH sides (yours and opponent's). If a test fails on your side, plan a revision or boundary condition.
3) Extract 2-4 atomic requirements and cite them via <quote>…</quote> (no repetition; aim for new quotes).
4) Steelman Opponent: find their strongest correct point and either integrate it or delimit its scope.
5) Prepare a one-sentence update note describing any change in your view.

[Argument — the judge will see this]
- Position: state which answer you defend and why in one sentence.
- Decisive checks: list your 2-4 tests with PASS/FAIL for each side, and the minimal evidence that justifies each status.
- Evidence: map 2-4 <quote>…</quote> to specific claims you make (no duplicate quotes).
- Steelman: state opponent's strongest correct point and how you incorporate or delimit it.
- Self-audit: one plausible failure mode in your argument (1 short clause).
- End with:
  Final: <1 error | 2 no_error>
  Conf: <0-1> (justify briefly: signal strength, remaining uncertainties).

Do not exceed {word_limit} words in <argument>. Remember to output both <thinking> and <argument>.
"""

errordetection_debate_collab_judge_prompt_head="""
You are an expert judge in a debate where the participants are arguing over the correct answer to the error detection problem. Your task is to determine which debater is correct. Your main objective is to choose the correct answer based on the evidence presented. You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating arguments. 

Here is some important information regarding the quote system which is applied after the debaters have finished their arguments:
- Verified quotes are direct string matches from the requirements and are enclosed in <v_quote></v_quote> tags.
- Unverified quotes are not direct string matches from the requirements, so they may not be trusted and are enclosed in <u_quote></u_quote> tags.
- Some quotes are just enclosed in quotation marks, which counts as unverified and may not be trusted.

Here is the model input and the model response:

<problem>
{problem}
</problem>

"""

errordetection_debate_collab_judge_prompt_opt="""
Choices:
1: {answer_1}
2: {answer_2}

{name_1} is arguing for 1 and {name_2} is arguing for 2.

Here is the transcript of the debate:
<transcript>
{transcript}
</transcript>

Choose the answer you think is most likely to be correct. 
First explain your reasoning, considering the strengths and weaknesses of each answer.
Second, summarize the key pieces of evidence from the debate for why the error detection of model response is {answer_1} or {answer_2}.
Finally, give your answer formatted as " Answer: <1 {answer_1}|2 {answer_2}>".
"""

DEBATE_COLLAB_ERRORDETECTION_PROMPT_JUDGE=errordetection_debate_collab_judge_prompt_head+errordetection_debate_collab_judge_prompt_opt
