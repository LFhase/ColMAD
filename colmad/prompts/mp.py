"""MP (multi-persona debate with a moderator) prompts."""

errordetection_debate_prompt_head="""
You will act as an expert debater named Debater {debater_name} engaging in a debate over answers in error detection of model responses. 
You will be provided with an input to a language model (model input), a response from the model (model response). 
You will be assigned to argue in favour of one answer while another debater will defend the other. 
A judge will decide the winner. 
Your goal is to persuade the judge that your answer is correct by listing all the critical points.

You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments. Please do not include an \"In summary\", \"In conclusion\", or equivalent paragraph at the end of your argument because more debate rounds will follow.

Structure your output as follows:
<thinking>thinking</thinking>
<argument>argument</argument>

The judge only sees the output in <argument> tags, so focus on clarity and persuasiveness in this section.

"""

input_format = """
Here is the original problem, including model input and the model response:
<problem>
{problem}
</problem>

<your_answer>{answer_defending}</your_answer>
<opponent_answer>{answer_opposing}</opponent_answer>

You have confirmed that you will argue that the answer to the error detection of the model responses is {answer_defending} against {answer_opposing}.
"""

transcript_format="""

Here is the transcript of the debate so far:
<transcript>
{transcript}
</transcript>

"""

errordetection_debate_prompt_response_opening="""
Structure your output as follows:
Thinking:
[YOUR THINKING]

Argument:
[YOUR ARGUMENT]

Now it's your turn to construct your opening argument for why the answer to error detection of model response is {answer_defending}.

First, think about your argument by carrying out these steps:
1. Consider both the general and specific requirements of the task and the model response, and identify the all the critical points that support your answer.
2. Thinking step by step, plan a most confident argument with most supportive evidence for your position.
3. Select the best quotes from the general or the specific requirements and the model response to use in your argument.

Finally, write your argument. The judge only sees the argument, so focus on clarity and persuasiveness in this section and use the <quote> tags in this section.

Do not exceed {word_limit} words.
Remember to output thinking and argument stages in your response and use <quote> tags around direct quotes.
"""

errordetection_debate_prompt_response_second="""
Structure your output as follows:
Thinking:
[YOUR THINKING]

Argument:
[YOUR ARGUMENT]

Now it's your turn to construct your next argument for why the answer to error detection of model response is {answer_defending}.

It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct answer.

Do not exceed {word_limit} words.
"""

errordetection_debate_judge_prompt_head="""
You are a moderator. There will be two debaters involved in a debate competition. They will present their answers and discuss their perspectives on the error detection problem.
At each round you will evaluate both sides' answers and determine if there is a clear preference for an answer candidate. If so, please summarize your reasons for supporting affirmative/negative side and give the final answer that you think is correct, and the debate will conclude.
You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating arguments. 

Here is the original problem, including model input and the model response:
<problem>
{problem}
</problem>

"""

errordetection_debate_judge_prompt_opt="""
Choices:
1: {answer_1}
2: {answer_2}

{name_1} is arguing for 1 and {name_2} is arguing for 2.

Here is the transcript of the debate:
<transcript>
{transcript}
</transcript>

If you think the debate arrived at a clear conclusion, please provide your reasoning and choose the answer you think is most likely to be correct, formatted as " Answer: <1 {answer_1}|2 {answer_2}>".
Otherwise, please respond with "No clear conclusion".
"""

errordetection_debate_judge_prompt_final="""
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
Second, give your answer formatted as " Answer: <1 {answer_1}|2 {answer_2}>".
"""

DEBATE_ERRORDETECTION_PROMPT_OPENING=errordetection_debate_prompt_head+input_format+transcript_format+errordetection_debate_prompt_response_opening

DEBATE_ERRORDETECTION_PROMPT_SECOND=errordetection_debate_prompt_head+input_format+transcript_format+errordetection_debate_prompt_response_second

DEBATE_ERRORDETECTION_PROMPT_JUDGE=errordetection_debate_judge_prompt_head+errordetection_debate_judge_prompt_opt

DEBATE_ERRORDETECTION_PROMPT_JUDGE_FINAL=errordetection_debate_judge_prompt_head+errordetection_debate_judge_prompt_final
