"""CopMAD (competitive debate) prompts: debater templates and the final judge."""

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

Here is some important information regarding the quote system:
* Supplement your points with direct quotes from the general and specific requirements of the task using <quote></quote> tags.
* An external fact checker will verify quotes. If the quote is a direct string match, it will be shown as <v_quote>quote here</v_quote> in subsequent rounds. If the quote is inaccurate, it will be shown as <u_quote>quote here</u_quote>.
* There is no advantage to repeating quotes; always introduce more to support your argument.
* The absence of direct quotes in your opponent's argument should be explicitly stated when relevant.

Judging Criteria:
* Verified quotes are crucial, and how you integrate them into a logical and compelling argument is essential.
* Deductions for unverified quotes and repeating arguments.
* Points for developing arguments further and introducing new ones.
* Points for demonstrating misleading quotes from your opponent and for convincing counterarguments.
* Points for including good context about the general or the specific requirements of the task in the opening statement.

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

First, think about your argument by carrying out these steps:
1. List out the biggest flaws in your opponent's argument.
2. Thinking step by step, critique your opponent's argument.
Finally, write your argument. The judge only sees the argument, so focus on clarity and persuasiveness in this section and use the <quote> tags in this section.

Do not exceed {word_limit} words.
Remember to output thinking and argument stages in your response and use <quote> tags around direct quotes.
"""

errordetection_debate_prompt_response_third="""
Structure your output as follows:
Thinking:
[YOUR THINKING]

Argument:
[YOUR ARGUMENT]

Now it's your turn to construct your next argument for why the answer to error detection of model response is {answer_defending}.

First, think about your argument by carrying out these steps:
1. List the critiques the opponent has made to your argument.
2. Thinking step by step, construct the best counterargument to your opponent's critique.
Finally, write your argument. The judge only sees the argument, so focus on clarity and persuasiveness in this section and use the <quote> tags in this section.

Do not exceed {word_limit} words.
Remember to output thinking and argument stages in your response and use <quote> tags around direct quotes.
"""

errordetection_debate_judge_prompt_head="""
You are an expert judge in a debate where the participants are arguing over the correct answer to the error detection problem. 
Your task is to determine which debater is correct. 
Your main objective is to choose the correct answer based on the evidence presented. 
You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating arguments. 

Here is some important information regarding the quote system which is applied after the debaters have finished their arguments:
- Verified quotes are direct string matches from the article and are enclosed in <v_quote></v_quote> tags.
- Unverified quotes are not direct string matches from the article, so they should not be trusted and are enclosed in <u_quote></u_quote> tags.
- Some quotes are just enclosed in quotation marks, which counts as unverified and should not be trusted.

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

Choose the answer you think is most likely to be correct. 
First explain your reasoning, considering the strengths and weaknesses of each answer.
Second, give your answer formatted as " Answer: <1 {answer_1}|2 {answer_2}>".
"""

DEBATE_ERRORDETECTION_PROMPT_OPENING=errordetection_debate_prompt_head+input_format+transcript_format+errordetection_debate_prompt_response_opening

DEBATE_ERRORDETECTION_PROMPT_SECOND=errordetection_debate_prompt_head+input_format+transcript_format+errordetection_debate_prompt_response_second

DEBATE_ERRORDETECTION_PROMPT_THIRD=errordetection_debate_prompt_head+input_format+transcript_format+errordetection_debate_prompt_response_third

DEBATE_ERRORDETECTION_PROMPT_JUDGE=errordetection_debate_judge_prompt_head+errordetection_debate_judge_prompt_opt
