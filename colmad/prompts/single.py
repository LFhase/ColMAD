"""Single-agent error-detection prompt (the round-0 prompt of every protocol)."""

input_format = """
===== Model Input Begins =====

{input}

===== Model Input Ends =====

===== Model Response Begins =====

{response}

===== Model Response Ends =====
"""

first_sentence = "We provide an input to a language model (model input) and a response from the model (model response)."

instruction_following_definition = "You need to check whether the model response follows all instructions and requirements in the model input."

baseline_errordetection_prompt_type1 = f"""{first_sentence} Your task is to detect errors in the model response. {instruction_following_definition} You need to classify the model response as an error if the model response does not follow any of the instructions or requirements in the model input.
"""

BASELINE_ERRORDETECTION_PROMPT_1 = baseline_errordetection_prompt_type1 + input_format + """
In your response, provide your explanation first and conclude your response with "Therefore, the model response contains an error." or "Therefore, the model response contains no error.""" + '"'
