# utils/prompts.py

# EXTRACT_ANSWER_PROMPT = """You will receive a response enclosed within <response> and </response> tags. Your task is to extract the final answer from the response that matches one of the options listed between <options> and </options>, based on the question provided between <question> and </question>. Do not interpret or infer beyond the provided text. Return your answer strictly in this format: ANSWER: <option_letter>. Do not output anything else. Make sure your extracted option_letter matches one of the provided options.

# <response>
# {answer}
# </response>

# <question>
# {question}
# </question>

# <options>
# {option}
# </options>"""

EXTRACT_ANSWER_PROMPT = """You will be given a text enclosed within <text> and </text> tags. Your task is to extract the final answer from the text such that the extracted answer matches exactly with one of the option letters listed between <options> and </options>: {keys}. Do not make any assumptions beyond the provided content. Return only the answer in this format: 'ANSWER: <option_letter>'. Do not output anything else. Make sure your extracted option_letter matches one of the provided options.

<text>
{answer}
</text>

<options>
{option}
</options>"""


# EXTRACT_ANSWER_PROMPT = """You will be given a text enclosed within <text> and </text> tags. Your task is to extract the final answer from the text such that the extracted answer exactly matches one of the options listed between <options> and </options>. Do not make any assumptions beyond the provided content. Put your final selected option_letter within \\boxed{{}}. Do not output anything else. Make sure your extracted option_letter matches one of the provided options.

# <text>
# {answer}
# </text>

# <options>
# {option}
# </options>"""