# utils/prompts.py

EXTRACT_ANSWER_PROMPT = """You will be given a response enclosed within <response> and </response> tags. Your task is to extract the final answer that matches one of the options listed between <options> and </options>, based on the question provided between <question> and </question>. Do not interpret or infer beyond the provided text. Return your answer strictly in this format: ANSWER: <option letter>. Do not output anything else.

<response>
{answer}
</response>

<question>
{question}
</question>

<options>
{option}
</options>"""
