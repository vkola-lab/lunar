# Prompt templates for different styles
# These will be filled using str.format, so remember to escape literal braces by doubling them.

TEMPLATE_THINK = """Below is the question and the corresponding answer options:
<question>
{question}
</question>

<options>
{options}
</options>

Thinking steps: 
Provide a comprehensive summary of the patient's presentation.
Analyze each of the provided answer options through a structured differential diagnosis.
Provide your final answer by selecting the best option from the provided options, and then place your selected option letter within \\boxed{{}}. Please select only one option."""

TEMPLATE = """Question: {question}

Answer Choices: 
{options}
"""

EXTRACT_ANSWER_PROMPT = """You will be given a text enclosed within <text> and </text> tags. 
Your task is to extract the "Final Answer" from the text and match it with one of the option letter or number listed between <options> and </options>. 
Look at the provided text to identify the "Final Answer" (or "Best Answer") and match it to the correct letter or number.
Do not try to come up with a new answer, just identify what is in the text.
Output your answer inside \\boxed{{}}. 
Do not output anything else.

Important: If you cannot identify the final answer symbol, output \\boxed{{0}}.

Example 1: 

Example text: 
"... the final answer is \\boxed{{\\text{{Dementia (DE)}}}}." 

Example options: 
"A. Normal Cognition 
B. Mild Cognitive Impairment 
C. Dementia"

Example output: "\\boxed{{C}}"

Example 2:

Example text: 
"... the final answer is \\boxed{{\\text{{Alzheimer's Disease}}}}." 

Example options: 
"1. Alzheimer's Disease
2. Lewy Body Dementia
3. Vascular Dementia"

Example output: "\\boxed{{1}}"

<text>
{answer}
</text>

<options>
{options}
</options>"""