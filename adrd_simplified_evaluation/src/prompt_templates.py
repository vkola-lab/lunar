# Prompt templates for different styles
# These will be filled using str.format, so remember to escape literal braces by doubling them.
TEMPLATE = """Question: {question}

Answer Choices: 
{options}
"""

EXTRACT_ANSWER_BOXED_PROMPT = """You will be given a text enclosed within <text> and </text> tags. 
Your task is to extract the final answer from the text and match it with one of the option letter (or number) listed between <options> and </options>. 
Do not try to come up with a new answer, just identify what is in the text. Do not make assumptions beyond the provided content.
Return only a single letter or number answer inside \\boxed{{}}.
Do not output anything else. 

Important: If you cannot identify the final answer symbol, output \\boxed{{-}}.

Example 1: 

Example 1 text: 
"... the final answer is \\boxed{{\\text{{Dementia (DE)}}}}." 

Example 1 options: "A. Normal Cognition B. Mild Cognitive Impairment C. Dementia"

Example 1 output: "\\boxed{{C}}"

Example 2:

Example 2 text: 
"... ANSWER: \\boxed{{\\text{{Alzheimer's Disease}}}}." 

Example 2 options: "1. Alzheimer's Disease 2. Lewy Body Dementia 3. Vascular Dementia"

Example 2 output: "\\boxed{{1}}"

<text>
{answer}
</text>

<options>
{options}
</options>"""


MULTILABEL = """Question: {question} Select all options that apply as comma separated option letters like \\boxed{{A, B, C}}.

Answer Choices: 
{options}
"""

EXTRACT_MULTI_ANSWER_BOXED_PROMPT = """You will be given a text enclosed within <text> and </text> tags. 
Your task is to extract the final answer from the text and match it with one of the option letter (or number) listed between <options> and </options>. 
Do not try to come up with a new answer, just identify what is in the text. Do not make assumptions beyond the provided content.
Return only comma separated letters answer inside \\boxed{{}}.
Do not output anything else. 

Important: If you cannot identify the final answer letter, output \\boxed{{-}}.

Example 1: 

Example 1 text: 
"... the final answer is \\boxed{{\\text{{C. Dementia (DE)}}}}." 

Example 1 options: "A. Normal Cognition B. Mild Cognitive Impairment C. Dementia"

Example 1 output: "\\boxed{{C}}"

Example 2:

Example 2 text: 
"... ANSWER: \\boxed{{\\text{{A. Alzheimer's Disease, C. Vascular Dementia}}}}." 

Example 2 options: "A. Alzheimer's Disease B. Lewy Body Dementia C. Vascular Dementia"

Example 2 output: "\\boxed{{A, C}}"

<text>
{answer}
</text>

<options>
{options}
</options>"""