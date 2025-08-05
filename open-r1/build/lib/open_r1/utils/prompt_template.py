# GRPO_TEMPLATE = """You are a highly skilled neurologist. You will receive a patient's case in the form of a JSON file between <patient data> and </patient data> tags. Your job is to analyise the patient data and answer the question provided between <question> and </question> tags by selecting the best option from the options provided between <options> and </options> tags following the template provided between <template> and </template> tags. Finally write your final answer between <final answer> and </final answer> tags. Follow these guidelines carefully:
    
# <guidelines>
# - Follow the template provided between <template> and </template> tags.
# - Give the presentation of the case between <presentation of case> and </presentation of case> tags.
# - Summarize imaging and biomarker studies between <imaging and biomarker studies> and </imaging and biomarker studies> tags.
# - Provide a differential diagnosis between <differential diagnosis> and </differential diagnosis> tags.
# - Write your final answer between <final answer> and </final answer> tags.
# </guidelines>

# <template>
# <presentation of case>
# - Generate a detailed, well-structured narrative summarizing the patient's case. If available, include:
#     * basic demographics
#     * medical history
#     * family history
#     * results of the neurological and physical examination
#     * neuropsychological testing
#     * psychiatric and functional assessments
#     * genetic testing
# - Include all information from the patient data without omissions. Provide as much detail as possible.
# - Do not draw any conclusions, make diagnoses, or interpret the data; focus solely on summarizing the facts as presented.
# </presentation of case>

# <imaging and biomarker studies>
# - List available imaging (e.g., MRI, PET, CT scans) and biomarker studies (e.g., CSF analysis, PET scans).
# - Include specific interpretations of each finding and discuss potential implications of these findings for diagnosis.
# </imaging and biomarker studies>

# <differential diagnosis>
# - Perform a detailed, prioritized differential diagnosis based on clinical assessment and imaging/biomarker findings.
# - Describe how the individual's clinical assessments lead you to consider each potential diagnosis.
# </differential diagnosis>

# <final answer> Write your final answer by picking an option from the given options in one word. Directly answer A/B/C. </final answer>
# </template>

# <patient data>
# {patient}
# </patient data>

# <question>
# {question}
# </question>

# <options>
# {options}
# </options>"""

# GRPO_TEMPLATE = """<patient_data>
# {patient}
# </patient_data>

# <question>
# {question}
# </question>

# <options>
# {options}
# </options>"""

GRPO_TEMPLATE_THINK = """Below is the background of the patient.
<patient_data>
{patient}
</patient_data>

Below is the question and the corresponding answer options:
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


GRPO_TEMPLATE = """Question: {patient}

{question}

Answer Choices: 
{options}
"""


SFT_TEMPLATE = """Below is the background of the patient.
<patient_data>
{patient}
</patient_data>

Below is the question and the corresponding answer options:
<question>
{question}
</question>

<options>
{options}
</options>"""