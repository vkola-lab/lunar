# TEMPLATE = """Below is the background of the patient.
# <patient_data>
# {patient}
# </patient_data>

# Below is the question and the corresponding answer options:
# <question>
# {question}
# </question>

# <options>
# {options}
# </options>

# Thinking steps: 
# Provide a comprehensive summary of the patient's presentation.
# Analyze each of the provided answer options through a structured differential diagnosis.
# Provide your final answer on a new line as 'Answer: <option letter>' by selecting the best option from the provided options."""

TEMPLATE = """Below is the background of the patient.
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


# TEMPLATE = """Below is the background of the patient.
# <patient_data>
# {patient}
# </patient_data>

# Below is the question and the corresponding answer options:
# <question>
# {question}
# </question>

# <options>
# {options}
# </options> 

# Provide your final answer on a new line as 'Answer: <option letter>' by selecting the best option from the provided options"""