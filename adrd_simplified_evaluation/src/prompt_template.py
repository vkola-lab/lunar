# TEMPLATE = """<patient data>
# {patient_summary}
# </patient data>

# <question>
# {question}. Provide your final answer on a new line as 'Answer: <option letter>' by selecting the best option from the provided options.
# </question>

# <options>
# {options}
# </options>"""

TEMPLATE = """<patient data>
{patient_summary}
</patient data>

<question>
{question}. Provide your final answer on a new line as 'Answer: <option letter>' by selecting the best option from the provided options.
</question>

<options>
A. Yes
B. No
</options>"""