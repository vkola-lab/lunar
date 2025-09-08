#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import markdown
import json


# In[10]:


# initialize df with the columns Variable / Field Name	Form Name	Section Header	Field Type	Field Label	Choices, Calculations, OR Slider Labels	Field Note	Text Validation Type OR Show Slider Number	Text Validation Min	Text Validation Max	Identifier?	Branching Logic (Show field only if...)	Required Field?	Custom Alignment	Question Number (surveys only)	Matrix Group Name	Matrix Ranking?	Field Annotation
cols = [
    "Variable / Field Name",
	"Form Name",
	"Section Header",
	"Field Type",
	"Field Label",
	"Choices, Calculations, OR Slider Labels",
	"Field Note",
	"Text Validation Type OR Show Slider Number",
	"Text Validation Min",
    "Text Validation Max",
	"Identifier?",
	"Branching Logic (Show field only if...)",
	"Required Field?",
	"Custom Alignment",
	"Question Number (surveys only)",
	"Matrix Group Name",
	"Matrix Ranking?",
	"Field Annotation",
]

# init the df
df = pd.DataFrame(columns=cols)


# In[11]:


#record id
new_row = {
    "Variable / Field Name": f"record_id",
    "Form Name": "llama_vs_qwen",
    "Field Type": "text",
    "Field Label": "Record ID",
    "Section Header": "<h>Identification<\h>",
}
df.loc[len(df)] = new_row
# name
new_row = {
    "Variable / Field Name": f"name",
    "Form Name": "llama_vs_qwen",
    "Field Type": "text",
    "Field Label": "Respondent Name",
    "Field Note": "Please enter your name",
    "Required Field?": "y",
    # "Section Header": "<h>Identification<\h>"
}
df.loc[len(df)] = new_row

# description
new_row = {
    "Variable / Field Name": f"description",
    "Form Name": "llama_vs_qwen",
    "Field Type": "descriptive",
    "Field Label": """<span style="font-weight:= normal;"=>(You will be provided with 10 cases. First, you will be presented with the data associated with the person. Then you will be given a series of specific queries along with two responses to the same query. Please review this information carefully.)</span>"""
}
df.loc[len(df)] = new_row



# <style>
#   /* A container to hold the panels in a row */
#   .container {
#     display: flex;         /* Places panels side by side */
#     gap: 20px;            /* Adds space between panels (optional) */
#     margin: 20px;         /* Some margin around the container */
#   }
# 
#   /* Each panel's styling */
#   .panel {
#     flex: 1;              /* Allows each panel to expand and fill available space (optional) */
#     min-width: 500px
#     max-width: 700px;     /* Alternatively, you can use a fixed width like 300px */
#     height: 400px;        /* Fixed height so scrolling can occur */
#     overflow-y: auto;     /* Vertical scrollbar will appear if content overflows */
#     border: 1px solid #ccc;  
#     padding: 10px;
#     box-sizing: border-box;
#   }
# 
#   /* (Optional) Just to show visible scroll area with background color */
#   .panel:nth-child(odd) {
#     background-color: #f9f9f9;
#   }
# </style>
# <div class="container">
#   <div class="panel">
#     <h2>Patient Data</h2>
#     <p>
#       example patient data
#     </p>
#     <p>
#       More data
#     </p>
#     <p>
#       more data
#     </p>
#   </div>
#   <div class="panel">
#     <h2>Summary 1</h2>
#     <p>
#       llm response
#     </p>
#     <p>
#       continued
#     </p>
#     <p>
#       more
#     </p>
#   </div>
#   <div class="panel">
#     <h2>Summary 2</h2>
#     <p>
#       llm response
#     </p>
#     <p>
#      more content
#     </p>
#   </div>
# </div>
# 
# 

# In[12]:


def generate_panels(patient_data, summary_1, summary_2):
  '''
  This function generates a HTML string that displays three panels side by side.

  input: 
    patient_data: str, a string containing the patient data
    summary_1: str, a string containing the first summary
    summary_2: str, a string containing the second summary

    * all inputs should be in HTML format, wrapped with <p> tags
  '''
  html = f"""
  <style>
    /* A container to hold the panels in a row */
    .container {{
      display: flex;         /* Places panels side by side */
      gap: 20px;            /* Adds space between panels (optional) */
      margin: 20px;         /* Some margin around the container */
    }}

    /* Each panel's styling */
    .panel {{
      flex: 1;              /* Allows each panel to expand and fill available space (optional) */
      min-width: 500px
      max-width: 700px;     /* Alternatively, you can use a fixed width like 300px */
      height: 400px;        /* Fixed height so scrolling can occur */
      overflow-y: auto;     /* Vertical scrollbar will appear if content overflows */
      border: 1px solid #ccc;  
      padding: 10px;
      box-sizing: border-box;
    }}

    /* (Optional) Just to show visible scroll area with background color */
    .panel:nth-child(odd) {{
      background-color: #f9f9f9;
    }}
  </style>
  <div class="container">
    <div class="panel">
      <h2>Patient Data</h2>
        {patient_data}
    </div>
    <div class="panel">
      <h2>Summary 1</h2>
        {summary_1}
    </div>
    <div class="panel">
      <h2>Summary 2</h2>
        {summary_2}
    </div>
  </div>
  """
  return html

def add_redcap_field(df, case_num, patient_data, question, summary_1, summary_2, choices):
  '''
  This function adds a new field to the ongoing REDCap dataframe.
  
  input:
    df: pd.DataFrame, the ongoing REDCap dataframe
    patient_data: dict, a dictionary containing the patient data
    question: str, the question to be added
    summary_1: str, the first summary
    summary_2: str, the second summary
    choices: list, a list of choices for the question
  '''

  # convert patient_data to html
  patient_data_html = dict_to_html(patient_data)
  patient_data_html = f"<p>{patient_data_html}</p>"

  # prepare the summary htmls
  summary_1_html = markdown.markdown(summary_1)
  summary_1_html = f"<p>*{question}*</p><p>{summary_1_html}</p>"
  summary_2_html = markdown.markdown(summary_2)
  summary_2_html = f"<p>*{question}*</p><p>{summary_2_html}</p>"

  # generate the panels
  panels_html = generate_panels(patient_data_html, summary_1_html, summary_2_html)

  # add the new field to the dataframe
  new_row = {
      "Variable / Field Name": f"field_{len(df)}",
      "Form Name": "llama_vs_qwen",
      "Field Type": "radio",
      "Field Label": question,
      "Choices, Calculations, OR Slider Labels": choices,
      "Field Note": panels_html,
      "Required Field?": "y",
      "Section Header": f"<h>Case {case_num}<\h>"
  }
  df.loc[len(df)] = new_row

  return df
  
def dict_to_html(nested_dict):
  # print(nested_dict)
  html_output = ''
  for key, value in nested_dict.items():
      if isinstance(value, dict):
          html_output += f"<h3><u>{key}</u></h3>"
          html_output += dict_to_html(value)
      elif isinstance(value, str) or isinstance(value, int) or isinstance(value, float):
          html_output += f"""<p>{key}: <span style="font-weight:= normal;"=>{value} </span> </p>"""
      else:
          raise ValueError(f"Value of type {type(value)} not supported")
  return html_output

def criterion_dict_to_choices_str(criterion_dict):
  choices = []
  for key, value in criterion_dict.items():
    choices.append(f"{key}, Level {key}: {value}")
  return " | ".join(choices)


# In[13]:


with open('acgme_neurology.json') as f:
  acgme_neurology = json.load(f)

acgme_neurology


# In[17]:


filler_text = """
Lorem ipsum odor amet, consectetuer adipiscing elit. Aliquam lorem etiam malesuada penatibus egestas accumsan. Aptent suscipit ultricies vivamus ultricies integer mus dui. Turpis pellentesque cras felis risus posuere justo cursus. Ut natoque fermentum parturient suspendisse tempus interdum auctor tristique aptent. Finibus ante porta venenatis integer nam. Ligula cursus sollicitudin mollis euismod curae et. Diam quis fusce quis lorem hac at volutpat porttitor faucibus. Diam nunc natoque et quis scelerisque posuere ante quisque.

Non penatibus tempus vulputate tempus aliquet; porta senectus. Finibus curae accumsan vivamus fusce dui vivamus class sollicitudin. Sed convallis senectus blandit condimentum lectus quam. Curae donec fermentum ante vehicula bibendum dolor. Feugiat sapien lobortis per maximus donec est ut. Gravida convallis hac proin scelerisque in. Himenaeos velit mattis efficitur euismod maximus imperdiet ex.

Placerat praesent potenti praesent imperdiet curabitur fusce massa aenean nibh. Per eget lacus ac integer himenaeos primis ac vitae. Vel porta nulla euismod pharetra augue. Tincidunt lacus enim curabitur tempor adipiscing consequat adipiscing ullamcorper. Tempus mattis varius habitasse velit nullam ultricies maximus massa. Convallis ad augue consectetur cras sapien. Vestibulum porta nam scelerisque elit ultricies turpis auctor nam. Porttitor semper conubia metus et nam dui nunc est egestas.

Suscipit turpis ipsum arcu egestas quisque donec. Consequat sit odio netus est suspendisse maximus nostra porttitor diam. Neque vestibulum rutrum inceptos ligula lectus maecenas vivamus suscipit. Faucibus potenti nisi libero nam ipsum vel aliquet. Parturient mi pulvinar lectus molestie sem auctor porttitor. Morbi libero mi leo habitant aliquet dis. Ultrices tincidunt ad nulla nostra velit netus.

Interdum amet consequat; auctor donec primis elementum. Risus venenatis proin vestibulum auctor ac imperdiet. Leo nostra felis tempor, porttitor nullam varius eu mus. Feugiat aliquam ipsum magna potenti vivamus. Placerat placerat parturient vitae porttitor sem praesent nam eget. Viverra cursus conubia ornare sit scelerisque. Cursus semper fringilla nisi nibh lacinia ultrices vitae. Convallis lectus etiam ante venenatis ut.
"""

filler_patient_data = json.loads('{"Subject Demographics": {"Living situation": "Lives with group", "Primary language": "English", "Level of independence": "Requires some assistance with basic activities", "Race": "White", "Second race": "None reported", "Hispanic/Latino ethnicity": "No", "Primary reason for coming to ADC": "To participate in a research study", "Principal referral source": "Other", "Is the subject left- or right-handed?": "Right-handed", "Type of residence": "Retirement community or independent group living", "Subject\'s age at visit": 97, "Marital status": "Never married (or marriage was annulled)", "Years of education": 18, "Subject\'s sex": "Female", "Subject\'s month of birth": 4, "Subject\'s year of birth": 1921, "Third race": "None reported", "Derived NIH race definitions": "White"}, "Co-participant Demographics": {"Co-participant\'s relationship to subject": "Other relative", "Derived NIH race definitions": "White", "Co-participant\'s year of birth": 1963.0, "Co-participant\'s month of birth": 4.0, "Is there a question about the co-participant\'s reliability?": "No", "Does the co-participant live with the subject?": "No", "If no, approximate frequency of telephone contact?": "Less than once a month", "Co-participant\'s sex": "Female", "If no, approximate frequency of in-person visits?": "Monthly"}, "Subject Family History": {"Indicator of father with cognitive impairment": "No report of father with cognitive impairment", "If other, specify": ".", "Specified other mutation": ".", "Indicator of mother with cognitive impairment": "No report of mother with cognitive impairment", "In this family, is there evidence for an AD mutation (from list of specific mutations)?": "No", "Indicator of first-degree family member with cognitive impairment": "No report of a first-degree family member with cognitive impairment", "In this family, is there evidence for an FTLD mutation (from list of specific mutations)?": "No", "If yes, Other (specify)": ".", "In this family, is there evidence for a mutation other than an AD or FTLD mutation?": "No"}, "Subject Medications": {"Total number of medications reported at each visit (range (0 - 40))": 2.0, "Subject taking any medications": "Yes", "Name of medications used within two weeks of UDS visit": "MULTIVITAMIN"}, "Neuropsychiatric Inventory Questionnaire (NPI-Q)": {"Nighttime behaviors in the last month": "Yes", "Delusions in the last month": "No", "Agitation or aggression severity": "Moderate (significant, but not a dramatic change)", "Anxiety severity": "Moderate (significant, but not a dramatic change)", "Agitation or aggression in the last month": "Yes", "Nighttime behaviors severity": "Mild (noticeable, but not a significant change)", "NPI-Q co-participant": "Other", "NPI-Q co-participant, other - specify": "neice", "Depression or dysphoria in the last month": "No", "Anxiety in the last month": "Yes", "Motor disturbance in the last month": "No", "Elation or euphoria in the last month": "No", "Disinhibition in the last month": "No", "Irritability or lability in the last month": "Yes", "Irritability or lability severity": "Severe (very marked or prominent, a dramatic change)", "Apathy or indifference severity": "Severe (very marked or prominent, a dramatic change)", "Apathy or indifference in the last month": "Yes", "Hallucinations in the last month": "No", "Appetite and eating problems in the last month": "Yes", "Appetite and eating severity": "Mild (noticeable, but not a significant change)"}, "Functional Assessment Scale (FAS)": {"In the past four weeks, did the subject have any difficulty or need help with: Playing a game of skill such as bridge or chess, working on a hobby": "Dependent", "In the past four weeks, did the subject have any difficulty or need help with: Keeping track of current events": "Dependent", "In the past four weeks, did the subject have any difficulty or need help with: Assembling tax records, business affairs, or other paper": "Dependent", "In the past four weeks, did the subject have any difficulty or need help with: Heating water, making a cup of coffee, turning off the stove": "Dependent", "In the past four weeks, did the subject have any difficulty or need help with: Remembering appointments, family occasions, idays, medications": "Dependent", "In the past four weeks, did the subject have any difficulty or need help with: Preparing a balanced meal": "Dependent", "In the past four weeks, did the subject have any difficulty or need help with: Shopping alone for clothes, household necessities, or groceries": "Dependent", "In the past four weeks, did the subject have any difficulty or need help with: Writing checks, paying bills, or balancing a checkbook": "Dependent"}, "Medical Conditions": {"Incontinence present - urinary": "Not assessed", "Atrial fibrillation present": "No", "Procedure: pacemaker and/or defibrillator within the past 12 months": "Not assessed", "Procedure: heart valve replacement or repair within the past 12 months": "Not assessed", "Diabetes present at visit": "No", "Hyposomnia/insomnia present": "Not assessed", "Myocardial infarct present within the past 12 months": "No", "Angina present": "No", "Hypertension present": "Yes", "Hypercholesterolemia present": "No", "b12 deficiency present": "No", "Antibody-mediated encephalopathy within the past 12 months": "Not assessed", "Percutaneous coronary intervention: angioplasty and/or stent within the past 12 months": "Not assessed", "Carotid procedure: angioplasty, endarterectomy, or stent within the past 12 months": "Not assessed", "Thyroid disease present": "Not assessed", "Arthritis present": "Not assessed", "REM sleep behavior disorder (RBD) present": "Not assessed", "Other medical conditions or procedures within the past 12 months not listed above": "No", "Incontinence present - bowel": "Not assessed", "Cancer present in the last 12 months (excluding non-melanoma skin cancer), primary or metastatic": "No", "Congestive heart failure present": "No", "Other sleep disorder present": "Not assessed", "Sleep apnea present": "Not assessed"}, "Genetic testing": {"APOE genotype": "(e3 e3)", "Number of APOE e4 alleles": "No e4 allele"}}')


# In[18]:


for case_num in range(1, 11):
    patient_data = filler_patient_data

    for section, data in acgme_neurology.items():
        prompt = data['prompt']
        acgme_data = data['ACGME']
        # add the section header and prompt
        new_row = {
            "Variable / Field Name": f"section_{section}",
            "Form Name": "llama_vs_qwen",
            "Field Type": "descriptive",
            "Field Label": f"<h>{section}</h>",
            "Field Note": f"<p>{prompt}</p>"
        }
        df.loc[len(df)] = new_row

        for response_data in acgme_data:
            label = response_data['label']
            criterion_dict = response_data['levels']
            criterion_str = criterion_dict_to_choices_str(criterion_dict)
            # add the new field
            df = add_redcap_field(df, case_num, patient_data, label, filler_text, filler_text, criterion_str)
        
df    


# In[20]:


df.to_csv('/projectnb/vkolagrp/spuduch/fmADRD/recap_v2.csv', index=False)

