import langchain
import langgraph
import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

#LOAD FILE
df = pd.read_csv("data/filtered_data/synthetic_personas.csv")
df_sample = df.sample(100, random_state=42)

#CREATE MODEL
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    api_key= os.getenv("GEMINI_API_KEY"),
    temperature=0
)

#ADD PROMPT
prompt = PromptTemplate(
    input_variables=[
        "report_number",
        "sen",
        "phase",
        "age",
        "gender",
        "secondary"
    ],
    template="""
EHCP REPORT #{report_number}

You are writing an EHCP-style education support summary based on the SEND Code of Practice (0–25).

The EHCP must include the following sections:

Section A: Views, interests, and aspirations
Section B: Special educational needs
Section C: Health needs
Section D: Social care needs
Section E: Outcomes sought
Section F: Special educational provision
Section G: Health provision
Section H1: Social care provision (CSDPA 1970)
Section H2: Additional social care provision
Section I: Placement
Section J: Personal budget / direct payments
Section K: Assessment evidence and advice

Student profile:
- SEN: {sen}
- Phase: {phase}
- Age group: {age}
- Gender: {gender}
- Secondary need: {secondary}

Instructions:
- Write a structured EHCP report using Sections A–K
- Display the report number at the top as "EHCP REPORT #{report_number}"
- Each section must be clearly labeled
- Use the student profile in all relevant sections
- Keep it realistic, concise, and educationally appropriate
- Do not invent specific medical diagnoses beyond the SEN category

Output format MUST be:

EHCP REPORT #{report_number}

Section A:
...

Section B:
...

...

Section K:
...
"""
)

#LOOP TO GENERATE REPORTS
reports = []

for i, row in df_sample.iterrows():

    formatted_prompt = prompt.format(
        report_number=i + 1,
        sen=row["sen_primary_need"],
        phase=row["phase_type_grouping"],
        age=row["age_group"],
        gender=row["gender"],
        secondary=row["secondary_need"]
    )

    response = model.invoke(formatted_prompt)

    reports.append({
        "report_number": i + 1,
        "sen": row["sen_primary_need"],
        "report": response.content
    })

output_df = pd.DataFrame(reports)
output_df.to_csv("ehcp_reports.csv", index=False)

print(output_df.head())