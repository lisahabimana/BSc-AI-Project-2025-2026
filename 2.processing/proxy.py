import json
import re
import time
import ast
from json_repair import repair_json
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
import pandas as pd
from pypdf import PdfReader
from langchain_core.prompts import PromptTemplate

#RETRIEVE KEYS FROM ENVIRONMENT
gpt_main_key = os.getenv("GPT_MAIN_KEY")

#LOAD MODEL
models = {
    "gpt52": ChatOpenAI( #gpt5
    model="gpt-5.2-pro",
    api_key=gpt_main_key,
    temperature=0
    )
}

#LOAD PROMPT
prompt = PromptTemplate(
    input_variables=["report"],
    template="""
You are an expert educational evaluator.

You MUST evaluate the report using the STRICT rubric below.
Do NOT guess. Follow definitions exactly.

RUBRIC

1. Does the OPP contain current and relevant information?
Score:
0 = Outdated or irrelevant information.
1 = Mostly current information.
2 = Fully current and educationally relevant information.
Notes:
Include recent assessment data and recent classroom observations.
Avoid including outdated school history unless it remains relevant.
Red flags: outdated class descriptions, expired IQ assessments without explanation, lengthy historical descriptions.

2. Is behavior described in observable and specific terms?
Score:
0 = Behavior is described abstractly or through labels.
1 = Behavior is partly concrete.
2 = Behavior is observable, specific, and measurable.
Notes:
Describe what the student actually does.
Good example: "Does not start a task independently without visual instructions."
Poor examples: "Is unmotivated." "Has behavioral problems."

3. Does the OPP lead to concrete educational actions?
Score:
0 = No concrete educational actions are described.
1 = Educational actions are only partly described.
2 = The OPP provides clear, directly applicable actions for teachers.
Notes:
Ask yourself: What should the teacher do differently tomorrow?
Recommendations should be immediately usable in practice.

4. Is there a logical connection between observations, analysis, support needs, and educational pathway?
Score:
0 = Sections are disconnected.
1 = Limited logical connection.
2 = Strong and consistent logical relationship throughout the OPP.
Notes:
Observations should explain the identified needs.
Support needs should justify the proposed educational pathway.

5. Is the language professional, clear, and easy to read?
Score:
0 = Vague language or excessive jargon.
1 = Generally understandable.
2 = Clear, concise, and professional throughout.
Notes:
Avoid judgmental language such as "lazy," "weak," or "can't."
Prefer need-focused language such as "Needs support with…" or "Demonstrates…"

6. Is the student's developmental functioning described appropriately?
Score:
0 = General or descriptive only.
1 = Partly concrete.
2 = Observable, context-specific, and educationally relevant.
Notes:
Consider:
Social-emotional functioning
Executive functioning
Communication
Task behavior
Independence
Good example: "Becomes overstimulated during classroom transitions."

7. Is academic functioning meaningfully described?
Score:
0 = Only assessment scores are reported.
1 = Scores are reported with limited interpretation.
2 = Scores are interpreted and linked to educational implications.
Notes:
Explain what the student can do in the classroom rather than only reporting test scores.

8. Are facilitating factors concrete and educationally relevant?
Score:
0 = General or irrelevant strengths.
1 = Reasonably concrete.
2 = Observable, functional, and relevant for learning.
Notes:
Describe strengths that help learning.
Good: "Benefits from visual supports."
Good: "Works more effectively with predictable routines."
Avoid personality traits such as "Is kind" or "Has a good sense of humor."

9. Are hindering factors concrete and educationally relevant?
Score:
0 = General descriptions or diagnostic labels only.
1 = Partly concrete.
2 = Observable, functional, and educationally relevant.
Notes:
Describe observable barriers instead of diagnoses.
Good: "Loses track during multi-step assignments."
Good: "Is distracted by auditory stimuli."
Avoid: ADHD, Autism, Behavioral problems.

10. Are the student's educational/support needs concrete and actionable?
Score:
0 = Problems are described instead of support needs.
1 = Support needs are partly specified.
2 = Support needs are concrete, student-specific, and directly actionable.
Notes:
Support needs should:
Be written as needs.
Describe who does what, where, when, and how.
Be individualized.
Go beyond standard classroom practice unless justified.
Allow teachers to act immediately.

11. Is the intensity of support clearly described?
Score:
0 = No indication of support intensity.
1 = General indication only.
2 = Clearly justified frequency and intensity.
Notes:
Specify when and how often support is required.
Examples:
Daily
During transitions
Continuous support when initiating tasks

12. Does the integrative profile provide meaningful analysis?
Score:
0 = Repetition or summary only.
1 = Limited analysis.
2 = Strong integrative analysis.
Notes:
A strong profile:
Connects multiple developmental domains.
Explains cause-and-effect relationships.
Links observations to support needs.
Describes educational implications.
Avoids repeating information already presented elsewhere.

13. Is the educational exit perspective logically justified?
Score:
0 = No justification.
1 = Limited justification.
2 = Clear and consistent justification.
Notes:
Consider:
Academic functioning
Executive functioning
Independence
Support needs

14. Is the document internally consistent?
Score:
0 = Major contradictions.
1 = Minor inconsistencies.
2 = Fully consistent throughout.
Notes:
Check that all sections support one another.
Examples of inconsistencies:
High independence but continuous supervision.
High educational pathway despite very low academic functioning.

15. Does the OPP avoid generic language?
Score:
0 = Many generic statements.
1 = Some generic statements.
2 = Almost entirely student-specific language.
Notes:
Replace vague statements with specific descriptions.
Avoid:
"Needs structure."
"Needs clarity."
"Positive feedback."
Instead, explain exactly what structure or feedback is needed.

16. Is the language primarily needs-focused rather than problem-focused?
Score:
0 = Frequent deficit-oriented language.
1 = Some problem-focused language.
2 = Primarily strengths- and needs-focused language.
Notes:
Describe observable behavior and educational needs rather than emphasizing deficits or diagnoses.

17. Is the OPP concise and focused?
Score:
0 = Overly long or unfocused.
1 = Reasonably concise.
2 = Concise and focused on relevant information.
Notes:
Include only information that contributes to understanding the student's educational needs and support. Avoid unnecessary repetition or extensive historical background.

TASK

Report:
{report}

Return ONLY valid JSON in this exact format:
{{
  "criteria": [17 integers between 0-2],
  "total_score": integer,
  "feedback": "string"
}}

Rules:
- Use double quotes only.
- Do not use single quotes.
- Do not wrap the JSON in markdown.
- Do not include any extra text before or after the JSON.
"""
)

#EXTRACT TEXT FROM PDF
def extract_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n\n".join(text).strip()

#DEBUG AND CLEAN FUNCTION FOR JSON
def clean_json_response(content):
    content = str(content).strip()
    content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    content = re.sub(r"^Here is the JSON:\s*", "", content, flags=re.IGNORECASE)
    return content.strip()


def extract_json_block(text):
    text = str(text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start:end + 1]

#LOAD FILES
pdf_dir = "data/reports/EHCP 25"
pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))

#OUTPUT DIRECRTORY
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

#LOOP THROUGH MODEL & REPORTS THEN SAVE EVALUATIONS OF MODEL (NL)
for model_name, model in models.items():
    results = []

    for idx, pdf_path in enumerate(pdf_files[:25], start=1):
        report = extract_pdf_text(pdf_path)

        if not report:
            print(f"Empty report: {pdf_path.name}")
            continue

        response = model.invoke(prompt.format(report=report))
        content = response.content

        if isinstance(content, list):
            content = "\n".join(
                item["text"] if isinstance(item, dict) and "text" in item else str(item)
                for item in content
            )

        content = clean_json_response(content)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            try:
                fixed = repair_json(content)
                data = json.loads(fixed)
            except Exception:
                print(f"Parse failed: {pdf_path.name}")
                continue

        if isinstance(data, list):
            dict_items = [item for item in data if isinstance(item, dict)]
            if not dict_items:
                print(f"List response without dict item: {pdf_path.name}")
                continue
            data = dict_items[-1]

        if not isinstance(data, dict):
            print(f"Still not a dict: {pdf_path.name} -> type={type(data)}")
            continue

        if not all(k in data for k in ("criteria", "total_score", "feedback")):
            print(f"Bad schema: {pdf_path.name}")
            print(f"Returned keys: {sorted(data.keys())}")
            print(f"Parsed object preview: {str(data)[:500]}")
            continue

        if not isinstance(data["criteria"], list) or len(data["criteria"]) != 17:
            print(f"Bad criteria length: {pdf_path.name}")
            print(f"criteria preview: {data.get('criteria')}")
            continue

        results.append({
            "report_number": idx,
            "pdf_name": pdf_path.name,
            "report": report,
            **{f"c{i+1}": data["criteria"][i] for i in range(17)},
            "total_score": data["total_score"],
            "feedback": data["feedback"],
            "raw_output": content,
        })

        print(f"{model_name}: processed {idx}/{min(1, len(pdf_files))} -> {pdf_path.name}")

    print(f"{model_name}: collected {len(results)} valid rows")

    if results:
        pd.DataFrame(results).to_csv(OUT_DIR / f"{model_name}_results.csv", index=False)
    else:
        print(f"No results to save for {model_name}")

print("Done")