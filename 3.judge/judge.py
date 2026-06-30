import json
import time
import os
from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from json_repair import repair_json

#RETRIEVE KEYS FROM ENVIRONMENT
gpt_main_key = os.getenv("GPT_MAIN_KEY")

#LOAD MODEL
model = {
    "gpt52": ChatOpenAI( #gpt5
    model="gpt-5.2-pro",
    api_key=gpt_main_key,
    temperature=0
    )
}

#LOAD FILES
INPUT_FILES = {
    "gpt":"output/evaluators_results/gpt5_results.csv",
    "opus":"output/evaluators_results/opus_results.csv",
    "glm":"output/evaluators_results/glm_results.csv",
    "qwen":"output/evaluators_results/qwen_results.csv",
    "minimax":"output/evaluators_results/minimax_results.csv"
}

#DIRECTORY
OUT_DIR = Path("output/judge_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

#LOAD PROMPT
prompt = PromptTemplate(
    input_variables=["report", "proxy","evaluator_output"],
    template="""
You are an expert evaluator of an LLM-generated assessment of an OPP report.

Judge the evaluator output against the proxy reference. 
Score correctness, consistency, completeness, and clarity based on how closely the evaluator matches the proxy’s rubric-based assessment.

This evaluation is for ONE report only.
Do not average across reports.

Score these criteria from 0 to 2:
1. correctness
2. consistency
3. completeness
4. clarity

Definitions:
- correctness: evaluator scores and feedback align with the proxy reference.
- consistency: evaluator scores and feedback do not contradict each other.
- completeness: all required rubric fields are present.
- clarity: feedback is understandable, specific, and usable.

Return ONLY valid JSON with exactly these keys:
{{
  "correctness": 0,
  "consistency": 0,
  "completeness": 0,
  "clarity": 0,
  "rationale": ""
}}

REPORT:
{report}

PROXY REFERENCE:
{proxy}

ASSESSMENT OUTPUT:
{evaluator_output}

"""
)

#CLEAN AND DEBUG JSON 
def clean_json_response(content):
    content = str(content).strip()
    content = content.replace("```json", "").replace("```", "").strip()
    return content

#SEND PROMPT + GET LLM response + EXTRACT JSON + DEBUG
def judge_row(report_text, proxy, evaluator_output):
    msg = prompt.format(
        report=report_text,
        proxy=proxy,
        evaluator_output=evaluator_output,
    )
    response = model.invoke(msg)

    content = response.content
    if isinstance(content, list):
        content = "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )

    content = clean_json_response(content)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        try:
            repaired = repair_json(content)
            data = json.loads(repaired)
        except Exception:
            return {
                "correctness": None,
                "consistency": None,
                "completeness": None,
                "clarity": None,
                "rationale": content,
            }

    if isinstance(data, list):
        dict_items = [item for item in data if isinstance(item, dict)]
        if dict_items:
            data = dict_items[0]
        else:
            return {
                "correctness": None,
                "consistency": None,
                "completeness": None,
                "clarity": None,
                "rationale": content,
            }

    if not isinstance(data, dict):
        return {
            "correctness": None,
            "consistency": None,
            "completeness": None,
            "clarity": None,
            "rationale": content,
        }

    return data

#CONVERT DF INTO JSON FOR LLM
def row_to_output(row):

    output = {f"c{i}": row.get(f"c{i}") for i in range(1, 18)}
    output["total_score"] = row.get("total_score")
    output["feedback"] = row.get("feedback", "")
    output["raw_output"] = row.get("raw_output", "")
    
    return output

#LLM JUDGE PIPELINE

proxy_df = pd.read_csv("output/proxy_results/gpt52_results.csv")
proxy_map = {row["report_number"]: row for _, row in proxy_df.iterrows()}

for model_name, file in INPUT_FILES.items():
    df = pd.read_csv(file)
    results = []

    for _, row in df.iterrows():
        report_number = row["report_number"]

        if report_number not in proxy_map:
            print(f"Missing proxy for report {report_number}")
            continue

        proxy_output = row_to_output(proxy_map[report_number])
        evaluator_output = row_to_output(row)

        judged = judge_row(
            row["report"],
            json.dumps(proxy_output, ensure_ascii=False),
            json.dumps(evaluator_output, ensure_ascii=False),
        )

        results.append({
            "model_name": model_name,
            "report_number": row["report_number"],
            "correctness": judged.get("correctness"),
            "consistency": judged.get("consistency"),
            "completeness": judged.get("completeness"),
            "clarity": judged.get("clarity"),
            "rationale": judged.get("rationale"),
        })

        print("RAW JUDGED:", judged)

        print(f"Done report {row['report_number']}")
        time.sleep(1)
        
    output_file = OUT_DIR / f"{model_name}_judged.csv"
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Saved to {output_file}")