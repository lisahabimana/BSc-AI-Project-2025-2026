import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

#LOAD FILES
INPUT_FILES = {
    "gpt":"evaluators_FINAL/gpt5_results.csv",
    "opus":"evaluators_FINAL/opus_results.csv",
    "glm":"evaluators_FINAL/glm_results.csv",
    "qwen":"evaluators_FINAL/qwen_results.csv",
    "minimax":"evaluators_FINAL/minimax_results.csv"
}

#DEBUG FCT
def is_nonempty(x):
    return pd.notna(x) and str(x).strip() != ""

def safe_int(x):
    try:
        return int(x)
    except:
        return np.nan
 
def extract_criterion_score(df):
    cols = [f"c{i}" for i in range(1, 18)]

    subset = df[cols].to_numpy().flatten().tolist()
    return subset

#COMPUTE CORRECTNESS
def correctness(df_ref, df_pred):
    ref = extract_criterion_score(df_ref)
    pred = extract_criterion_score(df_pred)

    kappa = cohen_kappa_score(ref, pred, weights="linear")

    print("Linear weighted kappa:", kappa)
    return kappa
#COMPUTE CONSISTENCY
def consistency(df):
    criteria_cols = [f"c{i}" for i in range(1, 18)]

    df["sum_match"] = df[criteria_cols].sum(axis=1) == df["total_score"]
    df["consistency_pass"] = df["sum_match"]

    return df["consistency_pass"].mean()

#COMPUTE COMPLETENESS
def completeness(df):
    criteria_cols = [f"c{i}" for i in range(1, 18)]
    required_cols = ["report_number", "pdf_name", "report", *criteria_cols, "total_score", "feedback", "raw_output"]

    df["filled_required_fields"] = df[required_cols].apply(lambda row: sum(is_nonempty(v) for v in row), axis=1)
    df["completeness_ratio"] = df["filled_required_fields"] / len(required_cols)

    return df["completeness_ratio"].mean()   

#COMPUTE CLARITY
def clarity(df):
    text = df["feedback"].astype(str)

    score = (
        text.str.contains("clear|well written|structured|concise", case=False).mean()
        -
        text.str.contains("unclear|confusing|vague|uncertain", case=False).mean()
    )

    return score

#READ CSV /LOAD FILE
proxy_ref = pd.read_csv("output/proxy_results/gpt52_results.csv")
summary = []

#COMPUTE METRICS *2 FOR COMPARISON WITH LLM JUDGE
for name, file in INPUT_FILES.items():
    df = pd.read_csv(file)

    summary.append({
        "model_name": name,
        "mean_correctness": correctness(df, proxy_ref)*2,
        "mean_consistency": consistency(df)*2,
        "mean_completeness": completeness(df)*2,
        "mean_clarity": clarity(df)*2,
        "n_rows": len(df),
    })

#SAVE
out_path = Path("output/benchmark/my_summary.csv")
pd.DataFrame(summary).to_csv(out_path, index=False)