import pandas as pd
import numpy as np

#LOAD PERFORMANCE FILE
perf = pd.read_csv("output/benchmark/model_summary.csv")
perf.columns = perf.columns.str.strip().str.lower()

#CONVERT COLUMNS
for c in ["mean_correctness", "mean_consistency", "mean_completeness", "mean_clarity"]:
    perf[c] = pd.to_numeric(perf[c], errors="coerce")

if "n_rows" not in perf.columns:
    perf["n_rows"] = 25
else:
    perf["n_rows"] = pd.to_numeric(perf["n_rows"], errors="coerce")

perf = perf.dropna(subset=["model_name"])

#COST TABLE (retrieved from consoles, no csv available for glm)
cost_df = pd.DataFrame([
    {"model_name": "gpt", "total_cost": 16.31},
    {"model_name": "qwen", "total_cost": 0.366784},
    {"model_name": "opus", "total_cost": 0.86},
    {"model_name": "minimax", "total_cost": 0.0668},
    {"model_name": "glm", "total_cost": 0.41},
])

#COMPUTE COST PER CALL
final = perf.merge(cost_df, on="model_name", how="left")
final["cost_per_call"] = final["total_cost"] / final["n_rows"]

final = final[[
    "model_name",
    "mean_correctness",
    "mean_consistency",
    "mean_completeness",
    "mean_clarity",
    "cost_per_call",
    "n_rows"
]]

#save into same file with added cost column
final.to_csv("output/benchmark/model_summary.csv", index=False)

print(final)