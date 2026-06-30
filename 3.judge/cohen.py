from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

#LOAD FILES
files = list(Path("output/evaluators_results").glob("*_results.csv"))
gpt5 = pd.read_csv("output/proxy_results/gpt52_results.csv")

#DIRECTORY
out_dir = Path("output/cohen")
out_dir.mkdir(parents=True, exist_ok=True)

#COMPUTE COHEN
overall_summary = []

for file in files:
    df = pd.read_csv(file)
    model_name = file.stem.replace("_results", "")

    summary = {"model_name": model_name}

    kappa_vals = []
    weighted_kappa_vals = []

    for i in range(1, 18):
        c = f"c{i}"

        #check val are differents
        y_true = df[c].reset_index(drop=True)
        y_pred = gpt5[c].reset_index(drop=True)

        # skip invalid k cases
        if len(set(y_true.dropna())) < 2 or len(set(y_pred.dropna())) < 2:
            summary[f"{c}_kappa"] = np.nan
            summary[f"{c}_weighted_kappa"] = np.nan
            continue

        kappa = cohen_kappa_score(y_true, y_pred)
        wkappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")

        summary[f"{c}_kappa"] = kappa
        summary[f"{c}_weighted_kappa"] = wkappa

        kappa_vals.append(kappa)
        weighted_kappa_vals.append(wkappa)

    # per-model full file (all criteria)
    pd.DataFrame([summary]).to_csv(
        out_dir / f"{model_name}_cohen.csv",
        index=False
    )

    # store ONLY overall values for final table
    overall_summary.append({
        "model_name": model_name,
        "overall_kappa": np.nanmean(kappa_vals),
        "overall_weighted_kappa": np.nanmean(weighted_kappa_vals)
    })

# FINAL SUMMARY FILE (ONLY OVERALL SCORES)
final_df = pd.DataFrame(overall_summary)
final_df.to_csv(out_dir / "ALL_MODELS_overall_kappa_summary.csv", index=False)

print(final_df)