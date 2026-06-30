from pathlib import Path
import pandas as pd

#LOAD FILES
files = list(Path("output/judge_results").glob("*_judged.csv"))
summary = []

#LOOP THROUGH FILE AND COMPUTE MEAN
for file in files:
    df = pd.read_csv(file)
    model_name = file.stem.replace("_judged", "")

    summary.append({
        "model_name": model_name,
        "mean_correctness": df["correctness"].mean(),
        "mean_consistency": df["consistency"].mean(),
        "mean_completeness": df["completeness"].mean(),
        "mean_clarity": df["clarity"].mean(),
        "n_rows": len(df),
    })

#SAVE
out_path = Path("output/judge_results_FINAL/model_summary.csv")
print("Saving to:", out_path.resolve())
print("Parent exists:", out_path.parent.exists())

pd.DataFrame(summary).to_csv(out_path, index=False)
print("Saved:", out_path.exists())
print((out_path).read_text())
print((Path("output/judge_results/model_summary.csv").resolve()))