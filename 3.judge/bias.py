from pathlib import Path
import pandas as pd
import numpy as np

#LOAD FILES
files = list(Path("output/evaluators_results").glob("*_results.csv"))
gpt5 = pd.read_csv("output/proxy_results/gpt52_results.csv")

#BIAS & STD PER C1-C17 
for file in files:
    df = pd.read_csv(file)
    model_name = file.stem.replace("_results", "")

    summary = {"model_name": model_name}

    for i in range(1, 18):
        c = f"c{i}"
        diff = df[c] - gpt5[c]

        summary[f"{c}_mean_bias"] = np.mean(diff)
        summary[f"{c}_std_bias"] = np.std(diff, ddof=1)   

    out_path = Path(f"output/bias/{model_name}_bias.csv")
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    
bias = list(Path("output/bias").glob("*_bias.csv"))

#OVERALL (MEAN) BIAS 
final_rows = []

for file in bias:
    df = pd.read_csv(file)

    model_name = df["model_name"].iloc[0]

    mean_cols = [c for c in df.columns if "mean_bias" in c]

    overall_bias = df[mean_cols].mean(axis=1).iloc[0]

    final_rows.append({
        "model_name": model_name,
        "overall_mean_bias": overall_bias
    })

final_df = pd.DataFrame(final_rows)

final_df.to_csv("output/bias/ALL_MODELS_summary.csv", index=False)

print(final_df)