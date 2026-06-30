import pandas as pd
import random

#LOAD DATA
df_primary = pd.read_csv("data/raw_data/.sen_age_sex_.csv.icloud")
df_secondary = pd.read_csv("0.data/raw_data/.sen_secondary_need_.csv.icloud")

#REMOVE "Total" ROWS
df_primary = df_primary[
    ~df_primary["sen_primary_need"].isin(["Total", "No identified SEN"])
]

df_primary = df_primary[
    ~df_primary["phase_type_grouping"].isin(["Total"])
]

df_secondary = df_secondary[
    ~df_secondary["primary_need"].isin(["Total", "No identified SEN"])
]

df_secondary = df_secondary[
    ~df_secondary["phase_type_grouping"].isin(["Total"])
]


#PRECOMPUTE AGE
age_cols = [c for c in df_primary.columns if c.startswith("age_") and c.endswith("_percent")]
age_labels = [c.replace("_percent", "") for c in age_cols]


def generate_person(df_primary, df_secondary):

    #Pick random sample
    row = df_primary.sample(1).iloc[0]

    sen = row["sen_primary_need"]
    phase = row["phase_type_grouping"]
    time_period = row["time_period"]

    #Gender
    gender = random.choices(
        ["female", "male"],
        weights=[
            row["pupil_sex_female_percent"],
            row["pupil_sex_male_percent"]
        ]
    )[0]

    #Age
    age_weights = [row[c] for c in age_cols]

    age_group = random.choices(age_labels, weights=age_weights)[0]

    #Secondary need
    sec_sen = None

    subset = df_secondary[
        (df_secondary["time_period"] == time_period) &
        (df_secondary["phase_type_grouping"] == phase) &
        (df_secondary["primary_need"] == sen)
    ]

    if len(subset) > 0:

        row_sec = subset.iloc[0]

        sec_cols = [
            c for c in subset.columns
            if c.startswith("secondary_need") and c.endswith("_percent")
        ]

        sec_labels = [c.replace("secondary_need_", "").replace("_percent", "") for c in sec_cols]

        sec_weights = [row_sec[c] for c in sec_cols]

        # CHECK
        if sum(sec_weights) > 0:
            sec_sen = random.choices(sec_labels, weights=sec_weights)[0]

    #RETURN PERSONA
    df_primary.to_csv("cleaned_primary.csv", index=False)
    df_secondary.to_csv("cleaned_secondary.csv", index=False)

    return {
        "sen_primary_need": sen,
        "phase_type_grouping": phase,
        "time_period": time_period,
        "gender": gender,
        "age_group": age_group,
        "secondary_need": sec_sen
    }

generate_person(df_primary, df_secondary)
#GENERATE DATASET
"""synthetic_data = [
    generate_person(df_primary, df_secondary)
    for _ in range(500)
]

synthetic_df = pd.DataFrame(synthetic_data)

synthetic_df.to_csv("synthetic_personas.csv", index=False)

print(synthetic_df.head())"""