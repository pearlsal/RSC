import pandas as pd
from pathlib import Path

BASE = Path("/Users/pearls/Library/Mobile Documents/com~apple~CloudDocs/work/csv")  # <-- change this

files = [
    ("EBC_ToothMuch_classification_summary.csv", "ToothMuch", "EBC"),
    ("EBC_PreciousGrape_classification_summary.csv", "PreciousGrape", "EBC"),
    ("EBC_Mimosapudica_classification_summary.csv", "MimosaPudica", "EBC"),

    ("ETC_toothmuch_classification_summary.csv", "ToothMuch", "ETC"),
    ("ETC_PreciousGrape_classification_summary.csv", "PreciousGrape", "ETC"),
    ("ETC_MimosaPudica_classification_summary.csv", "MimosaPudica", "ETC"),

    ("EOC_ToothMuch_classification_summary.csv", "ToothMuch", "EOC"),
    ("EOC_PreciousGrape_cclassification_summary.csv", "PreciousGrape", "EOC"),
    ("EOC_MimosaPudica_classification_summary.csv", "MimosaPudica", "EOC"),
]

dfs = []
for fname, animal, mode in files:
    df = pd.read_csv(BASE / fname)
    df["animal"] = animal
    df["mode"] = mode
    dfs.append(df)

long_df = pd.concat(dfs, ignore_index=True)

# --- WIDE with ALL metrics per mode ---
metric_cols = [c for c in long_df.columns if c not in ["animal", "mode", "cell_name"]]

wide = long_df.pivot_table(
    index=["animal", "cell_name"],
    columns="mode",
    values=metric_cols,
    aggfunc="first"
)

# flatten columns: firing_rate_hz_EBC, cc_correlation_ETC, etc.
wide.columns = [f"{col}_{mode}" for col, mode in wide.columns]
wide = wide.reset_index()

# --- STRICT flags: only count exact "EBC"/"ETC"/"EOC" ---
wide["is_EBC_strict"] = wide["classification_EBC"].eq("EBC")
wide["is_ETC_strict"] = wide["classification_ETC"].eq("ETC")
wide["is_EOC_strict"] = wide["classification_EOC"].eq("EOC")
wide["is_neither_strict"] = ~(wide["is_EBC_strict"] | wide["is_ETC_strict"] | wide["is_EOC_strict"])

wide.to_csv("EBC_ETC_EOC_all_animals_WIDE_with_metrics_STRICT.csv", index=False)

# counts summary
summary = (
    wide.groupby("animal")
    .agg(
        total_cells=("cell_name", "count"),
        total_EBC=("is_EBC_strict", "sum"),
        total_ETC=("is_ETC_strict", "sum"),
        total_EOC=("is_EOC_strict", "sum"),
        total_neither=("is_neither_strict", "sum"),
    )
    .reset_index()
)
summary.to_csv("EBC_ETC_EOC_counts_summary_STRICT.csv", index=False)

print("\n=== TOTALS (STRICT) ===")
print("Total cells:", len(wide))
print("EBC:", wide["is_EBC_strict"].sum())
print("ETC:", wide["is_ETC_strict"].sum())
print("EOC:", wide["is_EOC_strict"].sum())
print("Neither:", wide["is_neither_strict"].sum())
