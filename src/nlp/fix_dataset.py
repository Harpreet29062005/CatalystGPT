import pandas as pd
import os

# --- Paths ---
input_file = "data/processed/clean_dataset.csv"
output_file = "data/processed/clean_dataset_v2.csv"
os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv(input_file)
print(f"Original rows: {len(df)}")
print(f"Original unique formulas: {df['clean_formula'].nunique()}\n")

# ============================================================
# FIX 1: Remove corrupt papers (comparison/stability conditions)
# ============================================================
# paper17: values are comparison conditions ("at 240 mV for X, Y, Z")
# paper18: values are stability test conditions, not intrinsic
corrupt_papers = ["paper17.txt", "paper18.txt"]

before = len(df)
df = df[~df["paper"].isin(corrupt_papers)]
print(f"FIX 1 — Removed corrupt papers {corrupt_papers}")
print(f"  Rows removed: {before - len(df)}")
print(f"  Rows remaining: {len(df)}\n")

# ============================================================
# FIX 2: Group duplicates -> one row per unique formula
# ============================================================
# Take MEAN overpotential for each unique formula.
# This eliminates data leakage in cross-validation.
before = len(df)
grouped = df.groupby("clean_formula").agg({
    "overpotential_mV": "mean",
    "tafel_mV_per_dec": "mean",       # NaN if all missing
    "paper": lambda x: "|".join(sorted(set(x))),
    "n_metals": "first",
}).reset_index()

print(f"FIX 2 — Grouped by formula (1 row per catalyst)")
print(f"  Rows before: {before}")
print(f"  Rows after:  {len(grouped)}\n")

# ============================================================
# FIX 3: Recompute clean stats
# ============================================================
print(f"FINAL DATASET STATS")
print(f"  Total catalysts:  {len(grouped)}")
print(f"  Overpotential mean: {grouped['overpotential_mV'].mean():.2f} mV")
print(f"  Overpotential std:  {grouped['overpotential_mV'].std():.2f} mV")
print(f"  Overpotential min:  {grouped['overpotential_mV'].min():.2f} mV")
print(f"  Overpotential max:  {grouped['overpotential_mV'].max():.2f} mV")

# Save
grouped.to_csv(output_file, index=False)
print(f"\nSaved to: {output_file}")

# Show the new top 15
print(f"\n--- TOP 15 CATALYSTS (Lowest Average Overpotential) ---")
top = grouped.nsmallest(15, "overpotential_mV")
print(top[["clean_formula", "overpotential_mV", "tafel_mV_per_dec", "n_metals"]].to_string(index=False))

# Check for remaining duplicates
dups = grouped["clean_formula"].value_counts()
dups = dups[dups > 1]
if len(dups) > 0:
    print(f"\n⚠️ Still have duplicates: {len(dups)}")
else:
    print(f"\n✅ No duplicates remaining")