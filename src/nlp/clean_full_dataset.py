import pandas as pd
import os
import re

# --- Paths ---
input_file = "data/processed/extracted_data_v2.csv"
output_file = "data/processed/clean_dataset_full.csv"
os.makedirs("data/processed", exist_ok=True)

# --- Load the NEW extraction (73 rows from 16 papers) ---
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows from {input_file}")
print(f"Unique papers: {df['paper'].nunique()}")
print(f"Unique formulas (raw): {df['catalyst_formula'].nunique()}\n")

# --- Filter 1: Remove obviously corrupt papers ---
# paper17 & paper18 had comparison-condition values, not intrinsic
corrupt = ["paper17.txt", "paper18.txt"]
before = len(df)
df = df[~df["paper"].isin(corrupt)]
print(f"Filter 1 — Removed corrupt papers {corrupt}")
print(f"  Rows removed: {before - len(df)}, remaining: {len(df)}\n")

# --- Filter 2: Remove rows where overpotential and tafel are both NaN ---
before = len(df)
df = df[df["overpotential_mV"].notna() | df["tafel_mV_per_dec"].notna()]
print(f"Filter 2 — Removed rows with no data")
print(f"  Rows removed: {before - len(df)}, remaining: {len(df)}\n")

# --- Filter 3: Remove bug where overpotential == tafel ---
before = len(df)
mask = ~(
    (df["overpotential_mV"].notna())
    & (df["tafel_mV_per_dec"].notna())
    & (df["overpotential_mV"] == df["tafel_mV_per_dec"])
)
df = df[mask]
print(f"Filter 3 — Removed overpotential==tafel bug")
print(f"  Rows removed: {before - len(df)}, remaining: {len(df)}\n")

# --- Filter 4: Overpotential must be 5–500 mV ---
before = len(df)
df = df[df["overpotential_mV"].isna() | df["overpotential_mV"].between(5, 500)]
print(f"Filter 4 — Overpotential range 5–500 mV")
print(f"  Rows removed: {before - len(df)}, remaining: {len(df)}\n")

# --- Filter 5: Tafel must be 20–200 mV/dec ---
before = len(df)
df = df[df["tafel_mV_per_dec"].isna() | df["tafel_mV_per_dec"].between(20, 200)]
print(f"Filter 5 — Tafel range 20–200 mV/dec")
print(f"  Rows removed: {before - len(df)}, remaining: {len(df)}\n")

# --- Filter 6: Formula must have 3+ distinct real metals ---
METALS = set(["Pt", "Mo", "Pd", "Rh", "Ni", "Fe", "Co", "Mn", "Cr", "Cu",
              "Ru", "Ir", "Au", "Ag", "Ti", "V", "Zn", "Sn", "W", "Nb",
              "Ga", "In", "Ta", "Zr", "Hf", "Re", "Os", "Cd", "Al", "Mg"])

def get_metals(formula):
    if not isinstance(formula, str):
        return []
    symbols = re.findall(r"[A-Z][a-z]?", formula)
    return [s for s in symbols if s in METALS]

def count_distinct_metals(formula):
    return len(set(get_metals(formula)))

before = len(df)
df["n_metals"] = df["catalyst_formula"].apply(count_distinct_metals)
df = df[df["n_metals"] >= 3]
print(f"Filter 6 — Formula must have 3+ distinct metals")
print(f"  Rows removed: {before - len(df)}, remaining: {len(df)}\n")

# --- Filter 7: Normalize formula (sort elements alphabetically) ---
def normalize_formula(f):
    return "".join(sorted(set(get_metals(f))))

df["clean_formula"] = df["catalyst_formula"].apply(normalize_formula)

# --- Filter 8: Group by formula (one row per catalyst) ---
before = len(df)
df = df.groupby("clean_formula").agg({
    "overpotential_mV": "mean",
    "tafel_mV_per_dec": "mean",
    "paper": lambda x: "|".join(sorted(set(x))),
    "n_metals": "first",
}).reset_index()

print(f"Filter 8 — Grouped duplicates (one row per catalyst)")
print(f"  Rows before: {before}, after: {len(df)}\n")

# --- Sort by overpotential ---
df = df.sort_values("overpotential_mV", na_position="last")

# --- Save ---
df.to_csv(output_file, index=False)

print("=" * 65)
print("FINAL CLEAN DATASET")
print("=" * 65)
print(f"  Total catalysts:  {len(df)}")
print(f"  Overpotential mean: {df['overpotential_mV'].mean():.2f} mV")
print(f"  Overpotential std:  {df['overpotential_mV'].std():.2f} mV")
print(f"  Overpotential min:  {df['overpotential_mV'].min():.2f} mV")
print(f"  Overpotential max:  {df['overpotential_mV'].max():.2f} mV")
print(f"  Rows with Tafel:   {df['tafel_mV_per_dec'].notna().sum()}")
print(f"\nSaved to: {output_file}")

# --- Show all catalysts ---
print(f"\n--- ALL CLEAN CATALYSTS (sorted by overpotential) ---")
print(df[["clean_formula", "overpotential_mV", "tafel_mV_per_dec", "n_metals"]].to_string(index=False))