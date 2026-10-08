import pandas as pd
import re
import os

# --- Paths ---
input_file = "data/processed/extracted_data_v3.csv"
output_file = "data/processed/clean_dataset_v3.csv"
os.makedirs("data/processed", exist_ok=True)

# --- Load ---
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows from {input_file}")
print(f"Unique papers: {df['paper'].nunique()}")
print(f"Unique formulas (raw): {df['catalyst_formula'].nunique()}\n")

# --- Elements ---
METALS = set(["Pt", "Mo", "Pd", "Rh", "Ni", "Fe", "Co", "Mn", "Cr", "Cu",
              "Ru", "Ir", "Au", "Ag", "Ti", "V", "Zn", "Sn", "W", "Nb",
              "Ga", "In", "Ta", "Zr", "Hf", "Re", "Os", "Cd", "Al", "Mg",
              "Sc", "Y", "La", "Ce"])

def parse_metals(formula):
    """Extract metal symbols, ignoring stoichiometry numbers."""
    if not isinstance(formula, str):
        return []
    symbols = re.findall(r"[A-Z][a-z]?", formula)
    return [s for s in symbols if s in METALS]

def n_distinct_metals(formula):
    return len(set(parse_metals(formula)))

def normalize_formula(formula):
    """Return sorted unique metals without numbers."""
    return "".join(sorted(set(parse_metals(formula))))

# --- Filter 1: Remove corrupt papers ---
# paper17 and paper18 had comparison-condition values (from earlier analysis)
# Note: filename is now .pdf, not .txt
corrupt = ["paper17.pdf", "paper18.pdf"]
before = len(df)
df = df[~df["paper"].isin(corrupt)]
print(f"Filter 1 — Removed corrupt papers")
print(f"  Rows removed: {before - len(df)}, remaining: {len(df)}\n")

# --- Filter 2: Remove rows with no overpotential AND no Tafel ---
before = len(df)
df = df[df["overpotential_mV"].notna() | df["tafel_mV_per_dec"].notna()]
print(f"Filter 2 — Remove empty rows")
print(f"  Rows removed: {before - len(df)}, remaining: {len(df)}\n")

# --- Filter 3: Overpotential must be 5–300 mV ---
# HER catalysts typically have overpotential 5-300 mV. Above 300 is likely OER or wrong.
before = len(df)
df = df[df["overpotential_mV"].isna() | df["overpotential_mV"].between(5, 300)]
print(f"Filter 3 — Overpotential range 5–300 mV")
print(f"  Rows removed: {before - len(df)}, remaining: {len(df)}\n")

# --- Filter 4: Tafel must be 20–200 mV/dec ---
before = len(df)
df = df[df["tafel_mV_per_dec"].isna() | df["tafel_mV_per_dec"].between(20, 200)]
print(f"Filter 4 — Tafel range 20–200 mV/dec")
print(f"  Rows removed: {before - len(df)}, remaining: {len(df)}\n")

# --- Filter 5: Remove bug where overpotential == tafel ---
before = len(df)
mask = ~(
    df["overpotential_mV"].notna()
    & df["tafel_mV_per_dec"].notna()
    & (df["overpotential_mV"] == df["tafel_mV_per_dec"])
)
df = df[mask]
print(f"Filter 5 — Remove overpotential==tafel bug")
print(f"  Rows removed: {before - len(df)}, remaining: {len(df)}\n")

# --- Filter 6: Formula must have 3+ distinct metals ---
before = len(df)
df["n_metals"] = df["catalyst_formula"].apply(n_distinct_metals)
df = df[df["n_metals"] >= 3]
print(f"Filter 6 — Formula must have 3+ distinct metals")
print(f"  Rows removed: {before - len(df)}, remaining: {len(df)}\n")

# --- Filter 7: Normalize formula ---
df["clean_formula"] = df["catalyst_formula"].apply(normalize_formula)

# --- Filter 8: Group duplicates (one row per catalyst) ---
before = len(df)
df = df.groupby("clean_formula").agg({
    "overpotential_mV": "mean",
    "tafel_mV_per_dec": "mean",
    "paper": lambda x: "|".join(sorted(set(x))),
    "n_metals": "first",
    "source": lambda x: "|".join(sorted(set(x))),
}).reset_index()

print(f"Filter 8 — Grouped duplicates")
print(f"  Rows before: {before}, after: {len(df)}\n")

# --- Sort ---
df = df.sort_values("overpotential_mV", na_position="last")

# --- Save ---
df.to_csv(output_file, index=False)

print("=" * 65)
print("FINAL CLEAN DATASET (V3)")
print("=" * 65)
print(f"  Total catalysts:    {len(df)}")
print(f"  Overpotential mean: {df['overpotential_mV'].mean():.2f} mV")
print(f"  Overpotential std:  {df['overpotential_mV'].std():.2f} mV")
print(f"  Overpotential min:  {df['overpotential_mV'].min():.2f} mV")
print(f"  Overpotential max:  {df['overpotential_mV'].max():.2f} mV")
print(f"  Rows with Tafel:    {df['tafel_mV_per_dec'].notna().sum()}")
print(f"\nSaved to: {output_file}")

# --- Show all catalysts ---
print(f"\n--- ALL CLEAN CATALYSTS ---")
print(df[["clean_formula", "overpotential_mV", "tafel_mV_per_dec", "n_metals", "source"]].to_string(index=False))