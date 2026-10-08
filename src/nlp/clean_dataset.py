import pandas as pd
import re
import os

# Paths
input_file = "data/processed/extracted_data_v2.csv"
output_file = "data/processed/clean_dataset.csv"
os.makedirs("data/processed", exist_ok=True)

# Load
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows from {input_file}")

# Known metals for HEA HER catalysts
METALS = set(["Pt", "Mo", "Pd", "Rh", "Ni", "Fe", "Co", "Mn", "Cr", "Cu",
              "Ru", "Ir", "Au", "Ag", "Ti", "V", "Zn", "Sn", "W", "Nb",
              "Ga", "In", "Ta", "Zr", "Hf", "Re", "Os", "Cd", "Al", "Mg",
              "Sc", "Y", "La", "Ce", "Nd", "Sm", "Gd", "Er"])

def get_metals(formula):
    """Extract metal symbols from a formula string."""
    if not isinstance(formula, str):
        return []
    symbols = re.findall(r"[A-Z][a-z]?", formula)
    return [s for s in symbols if s in METALS]

def count_distinct_metals(formula):
    return len(set(get_metals(formula)))

def is_valid_formula(formula):
    """Formula must have 3+ distinct metals."""
    return count_distinct_metals(formula) >= 3

def is_valid_overpotential(x):
    """Overpotential must be between 5 and 500 mV."""
    if pd.isna(x):
        return False
    return 5 <= x <= 500

def is_valid_tafel(x):
    """Tafel slope must be between 20 and 200 mV/dec."""
    if pd.isna(x):
        return False
    return 20 <= x <= 200

# --- Apply filters ---
print(f"\nApplying filters...")

# 1. Valid formula
df["n_metals"] = df["catalyst_formula"].apply(count_distinct_metals)
df = df[df["n_metals"] >= 3]
print(f"After formula filter (3+ metals): {len(df)} rows")

# 2. Remove rows where BOTH overpotential and tafel are missing
df = df[df["overpotential_mV"].notna() | df["tafel_mV_per_dec"].notna()]
print(f"After removing empty rows: {len(df)} rows")

# 3. Remove where overpotential == tafel (extraction bug)
mask = ~((df["overpotential_mV"].notna()) & 
         (df["tafel_mV_per_dec"].notna()) & 
         (df["overpotential_mV"] == df["tafel_mV_per_dec"]))
df = df[mask]
print(f"After removing o.p.==tafel bug: {len(df)} rows")

# 4. Valid overpotential range
df = df[df["overpotential_mV"].isna() | df["overpotential_mV"].apply(is_valid_overpotential)]
print(f"After overpotential range filter: {len(df)} rows")

# 5. Valid tafel range
df = df[df["tafel_mV_per_dec"].isna() | df["tafel_mV_per_dec"].apply(is_valid_tafel)]
print(f"After tafel range filter: {len(df)} rows")

# 6. Deduplicate
df = df.drop_duplicates(subset=["paper", "catalyst_formula", "overpotential_mV", "tafel_mV_per_dec"])
print(f"After deduplication: {len(df)} rows")

# 7. Normalize formula (sort elements alphabetically)
def normalize_formula(f):
    metals = sorted(set(get_metals(f)))
    return "".join(metals)

df["clean_formula"] = df["catalyst_formula"].apply(normalize_formula)

# Final sort
df = df.sort_values(by=["paper", "overpotential_mV"], na_position="last")

# Save
df.to_csv(output_file, index=False)
print(f"\n✅ Clean dataset saved to: {output_file}")
print(f"✅ Total rows: {len(df)}")
print(f"✅ Unique papers: {df['paper'].nunique()}")
print(f"✅ Unique catalysts: {df['clean_formula'].nunique()}")
print(f"✅ Rows with overpotential: {df['overpotential_mV'].notna().sum()}")
print(f"✅ Rows with Tafel slope: {df['tafel_mV_per_dec'].notna().sum()}")

# Show the top catalysts (lowest overpotential)
print(f"\n--- TOP 15 CATALYSTS (Lowest Overpotential) ---")
top = df.dropna(subset=["overpotential_mV"]).nsmallest(15, "overpotential_mV")
print(top[["clean_formula", "overpotential_mV", "tafel_mV_per_dec", "paper"]].to_string(index=False))