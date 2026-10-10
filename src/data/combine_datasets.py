"""
Combine Gorsse (180) and Custom NLP (54) datasets into a unified dataset.

Both targets represent HER activity but differ slightly:
- Gorsse: onset_potential (where HER begins)
- Custom: overpotential_mV (at 10 mA/cm2)

We combine them because their distributions are similar (mean ~90 mV, std ~90 mV),
documenting this as a limitation.

Output: data/processed/combined_dataset.csv
  Columns: formula, target_mV, source, original_target_name
"""

import pandas as pd
import os
import re

os.makedirs("data/processed", exist_ok=True)

# ============================================================
# LOAD GORSSE DATASET
# ============================================================
gorsse_file = "data/processed/gorsse_180_catalysts.csv"
gorsse = pd.read_csv(gorsse_file)

print(f"Gorsse dataset loaded: {len(gorsse)} rows")
print(f"Columns: {list(gorsse.columns)[:5]}...\n")

gorsse_std = pd.DataFrame({
    "formula": gorsse["Formula"].astype(str).str.strip(),
    "target_mV": gorsse["onset_potential"],
    "source": "gorsse",
    "original_target_name": "onset_potential",
})

# ============================================================
# LOAD CUSTOM NLP DATASET
# ============================================================
custom_file = "data/processed/clean_dataset_v3.csv"
custom = pd.read_csv(custom_file)

print(f"Custom NLP dataset loaded: {len(custom)} rows")
print(f"Columns: {list(custom.columns)[:5]}...\n")

custom_std = pd.DataFrame({
    "formula": custom["clean_formula"].astype(str).str.strip(),
    "target_mV": custom["overpotential_mV"],
    "source": "custom_nlp",
    "original_target_name": "overpotential_mV",
})

# ============================================================
# COMBINE
# ============================================================
combined = pd.concat([gorsse_std, custom_std], ignore_index=True)

print(f"Combined (before cleaning): {len(combined)} rows")

# Drop rows with missing targets or formulas
before = len(combined)
combined = combined.dropna(subset=["target_mV", "formula"])
print(f"After removing missing targets: {len(combined)} (removed {before - len(combined)})")

# Filter to reasonable range: 5–400 mV
before = len(combined)
combined = combined[(combined["target_mV"] >= 5) & (combined["target_mV"] <= 400)]
print(f"After range filter (5-400 mV): {len(combined)} (removed {before - len(combined)})")

# Filter to 3+ distinct metals
METALS = {"Pt", "Mo", "Pd", "Rh", "Ni", "Fe", "Co", "Mn", "Cr", "Cu",
          "Ru", "Ir", "Au", "Ag", "Ti", "V", "Zn", "Sn", "W", "Nb",
          "Ga", "In", "Ta", "Zr", "Hf", "Re", "Os", "Cd", "Al", "Mg"}

def count_metals(formula):
    if not isinstance(formula, str):
        return 0
    symbols = re.findall(r"[A-Z][a-z]?", formula)
    return len(set(s for s in symbols if s in METALS))

combined["n_metals"] = combined["formula"].apply(count_metals)

before = len(combined)
combined = combined[combined["n_metals"] >= 3]
print(f"After 3+ metals filter: {len(combined)} (removed {before - len(combined)})")

# ============================================================
# DEDUPLICATE
# ============================================================
# If the same formula appears in both datasets, keep the Gorsse one
# (curated dataset takes priority)
before = len(combined)
combined = combined.sort_values("source", key=lambda s: s.map({"gorsse": 0, "custom_nlp": 1}))
combined = combined.drop_duplicates(subset=["formula"], keep="first")
print(f"After deduplication: {len(combined)} (removed {before - len(combined)})")

# ============================================================
# SAVE
# ============================================================
combined = combined.sort_values("target_mV").reset_index(drop=True)

output_file = "data/processed/combined_dataset.csv"
combined.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"FINAL COMBINED DATASET")
print(f"{'='*60}")
print(f"  Total catalysts: {len(combined)}")
print(f"  From Gorsse:     {(combined['source'] == 'gorsse').sum()}")
print(f"  From Custom NLP: {(combined['source'] == 'custom_nlp').sum()}")
print(f"\n  Target range:  {combined['target_mV'].min():.2f} – {combined['target_mV'].max():.2f} mV")
print(f"  Target mean:   {combined['target_mV'].mean():.2f} mV")
print(f"  Target std:    {combined['target_mV'].std():.2f} mV")
print(f"\nSaved to: {output_file}")

# Show top 10 lowest overpotential
print(f"\n--- Top 10 Best Catalysts (lowest target) ---")
top10 = combined.nsmallest(10, "target_mV")
print(top10[["formula", "target_mV", "source"]].to_string(index=False))

# Show source distribution by target bins
print(f"\n--- Distribution by source ---")
for src in ["gorsse", "custom_nlp"]:
    sub = combined[combined["source"] == src]
    print(f"  {src}: n={len(sub)}, mean={sub['target_mV'].mean():.1f} mV, std={sub['target_mV'].std():.1f} mV")