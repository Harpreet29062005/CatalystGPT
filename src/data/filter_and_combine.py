"""
Aggressive filtering of NLP-extracted catalysts before combining with Gorsse.

Filters applied to custom NLP data:
- Keep only overpotential in [5, 120] mV (typical HER range)
- Remove suspicious entries where the same formula appears 3+ times with very different values
- Keep all Gorsse data as-is (curated, trustworthy)

Output: data/processed/combined_dataset_filtered.csv
"""

import pandas as pd
import os
import re

os.makedirs("data/processed", exist_ok=True)

# ============================================================
# LOAD GORSSE
# ============================================================
gorsse = pd.read_csv("data/processed/gorsse_180_catalysts.csv")
gorsse_std = pd.DataFrame({
    "formula": gorsse["Formula"].astype(str).str.strip(),
    "target_mV": gorsse["onset_potential"],
    "source": "gorsse",
})
print(f"Gorsse: {len(gorsse_std)} rows")


# ============================================================
# LOAD CUSTOM NLP
# ============================================================
custom = pd.read_csv("data/processed/clean_dataset_v3.csv")
print(f"Custom NLP raw: {len(custom)} rows")

custom_std = pd.DataFrame({
    "formula": custom["clean_formula"].astype(str).str.strip(),
    "target_mV": custom["overpotential_mV"],
    "source": "custom_nlp",
})

# --- Filter 1: Keep only overpotential in [5, 120] mV ---
before = len(custom_std)
custom_std = custom_std[(custom_std["target_mV"] >= 5) & (custom_std["target_mV"] <= 120)]
print(f"  After [5, 120] mV filter: {len(custom_std)} (removed {before - len(custom_std)})")

# --- Filter 2: Remove very short formulas (< 3 metals) ---
METALS = {"Pt", "Mo", "Pd", "Rh", "Ni", "Fe", "Co", "Mn", "Cr", "Cu",
          "Ru", "Ir", "Au", "Ag", "Ti", "V", "Zn", "Sn", "W", "Nb",
          "Ga", "In", "Ta", "Zr", "Hf", "Re", "Os", "Cd", "Al", "Mg"}

def count_metals(f):
    if not isinstance(f, str):
        return 0
    return len(set(s for s in re.findall(r"[A-Z][a-z]?", f) if s in METALS))

custom_std["n_metals"] = custom_std["formula"].apply(count_metals)
before = len(custom_std)
custom_std = custom_std[custom_std["n_metals"] >= 3]
print(f"  After 3+ metals filter: {len(custom_std)} (removed {before - len(custom_std)})")

# --- Filter 3: Remove entries where same formula has std dev > 40 mV ---
grouped = custom_std.groupby("formula")["target_mV"].agg(["mean", "std", "count"])
suspicious = grouped[(grouped["std"] > 40) & (grouped["count"] >= 2)].index.tolist()
before = len(custom_std)
custom_std = custom_std[~custom_std["formula"].isin(suspicious)]
print(f"  After removing high-variance formulas: {len(custom_std)} (removed {before - len(custom_std)})")

# --- Group duplicates: take mean overpotential per formula ---
custom_clean = custom_std.groupby("formula").agg({
    "target_mV": "mean",
    "source": "first",
}).reset_index()
print(f"  After grouping duplicates: {len(custom_clean)} unique custom catalysts\n")


# ============================================================
# COMBINE
# ============================================================
combined = pd.concat([gorsse_std, custom_clean], ignore_index=True)

# Clean target values
combined = combined.dropna(subset=["target_mV", "formula"])
combined = combined[(combined["target_mV"] >= 5) & (combined["target_mV"] <= 400)]

# 3+ metals
combined["n_metals"] = combined["formula"].apply(count_metals)
combined = combined[combined["n_metals"] >= 3]

# Deduplicate — prefer Gorsse for overlaps
combined = combined.sort_values("source", key=lambda s: s.map({"gorsse": 0, "custom_nlp": 1}))
before = len(combined)
combined = combined.drop_duplicates(subset=["formula"], keep="first")
removed = before - len(combined)

combined = combined.sort_values("target_mV").reset_index(drop=True)
combined = combined[["formula", "target_mV", "source", "n_metals"]]

# ============================================================
# SAVE
# ============================================================
output_file = "data/processed/combined_dataset_filtered.csv"
combined.to_csv(output_file, index=False)

print(f"{'='*60}")
print(f"FILTERED COMBINED DATASET")
print(f"{'='*60}")
print(f"  Total:           {len(combined)}")
print(f"  From Gorsse:     {(combined['source'] == 'gorsse').sum()}")
print(f"  From Custom NLP: {(combined['source'] == 'custom_nlp').sum()}")
print(f"  Duplicates removed: {removed}")
print(f"\n  Target range: {combined['target_mV'].min():.2f} – {combined['target_mV'].max():.2f} mV")
print(f"  Target mean:  {combined['target_mV'].mean():.2f} mV")
print(f"  Target std:   {combined['target_mV'].std():.2f} mV")
print(f"\nSaved to: {output_file}")

print(f"\n--- Custom NLP catalysts kept ({len(custom_clean)}) ---")
print(custom_clean.sort_values("target_mV").head(20).to_string(index=False))