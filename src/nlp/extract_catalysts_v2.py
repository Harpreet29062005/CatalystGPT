import spacy
import re
import os
import glob
import pandas as pd

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Paths
raw_text_folder = "data/raw"
output_file = "data/processed/extracted_data_v2.csv"
os.makedirs("data/processed", exist_ok=True)

text_files = glob.glob(os.path.join(raw_text_folder, "*.txt"))
print(f"Found {len(text_files)} text files to process.\n")

# Full list of metal elements used in HEA HER catalysts
METALS = ["Pt", "Mo", "Pd", "Rh", "Ni", "Fe", "Co", "Mn", "Cr", "Cu",
          "Ru", "Ir", "Au", "Ag", "Ti", "V", "Zn", "Sn", "W", "Nb",
          "Ga", "In", "Ta", "Zr", "Hf", "Re", "Os", "Cd", "Al", "Mg"]

# Regex: a full chemical formula with element+optional number, repeated 3+ times
# Examples that should match: PtMoPdRhNi, Pt28Mo6Pd28Rh27Ni15, PdMoGaInNi
FORMULA_PATTERN = re.compile(
    r"\b((?:[A-Z][a-z]?\d*){3,})\b"
)

# Overpotential pattern — capture the number and the current density context
OVERPOTENTIAL_PATTERN = re.compile(r"(\d+\.?\d*)\s*mV", re.IGNORECASE)

# Tafel slope
TAFEL_PATTERN = re.compile(r"(\d+\.?\d*)\s*mV\s*/?\s*dec", re.IGNORECASE)

# Sentences to EXCLUDE (they refer to full-cell, not HER half-cell)
EXCLUDE_TERMS = [
    "full cell", "water splitting cell", "cell voltage",
    "1.6 v", "1.5 v", "1.7 v", "1.8 v",
    "100 ma", "500 ma", "1000 ma",
    "overall water splitting", "two-electrode",
]

def clean_sentence(s):
    """Remove newlines and extra spaces from a sentence."""
    s = s.replace("\\n", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def is_real_hea(formula):
    """
    Check that the formula contains at least 3 distinct metal elements.
    Rejects things like 'V' or 'NiV'.
    """
    # Extract element symbols from the formula
    found = re.findall(r"[A-Z][a-z]?", formula)
    # Keep only real metals
    real_metals = [e for e in found if e in METALS]
    # Must have at least 3 distinct metals for a HEA
    return len(set(real_metals)) >= 3

# Storage
results = []

for text_file in text_files:
    filename = os.path.basename(text_file)
    print(f"Processing: {filename}")

    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    doc = nlp(text[:150000])

    for sent in doc.sents:
        sent_text = clean_sentence(sent.text)
        lower = sent_text.lower()

        # Must mention overpotential or tafel
        if "overpotential" not in lower and "tafel" not in lower:
            continue

        # Skip sentences about full cells
        if any(term in lower for term in EXCLUDE_TERMS):
            continue

        # Extract overpotential and Tafel
        op_match = OVERPOTENTIAL_PATTERN.search(sent_text)
        tafel_match = TAFEL_PATTERN.search(sent_text)
        overpotential = float(op_match.group(1)) if op_match else None
        tafel = float(tafel_match.group(1)) if tafel_match else None

        # Find candidate formulas in this sentence
        formulas = FORMULA_PATTERN.findall(sent_text)
        valid_formulas = [f for f in formulas if is_real_hea(f)]

        if not valid_formulas:
            continue

        for formula in valid_formulas:
            results.append({
                "paper": filename,
                "catalyst_formula": formula,
                "overpotential_mV": overpotential,
                "tafel_mV_per_dec": tafel,
                "sentence": sent_text[:350],
            })

    print(f"  -> Running total: {len(results)} entries")

# Convert to DataFrame
df = pd.DataFrame(results)

# Deduplicate: keep one row per (paper, formula, overpotential, tafel)
if not df.empty:
    df = df.drop_duplicates(
        subset=["paper", "catalyst_formula", "overpotential_mV", "tafel_mV_per_dec"]
    )

# Sort by paper then overpotential
df = df.sort_values(by=["paper", "overpotential_mV"], na_position="last")

# Save
df.to_csv(output_file, index=False)

print(f"\nExtraction v2 complete!")
print(f"Total unique entries: {len(df)}")
print(f"Saved to: {output_file}")
print(f"\nFull results:")
print(df.to_string())

# Quick summary
print(f"\n--- SUMMARY ---")
print(f"Papers processed: {df['paper'].nunique() if not df.empty else 0}")
print(f"Unique catalysts: {df['catalyst_formula'].nunique() if not df.empty else 0}")
print(f"Rows with overpotential: {df['overpotential_mV'].notna().sum() if not df.empty else 0}")
print(f"Rows with Tafel slope: {df['tafel_mV_per_dec'].notna().sum() if not df.empty else 0}")