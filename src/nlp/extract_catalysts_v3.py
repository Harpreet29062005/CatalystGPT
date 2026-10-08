import pdfplumber
import pandas as pd
import re
import os
import glob

# --- Paths ---
pdf_folder = "data/papers"
output_file = "data/processed/extracted_data_v3.csv"
os.makedirs("data/processed", exist_ok=True)

# --- Elements for HEA catalysts ---
METALS = set(["Pt", "Mo", "Pd", "Rh", "Ni", "Fe", "Co", "Mn", "Cr", "Cu",
              "Ru", "Ir", "Au", "Ag", "Ti", "V", "Zn", "Sn", "W", "Nb",
              "Ga", "In", "Ta", "Zr", "Hf", "Re", "Os", "Cd", "Al", "Mg",
              "Sc", "Y", "La", "Ce"])

# --- Regex patterns ---
# Formula: 3+ element symbols (with optional numbers)
FORMULA_PATTERN = re.compile(r"\b((?:[A-Z][a-z]?\d*){3,})\b")

# Number + unit patterns
OP_PATTERNS = [
    re.compile(r"(\d+\.?\d*)\s*mV", re.IGNORECASE),
    re.compile(r"η[\s_]*[=:]?\s*(\d+\.?\d*)", re.IGNORECASE),
    re.compile(r"overpotential[^\d]{0,20}(\d+\.?\d*)", re.IGNORECASE),
]

TAFEL_PATTERNS = [
    re.compile(r"(\d+\.?\d*)\s*mV\s*/?\s*dec", re.IGNORECASE),
    re.compile(r"(\d+\.?\d*)\s*mV\s*dec", re.IGNORECASE),
    re.compile(r"Tafel\s*slope[^\d]{0,20}(\d+\.?\d*)", re.IGNORECASE),
]

# Elements in a formula
ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")

def extract_formula(s):
    """Try to pull a formula from a string."""
    if not s:
        return None
    s = str(s).strip()
    # Look for formula pattern
    matches = FORMULA_PATTERN.findall(s)
    best = None
    for m in matches:
        # Count distinct metals
        elems = ELEMENT_RE.findall(m)
        metals = [e for e in elems if e in METALS]
        if len(set(metals)) >= 3:
            # Prefer the longest (most likely HEA)
            if best is None or len(m) > len(best):
                best = m
    return best

def extract_number(text, patterns):
    """Try each pattern, return the first numeric match."""
    if not text:
        return None
    for pat in patterns:
        m = pat.search(str(text))
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    return None

# --- Process each PDF ---
pdf_files = sorted(glob.glob(os.path.join(pdf_folder, "*.pdf")))
print(f"Found {len(pdf_files)} PDFs\n")

results = []

for pdf_path in pdf_files:
    filename = os.path.basename(pdf_path)
    print(f"Processing: {filename}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Limit to first 30 pages (data is usually near the top)
            for page_num, page in enumerate(pdf.pages[:30]):
                # --- Process TABLES ---
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []

                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    # Get header row
                    header = table[0] if table[0] else []
                    header_joined = " ".join([str(h) if h else "" for h in header]).lower()

                    # Only process tables that look like catalyst tables
                    has_formula_col = False
                    has_op_col = False
                    has_tafel_col = False

                    for ci, h in enumerate(header):
                        h_lower = str(h).lower() if h else ""
                        if "catalyst" in h_lower or "alloy" in h_lower or "sample" in h_lower or "electrode" in h_lower:
                            has_formula_col = True
                        if "overpotential" in h_lower or "η" in h_lower:
                            has_op_col = True
                        if "tafel" in h_lower:
                            has_tafel_col = True

                    # Process each data row
                    for row in table[1:]:
                        if not row:
                            continue

                        formula = None
                        op_val = None
                        tafel_val = None

                        for cell in row:
                            cell_str = str(cell) if cell else ""

                            # Try formula
                            f = extract_formula(cell_str)
                            if f and not formula:
                                formula = f

                            # Try overpotential
                            if op_val is None:
                                n = extract_number(cell_str, OP_PATTERNS)
                                if n and 5 <= n <= 500:
                                    op_val = n

                            # Try tafel
                            if tafel_val is None:
                                n = extract_number(cell_str, TAFEL_PATTERNS)
                                if n and 20 <= n <= 200:
                                    tafel_val = n

                        if formula and (op_val or tafel_val):
                            results.append({
                                "paper": filename,
                                "page": page_num + 1,
                                "source": "table",
                                "catalyst_formula": formula,
                                "overpotential_mV": op_val,
                                "tafel_mV_per_dec": tafel_val,
                                "context": str(row)[:200],
                            })

                # --- Process TEXT (sentences) ---
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                # Split into sentences
                sentences = re.split(r"[.!?]\s", text)

                for sent in sentences:
                    lower = sent.lower()
                    if "overpotential" not in lower and "tafel" not in lower and "η" not in lower:
                        continue

                    formula = extract_formula(sent)
                    if not formula:
                        continue

                    op_val = extract_number(sent, OP_PATTERNS)
                    tafel_val = extract_number(sent, TAFEL_PATTERNS)

                    # Validate ranges
                    if op_val and not (5 <= op_val <= 500):
                        op_val = None
                    if tafel_val and not (20 <= tafel_val <= 200):
                        tafel_val = None

                    if op_val or tafel_val:
                        results.append({
                            "paper": filename,
                            "page": page_num + 1,
                            "source": "text",
                            "catalyst_formula": formula,
                            "overpotential_mV": op_val,
                            "tafel_mV_per_dec": tafel_val,
                            "context": sent[:200],
                        })

    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    print(f"  -> Running total: {len(results)} entries")

# --- Save ---
df = pd.DataFrame(results)
df.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"EXTRACTION V3 COMPLETE")
print(f"{'='*60}")
print(f"Total raw entries: {len(df)}")
print(f"From tables: {(df['source']=='table').sum()}")
print(f"From text:   {(df['source']=='text').sum()}")
print(f"Unique papers: {df['paper'].nunique()}")
print(f"Unique formulas: {df['catalyst_formula'].nunique()}")
print(f"Rows with overpotential: {df['overpotential_mV'].notna().sum()}")
print(f"Rows with Tafel: {df['tafel_mV_per_dec'].notna().sum()}")
print(f"\nSaved to: {output_file}")

# Show sample
print(f"\nSample of first 10 rows:")
print(df.head(10)[["paper", "source", "catalyst_formula", "overpotential_mV", "tafel_mV_per_dec"]].to_string(index=False))