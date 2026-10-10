import os
import pandas as pd
import re
from cathub.query import CathubQuery

# --- Paths ---
output_folder = "data/raw"
output_file = os.path.join(output_folder, "catalysis_hub_raw.csv")
os.makedirs(output_folder, exist_ok=True)

# --- API Key ---
# Option 1: Set the CATHUB_API_KEY environment variable (recommended)
# Option 2: Pass it directly: client = CathubQuery(api_key='your_key_here')
API_KEY = os.environ.get("CATHUB_API_KEY", None)

if not API_KEY:
    print("=" * 70)
    print("⚠️  CATHUB_API_KEY not set.")
    print("=" * 70)
    print("To get an API key:")
    print("  1. Go to https://www.catalysis-hub.org")
    print("  2. Sign in (or create account)")
    print("  3. Find API key in your profile/settings")
    print("  4. Set it: set CATHUB_API_KEY=your_key_here")
    print("=" * 70)
    print("\nAttempting query without key (may fail)...\n")

# --- Query using official CathubQuery class ---
print("Querying Catalysis-Hub via CathubQuery...\n")

try:
    if API_KEY:
        client = CathubQuery(api_key=API_KEY)
    else:
        client = CathubQuery()

    # Fetch reactions (all available, then filter)
    df = client.get_dataframe()

    print(f"✅ Query successful. Fetched {len(df)} reactions.\n")

except Exception as e:
    print(f"❌ Query failed: {e}")
    print("\nThis means we need the API key.")
    print("Please get your key from https://www.catalysis-hub.org and rerun.")
    exit(1)

# --- Filter for multinary (HEA-relevant) compositions ---
METALS = {"Pt", "Mo", "Pd", "Rh", "Ni", "Fe", "Co", "Mn", "Cr", "Cu",
          "Ru", "Ir", "Au", "Ag", "Ti", "V", "Zn", "Sn", "W", "Nb",
          "Ga", "In", "Ta", "Zr", "Hf", "Re", "Os", "Cd", "Al", "Mg"}

def count_metals(comp):
    if not isinstance(comp, str):
        return 0
    symbols = re.findall(r"[A-Z][a-z]?", comp)
    return len(set(s for s in symbols if s in METALS))

if 'surface_composition' in df.columns:
    df["n_metals"] = df["surface_composition"].apply(count_metals)
    multinary = df[df["n_metals"] >= 3]
else:
    print("⚠️ 'surface_composition' column not found. Available columns:")
    print(list(df.columns))
    multinary = pd.DataFrame()

print(f"Total reactions: {len(df)}")
print(f"Multinary (3+ metals): {len(multinary)}\n")

# --- Save ---
df.to_csv(output_file, index=False)
print(f"✅ Saved all data to: {output_file}")

if len(multinary) > 0:
    multinary_file = os.path.join(output_folder, "catalysis_hub_multinary.csv")
    multinary.to_csv(multinary_file, index=False)
    print(f"✅ Saved multinary subset to: {multinary_file}")
    print(f"\n--- Top 10 multinary compositions ---")
    cols = [c for c in ["surface_composition", "facet", "reaction_energy"] if c in multinary.columns]
    if cols:
        print(multinary[cols].head(10).to_string(index=False))
else:
    print("\n⚠️ No multinary compositions found.")
    print("Catalysis-Hub is primarily pure metals and bimetallics.")
    print("This is expected — it means we need other sources for HEA data.")

# --- Quick stats ---
print(f"\n--- Summary ---")
if 'surface_composition' in df.columns:
    print(f"Distinct compositions: {df['surface_composition'].nunique()}")
if 'facet' in df.columns:
    print(f"Distinct facets: {df['facet'].nunique()}")