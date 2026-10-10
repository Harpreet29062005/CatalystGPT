"""
Novelty Check for Discovery Engine Candidates

Loads the top 10 earth-abundant candidates and:
1. Shows each candidate's elements
2. Checks against Gorsse training data (exact + element-set overlap)
3. Generates Google Scholar search queries for manual verification
"""

import pandas as pd
import re
import os

# --- Load candidates ---
candidates_file = "results/earth_abundant_candidates.csv"
df = pd.read_csv(candidates_file)

print("=" * 70)
print("NOVELTY CHECK — Top 10 Earth-Abundant Candidates")
print("=" * 70)
print()

# --- Load Gorsse training data ---
gorsse = pd.read_csv("data/processed/gorsse_180_catalysts.csv")

def normalize(f):
    syms = re.findall(r"[A-Z][a-z]?", str(f))
    return "".join(sorted(set(syms)))

gorsse_forms = [str(f).strip() for f in gorsse["Formula"].dropna()]
gorsse_norm = set(normalize(f) for f in gorsse_forms)

# --- Check each candidate ---
top10 = df.head(10).reset_index(drop=True)

print(f"{'#':<4}{'Formula':<35}{'Elements':<40}{'In Gorsse?'}")
print("-" * 95)

search_queries = []

for i, row in top10.iterrows():
    f = row["formula"]
    pred = row["predicted_mV"]
    syms = sorted(set(re.findall(r"[A-Z][a-z]?", f)))
    elem_str = "+".join(syms)
    norm = normalize(f)
    
    # Check exact match
    exact_match = any(normalize(g) == norm for g in gorsse_forms)
    
    print(f"{i+1:<4}{f:<35}{elem_str:<40}{'YES' if exact_match else 'NO'}")

    # Build search query
    query = " OR ".join(f'"{e}"' for e in syms[:4])  # limit for length
    search_queries.append((f, pred, elem_str))

print()
print("=" * 70)
print("GOOGLE SCHOLAR SEARCH QUERIES")
print("=" * 70)
print()
print("For each candidate below, open Google Scholar and search:")
print()

for i, (f, pred, elem_str) in enumerate(search_queries, 1):
    print(f"--- Candidate #{i}: {f} (predicted {pred:.2f} mV) ---")
    print(f"    Query 1: \"high entropy alloy\" \"{elem_str}\" hydrogen evolution")
    print(f"    Query 2: {f} HER catalyst")
    print()

print()
print("=" * 70)
print("HOW TO INTERPRET RESULTS")
print("=" * 70)
print()
print("For each candidate, check the top 5 Google Scholar results:")
print()
print("  ✅ NOVEL   = No paper has the same element combination + HER activity")
print("  ⚠️ SIMILAR = Same elements reported but different stoichiometry")
print("  ❌ KNOWN   = Exact same composition already published")
print()
print("If 3+ candidates are NOVEL, we have a genuine discovery.")