"""
Build graphs from the combined dataset (193 catalysts).
Uses composition-aware features (node_dim=10, edge_dim=3) to match EdgeGNN.
"""

import pandas as pd
import torch
import numpy as np
import os
import re
from pymatgen.core import Element
from torch_geometric.data import Data

# --- Paths ---
input_file = "data/processed/combined_dataset_filtered.csv"
output_file = "data/processed/graphs_combined.pt"
os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows from {input_file}")
print(f"Columns: {list(df.columns)}\n")


def safe_float(value, default):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def parse_formula_with_stoich(formula):
    """Parse a formula like 'Pt28Mo6Pd28Rh27Ni15' into [(element, stoich)]."""
    matches = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", str(formula))
    result = []
    for sym, num in matches:
        if not sym:
            continue
        try:
            n = float(num) if num else 1.0
        except ValueError:
            n = 1.0
        result.append((sym, n))
    return result


def get_node_features(element_symbol, composition_fraction):
    """10 features per node: 9 atomic + composition fraction."""
    try:
        el = Element(element_symbol)
    except Exception:
        return None
    return [
        float(el.Z),
        safe_float(el.atomic_radius, 1.5),
        safe_float(el.X, 2.0),
        float(el.group) if el.group else 8.0,
        float(el.row) if el.row else 4.0,
        safe_float(el.electron_affinity, 0.0),
        safe_float(el.ionization_energy, 0.0),
        safe_float(el.melting_point, 1000.0),
        safe_float(el.density_of_solid, 5.0),
        float(composition_fraction),
    ]


def build_graph(formula, target_value):
    parsed = parse_formula_with_stoich(formula)
    if not parsed:
        return None

    total = sum(n for _, n in parsed)
    if total <= 0:
        return None

    # Deduplicate elements, keep composition fraction
    seen = {}
    for sym, n in parsed:
        if sym in seen:
            continue
        frac = n / total
        feats = get_node_features(sym, frac)
        if feats is not None:
            seen[sym] = feats

    if len(seen) < 3:
        return None

    syms = list(seen.keys())
    feats_list = list(seen.values())
    x = torch.tensor(feats_list, dtype=torch.float)

    n = len(syms)
    edge_index, edge_attr = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_index.append([i, j])
                xi, xj = feats_list[i], feats_list[j]
                en_diff = abs(xi[2] - xj[2])
                rad_diff = abs(xi[1] - xj[1])
                comp_avg = (xi[9] + xj[9]) / 2.0
                edge_attr.append([en_diff, rad_diff, comp_avg])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor([[target_value]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.formula = str(formula)
    data.source = str(target_value)
    data.n_elements = n
    return data


# --- Build all graphs ---
graphs = []
failed = []

for idx, row in df.iterrows():
    formula = str(row["formula"])
    target = row["target_mV"]

    if pd.isna(target) or pd.isna(formula):
        failed.append(f"row_{idx}_missing")
        continue

    try:
        g = build_graph(formula, float(target))
        if g is not None:
            g.source = str(row.get("source", "unknown"))
            graphs.append(g)
        else:
            failed.append(formula)
    except Exception as e:
        failed.append(f"{formula} ({e})")

print(f"Built {len(graphs)} graphs")
print(f"Failed: {len(failed)}\n")

# --- Save ---
torch.save(graphs, output_file)
print(f"Saved graphs to: {output_file}\n")

# --- Summary ---
if graphs:
    sample = graphs[0]
    print(f"Sample graph: {sample.formula}")
    print(f"  Nodes:      {sample.x.shape[0]}")
    print(f"  Node dim:   {sample.x.shape[1]}")
    print(f"  Edges:      {sample.edge_index.shape[1]}")
    print(f"  Edge dim:   {sample.edge_attr.shape[1]}")
    print(f"  Target:     {sample.y.item():.2f} mV\n")

    y_vals = np.array([g.y.item() for g in graphs])
    print("Target distribution:")
    print(f"  Min:  {y_vals.min():.2f} mV")
    print(f"  Max:  {y_vals.max():.2f} mV")
    print(f"  Mean: {y_vals.mean():.2f} mV")
    print(f"  Std:  {y_vals.std():.2f} mV")

    # Source breakdown
    from collections import Counter
    sources = Counter(g.source for g in graphs)
    print(f"\nSource breakdown:")
    for src, count in sources.most_common():
        print(f"  {src}: {count}")

    dims = set(g.x.shape[1] for g in graphs)
    print(f"\nNode feature dims: {dims}")
    edge_dims = set(g.edge_attr.shape[1] for g in graphs)
    print(f"Edge feature dims: {edge_dims}")