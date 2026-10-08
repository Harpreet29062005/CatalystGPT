import pandas as pd
import torch
import numpy as np
import os
import re
from pymatgen.core import Element
from torch_geometric.data import Data

# --- Paths ---
input_file = "data/processed/gorsse_180_catalysts.csv"
output_file = "data/processed/graphs_v4.pt"
os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows from {input_file}\n")

target_col = "onset_potential"
formula_col = "Formula"


def safe_float(value, default):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def parse_formula_with_stoich(formula):
    """
    Parse a formula like 'Al20Cu20Fe20Mo20Ni20' into:
    [(element, stoich_float), ...]
    Handles formulas with or without numbers.
    """
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
    """Features for a single atom node."""
    try:
        el = Element(element_symbol)
    except Exception:
        return None

    return [
        float(el.Z),                              # atomic number
        safe_float(el.atomic_radius, 1.5),        # radius
        safe_float(el.X, 2.0),                    # electronegativity
        float(el.group) if el.group else 8.0,     # group
        float(el.row) if el.row else 4.0,         # period
        safe_float(el.electron_affinity, 0.0),    # electron affinity
        safe_float(el.ionization_energy, 0.0),    # ionization energy
        safe_float(el.melting_point, 1000.0),     # melting point
        safe_float(el.density_of_solid, 5.0),     # density
        float(composition_fraction),              # <<< NEW: composition fraction
    ]


def build_graph(formula, target_value):
    parsed = parse_formula_with_stoich(formula)
    if not parsed:
        return None

    # Total stoichiometry for normalization
    total = sum(n for _, n in parsed)
    if total <= 0:
        return None

    # Build node features with composition fraction
    node_feats = []
    valid_elements = []
    for sym, n in parsed:
        frac = n / total
        feats = get_node_features(sym, frac)
        if feats is not None:
            node_feats.append(feats)
            valid_elements.append(sym)

    if len(valid_elements) < 3:
        return None

    # Deduplicate elements if a formula lists them twice
    seen = {}
    for sym, feats in zip(valid_elements, node_feats):
        if sym in seen:
            continue
        seen[sym] = feats
    unique_syms = list(seen.keys())
    unique_feats = list(seen.values())

    x = torch.tensor(unique_feats, dtype=torch.float)

    n = len(unique_syms)
    edge_index = []
    edge_attr = []

    for i in range(n):
        for j in range(n):
            if i != j:
                edge_index.append([i, j])
                xi, xj = unique_feats[i], unique_feats[j]
                en_diff = abs(xi[2] - xj[2])
                rad_diff = abs(xi[1] - xj[1])
                # Composition-weighted average difference
                comp_avg = (xi[9] + xj[9]) / 2.0
                edge_attr.append([en_diff, rad_diff, comp_avg])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor([[target_value]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.formula = str(formula)
    data.n_elements = n
    return data


# --- Build graphs ---
graphs = []
failed = []

for idx, row in df.iterrows():
    formula = row[formula_col]
    target = row[target_col]

    if pd.isna(target) or pd.isna(formula):
        failed.append(f"row_{idx}")
        continue

    try:
        g = build_graph(formula, float(target))
        if g is not None:
            graphs.append(g)
        else:
            failed.append(str(formula))
    except Exception as e:
        failed.append(f"{formula} ({e})")

print(f"Built {len(graphs)} graphs")
print(f"Failed: {len(failed)}\n")

torch.save(graphs, output_file)
print(f"Saved graphs to: {output_file}\n")

if graphs:
    sample = graphs[0]
    print(f"Sample graph: {sample.formula}")
    print(f"  Nodes:      {sample.x.shape[0]}")
    print(f"  Node dim:   {sample.x.shape[1]}   (was 9, now 10 with composition)")
    print(f"  Edges:      {sample.edge_index.shape[1]}")
    print(f"  Edge dim:   {sample.edge_attr.shape[1]}   (was 2, now 3)")
    print(f"  Target:     {sample.y.item():.2f} mV\n")

    y_vals = [g.y.item() for g in graphs]
    print("Target distribution:")
    print(f"  Min:  {min(y_vals):.2f} mV")
    print(f"  Max:  {max(y_vals):.2f} mV")
    print(f"  Mean: {np.mean(y_vals):.2f} mV")
    print(f"  Std:  {np.std(y_vals):.2f} mV")

    dims = set(g.x.shape[1] for g in graphs)
    print(f"\nNode feature dims: {dims}")
    edge_dims = set(g.edge_attr.shape[1] for g in graphs)
    print(f"Edge feature dims: {edge_dims}")

    # Show a sample with composition fractions
    print(f"\nExample graph details:")
    print(f"  Formula: {sample.formula}")
    print(f"  Node 0 (composition fraction + atomic number): "
          f"{sample.x[0].tolist()}")