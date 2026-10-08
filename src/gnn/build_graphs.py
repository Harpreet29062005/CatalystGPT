import pandas as pd
import torch
import numpy as np
import os
import re
from pymatgen.core import Element
from torch_geometric.data import Data

# --- Paths ---
input_file = "data/processed/clean_dataset.csv"
output_file = "data/processed/graphs.pt"
os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows from {input_file}\n")


# --- Helper: safely convert to float ---
def safe_float(value, default):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


# --- Node feature extractor ---
def get_node_features(element_symbol):
    try:
        el = Element(element_symbol)
    except Exception:
        return None

    features = [
        float(el.Z),                              # atomic number
        safe_float(el.atomic_radius, 1.5),        # atomic radius (Angstrom)
        safe_float(el.X, 2.0),                    # electronegativity
        float(el.group) if el.group else 8.0,     # group
        float(el.row) if el.row else 4.0,         # period
        safe_float(el.electron_affinity, 0.0),    # electron affinity
        safe_float(el.ionization_energy, 0.0),    # first ionization energy
        safe_float(el.melting_point, 1000.0),     # melting point (K)
        safe_float(el.density_of_solid, 5.0),     # density (g/cm3)
    ]
    return features


# --- Parse formula into unique elements ---
def parse_formula(formula):
    symbols = re.findall(r"[A-Z][a-z]?", formula)
    seen = set()
    unique = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


# --- Build a single graph ---
def build_graph(formula, overpotential):
    elements = parse_formula(formula)

    node_feats = []
    valid_elements = []
    for el in elements:
        feats = get_node_features(el)
        if feats is not None:
            node_feats.append(feats)
            valid_elements.append(el)

    # Need at least 3 distinct metals to be a valid HEA graph
    if len(valid_elements) < 3:
        return None

    x = torch.tensor(node_feats, dtype=torch.float)

    # Build fully-connected edges (no self-loops)
    n = len(valid_elements)
    edge_index = []
    edge_attr = []

    for i in range(n):
        for j in range(n):
            if i != j:
                edge_index.append([i, j])
                xi, xj = node_feats[i], node_feats[j]
                en_diff = abs(xi[2] - xj[2])
                rad_diff = abs(xi[1] - xj[1])
                edge_attr.append([en_diff, rad_diff])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.tensor([[overpotential]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.formula = formula
    data.n_elements = n
    return data


# --- Build all graphs ---
graphs = []
failed = []

for idx, row in df.iterrows():
    formula = str(row["clean_formula"])
    overpotential = row["overpotential_mV"]

    if pd.isna(overpotential):
        continue

    try:
        g = build_graph(formula, float(overpotential))
        if g is not None:
            graphs.append(g)
        else:
            failed.append(formula)
    except Exception as e:
        failed.append(f"{formula} ({e})")

print(f"Built {len(graphs)} graphs")
print(f"Failed: {len(failed)}\n")

if failed:
    print("Failed formulas (first 10):")
    for f in failed[:10]:
        print(f"  - {f}")
    print()

# --- Save ---
torch.save(graphs, output_file)
print(f"Saved graphs to: {output_file}\n")

# --- Summary ---
if graphs:
    sample = graphs[0]
    print(f"Sample graph: {sample.formula}")
    print(f"  Nodes:            {sample.x.shape[0]}")
    print(f"  Node feat dim:    {sample.x.shape[1]}")
    print(f"  Edges:            {sample.edge_index.shape[1]}")
    print(f"  Edge feat dim:    {sample.edge_attr.shape[1]}")
    print(f"  Target (y):       {sample.y.item()} mV\n")

    y_vals = [g.y.item() for g in graphs]
    print("Overpotential distribution:")
    print(f"  Min:  {min(y_vals):.2f} mV")
    print(f"  Max:  {max(y_vals):.2f} mV")
    print(f"  Mean: {np.mean(y_vals):.2f} mV")
    print(f"  Std:  {np.std(y_vals):.2f} mV")

    # Check node feature dim consistency
    dims = set(g.x.shape[1] for g in graphs)
    print(f"\nNode feature dims across all graphs: {dims}")
    print(f"(Should be a single value)")