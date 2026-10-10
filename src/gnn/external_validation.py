"""
External Validation: Test the Gorsse-trained EdgeGNN on the 
54 custom NLP-extracted catalysts.

This tests whether the model generalizes to a completely independent dataset
with a different target definition (overpotential vs onset potential).

We embed the graph builder with composition features to match the model.
"""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import NNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pymatgen.core import Element


# ============================================================
# EDGE GNN ARCHITECTURE (matches best_gnn_v4.pt)
# ============================================================
class EdgeGNN(nn.Module):
    def __init__(self, node_dim=10, edge_dim=3, hidden_dim=64, dropout=0.15):
        super().__init__()
        self.dropout = dropout

        edge_net1 = nn.Sequential(
            nn.Linear(edge_dim, 32), nn.ReLU(),
            nn.Linear(32, node_dim * hidden_dim),
        )
        self.conv1 = NNConv(node_dim, hidden_dim, edge_net1, aggr="mean")

        edge_net2 = nn.Sequential(
            nn.Linear(edge_dim, 32), nn.ReLU(),
            nn.Linear(32, hidden_dim * hidden_dim),
        )
        self.conv2 = NNConv(hidden_dim, hidden_dim, edge_net2, aggr="mean")

        edge_net3 = nn.Sequential(
            nn.Linear(edge_dim, 32), nn.ReLU(),
            nn.Linear(32, hidden_dim * hidden_dim),
        )
        self.conv3 = NNConv(hidden_dim, hidden_dim, edge_net3, aggr="mean")

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x); x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x); x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x); x = F.relu(x)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)


# ============================================================
# GRAPH BUILDER (with composition features)
# ============================================================
def safe_float(value, default):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def parse_formula_with_stoich(formula):
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


def build_graph(formula, target_value=0.0):
    parsed = parse_formula_with_stoich(formula)
    if not parsed:
        return None

    total = sum(n for _, n in parsed)
    if total <= 0:
        return None

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
    data.n_elements = n
    return data


# ============================================================
# MAIN
# ============================================================
print("=" * 70)
print("EXTERNAL VALIDATION")
print("Gorsse-trained model vs Custom NLP-extracted catalysts")
print("=" * 70)
print()

# --- Load custom dataset ---
custom_file = "data/processed/clean_dataset_v3.csv"
df = pd.read_csv(custom_file)
print(f"Loaded custom dataset: {len(df)} catalysts\n")

# --- Load trained model ---
model_path = "results/models/best_gnn_v4.pt"
if not os.path.exists(model_path):
    print(f"ERROR: Model checkpoint not found: {model_path}")
    exit(1)

ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
y_mean = float(ckpt["y_mean"])
y_std = float(ckpt["y_std"])
node_dim = int(ckpt.get("node_dim", 10))
edge_dim = int(ckpt.get("edge_dim", 3))

print(f"Model checkpoint loaded:")
print(f"  y_mean (Gorsse): {y_mean:.2f} mV")
print(f"  y_std  (Gorsse): {y_std:.2f} mV")
print(f"  node_dim: {node_dim}")
print(f"  edge_dim: {edge_dim}\n")

model = EdgeGNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=64, dropout=0.15)
model.load_state_dict(ckpt["model_state"])
model.eval()

# --- Build graphs from custom dataset ---
graphs = []
skipped = []

for _, row in df.iterrows():
    formula = str(row["clean_formula"])
    target = row["overpotential_mV"]
    if pd.isna(target):
        continue
    g = build_graph(formula, float(target))
    if g is not None:
        g.y_actual = float(target)
        graphs.append(g)
    else:
        skipped.append(formula)

print(f"Built {len(graphs)} graphs from custom dataset")
print(f"Skipped (invalid): {len(skipped)}\n")

if len(graphs) == 0:
    print("No valid graphs. Exiting.")
    exit(1)

# --- Predict ---
loader = DataLoader(graphs, batch_size=16, shuffle=False)

preds_norm = []
targets_actual = []
formulas_list = []

with torch.no_grad():
    for batch in loader:
        out = model(batch)
        preds_norm.extend(out.cpu().numpy().flatten())
        targets_actual.extend(batch.y_actual.cpu().numpy().flatten())
        for i in range(batch.num_graphs):
            formulas_list.append(batch.formula[i] if isinstance(batch.formula, list) else "?")

# Handle batch.formula properly
formulas_list = [g.formula for g in graphs]

preds_norm = np.array(preds_norm)
targets_actual = np.array(targets_actual)

# Convert predictions back to mV using Gorsse normalization
preds_mv = preds_norm * y_std + y_mean

# ============================================================
# Metrics
# ============================================================
print("=" * 70)
print("EXTERNAL VALIDATION RESULTS")
print("=" * 70)

mae = mean_absolute_error(targets_actual, preds_mv)
rmse = np.sqrt(mean_squared_error(targets_actual, preds_mv))
try:
    r2 = r2_score(targets_actual, preds_mv)
except Exception:
    r2 = float("nan")

# Baseline: predict mean of custom dataset
custom_mean = targets_actual.mean()
baseline_preds = np.full_like(targets_actual, custom_mean)
baseline_mae = mean_absolute_error(targets_actual, baseline_preds)

print(f"\nCustom dataset statistics:")
print(f"  N catalysts: {len(graphs)}")
print(f"  Actual range: {targets_actual.min():.2f} – {targets_actual.max():.2f} mV")
print(f"  Actual mean:  {targets_actual.mean():.2f} mV")
print(f"  Actual std:   {targets_actual.std():.2f} mV")

print(f"\nPrediction statistics (from Gorsse-trained model):")
print(f"  Predicted range: {preds_mv.min():.2f} – {preds_mv.max():.2f} mV")
print(f"  Predicted mean:  {preds_mv.mean():.2f} mV")
print(f"  Predicted std:   {preds_mv.std():.2f} mV")

print(f"\nMetrics:")
print(f"  MAE:  {mae:.2f} mV")
print(f"  RMSE: {rmse:.2f} mV")
print(f"  R²:   {r2:.3f}")
print(f"\nBaseline (predict custom mean {custom_mean:.2f} mV):")
print(f"  Baseline MAE: {baseline_mae:.2f} mV")
print(f"  Improvement over baseline: {100 * (baseline_mae - mae) / baseline_mae:.1f}%")

# --- Save results ---
results_df = pd.DataFrame({
    "formula": formulas_list,
    "actual_overpotential_mV": targets_actual,
    "predicted_from_gorsse_model_mV": preds_mv,
    "error_mV": preds_mv - targets_actual,
})
output_path = "results/external_validation.csv"
os.makedirs("results", exist_ok=True)
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

# Show sample
print(f"\n--- Sample predictions (first 15) ---")
print(results_df.head(15).to_string(index=False))

# --- Honest interpretation ---
print("\n" + "=" * 70)
print("HONEST INTERPRETATION")
print("=" * 70)
if r2 > 0.3:
    print("✅ The model GENERALIZES to the independent custom dataset.")
    print("   This is a strong positive result — the model is not overfit.")
elif r2 > 0:
    print("⚠️ The model shows WEAK transfer.")
    print("   Some signal transfers, but the small dataset limits confidence.")
elif r2 > -1:
    print("⚠️ The model does NOT transfer well.")
    print("   Predictions are barely better than random.")
else:
    print("❌ The model FAILS on the external dataset.")
    print("   Likely cause: onset potential vs overpotential target mismatch.")
    print("   This is an honest negative result worth reporting.")
print("=" * 70)