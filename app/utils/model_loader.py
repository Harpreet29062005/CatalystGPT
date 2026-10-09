"""
CatalystGPT — Model Loader
Self-contained module: includes EdgeGNN architecture, graph builder, and prediction function.
No external imports from src/gnn/ needed — safe for cloud deployment.
"""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from pymatgen.core import Element


# ============================================================
# EdgeGNN MODEL ARCHITECTURE
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
# GRAPH BUILDER
# ============================================================
def safe_float(value, default):
    try:
        if value is None or (isinstance(value, float) and torch.isnan(torch.tensor(value))):
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
# MODEL LOADER
# ============================================================
class CatalystPredictor:
    """Loads the trained EdgeGNN and predicts HER onset potential."""

    def __init__(self, checkpoint_path="results/models/best_gnn_v4.pt"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cpu")
        self.model = None
        self.y_mean = None
        self.y_std = None
        self.loaded = False
        self.load_error = None
        self._load()

    def _load(self):
        try:
            if not os.path.exists(self.checkpoint_path):
                self.load_error = f"Checkpoint not found: {self.checkpoint_path}"
                return

            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

            self.y_mean = float(ckpt["y_mean"])
            self.y_std = float(ckpt["y_std"])
            node_dim = int(ckpt.get("node_dim", 10))
            edge_dim = int(ckpt.get("edge_dim", 3))

            self.model = EdgeGNN(
                node_dim=node_dim, edge_dim=edge_dim,
                hidden_dim=64, dropout=0.15,
            ).to(self.device)

            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            self.loaded = True
        except Exception as e:
            self.load_error = str(e)

    def predict(self, formula):
        """Predict onset potential (mV) for a single catalyst formula."""
        if not self.loaded:
            return {"error": self.load_error or "Model not loaded"}

        graph = build_graph(formula, target_value=0.0)
        if graph is None:
            return {"error": "Invalid formula. Need 3+ distinct metal elements."}

        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        graph = graph.to(self.device)

        with torch.no_grad():
            pred_norm = self.model(graph).item()

        pred_mv = pred_norm * self.y_std + self.y_mean
        return {"formula": formula, "prediction_mV": pred_mv}


# ============================================================
# SELF-TEST
# ============================================================
if __name__ == "__main__":
    print("Loading CatalystPredictor...")
    predictor = CatalystPredictor()

    if not predictor.loaded:
        print(f"FAILED to load: {predictor.load_error}")
    else:
        print(f"Model loaded successfully")
        print(f"  y_mean = {predictor.y_mean:.2f} mV")
        print(f"  y_std  = {predictor.y_std:.2f} mV\n")

        test_formulas = ["PtPdNiCoMn", "PtMoPdRhNi", "CoFeIrNiPtZn", "FeNiPt"]
        for f in test_formulas:
            result = predictor.predict(f)
            if "error" in result:
                print(f"  {f:<20} ERROR: {result['error']}")
            else:
                print(f"  {f:<20} -> {result['prediction_mV']:.2f} mV")