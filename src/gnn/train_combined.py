"""
Train EdgeGNN on the combined dataset (193 catalysts: Gorsse 139 + custom NLP 54).

Uses 5-fold cross-validation with nested early stopping.
Compares against mean baseline and Random Forest.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import random
import warnings
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model_edge import EdgeGNN

warnings.filterwarnings("ignore")

# --- Reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- Paths ---
graphs_file = "data/processed/graphs_combined.pt"
model_save = "results/models/best_gnn_combined.pt"
os.makedirs("results/models", exist_ok=True)

# --- Load ---
print("Loading combined graphs...")
graphs = torch.load(graphs_file, weights_only=False)
print(f"Loaded {len(graphs)} graphs\n")

# --- Config ---
N_FOLDS = 5
EPOCHS = 500
PATIENCE = 60
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
HIDDEN_DIM = 64
DROPOUT = 0.2
WEIGHT_DECAY = 5e-5
DEVICE = torch.device("cpu")


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y_norm.squeeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


def predict(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            pred = model(batch)
            preds.extend(pred.cpu().numpy().flatten())
            targets.extend(batch.y_norm.squeeze(-1).cpu().numpy().flatten())
    return np.array(preds), np.array(targets)


# =============================================================
# 5-FOLD CROSS-VALIDATION
# =============================================================
print("=" * 70)
print("COMBINED DATASET — 5-fold outer, inner validation, no leakage")
print("=" * 70)

outer_kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

gnn_preds_all = []
gnn_targets_all = []
fold_maes = []

for fold, (train_pool_idx, test_idx) in enumerate(outer_kf.split(graphs)):
    train_pool = [graphs[i] for i in train_pool_idx]
    test_set = [graphs[i] for i in test_idx]

    # Normalize using training statistics only
    y_train = np.array([g.y.item() for g in train_pool])
    y_mean = y_train.mean()
    y_std = y_train.std() if y_train.std() > 0 else 1.0

    for g in train_pool + test_set:
        g.y_norm = torch.tensor(
            [[(g.y.item() - y_mean) / y_std]], dtype=torch.float
        )

    # Inner split
    n_val = max(4, int(len(train_pool) * 0.15))
    train_idx_in, val_idx_in = train_test_split(
        list(range(len(train_pool))), test_size=n_val, random_state=SEED
    )
    train_inner = [train_pool[i] for i in train_idx_in]
    val_inner = [train_pool[i] for i in val_idx_in]

    train_loader = DataLoader(train_inner, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_inner, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = EdgeGNN(
        node_dim=10, edge_dim=3, hidden_dim=HIDDEN_DIM, dropout=DROPOUT
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )
    criterion = nn.SmoothL1Loss()

    best_val = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion)
        vp, vt = predict(model, val_loader)
        vloss = ((vp - vt) ** 2).mean()
        scheduler.step(vloss)
        if vloss < best_val:
            best_val = vloss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    tp, tt = predict(model, test_loader)
    tp_mv = tp * y_std + y_mean
    tt_mv = tt * y_std + y_mean

    gnn_preds_all.extend(tp_mv)
    gnn_targets_all.extend(tt_mv)
    fold_maes.append(mean_absolute_error(tt_mv, tp_mv))
    print(f"  Fold {fold+1}: test MAE = {fold_maes[-1]:.2f} mV (n={len(test_set)})")

gnn_preds_all = np.array(gnn_preds_all)
gnn_targets_all = np.array(gnn_targets_all)


# =============================================================
# BASELINE 1: Mean of training set
# =============================================================
base_preds, base_targets = [], []
for train_idx, test_idx in outer_kf.split(graphs):
    m = np.array([graphs[i].y.item() for i in train_idx]).mean()
    for i in test_idx:
        base_preds.append(m)
        base_targets.append(graphs[i].y.item())
base_preds = np.array(base_preds)
base_targets = np.array(base_targets)
base_mae = mean_absolute_error(base_targets, base_preds)
base_rmse = np.sqrt(mean_squared_error(base_targets, base_preds))


# =============================================================
# BASELINE 2: Random Forest
# =============================================================
METALS = ["Pt", "Mo", "Pd", "Rh", "Ni", "Fe", "Co", "Mn", "Cr", "Cu",
          "Ru", "Ir", "Au", "Ag", "Ti", "V", "Zn", "Sn", "W", "Nb",
          "Ga", "In", "Ta", "Zr", "Hf", "Re", "Os", "Cd", "Al", "Mg"]

from pymatgen.core import Element as PmgElement
import re as _re

def feats(formula):
    elems = list(set(_re.findall(r"[A-Z][a-z]?", formula)))
    f = {}
    for m in METALS:
        f[f"has_{m}"] = 1 if m in elems else 0
    f["n_elements"] = len(elems)
    for prop in ["Z", "X", "ionization_energy", "electron_affinity",
                 "melting_point", "atomic_radius"]:
        vals = []
        for e in elems:
            try:
                el = PmgElement(e)
                v = getattr(el, prop, 0)
                if v is None:
                    v = 0
                vals.append(float(v))
            except Exception:
                continue
        if not vals:
            vals = [0]
        f[f"{prop}_mean"] = np.mean(vals)
        f[f"{prop}_std"] = np.std(vals)
        f[f"{prop}_max"] = np.max(vals)
        f[f"{prop}_min"] = np.min(vals)
    return f

X_rf = np.array([list(feats(g.formula).values()) for g in graphs])
y_rf = np.array([g.y.item() for g in graphs])

rf_preds = np.zeros(len(graphs))
for train_idx, test_idx in outer_kf.split(graphs):
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=2,
        random_state=SEED, n_jobs=-1
    )
    rf.fit(X_rf[train_idx], y_rf[train_idx])
    rf_preds[test_idx] = rf.predict(X_rf[test_idx])

rf_mae = mean_absolute_error(y_rf, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_rf, rf_preds))
rf_r2 = r2_score(y_rf, rf_preds)


# =============================================================
# FINAL RESULTS
# =============================================================
gnn_mae = mean_absolute_error(gnn_targets_all, gnn_preds_all)
gnn_rmse = np.sqrt(mean_squared_error(gnn_targets_all, gnn_preds_all))
gnn_r2 = r2_score(gnn_targets_all, gnn_preds_all)

print()
print("=" * 70)
print("FINAL RESULTS — COMBINED DATASET")
print("=" * 70)
print(f"{'Model':<25}{'MAE (mV)':<15}{'RMSE (mV)':<15}{'R2':<10}")
print("-" * 70)
print(f"{'Baseline (mean)':<25}{base_mae:<15.2f}{base_rmse:<15.2f}{'N/A':<10}")
print(f"{'Random Forest':<25}{rf_mae:<15.2f}{rf_rmse:<15.2f}{rf_r2:<10.3f}")
print(f"{'EdgeGNN (combined)':<25}{gnn_mae:<15.2f}{gnn_rmse:<15.2f}{gnn_r2:<10.3f}")

print(f"\nGNN improvement over mean baseline: "
      f"{100*(base_mae - gnn_mae)/base_mae:.1f}%")
print(f"GNN improvement over RF:            "
      f"{100*(rf_mae - gnn_mae)/rf_mae:.1f}%")
print(f"\nPer-fold GNN MAE: {[f'{m:.1f}' for m in fold_maes]}")
print(f"Mean ± std: {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f} mV")

# Save predictions
np.save("results/predictions_combined.npy",
        np.column_stack([gnn_targets_all, gnn_preds_all, base_preds, rf_preds]))
print(f"\nPredictions saved to: results/predictions_combined.npy")

# Train final production model
print("\nTraining final production model on all data...")
all_y = np.array([g.y.item() for g in graphs])
y_mean_f = all_y.mean()
y_std_f = all_y.std()
for g in graphs:
    g.y_norm = torch.tensor(
        [[(g.y.item() - y_mean_f) / y_std_f]], dtype=torch.float
    )

full_loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
final_model = EdgeGNN(
    node_dim=10, edge_dim=3, hidden_dim=HIDDEN_DIM, dropout=DROPOUT
).to(DEVICE)
opt = torch.optim.Adam(
    final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
crit = nn.SmoothL1Loss()
for epoch in range(EPOCHS):
    train_epoch(final_model, full_loader, opt, crit)

torch.save({
    "model_state": final_model.state_dict(),
    "y_mean": float(y_mean_f),
    "y_std": float(y_std_f),
    "n_training_samples": len(graphs),
    "node_dim": 10,
    "edge_dim": 3,
}, model_save)
print(f"Final model saved to: {model_save}")