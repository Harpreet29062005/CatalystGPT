import numpy as np
import matplotlib.pyplot as plt
import os
import re
import torch
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from pymatgen.core import Element as PmgElement

from model_edge import EdgeGNN
import torch.nn as nn
from torch_geometric.loader import DataLoader

# --- Reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Paths ---
graphs_file = "data/processed/graphs_v4.pt"
output_folder = "results/figures"
os.makedirs(output_folder, exist_ok=True)

# --- Load graphs ---
print("Loading graphs...")
graphs = torch.load(graphs_file, weights_only=False)
graphs = [g for g in graphs if g.y.item() <= 400]
print(f"Loaded {len(graphs)} graphs (after outlier removal)\n")

DEVICE = torch.device("cpu")
N_FOLDS = 5
BATCH_SIZE = 16
EPOCHS = 500
PATIENCE = 60
LEARNING_RATE = 3e-4
HIDDEN_DIM = 64
DROPOUT = 0.2
WEIGHT_DECAY = 5e-5


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
            p = model(batch)
            preds.extend(p.cpu().numpy().flatten())
            targets.extend(batch.y_norm.squeeze(-1).cpu().numpy().flatten())
    return np.array(preds), np.array(targets)


# --- Feature extraction for Random Forest ---
METALS = ["Pt", "Mo", "Pd", "Rh", "Ni", "Fe", "Co", "Mn", "Cr", "Cu",
          "Ru", "Ir", "Au", "Ag", "Ti", "V", "Zn", "Sn", "W", "Nb",
          "Ga", "In", "Ta", "Zr", "Hf", "Re", "Os", "Cd", "Al", "Mg"]

def feats(formula):
    elems = list(set(re.findall(r"[A-Z][a-z]?", formula)))
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

feature_names = list(feats(graphs[0].formula).keys())
X_rf = np.array([[feats(g.formula)[k] for k in feature_names] for g in graphs])
y_all = np.array([g.y.item() for g in graphs])


# =============================================================
# 5-fold CV — collect predictions in the SAME order
# =============================================================
print("Running 5-fold CV to collect aligned predictions...")
outer_kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# We'll store predictions indexed by graph i (0..N-1)
gnn_preds_aligned = np.zeros(len(graphs))
rf_preds_aligned = np.zeros(len(graphs))
base_preds_aligned = np.zeros(len(graphs))
targets_aligned = np.zeros(len(graphs))

for fold, (train_pool_idx, test_idx) in enumerate(outer_kf.split(graphs)):
    print(f"  Fold {fold+1}...")
    train_pool = [graphs[i] for i in train_pool_idx]
    test_set = [graphs[i] for i in test_idx]

    # Normalization from training only
    y_train = np.array([g.y.item() for g in train_pool])
    y_mean = y_train.mean()
    y_std = y_train.std() if y_train.std() > 0 else 1.0

    for g in train_pool + test_set:
        g.y_norm = torch.tensor([[(g.y.item() - y_mean) / y_std]], dtype=torch.float)

    # Inner validation for early stopping
    n_val = max(4, int(len(train_pool) * 0.15))
    tr_in, val_in = train_test_split(
        list(range(len(train_pool))), test_size=n_val, random_state=SEED
    )
    train_inner = [train_pool[i] for i in tr_in]
    val_inner = [train_pool[i] for i in val_in]

    train_loader = DataLoader(train_inner, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_inner, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # ----- GNN -----
    model = EdgeGNN(node_dim=10, edge_dim=3, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)
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

    # Store in aligned arrays
    for k, idx in enumerate(test_idx):
        gnn_preds_aligned[idx] = tp_mv[k]

    # ----- RF -----
    rf = RandomForestRegressor(n_estimators=300, max_depth=10,
                               min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    rf.fit(X_rf[train_pool_idx], y_all[train_pool_idx])
    rf_p = rf.predict(X_rf[test_idx])
    for k, idx in enumerate(test_idx):
        rf_preds_aligned[idx] = rf_p[k]

    # ----- Baseline (mean of training) -----
    for idx in test_idx:
        base_preds_aligned[idx] = y_mean

    # Store targets
    for idx in test_idx:
        targets_aligned[idx] = graphs[idx].y.item()


# =============================================================
# Metrics
# =============================================================
def metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<22} MAE={mae:6.2f}  RMSE={rmse:6.2f}  R2={r2:7.3f}")
    return mae, rmse, r2

print()
print("=" * 60)
print("FINAL METRICS (aligned, honest)")
print("=" * 60)
base_m = metrics(targets_aligned, base_preds_aligned, "Baseline (mean)")
rf_m   = metrics(targets_aligned, rf_preds_aligned, "Random Forest")
gnn_m  = metrics(targets_aligned, gnn_preds_aligned, "EdgeGNN (proposed)")
print("=" * 60)


# =============================================================
# PLOT 1: Parity plots
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models = [
    ("Baseline (mean)", base_preds_aligned, base_m, "#888888"),
    ("Random Forest", rf_preds_aligned, rf_m, "#2E86AB"),
    ("EdgeGNN (proposed)", gnn_preds_aligned, gnn_m, "#D62246"),
]

for ax, (name, preds, (mae, rmse, r2), color) in zip(axes, models):
    ax.scatter(targets_aligned, preds, alpha=0.55, s=28, color=color,
               edgecolor="k", linewidth=0.3)

    lo = min(targets_aligned.min(), preds.min()) - 10
    hi = max(targets_aligned.max(), preds.max()) + 10
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Actual onset potential (mV)", fontsize=11)
    ax.set_ylabel("Predicted onset potential (mV)", fontsize=11)
    ax.set_title(f"{name}\nMAE={mae:.1f} mV | R²={r2:.3f}", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

plt.suptitle("CatalystGPT: Predicted vs Actual HER Onset Potential",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "parity_plots.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_folder}/parity_plots.png")


# =============================================================
# PLOT 2: Residuals for GNN
# =============================================================
residuals = gnn_preds_aligned - targets_aligned
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(targets_aligned, residuals, alpha=0.5, s=25, color="#D62246",
           edgecolor="k", linewidth=0.3)
ax.axhline(0, color="k", linestyle="--", linewidth=1)
ax.set_xlabel("Actual onset potential (mV)", fontsize=11)
ax.set_ylabel("Residual (mV)", fontsize=11)
ax.set_title("EdgeGNN Residuals vs Actual", fontsize=12)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(residuals, bins=30, color="#D62246", alpha=0.7, edgecolor="black")
ax.axvline(0, color="k", linestyle="--", linewidth=1)
ax.axvline(residuals.mean(), color="blue", linewidth=1,
           label=f"Mean = {residuals.mean():.1f} mV")
ax.set_xlabel("Residual (mV)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("EdgeGNN Residual Distribution", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "residuals.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_folder}/residuals.png")


# =============================================================
# PLOT 3: Feature importance
# =============================================================
print("\nComputing feature importance...")
rf_full = RandomForestRegressor(n_estimators=500, max_depth=10,
                                min_samples_leaf=2, random_state=SEED, n_jobs=-1)
rf_full.fit(X_rf, y_all)
imp = rf_full.feature_importances_
top_idx = np.argsort(imp)[-15:][::-1]
top_features = [feature_names[i] for i in top_idx]
top_values = [imp[i] for i in top_idx]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(top_features)), top_values, color="#2E86AB", edgecolor="black")
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Feature importance", fontsize=11)
ax.set_title("Top 15 Features for HER Activity Prediction (Random Forest)", fontsize=12)
ax.grid(True, alpha=0.3, axis="x")
for i, (bar, val) in enumerate(zip(bars, top_values)):
    ax.text(val, i, f" {val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_folder}/feature_importance.png")


# =============================================================
# Metrics summary text
# =============================================================
summary_path = os.path.join(output_folder, "metrics_summary.txt")
with open(summary_path, "w") as f:
    f.write("CatalystGPT — Final Metrics Summary\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"{'Model':<25}{'MAE':<12}{'RMSE':<12}{'R2':<10}\n")
    f.write("-" * 55 + "\n")
    f.write(f"{'Baseline (mean)':<25}{base_m[0]:<12.2f}{base_m[1]:<12.2f}{'N/A':<10}\n")
    f.write(f"{'Random Forest':<25}{rf_m[0]:<12.2f}{rf_m[1]:<12.2f}{rf_m[2]:<10.3f}\n")
    f.write(f"{'EdgeGNN (proposed)':<25}{gnn_m[0]:<12.2f}{gnn_m[1]:<12.2f}{gnn_m[2]:<10.3f}\n")
    f.write(f"\nSamples: {len(graphs)} catalysts\n")
    f.write("Evaluation: 5-fold cross-validation, nested early stopping\n")
    f.write("\nTop 5 features:\n")
    for i in range(5):
        f.write(f"  {i+1}. {top_features[i]} ({top_values[i]:.4f})\n")

print(f"Saved: {summary_path}")
print("\nAll plots regenerated with correct alignment!")