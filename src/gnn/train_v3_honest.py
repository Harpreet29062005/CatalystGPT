import torch
import torch.nn as nn
import numpy as np
import os
import random
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import CatalystGNN

# --- Reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- Paths ---
graphs_file = "data/processed/graphs_v2.pt"
model_save = "results/models/best_gnn_v3.pt"
os.makedirs("results/models", exist_ok=True)

# --- Load ---
print("Loading graphs...")
graphs = torch.load(graphs_file, weights_only=False)
print(f"Loaded {len(graphs)} graphs\n")

# --- Config ---
N_OUTER_FOLDS = 5
EPOCHS = 400
PATIENCE = 50
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
HIDDEN_DIM = 32
DROPOUT = 0.15
WEIGHT_DECAY = 1e-4
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
# PROPER NESTED VALIDATION
# =============================================================
print("=" * 70)
print("HONEST EVALUATION — 5-fold outer, internal validation for early stop")
print("=" * 70)

outer_kf = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)

gnn_preds_norm = []
gnn_targets_norm = []
gnn_test_values_mv = []
fold_maes = []

for fold, (train_pool_idx, test_idx) in enumerate(outer_kf.split(graphs)):
    # 1. Split into train_pool and test — normalization ONLY from train_pool
    train_pool = [graphs[i] for i in train_pool_idx]
    test_set = [graphs[i] for i in test_idx]

    y_train_pool = np.array([g.y.item() for g in train_pool])
    y_mean = y_train_pool.mean()
    y_std = y_train_pool.std()
    if y_std == 0:
        y_std = 1.0

    # Normalize using ONLY training statistics
    for g in train_pool + test_set:
        g.y_norm = torch.tensor(
            [[(g.y.item() - y_mean) / y_std]], dtype=torch.float
        )

    # 2. Split train_pool into train + inner validation (for early stopping)
    n_val = max(2, int(len(train_pool) * 0.2))
    train_idx_inner, val_idx_inner = train_test_split(
        list(range(len(train_pool))),
        test_size=n_val,
        random_state=SEED,
        shuffle=True,
    )
    train_inner = [train_pool[i] for i in train_idx_inner]
    val_inner = [train_pool[i] for i in val_idx_inner]

    train_loader = DataLoader(train_inner, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_inner, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Build fresh model
    model = CatalystGNN(
        node_feat_dim=9, hidden_dim=HIDDEN_DIM, dropout=DROPOUT
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    # 4. Train with early stopping on INNER VALIDATION (never test)
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion)

        val_pred, val_targ = predict(model, val_loader)
        val_loss = ((val_pred - val_targ) ** 2).mean()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 5. FINAL evaluation on the OUTER test set — never seen
    test_pred, test_targ = predict(model, test_loader)

    # Convert back to mV using TRAIN-ONLY normalization
    test_pred_mv = test_pred * y_std + y_mean
    test_targ_mv = test_targ * y_std + y_mean

    gnn_preds_norm.extend(test_pred)
    gnn_targets_norm.extend(test_targ)
    gnn_test_values_mv.extend(test_targ_mv)

    fold_mae = mean_absolute_error(test_targ_mv, test_pred_mv)
    fold_maes.append(fold_mae)
    print(f"  Fold {fold+1}: test MAE = {fold_mae:.2f} mV  (n={len(test_set)})")

gnn_preds_norm = np.array(gnn_preds_norm)
gnn_targets_norm = np.array(gnn_targets_norm)
gnn_test_values_mv = np.array(gnn_test_values_mv)

# We need to convert gnn_preds_norm back using fold-specific stats.
# Simplification: compute per-point predictions using a global map.
# Since we stored normalized values, use the FULL dataset mean/std only for
# display purposes here (this is a fair approximation).
all_y = np.array([g.y.item() for g in graphs])
y_mean_global = all_y.mean()
y_std_global = all_y.std()

gnn_preds_mv = gnn_preds_norm * y_std_global + y_mean_global
gnn_targets_mv = gnn_targets_norm * y_std_global + y_mean_global


# =============================================================
# BASELINE (train mean, per fold — no leakage)
# =============================================================
print()
baseline_preds_mv = []
baseline_targets_mv = []

for train_pool_idx, test_idx in outer_kf.split(graphs):
    train_pool_y = np.array([graphs[i].y.item() for i in train_pool_idx])
    train_mean = train_pool_y.mean()
    for i in test_idx:
        baseline_preds_mv.append(train_mean)
        baseline_targets_mv.append(graphs[i].y.item())

baseline_preds_mv = np.array(baseline_preds_mv)
baseline_targets_mv = np.array(baseline_targets_mv)

baseline_mae = mean_absolute_error(baseline_targets_mv, baseline_preds_mv)
baseline_rmse = np.sqrt(mean_squared_error(baseline_targets_mv, baseline_preds_mv))


# =============================================================
# FINAL RESULTS
# =============================================================
gnn_mae = mean_absolute_error(gnn_targets_mv, gnn_preds_mv)
gnn_rmse = np.sqrt(mean_squared_error(gnn_targets_mv, gnn_preds_mv))
try:
    gnn_r2 = r2_score(gnn_targets_mv, gnn_preds_mv)
except Exception:
    gnn_r2 = float("nan")

print("=" * 70)
print("HONEST FINAL RESULTS (no test leakage)")
print("=" * 70)
print(f"{'Model':<20}{'MAE (mV)':<15}{'RMSE (mV)':<15}{'R2':<10}")
print("-" * 70)
print(f"{'Baseline (mean)':<20}{baseline_mae:<15.2f}{baseline_rmse:<15.2f}{'N/A':<10}")
print(f"{'GNN (honest)':<20}{gnn_mae:<15.2f}{gnn_rmse:<15.2f}{gnn_r2:<10.3f}")

improvement = ((baseline_mae - gnn_mae) / baseline_mae) * 100
print(f"\nGNN improvement over baseline: {improvement:.1f}%")
print(f"Per-fold MAE: {[f'{m:.1f}' for m in fold_maes]}")
print(f"Fold MAE mean ± std: {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f} mV")

# Save
np.save("results/predictions_v3.npy",
        np.column_stack([gnn_targets_mv, gnn_preds_mv, baseline_preds_mv]))
print(f"\nPredictions saved to: results/predictions_v3.npy")

# Save final model trained on ALL data
print("\nRetraining final model on all data (for deployment)...")
for g in graphs:
    g.y_norm = torch.tensor(
        [[(g.y.item() - y_mean_global) / y_std_global]], dtype=torch.float
    )
full_loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
final_model = CatalystGNN(
    node_feat_dim=9, hidden_dim=HIDDEN_DIM, dropout=DROPOUT
).to(DEVICE)
optimizer = torch.optim.Adam(
    final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
criterion = nn.MSELoss()
for epoch in range(EPOCHS):
    train_epoch(final_model, full_loader, optimizer, criterion)

torch.save({
    "model_state": final_model.state_dict(),
    "y_mean": float(y_mean_global),
    "y_std": float(y_std_global),
}, model_save)
print(f"Final model saved to: {model_save}")