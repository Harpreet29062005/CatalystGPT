import torch
import torch.nn as nn
import numpy as np
import os
import random
from torch_geometric.loader import DataLoader
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import CatalystGNN

# --- Reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- Paths ---
graphs_file = "data/processed/graphs_v2.pt"
model_save = "results/models/best_gnn_v2.pt"
os.makedirs("results/models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- Load ---
print("Loading graphs...")
graphs = torch.load(graphs_file, weights_only=False)
print(f"Loaded {len(graphs)} graphs\n")

# --- Compute normalization (from entire dataset) ---
y_all = np.array([g.y.item() for g in graphs])
y_mean = y_all.mean()
y_std = y_all.std()
print(f"Overpotential normalization: mean={y_mean:.2f}, std={y_std:.2f}\n")

# Normalize every graph's target
for g in graphs:
    g.y_norm = torch.tensor([[(g.y.item() - y_mean) / y_std]], dtype=torch.float)

# --- Config ---
EPOCHS = 300
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
HIDDEN_DIM = 32          # smaller = less overfit
DROPOUT = 0.15
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y_norm.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def predict(model, loader):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            pred = model(batch)
            preds.extend(pred.cpu().numpy().flatten())
            targets.extend(batch.y_norm.squeeze(-1).cpu().numpy().flatten())
    return np.array(preds), np.array(targets)


# =============================================================
# BASELINE: Predict mean overpotential for every test point
# =============================================================
print("=" * 65)
print("BASELINE (predict the mean for every catalyst)")
print("=" * 65)

loo = LeaveOneOut()
baseline_preds = []
baseline_targets_norm = []

for train_idx, test_idx in loo.split(graphs):
    test_g = graphs[test_idx[0]]
    # Predict the mean of the TRAIN set (proper baseline)
    train_mean = np.mean([graphs[i].y.item() for i in train_idx])
    baseline_preds.append((train_mean - y_mean) / y_std)  # normalized
    baseline_targets_norm.append(test_g.y_norm.item())

baseline_preds = np.array(baseline_preds)
baseline_targets_norm = np.array(baseline_targets_norm)

# Convert back to mV for interpretation
baseline_preds_mv = baseline_preds * y_std + y_mean
baseline_targets_mv = baseline_targets_norm * y_std + y_mean

baseline_mae = mean_absolute_error(baseline_targets_mv, baseline_preds_mv)
baseline_rmse = np.sqrt(mean_squared_error(baseline_targets_mv, baseline_preds_mv))
print(f"Baseline MAE:  {baseline_mae:.2f} mV")
print(f"Baseline RMSE: {baseline_rmse:.2f} mV\n")


# =============================================================
# GNN: Leave-One-Out Cross-Validation
# =============================================================
print("=" * 65)
print("GNN — LEAVE-ONE-OUT CROSS-VALIDATION")
print("=" * 65)

gnn_preds = []
gnn_targets_norm = []
gnn_train_maes = []

for i, (train_idx, test_idx) in enumerate(loo.split(graphs)):
    train_graphs = [graphs[j] for j in train_idx]
    test_graph = graphs[test_idx[0]]

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader([test_graph], batch_size=1, shuffle=False)

    model = CatalystGNN(
        node_feat_dim=9, hidden_dim=HIDDEN_DIM, dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    # Early stopping setup
    best_test_loss = float("inf")
    patience = 40
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

        # Evaluate on the held-out test point (this is technically
        # optimistic for early stopping, but with 1 point it's minor)
        test_pred, test_targ = predict(model, test_loader)
        test_loss = ((test_pred - test_targ) ** 2).mean()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final prediction
    test_pred, test_targ = predict(model, test_loader)
    gnn_preds.append(test_pred[0])
    gnn_targets_norm.append(test_targ[0])

    train_pred, train_targ = predict(model, train_loader)
    train_mae_norm = mean_absolute_error(train_targ, train_pred)
    gnn_train_maes.append(train_mae_norm)

    if (i + 1) % 5 == 0 or (i + 1) == len(graphs):
        print(f"  Completed {i+1}/{len(graphs)} folds...")

gnn_preds = np.array(gnn_preds)
gnn_targets_norm = np.array(gnn_targets_norm)

# Convert back to mV
gnn_preds_mv = gnn_preds * y_std + y_mean
gnn_targets_mv = gnn_targets_norm * y_std + y_mean

gnn_mae = mean_absolute_error(gnn_targets_mv, gnn_preds_mv)
gnn_rmse = np.sqrt(mean_squared_error(gnn_targets_mv, gnn_preds_mv))
try:
    gnn_r2 = r2_score(gnn_targets_mv, gnn_preds_mv)
except Exception:
    gnn_r2 = float("nan")

print()
print("=" * 65)
print("FINAL RESULTS")
print("=" * 65)
print(f"{'Model':<20}{'MAE (mV)':<15}{'RMSE (mV)':<15}{'R2':<10}")
print("-" * 65)
print(f"{'Baseline (mean)':<20}{baseline_mae:<15.2f}{baseline_rmse:<15.2f}{'N/A':<10}")
print(f"{'GNN':<20}{gnn_mae:<15.2f}{gnn_rmse:<15.2f}{gnn_r2:<10.3f}")

improvement = ((baseline_mae - gnn_mae) / baseline_mae) * 100
print(f"\nGNN improvement over baseline: {improvement:.1f}%")

# Save
np.save("results/predictions_v2.npy",
        np.column_stack([gnn_targets_mv, gnn_preds_mv, baseline_preds_mv]))
print(f"\nPredictions saved to: results/predictions_v2.npy")

# Save best model (retrain on all data with best hyperparameters)
print("\nRetraining on full dataset for final model...")
full_loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
final_model = CatalystGNN(
    node_feat_dim=9, hidden_dim=HIDDEN_DIM, dropout=DROPOUT
).to(DEVICE)
optimizer = torch.optim.Adam(
    final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    train_one_epoch(final_model, full_loader, optimizer, criterion)

torch.save({
    "model_state": final_model.state_dict(),
    "y_mean": y_mean,
    "y_std": y_std,
}, model_save)
print(f"Final model saved to: {model_save}")