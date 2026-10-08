import torch
import torch.nn as nn
import numpy as np
import os
import random
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import CatalystGNN

# --- Reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- Paths ---
graphs_file = "data/processed/graphs.pt"
model_save = "results/models/best_gnn.pt"
os.makedirs("results/models", exist_ok=True)

# --- Load graphs ---
print("Loading graphs...")
graphs = torch.load(graphs_file, weights_only=False)
print(f"Loaded {len(graphs)} graphs\n")

# --- Shuffle ---
random.shuffle(graphs)

# --- Config ---
N_FOLDS = 5
EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
DROPOUT = 0.2
DEVICE = torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            pred = model(batch)
            preds.extend(pred.cpu().numpy().flatten())
            targets.extend(batch.y.squeeze(-1).cpu().numpy().flatten())
    return np.array(preds), np.array(targets)


# --- K-Fold Cross-Validation ---
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

fold_results = []
all_preds = []
all_targets = []
best_overall_mae = float("inf")
best_model_state = None

print(f"Starting {N_FOLDS}-fold cross-validation...\n")
print(f"{'Fold':<6}{'Train MAE':<14}{'Test MAE':<14}{'Test RMSE':<14}{'Test R2':<10}")
print("-" * 60)

for fold, (train_idx, test_idx) in enumerate(kf.split(graphs)):
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    # Fresh model for each fold
    model = CatalystGNN(
        node_feat_dim=9, hidden_dim=HIDDEN_DIM, dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

    # Evaluate
    train_preds, train_targets = evaluate(model, train_loader)
    test_preds, test_targets = evaluate(model, test_loader)

    train_mae = mean_absolute_error(train_targets, train_preds)
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    test_r2 = r2_score(test_targets, test_preds) if len(test_targets) > 1 else float("nan")

    fold_results.append({
        "fold": fold + 1,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
    })

    all_preds.extend(test_preds)
    all_targets.extend(test_targets)

    print(f"{fold+1:<6}{train_mae:<14.2f}{test_mae:<14.2f}{test_rmse:<14.2f}{test_r2:<10.3f}")

    if test_mae < best_overall_mae:
        best_overall_mae = test_mae
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

# --- Summary ---
print("\n" + "=" * 60)
print("CROSS-VALIDATION SUMMARY")
print("=" * 60)

avg_train_mae = np.mean([r["train_mae"] for r in fold_results])
avg_test_mae = np.mean([r["test_mae"] for r in fold_results])
avg_test_rmse = np.mean([r["test_rmse"] for r in fold_results])
avg_test_r2 = np.mean([r["test_r2"] for r in fold_results])

print(f"Average Train MAE:  {avg_train_mae:.2f} mV")
print(f"Average Test MAE:   {avg_test_mae:.2f} mV")
print(f"Average Test RMSE:  {avg_test_rmse:.2f} mV")
print(f"Average Test R2:    {avg_test_r2:.3f}")

# Overall metrics on all predictions pooled
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
overall_mae = mean_absolute_error(all_targets, all_preds)
overall_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
overall_r2 = r2_score(all_targets, all_preds)

print(f"\nOverall MAE (all folds):  {overall_mae:.2f} mV")
print(f"Overall RMSE (all folds): {overall_rmse:.2f} mV")
print(f"Overall R2 (all folds):   {overall_r2:.3f}")

# --- Save best model ---
if best_model_state is not None:
    torch.save(best_model_state, model_save)
    print(f"\nBest model saved to: {model_save}")
    print(f"Best fold MAE: {best_overall_mae:.2f} mV")

# --- Save predictions for plotting ---
np.save("results/predictions.npy", np.column_stack([all_targets, all_preds]))
np.save("results/fold_results.npy", fold_results, allow_pickle=True)
print(f"Predictions saved to: results/predictions.npy")
print(f"Fold results saved to: results/fold_results.npy")