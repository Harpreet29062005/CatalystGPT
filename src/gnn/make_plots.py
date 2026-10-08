import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from pymatgen.core import Element as PmgElement
import re
import torch

# --- Paths ---
preds_file = "results/predictions_v4.npy"
graphs_file = "data/processed/graphs_v4.pt"
output_folder = "results/figures"
os.makedirs(output_folder, exist_ok=True)

# --- Load predictions ---
data = np.load(preds_file)
gnn_targets = data[:, 0]
gnn_preds = data[:, 1]
base_preds = data[:, 2]
rf_preds = data[:, 3]

print(f"Loaded {len(gnn_targets)} predictions")
print(f"Target range: {gnn_targets.min():.1f} – {gnn_targets.max():.1f} mV\n")

# --- Metrics ---
def metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<20} MAE={mae:6.2f}  RMSE={rmse:6.2f}  R2={r2:6.3f}")
    return mae, rmse, r2

print("=" * 55)
print("FINAL METRICS")
print("=" * 55)
base_metrics = metrics(gnn_targets, base_preds, "Baseline (mean)")
rf_metrics = metrics(gnn_targets, rf_preds, "Random Forest")
gnn_metrics = metrics(gnn_targets, gnn_preds, "EdgeGNN")
print("=" * 55)
print()

# =============================================================
# PLOT 1: PARITY PLOTS
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models = [
    ("Baseline (mean)", base_preds, base_metrics, "#888888"),
    ("Random Forest", rf_preds, rf_metrics, "#2E86AB"),
    ("EdgeGNN (proposed)", gnn_preds, gnn_metrics, "#D62246"),
]

for ax, (name, preds, (mae, rmse, r2), color) in zip(axes, models):
    ax.scatter(gnn_targets, preds, alpha=0.5, s=25, color=color, edgecolor="k", linewidth=0.3)

    lo = min(gnn_targets.min(), preds.min()) - 10
    hi = max(gnn_targets.max(), preds.max()) + 10
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="Perfect")

    ax.set_xlabel("Actual onset potential (mV)", fontsize=11)
    ax.set_ylabel("Predicted onset potential (mV)", fontsize=11)
    ax.set_title(f"{name}\nMAE={mae:.1f} mV | R²={r2:.3f}", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(loc="upper left", fontsize=9)

plt.suptitle("CatalystGPT: Predicted vs Actual HER Onset Potential", fontsize=13, y=1.02)
plt.tight_layout()
parity_path = os.path.join(output_folder, "parity_plots.png")
plt.savefig(parity_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {parity_path}")


# =============================================================
# PLOT 2: RESIDUAL PLOT for GNN
# =============================================================
residuals = gnn_preds - gnn_targets

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Residual scatter
ax = axes[0]
ax.scatter(gnn_targets, residuals, alpha=0.5, s=25, color="#D62246", edgecolor="k", linewidth=0.3)
ax.axhline(0, color="k", linestyle="--", linewidth=1)
ax.set_xlabel("Actual onset potential (mV)", fontsize=11)
ax.set_ylabel("Residual (predicted − actual) (mV)", fontsize=11)
ax.set_title("EdgeGNN Residuals vs Actual", fontsize=12)
ax.grid(True, alpha=0.3)

# Residual histogram
ax = axes[1]
ax.hist(residuals, bins=30, color="#D62246", alpha=0.7, edgecolor="black")
ax.axvline(0, color="k", linestyle="--", linewidth=1)
ax.axvline(residuals.mean(), color="blue", linestyle="-", linewidth=1, label=f"Mean = {residuals.mean():.1f} mV")
ax.set_xlabel("Residual (mV)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("EdgeGNN Residual Distribution", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
residual_path = os.path.join(output_folder, "residuals.png")
plt.savefig(residual_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {residual_path}")


# =============================================================
# PLOT 3: FEATURE IMPORTANCE (from Random Forest)
# =============================================================
print("\nComputing feature importance from Random Forest...")

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

# Load graphs for formulas
graphs = torch.load(graphs_file, weights_only=False)
graphs = [g for g in graphs if g.y.item() <= 400]

feature_dicts = [feats(g.formula) for g in graphs]
feature_names = list(feature_dicts[0].keys())
X_rf = np.array([[fd[k] for k in feature_names] for fd in feature_dicts])
y_rf = np.array([g.y.item() for g in graphs])

rf_full = RandomForestRegressor(
    n_estimators=500, max_depth=10, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
rf_full.fit(X_rf, y_rf)

importances = rf_full.feature_importances_
top_idx = np.argsort(importances)[-15:][::-1]
top_features = [feature_names[i] for i in top_idx]
top_values = [importances[i] for i in top_idx]

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
importance_path = os.path.join(output_folder, "feature_importance.png")
plt.savefig(importance_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {importance_path}")


# =============================================================
# SUMMARY TABLE (text file for reference)
# =============================================================
summary_path = os.path.join(output_folder, "metrics_summary.txt")
with open(summary_path, "w") as f:
    f.write("CatalystGPT — Final Metrics Summary\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"{'Model':<25}{'MAE':<12}{'RMSE':<12}{'R²':<10}\n")
    f.write("-" * 55 + "\n")
    f.write(f"{'Baseline (mean)':<25}{base_metrics[0]:<12.2f}{base_metrics[1]:<12.2f}{'N/A':<10}\n")
    f.write(f"{'Random Forest':<25}{rf_metrics[0]:<12.2f}{rf_metrics[1]:<12.2f}{rf_metrics[2]:<10.3f}\n")
    f.write(f"{'EdgeGNN (proposed)':<25}{gnn_metrics[0]:<12.2f}{gnn_metrics[1]:<12.2f}{gnn_metrics[2]:<10.3f}\n")
    f.write("\n")
    f.write("Sample count: 173 catalysts (after removing 4 outliers >400 mV)\n")
    f.write("Evaluation: 5-fold cross-validation, nested early stopping\n")
    f.write("Feature importance (top 5):\n")
    for i in range(5):
        f.write(f"  {i+1}. {top_features[i]} ({top_values[i]:.4f})\n")

print(f"Saved: {summary_path}")
print(f"\nAll plots saved to: {output_folder}/")
print("\nAnalysis complete!")