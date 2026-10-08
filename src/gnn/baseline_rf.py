import pandas as pd
import numpy as np
import os
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pymatgen.core import Element

# --- Reproducibility ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# --- Load ---
df = pd.read_csv("data/processed/clean_dataset_v2.csv")
print(f"Loaded {len(df)} catalysts\n")

METALS = ["Pt", "Mo", "Pd", "Rh", "Ni", "Fe", "Co", "Mn", "Cr", "Cu",
          "Ru", "Ir", "Au", "Ag", "Ti", "V", "Zn", "Sn", "W", "Nb",
          "Ga", "In", "Ta", "Zr", "Hf", "Re", "Os", "Cd", "Al", "Mg"]

import re

def parse_formula(formula):
    return re.findall(r"[A-Z][a-z]?", formula)

def safe(x, d):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return d
        return float(x)
    except Exception:
        return d

def features_from_formula(formula):
    elements = list(set(parse_formula(formula)))
    feats = {}

    # 1. One-hot presence of each metal
    for m in METALS:
        feats[f"has_{m}"] = 1 if m in elements else 0

    # 2. Number of elements
    feats["n_elements"] = len(elements)

    # 3. Mean / max / min of atomic properties
    props = {"Z": [], "radius": [], "X": [], "ionization": [], "affinity": [], "melt": [], "group": []}
    for el in elements:
        try:
            e = Element(el)
            props["Z"].append(float(e.Z))
            props["radius"].append(safe(e.atomic_radius, 1.5))
            props["X"].append(safe(e.X, 2.0))
            props["ionization"].append(safe(e.ionization_energy, 0))
            props["affinity"].append(safe(e.electron_affinity, 0))
            props["melt"].append(safe(e.melting_point, 1000))
            props["group"].append(float(e.group) if e.group else 8.0)
        except Exception:
            continue

    for name, vals in props.items():
        if len(vals) == 0:
            vals = [0]
        feats[f"{name}_mean"] = np.mean(vals)
        feats[f"{name}_max"] = np.max(vals)
        feats[f"{name}_min"] = np.min(vals)
        feats[f"{name}_std"] = np.std(vals)

    return feats


# --- Build feature matrix ---
rows = []
y = []
formulas = []

for _, row in df.iterrows():
    f = str(row["clean_formula"])
    if pd.isna(row["overpotential_mV"]):
        continue
    rows.append(features_from_formula(f))
    y.append(float(row["overpotential_mV"]))
    formulas.append(f)

X = pd.DataFrame(rows).fillna(0)
y = np.array(y)

print(f"Feature matrix: {X.shape}")
print(f"Target: {len(y)} values, mean={y.mean():.2f}, std={y.std():.2f}\n")


# --- LEAVE-ONE-OUT Cross Validation ---
loo = LeaveOneOut()
preds = np.zeros(len(y))

# Suppress verbose training
import warnings
warnings.filterwarnings("ignore")

for i, (train_idx, test_idx) in enumerate(loo.split(X)):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X.iloc[train_idx], y[train_idx])
    preds[test_idx[0]] = model.predict(X.iloc[test_idx])[0]

# --- Baseline (predict train mean) ---
baseline_preds = np.zeros(len(y))
for i, (train_idx, test_idx) in enumerate(loo.split(X)):
    baseline_preds[test_idx[0]] = y[train_idx].mean()

# --- Metrics ---
rf_mae = mean_absolute_error(y, preds)
rf_rmse = np.sqrt(mean_squared_error(y, preds))
rf_r2 = r2_score(y, preds)

base_mae = mean_absolute_error(y, baseline_preds)
base_rmse = np.sqrt(mean_squared_error(y, baseline_preds))

print("=" * 65)
print("RESULTS — Leave-One-Out Cross-Validation")
print("=" * 65)
print(f"{'Model':<25}{'MAE (mV)':<15}{'RMSE (mV)':<15}{'R2':<10}")
print("-" * 65)
print(f"{'Baseline (mean)':<25}{base_mae:<15.2f}{base_rmse:<15.2f}{'N/A':<10}")
print(f"{'Random Forest':<25}{rf_mae:<15.2f}{rf_rmse:<15.2f}{rf_r2:<10.3f}")

improvement = ((base_mae - rf_mae) / base_mae) * 100
print(f"\nRandom Forest improvement over baseline: {improvement:.1f}%")

# --- Feature importance ---
model_full = RandomForestRegressor(
    n_estimators=300, max_depth=6, min_samples_leaf=2,
    random_state=SEED, n_jobs=-1,
)
model_full.fit(X, y)
importances = pd.Series(model_full.feature_importances_, index=X.columns)
top = importances.nlargest(15)

print(f"\nTop 15 most important features:")
for feat, val in top.items():
    print(f"  {feat:<25} {val:.4f}")

# --- Sample predictions ---
print(f"\nSample predictions (target vs predicted):")
for i in range(min(10, len(y))):
    print(f"  {formulas[i]:<20} target={y[i]:>6.1f} mV   predicted={preds[i]:>6.1f} mV")

# --- Save ---
os.makedirs("results", exist_ok=True)
np.save("results/rf_predictions.npy", np.column_stack([y, preds, baseline_preds]))
print(f"\nPredictions saved to: results/rf_predictions.npy")