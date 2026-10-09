"""
CatalystGPT — Streamlit Web Application
AI for High-Entropy Alloy Catalyst Discovery in Green Hydrogen
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import random

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from model_loader import CatalystPredictor, parse_formula_with_stoich

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CatalystGPT",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1F3A6E, #2E86AB, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: 800;
        color: #1F3A6E;
    }
    .prediction-label {
        font-size: 1rem;
        color: #555;
        margin-bottom: 0.5rem;
    }
    .status-good {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .status-ok {
        background: #fff3cd;
        color: #856404;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .status-bad {
        background: #f8d7da;
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .footer {
        text-align: center;
        color: #888;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL (cached)
# ============================================================
@st.cache_resource
def load_predictor():
    return CatalystPredictor("results/models/best_gnn_v4.pt")

predictor = load_predictor()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🧪 CatalystGPT")
    st.markdown("*AI for HEA Catalyst Discovery*")
    st.divider()

    if predictor.loaded:
        st.success("Model loaded ✓")
        st.caption(f"Training samples: 173")
        st.caption(f"Mean onset: {predictor.y_mean:.1f} mV")
        st.caption(f"Std: {predictor.y_std:.1f} mV")
    else:
        st.error(f"Model not loaded: {predictor.load_error}")

    st.divider()
    st.markdown("### 📊 Model Performance")
    st.markdown("""
    - **Test R²**: 0.427
    - **Test MAE**: 49.35 mV
    - **Baseline MAE**: 79.36 mV
    - **Improvement**: 37.8%
    """)

    st.divider()
    st.markdown("### 🔗 Links")
    st.markdown("[📂 GitHub Repo](https://github.com/Harpreet29062005/CatalystGPT)")
    st.caption("Built by Harpreet Thappa")
    st.caption("NIT Srinagar — Chemical Engineering")


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">CatalystGPT</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-Driven Discovery of High-Entropy Alloy Catalysts for Green Hydrogen</div>',
    unsafe_allow_html=True,
)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "🔍 Discover", "📖 About"])


# ============================================================
# TAB 1: PREDICT
# ============================================================
with tab1:
    st.markdown("### Predict HER Activity for a Single Catalyst")
    st.markdown(
        "Enter a High-Entropy Alloy composition to predict its onset potential "
        "for the Hydrogen Evolution Reaction (HER). Lower values = better catalyst."
    )
    st.markdown("**Example valid inputs:** `PtPdNiCoMn` · `PtMoPdRhNi` · `CoFeIrNiPtZn` · `IrPdPtRhRu`")
    st.divider()

    col1, col2 = st.columns([3, 1])
    with col1:
        formula_input = st.text_input(
            "Catalyst composition:",
            value="PtMoPdRhNi",
            help="Format: element symbols concatenated. e.g., PtMoPdRhNi means 5 metals",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔬 Predict", use_container_width=True, type="primary")

    if predict_btn:
        if not predictor.loaded:
            st.error("Model is not loaded. Cannot predict.")
        else:
            with st.spinner("Running prediction..."):
                result = predictor.predict(formula_input.strip())

            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                pred = result["prediction_mV"]

                # Prediction card
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div class="prediction-label">Predicted onset potential</div>
                        <div class="prediction-value">{pred:.2f} mV</div>
                        {f'<div class="status-good">🟢 Excellent — better than Pt/C (~30 mV)</div>'
                         if pred < 30 else
                         f'<div class="status-ok">🟡 Good — competitive with Pt/C (~30 mV)</div>'
                         if pred < 60 else
                         f'<div class="status-bad">🔴 Weak — above Pt/C benchmark</div>'}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Element breakdown
                st.markdown("### Element Breakdown")
                parsed = parse_formula_with_stoich(formula_input.strip())
                total = sum(n for _, n in parsed)
                from pymatgen.core import Element

                rows = []
                for sym, n in parsed:
                    try:
                        el = Element(sym)
                        rows.append({
                            "Element": sym,
                            "Fraction": f"{100 * n / total:.1f}%",
                            "Atomic number": int(el.Z),
                            "Electronegativity": round(float(el.X), 2) if el.X else "—",
                            "Atomic radius (Å)": round(float(el.atomic_radius), 2) if el.atomic_radius else "—",
                            "Group": int(el.group) if el.group else "—",
                            "Period": int(el.row) if el.row else "—",
                        })
                    except Exception:
                        rows.append({"Element": sym, "Fraction": f"{100*n/total:.1f}%"})

                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Benchmark comparison
                st.markdown("### Benchmark Comparison")
                benchmarks = {
                    "Pt/C (commercial)": 30.0,
                    "IrO₂ (reference)": 45.0,
                    "This prediction": pred,
                }
                bench_df = pd.DataFrame({
                    "Catalyst": list(benchmarks.keys()),
                    "Onset potential (mV)": list(benchmarks.values()),
                })
                st.bar_chart(bench_df.set_index("Catalyst"), color="#2E86AB")


# ============================================================
# TAB 2: DISCOVER
# ============================================================
with tab2:
    st.markdown("### Discover New Candidate Catalysts")
    st.markdown(
        "The AI generates random HEA compositions from a library of metals and "
        "predicts their performance. It then ranks them to find the most promising candidates."
    )
    st.divider()

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        n_candidates = st.slider("Number of candidates to generate", 100, 5000, 1000, 100)
    with col2:
        elements_pool = st.multiselect(
            "Elements to include:",
            ["Pt", "Ir", "Pd", "Rh", "Ru", "Ni", "Co", "Fe", "Cu", "Mn", "Cr", "Mo", "W", "Zn", "Ga", "Sn", "Al", "V"],
            default=["Pt", "Pd", "Rh", "Ir", "Ni", "Co", "Fe", "Cu"],
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        discover_btn = st.button("🚀 Discover", use_container_width=True, type="primary")

    if discover_btn:
        if not predictor.loaded:
            st.error("Model is not loaded.")
        elif len(elements_pool) < 3:
            st.warning("Please select at least 3 elements.")
        else:
            with st.spinner(f"Generating and screening {n_candidates} candidates..."):
                results = []
                progress = st.progress(0)

                for i in range(n_candidates):
                    n_elems = random.choice([4, 5, 5, 5, 6])
                    chosen = random.sample(elements_pool, min(n_elems, len(elements_pool)))
                    # Random stoichiometry
                    weights = [random.uniform(0.5, 2.0) for _ in chosen]
                    formula = "".join(f"{e}{int(w*10)}" for e, w in zip(chosen, weights))

                    result = predictor.predict(formula)
                    if "error" not in result:
                        results.append({
                            "formula": formula,
                            "prediction_mV": result["prediction_mV"],
                            "n_elements": len(chosen),
                        })

                    if (i + 1) % 50 == 0:
                        progress.progress((i + 1) / n_candidates)

                progress.empty()

            if not results:
                st.error("No valid predictions generated. Try different elements.")
            else:
                df = pd.DataFrame(results).sort_values("prediction_mV").reset_index(drop=True)
                df.index += 1

                # Filter to HEA (3+ elements)
                df = df[df["n_elements"] >= 3]

                st.success(f"✅ Generated and screened {len(df)} valid candidates")
                st.markdown("### 🏆 Top 20 Predicted Catalysts")

                top20 = df.head(20)
                st.dataframe(
                    top20.rename(columns={
                        "formula": "Catalyst Formula",
                        "prediction_mV": "Predicted Onset (mV)",
                        "n_elements": "# Elements",
                    }).style.format({"Predicted Onset (mV)": "{:.2f}"}),
                    use_container_width=True,
                )

                # Download
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download all results as CSV",
                    data=csv,
                    file_name="catalystgpt_discoveries.csv",
                    mime="text/csv",
                )

                # Statistics
                st.markdown("### 📊 Distribution")
                st.caption(
                    f"Mean: {df['prediction_mV'].mean():.1f} mV · "
                    f"Median: {df['prediction_mV'].median():.1f} mV · "
                    f"Best: {df['prediction_mV'].min():.1f} mV"
                )
                st.bar_chart(df.head(30).set_index("formula")["prediction_mV"])


# ============================================================
# TAB 3: ABOUT
# ============================================================
with tab3:
    st.markdown("### About CatalystGPT")
    st.markdown("""
    **CatalystGPT** is an end-to-end computational pipeline for the discovery of 
    **High-Entropy Alloy (HEA) catalysts** for the **Hydrogen Evolution Reaction (HER)** — 
    a critical reaction in green hydrogen production.

    #### 🔬 Pipeline Architecture

    1. **Literature Mining** — NLP pipeline extracts catalyst data from 75 scientific papers
    2. **Dataset Curation** — Combines with a public 180-catalyst curated dataset
    3. **Graph Representation** — Each catalyst is a graph: nodes = metal atoms (10 atomic features), 
       edges = pairwise interactions (3 features including composition-weighted average)
    4. **EdgeGNN Model** — Edge-conditioned Graph Neural Network using NNConv layers
    5. **Rigorous Evaluation** — 5-fold cross-validation with nested early stopping
    """)

    st.divider()

    st.markdown("### 📊 Model Performance")
    metrics_df = pd.DataFrame({
        "Model": ["Baseline (mean)", "Random Forest", "EdgeGNN (this app)"],
        "MAE (mV)": [79.36, 45.56, 49.35],
        "RMSE (mV)": [95.63, 70.71, 71.99],
        "R²": ["-0.011", "0.447", "0.427"],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("### ⚠️ Honest Limitations")
    st.markdown("""
    - **Small dataset:** 173 catalysts is a proof-of-concept size; real ML studies use 1,000–100,000+
    - **Composition-only features:** Crystal structure and experimental conditions not included
    - **Onset potential proxy:** Different from overpotential @ 10 mA/cm²
    - **No experimental validation:** All results are computational
    """)

    st.divider()

    st.markdown("### 👤 Author")
    st.markdown("""
    **Harpreet Thappa**  
    Chemical Engineering, 5th Semester  
    National Institute of Technology (NIT) Srinagar  
    📧 happythappa5@gmail.com
    """)

    st.divider()

    st.markdown("### 🔗 Links")
    st.markdown("[GitHub Repository](https://github.com/Harpreet29062005/CatalystGPT)")


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer">'
    'Built with ❤️ using Streamlit + PyTorch Geometric · '
    'Harpreet Thappa, NIT Srinagar · 2026'
    '</div>',
    unsafe_allow_html=True,
)