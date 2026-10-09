from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

# --- Output path ---
output_dir = "docs/report"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "CatalystGPT_Paper.docx")

doc = Document()

# =============================================================
# BASE STYLES
# =============================================================
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(11)
style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
style.paragraph_format.space_after = Pt(6)


def set_heading(text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x6E)
    return p


def add_paragraph(text, italic=False, bold=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = italic
    run.bold = bold
    return p


def add_bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
        p.add_run(text)
    else:
        p.add_run(text)
    return p


def add_table(headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.name = "Times New Roman"
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for row in rows:
        cells = table.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            r = p.add_run(str(v))
            r.font.name = "Times New Roman"
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    return table


# =============================================================
# TITLE PAGE
# =============================================================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("CatalystGPT: An AI-Driven Pipeline for\nHigh-Entropy Alloy Catalyst Discovery\nin Green Hydrogen Production")
run.bold = True
run.font.size = Pt(18)
run.font.name = "Times New Roman"

doc.add_paragraph()

author = doc.add_paragraph()
author.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = author.add_run("Harpreet Thappa")
run.bold = True
run.font.size = Pt(13)

affil = doc.add_paragraph()
affil.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = affil.add_run("Department of Chemical Engineering\nNational Institute of Technology Srinagar, India\nhappythappa5@gmail.com")
run.font.size = Pt(11)

doc.add_paragraph()
date = doc.add_paragraph()
date.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = date.add_run("Submitted: October 2026")
run.italic = True

doc.add_page_break()

# =============================================================
# ABSTRACT
# =============================================================
set_heading("Abstract", level=1)

add_paragraph(
    "The discovery of efficient, low-cost catalysts for the Hydrogen Evolution Reaction (HER) is "
    "critical for scalable green hydrogen production. High-Entropy Alloys (HEAs) — alloys containing "
    "five or more principal elements — have emerged as promising HER catalysts due to their tunable "
    "composition and multi-site activity. However, the vast compositional space of HEAs (on the order "
    "of 10^10 possible combinations) makes exhaustive experimental screening infeasible. In this work, "
    "we present CatalystGPT, an end-to-end computational pipeline that (i) automatically mines "
    "scientific literature using natural language processing (NLP) to extract HEA-HER performance data, "
    "and (ii) trains a composition-aware Edge-conditioned Graph Neural Network (EdgeGNN) to predict "
    "HER onset potential directly from alloy composition. We curate a dataset of 173 unique HEA "
    "catalysts and evaluate our model using rigorous 5-fold cross-validation with nested early stopping. "
    "Our EdgeGNN achieves a mean absolute error (MAE) of 49.35 mV and R^2 of 0.427, representing a "
    "37.8% reduction in MAE over the trivial mean baseline (79.36 mV). A Random Forest baseline "
    "achieves comparable performance (MAE = 45.56 mV, R^2 = 0.447), demonstrating that composition-only "
    "features capture substantial HER activity trends."
)

p = doc.add_paragraph()
r = p.add_run("Keywords: ")
r.bold = True
p.add_run("High-Entropy Alloys, Hydrogen Evolution Reaction, Graph Neural Networks, "
          "Materials Discovery, Natural Language Processing, Green Hydrogen")

doc.add_paragraph()

# =============================================================
# 1. INTRODUCTION
# =============================================================
set_heading("1. Introduction", level=1)

set_heading("1.1 Motivation", level=2)
add_paragraph(
    "Hydrogen produced via water electrolysis powered by renewable energy — commonly called green "
    "hydrogen — is a cornerstone of global decarbonization strategies. The Hydrogen Evolution Reaction "
    "(HER) is the cathodic half-reaction in water electrolysis. Its efficiency is limited by the sluggish "
    "kinetics of the catalyst. Platinum-group metals (Pt, Ir) are the state-of-the-art HER catalysts, but "
    "their scarcity and high cost prohibit large-scale deployment."
)
add_paragraph(
    "High-Entropy Alloys (HEAs) — solid solutions containing five or more principal metallic elements in "
    "near-equimolar ratios — have emerged as promising alternatives. Their unique properties include:"
)
add_bullet(" Continuous variation across a vast compositional space.", "Tunable composition:")
add_bullet(" Different metal sites can catalyze different elementary steps.", "Multi-site synergy:")
add_bullet(" Entropy stabilization resists phase segregation.", "Enhanced stability:")
add_paragraph(
    "However, the HEA compositional space is astronomically large. Even with a modest set of 20 metals "
    "in 5-element alloys, the number of possible combinations exceeds 15,504. Systematic experimental "
    "exploration is intractable."
)

set_heading("1.2 Contributions", level=2)
add_paragraph("We present CatalystGPT, a computational pipeline with the following contributions:")
add_bullet(" An NLP-based literature mining pipeline that extracts catalyst compositions, overpotentials, "
           "and Tafel slopes from PDFs of scientific papers, using spaCy and pdfplumber.", "1.")
add_bullet(" A composition-aware graph representation of HEA catalysts, encoding stoichiometric "
           "fractions and pairwise metal-metal interaction features.", "2.")
add_bullet(" An Edge-conditioned Graph Neural Network (EdgeGNN) that uses NNConv layers to learn from "
           "both node (atomic) and edge (pairwise) features.", "3.")
add_bullet(" Honest, reproducible evaluation using 5-fold cross-validation with nested early stopping "
           "on 173 curated catalysts, benchmarked against mean baseline and Random Forest.", "4.")
add_bullet(" Full open-source release of code, datasets, and trained models.", "5.")

set_heading("1.3 Key Finding", level=2)
add_paragraph(
    "Our EdgeGNN achieves comparable performance to Random Forest (R^2 = 0.427 vs. 0.447) using only "
    "composition information, demonstrating that graph-based deep learning can capture HER activity "
    "trends from HEA composition alone."
)

# =============================================================
# 2. RELATED WORK
# =============================================================
set_heading("2. Related Work", level=1)

set_heading("2.1 Machine Learning for HER Catalysts", level=2)
add_paragraph(
    "Recent work has applied ML to catalyst discovery across multiple material classes. Gradient boosting "
    "models (XGBoost, CatBoost) have been used to predict HER overpotential from elemental descriptors. "
    "Graph Neural Networks (GNNs) have shown particular promise for crystalline materials, where atomic "
    "connectivity is naturally represented as a graph."
)

set_heading("2.2 HEA Catalyst Datasets", level=2)
add_paragraph(
    "The recent Scientific Data dataset by Gorsse et al. (2026) compiled 180 multinary alloy HER "
    "catalysts from the literature with harmonized metadata — the first large-scale open dataset "
    "specifically for composition-only HEA modeling. We use this dataset as our primary training set."
)

set_heading("2.3 Literature Mining for Materials", level=2)
add_paragraph(
    "NLP-based extraction from scientific literature has gained traction as a way to build large "
    "materials datasets automatically. Prior work has focused on named entity recognition of chemical "
    "formulas and property extraction from tables. Our pipeline builds on these ideas and additionally "
    "handles composition-weighted graph construction downstream."
)

# =============================================================
# 3. METHODOLOGY
# =============================================================
set_heading("3. Methodology", level=1)

set_heading("3.1 Data Collection", level=2)
add_paragraph(
    "We combine two data sources: (1) the public curated dataset from Gorsse et al. (2026) containing "
    "180 HEA HER catalysts with onset potential and Tafel slope, and (2) a custom NLP-extracted "
    "dataset from 75 scientific papers, from which 23 clean catalysts were extracted after filtering."
)

set_heading("3.2 Dataset Curation", level=2)
add_paragraph("Combining the two sources and applying filters (3+ metal elements, onset potential in "
              "[5, 400] mV, deduplication) resulted in a final dataset of 173 unique HEA catalysts.")

set_heading("3.3 Graph Representation", level=2)
add_paragraph("Each catalyst is represented as a fully-connected directed graph:")
add_bullet(" One per unique metal element, with 10 features: atomic number, atomic radius, "
           "electronegativity, group, period, electron affinity, ionization energy, melting point, "
           "density, and composition fraction.", "Nodes:")
add_bullet(" All ordered pairs, with 3 features: electronegativity difference, atomic radius "
           "difference, and composition-weighted average.", "Edges:")

set_heading("3.4 EdgeGNN Architecture", level=2)
add_paragraph("The model consists of:")
add_bullet(" 3 NNConv layers with hidden dimension 64, using edge features to generate dynamic "
           "weight matrices.", "Convolutions:")
add_bullet(" After each convolution, followed by ReLU and dropout (0.2).", "Batch normalization:")
add_bullet(" Mean + max pooling concatenated into a 128-dim graph embedding.", "Pooling:")
add_bullet(" 3-layer MLP (128 → 64 → 32 → 1).", "MLP head:")
add_paragraph("Total parameters: approximately 311,000.")

set_heading("3.5 Training Protocol", level=2)
add_bullet(" 5-fold cross-validation (outer split).")
add_bullet(" Nested inner validation (15% of training set) for early stopping.")
add_bullet(" Normalization computed only from training fold statistics (no leakage).")
add_bullet(" SmoothL1Loss as the training objective.")
add_bullet(" Adam optimizer with learning rate 3e-4, ReduceLROnPlateau scheduler.")
add_bullet(" Gradient clipping at norm 1.0; early stopping with patience of 60 epochs.")

set_heading("3.6 Baselines", level=2)
add_bullet(" Predicts the training-set mean onset potential for every catalyst.", "Mean predictor:")
add_bullet(" 300 trees, max depth 10, trained on 90+ elemental descriptors.", "Random Forest:")

# =============================================================
# 4. RESULTS
# =============================================================
set_heading("4. Results", level=1)

set_heading("4.1 Overall Performance", level=2)
add_paragraph("All three models were evaluated using identical 5-fold cross-validation splits:")
add_table(
    ["Model", "MAE (mV)", "RMSE (mV)", "R²"],
    [
        ["Baseline (mean)", "79.36", "95.63", "-0.011"],
        ["Random Forest", "45.56", "70.71", "0.447"],
        ["EdgeGNN (proposed)", "49.35", "71.99", "0.427"],
    ],
)

set_heading("4.2 Per-Fold EdgeGNN Performance", level=2)
add_table(
    ["Fold", "Test MAE (mV)"],
    [
        ["1", "35.51"],
        ["2", "58.86"],
        ["3", "41.92"],
        ["4", "60.00"],
        ["5", "50.81"],
        ["Mean ± std", "49.42 ± 9.51"],
    ],
)

set_heading("4.3 Comparison with Baselines", level=2)
add_bullet(" The EdgeGNN reduces MAE by 37.8% (79.36 → 49.35 mV).",
           "vs. Mean baseline:")
add_bullet(" The EdgeGNN achieves 8.3% higher MAE but nearly identical R² "
           "(0.427 vs. 0.447).", "vs. Random Forest:")

set_heading("4.4 Feature Importance", level=2)
add_paragraph("Random Forest feature importance analysis identified the top descriptors:")
add_bullet(" Minimum ionization energy")
add_bullet(" Minimum melting point")
add_bullet(" Mean ionization energy")
add_bullet(" Ionization energy standard deviation")
add_bullet(" Electron affinity standard deviation")
add_paragraph(
    "These results suggest that electronic structure heterogeneity — captured by dispersion of "
    "ionization energies and electron affinities — is a dominant driver of HER activity in HEAs."
)

# --- Insert figures ---
set_heading("4.5 Figures", level=2)

figures = [
    ("results/figures/parity_plots.png",
     "Figure 1. Parity plots showing predicted vs actual onset potential for baseline, Random Forest, and EdgeGNN."),
    ("results/figures/residuals.png",
     "Figure 2. EdgeGNN residuals plotted against actual values (left) and histogram of residual distribution (right)."),
    ("results/figures/feature_importance.png",
     "Figure 3. Top 15 features ranked by Random Forest importance for HER activity prediction."),
]

for path, caption in figures:
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(caption)
        r.italic = True
        r.font.size = Pt(10)
    else:
        add_paragraph(f"[Figure not found: {path}]", italic=True)

# =============================================================
# 5. DISCUSSION
# =============================================================
set_heading("5. Discussion", level=1)

set_heading("5.1 Interpretation", level=2)
add_paragraph(
    "Both Random Forest and EdgeGNN substantially outperform the trivial mean baseline, confirming "
    "that HEA composition carries predictive signal for HER activity. The near-equivalent performance "
    "of the two models suggests that: (1) for datasets of this size (~173 samples), simple models can "
    "match complex ones; (2) the graph representation provides no additional information beyond what "
    "tabular compositional features capture; (3) compositional heterogeneity (e.g., ionization energy "
    "spread) is more predictive than pairwise metal interactions."
)

set_heading("5.2 Why the GNN Does Not Outperform RF", level=2)
add_bullet(" 173 samples is 10–100× smaller than typical GNN training sets.", "Small dataset:")
add_bullet(" Without crystal structure, all metals are equally connected. This provides no "
           "connectivity information.", "Fully-connected graph:")
add_bullet(" Crystal structure, morphology, and experimental conditions are not encoded.",
           "Composition-only features:")

set_heading("5.3 The NLP Pipeline", level=2)
add_paragraph(
    "While our custom NLP extraction produced only 23 catalysts (vs. 180 in the curated dataset), the "
    "pipeline itself is a reproducible, extensible tool for automated materials literature mining. With "
    "improved extraction accuracy and a larger corpus, this approach can scale to thousands of papers."
)

# =============================================================
# 6. LIMITATIONS
# =============================================================
set_heading("6. Limitations", level=1)
add_paragraph("We explicitly acknowledge the following limitations:")
add_bullet(" 173 catalysts is insufficient for confident deep learning.", "Small dataset:")
add_bullet(" Real HER activity depends on crystal structure, morphology, and conditions.",
           "Composition-only modeling:")
add_bullet(" The onset potential target differs from the more common overpotential at "
           "10 mA/cm².", "Onset potential proxy:")
add_bullet(" The curated dataset and our NLP-extracted dataset differ in measurement "
           "conditions.", "Heterogeneous data sources:")
add_bullet(" All results are computational; experimental confirmation is future work.",
           "No experimental validation:")

# =============================================================
# 7. CONCLUSION
# =============================================================
set_heading("7. Conclusion and Future Work", level=1)
add_paragraph(
    "We presented CatalystGPT, an end-to-end pipeline for HEA catalyst discovery that combines "
    "NLP-based literature mining with a composition-aware Graph Neural Network. On 173 curated HEA "
    "catalysts, our EdgeGNN achieves R² = 0.427, competitive with Random Forest (R² = 0.447) and "
    "substantially better than the mean baseline."
)

set_heading("Future Directions", level=2)
add_bullet(" Scale the dataset to 1,000+ catalysts by improving NLP extraction.", "1.")
add_bullet(" Add structural features (crystal structure, coordination number, d-band center).", "2.")
add_bullet(" Multi-target learning (predict overpotential, Tafel slope, and stability).", "3.")
add_bullet(" Experimental validation of top-predicted catalysts.", "4.")
add_bullet(" Deploy a web interface for the research community.", "5.")

# =============================================================
# REFERENCES
# =============================================================
doc.add_page_break()
set_heading("References", level=1)

refs = [
    "Gorsse, S., Tang, B., Tang, Y., Ma, M., & Liu, Z. (2026). Curated dataset of multinary alloy HER catalysts for composition-only modelling with Magpie descriptors and GP baselines. Scientific Data.",
    "George, E. P., Raabe, D., & Ritchie, R. O. (2019). High-entropy alloys. Nature Reviews Materials, 4(8), 515–534.",
    "Sanchez-Lengeling, B., & Aspuru-Guzik, A. (2018). Inverse molecular design using machine learning. Science, 361(6400), 360–365.",
    "Zheng, Y., Jiao, Y., Qiao, S. Z., et al. (2016). Hydrogen evolution reaction in alkaline media. Angewandte Chemie, 55(52), 16081–16083.",
    "Xie, T., & Grossman, J. C. (2018). Crystal Graph Convolutional Neural Networks for an accurate and interpretable prediction of material properties. Physical Review Letters, 120(14), 145301.",
    "Hu, W., et al. (2021). Open Graph Benchmark: Datasets for machine learning on graphs. NeurIPS.",
    "Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural Message Passing for Quantum Chemistry. ICML.",
    "Belsky, A., Hellenbrandt, M., Karen, V. L., & Luksch, P. (2002). New developments in the Inorganic Crystal Structure Database. Acta Crystallographica Section B, 58(3), 364–369.",
]

for i, r in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    p.add_run(f"[{i}] {r}")

# =============================================================
# APPENDICES
# =============================================================
doc.add_page_break()
set_heading("Appendix A: Reproducibility", level=1)
add_paragraph("All code is available at: https://github.com/Harpreet29062005/CatalystGPT")
add_paragraph("Environment: Python 3.13, PyTorch 2.14, PyTorch Geometric 2.8, pymatgen 2026.9")
add_paragraph("To reproduce main results, run:")
p = doc.add_paragraph()
p.paragraph_format.left_indent = Inches(0.4)
r = p.add_run("python src/gnn/build_graphs_v4.py\n"
              "set PYTHONPATH=%CD%\\src\\gnn\n"
              "python src/gnn/train_v4.py\n"
              "python src/gnn/make_plots_v2.py")
r.font.name = "Consolas"

set_heading("Appendix B: Dataset Statistics", level=1)
add_bullet(" 173 catalysts", "Total:")
add_bullet(" 5.00 – 385.00 mV", "Onset potential range:")
add_bullet(" 103.62 mV", "Onset potential mean:")
add_bullet(" 120.83 mV", "Onset potential std:")
add_paragraph("Element count distribution:")
add_table(
    ["Element Count", "Number of Catalysts"],
    [
        ["3 elements", "15"],
        ["4 elements", "27"],
        ["5 elements", "115 (true HEA)"],
        ["6 elements", "16"],
        ["7+ elements", "4"],
    ],
)

# =============================================================
# SAVE
# =============================================================
doc.save(output_path)
print(f"\nDocument created: {output_path}")
print(f"Total paragraphs: {len(doc.paragraphs)}")
print(f"Total tables: {len(doc.tables)}")
print(f"Total figures embedded: {sum(1 for f, _ in figures if os.path.exists(f))}")