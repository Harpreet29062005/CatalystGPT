# 🚀 Live Demo

**Try it now:** [**catalystgpt.streamlit.app**](https://catalystgpt.streamlit.app)

[![Live App](https://img.shields.io/badge/Live%20App-catalystgpt.streamlit.app-success?style=for-the-badge&logo=streamlit)](https://catalystgpt.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-CatalystGPT-black?style=for-the-badge&logo=github)](https://github.com/Harpreet29062005/CatalystGPT)

**Enter a catalyst composition → get a predicted HER onset potential in seconds.**

Try: `PtMoPdRhNi` · `CoFeIrNiPtZn` · `IrPdPtRhRu`

---

\# CatalystGPT



\### AI-Driven Discovery of High-Entropy Alloy Catalysts for Green Hydrogen



\[!\[Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)

\[!\[PyTorch](https://img.shields.io/badge/PyTorch-2.14-red.svg)](https://pytorch.org/)

\[!\[PyG](https://img.shields.io/badge/PyTorch\_Geometric-2.8-orange.svg)](https://pyg.org/)

\[!\[License](https://img.shields.io/badge/License-Academic-green.svg)]()



\---



\## 🎯 Overview



CatalystGPT is an end-to-end computational pipeline that accelerates the discovery of \*\*High-Entropy Alloy (HEA) catalysts\*\* for the \*\*Hydrogen Evolution Reaction (HER)\*\* — a key reaction in green hydrogen production.



The project combines:

1\. \*\*Automated literature mining\*\* (NLP) to extract catalyst data from scientific papers

2\. \*\*Graph Neural Networks\*\* to predict catalytic activity from composition alone

3\. \*\*Rigorous, honest evaluation\*\* against multiple baselines



\---



\## 📊 Key Results



Trained and evaluated on \*\*173 unique HEA catalysts\*\* using 5-fold cross-validation with no data leakage.



| Model | MAE (mV) | RMSE (mV) | R² |

|-------|----------|-----------|-----|

| Baseline (predict mean) | 79.36 | 95.63 | −0.011 |

| Random Forest | \*\*45.56\*\* | \*\*70.71\*\* | \*\*0.447\*\* |

| EdgeGNN (proposed) | 49.35 | 71.99 | 0.427 |



\*\*Key finding:\*\* Both Random Forest and the EdgeGNN substantially outperform the trivial baseline (37–43% reduction in MAE). The EdgeGNN achieves comparable performance to Random Forest using only composition information — no experimental features required.



\---



\## 🔬 Pipeline Architecture

+-------------------------------------------------------------+

| 1. LITERATURE MINING (NLP) |

| - 75 scientific papers downloaded (arXiv + manual) |

| - PDF text extraction (pdfplumber) |

| - spaCy-based catalyst entity recognition |

| - Regex extraction of overpotential + Tafel slopes |

+-------------------------------------------------------------+

|

v

+-------------------------------------------------------------+

| 2. DATASET CURATION |

| - Filtering by formula validity (3+ metals) |

| - Outlier removal (onset potential 5-400 mV) |

| - Duplicate grouping (one row per catalyst) |

| - Combined with curated public dataset (180 catalysts) |

+-------------------------------------------------------------+

|

v

+-------------------------------------------------------------+

| 3. GRAPH REPRESENTATION |

| - Nodes = metal elements with 10 features: |

| atomic number, radius, electronegativity, group, |

| period, electron affinity, ionization energy, |

| melting point, density, composition fraction |

| - Edges = pairwise metal-metal interactions |

| - Edge features = delta electronegativity, delta radius,|

| composition-weighted average |

+-------------------------------------------------------------+

|

v

+-------------------------------------------------------------+

| 4. GRAPH NEURAL NETWORK (EdgeGNN) |

| - 3 x NNConv layers (uses edge features dynamically) |

| - Batch normalization + ReLU + dropout |

| - Mean + Max global pooling |

| - 3-layer MLP head -> scalar overpotential prediction |

+-------------------------------------------------------------+

|

v

+-------------------------------------------------------------+

| 5. EVALUATION |

| - 5-fold cross-validation |

| - Nested early stopping (inner validation) |

| - No test-set leakage |

| - Baselines: mean predictor, Random Forest |

+-------------------------------------------------------------+


\---



\## Repository Structure

CatalystGPT/

|

|-- data/

| |-- papers/ # 75 PDFs of scientific papers

| |-- raw/ # Extracted text files

| |-- processed/ # Cleaned datasets, graphs (.pt files)

|

|-- src/

| |-- nlp/ # Literature mining pipeline

| | |-- extract\_text.py

| | |-- extract\_catalysts\_v2.py

| | |-- extract\_catalysts\_v3.py # Table-aware extraction

| | |-- clean\_full\_dataset.py

| | |-- download\_papers.py

| |

| |-- gnn/ # Graph Neural Network pipeline

| |-- model.py # Original GIN model

| |-- model\_edge.py # EdgeGNN (final model)

| |-- build\_graphs\_v4.py # Composition-aware graphs

| |-- train\_v4.py # Training with CV

| |-- make\_plots\_v2.py # Analysis and figures

|

|-- results/

| |-- figures/ # Parity plots, residuals, importance

| |-- models/ # Trained model checkpoints

|

|-- docs/

| |-- report/ # Paper draft, technical report

| |-- presentation/ # Slide decks

|

|-- requirements.txt

|-- README.md





\---



\## Technology Stack



| Category | Tools |

|----------|-------|

| \*\*Language\*\* | Python 3.13 |

| \*\*NLP\*\* | spaCy 3.8, pdfplumber, BeautifulSoup |

| \*\*Deep Learning\*\* | PyTorch 2.14, PyTorch Geometric 2.8 |

| \*\*Materials Science\*\* | pymatgen, ASE |

| \*\*Data Science\*\* | NumPy, Pandas, Scikit-learn |

| \*\*Visualization\*\* | Matplotlib |

| \*\*Version Control\*\* | Git, GitHub |



\---



\## Installation



```bash

\# Clone the repository

git clone https://github.com/Harpreet29062005/CatalystGPT.git

cd CatalystGPT



\# Create and activate virtual environment

python -m venv venv

venv\\Scripts\\activate   # Windows

\# source venv/bin/activate   # Linux/Mac



\# Install dependencies

pip install -r requirements.txt


How to Run
Option 1: Train on the curated dataset (recommended)
# Build graphs from the 180-catalyst dataset

python src/gnn/build\_graphs\_v4.py



\# Train EdgeGNN with cross-validation

set PYTHONPATH=%CD%\\src\\gnn

python src/gnn/train\_v4.py



\# Generate analysis plots

python src/gnn/make\_plots\_v2.py
Option 2: Extract data from scientific papers (NLP pipeline)
# Extract text from PDFs

python src/nlp/extract\_text.py



\# Extract catalyst data via NLP

python src/nlp/extract\_catalysts\_v3.py



\# Clean the extracted dataset

python src/nlp/clean\_v3\_dataset.py
Sample Results

The following figures are generated by make\_plots\_v2.py:



results/figures/parity\_plots.png — Predicted vs actual onset potential for all three models



results/figures/residuals.png — Residual distribution for EdgeGNN



results/figures/feature\_importance.png — Top features driving HER activity



Dataset

Sources

Curated public dataset (primary): 180 multinary alloy HER catalysts from Gorsse et al. (2026), Scientific Data.



Custom NLP extraction (secondary): 75 papers mined for HEA-HER data using the pipeline in src/nlp/.



Target variable

onset\_potential (mV) — the potential at which HER begins. Lower values indicate better catalytic activity.



Distribution

Range: 5-385 mV (after outlier removal)



Mean: \~103 mV



Samples used for training: 173



Honest Limitations

This project reports honest, reproducible results. Important caveats:



Small dataset: 173 samples is small for deep learning. Performance may improve substantially with 1000+ catalysts.



Composition-only features: The model does not use crystal structure, morphology, or experimental conditions (pH, electrolyte), which are known to affect HER activity.



NLP extraction quality: The custom pipeline achieved \~30% extraction precision on reviewed papers — the primary value is the pipeline itself, not the extracted dataset.



Onset potential proxy: The target onset\_potential is used as a proxy for HER activity. It differs from the more common overpotential @ 10 mA/cm².



Author

Harpreet Thappa

Chemical Engineering, 5th Semester

National Institute of Technology (NIT) Srinagar

Email: happythappa5@gmail.com



License

This project is released for academic and research purposes.



Acknowledgments

Dataset: sgorsse/HER-alloy-catalysts-dataset-GPR



Frameworks: PyTorch, PyTorch Geometric, spaCy, pymatgen

