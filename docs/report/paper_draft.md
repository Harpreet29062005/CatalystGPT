\# CatalystGPT: An AI-Driven Pipeline for High-Entropy Alloy Catalyst Discovery in Green Hydrogen Production



\*\*Harpreet Thappa\*\*

Department of Chemical Engineering

National Institute of Technology Srinagar, India

happythappa5@gmail.com



\---



\## Abstract



The discovery of efficient, low-cost catalysts for the Hydrogen Evolution Reaction (HER) is critical for scalable green hydrogen production. High-Entropy Alloys (HEAs) — alloys containing five or more principal elements — have emerged as promising HER catalysts due to their tunable composition and multi-site activity. However, the vast compositional space of HEAs (on the order of 10^10 possible combinations) makes exhaustive experimental screening infeasible. In this work, we present \*\*CatalystGPT\*\*, an end-to-end computational pipeline that (i) automatically mines scientific literature using natural language processing (NLP) to extract HEA-HER performance data, and (ii) trains a \*\*composition-aware Edge-conditioned Graph Neural Network (EdgeGNN)\*\* to predict HER onset potential directly from alloy composition. We curate a dataset of \*\*173 unique HEA catalysts\*\* and evaluate our model using rigorous 5-fold cross-validation with nested early stopping. Our EdgeGNN achieves a mean absolute error (MAE) of \*\*49.35 mV\*\* and R^2 of \*\*0.427\*\*, representing a \*\*37.8% reduction in MAE\*\* over the trivial mean baseline (79.36 mV). A Random Forest baseline achieves comparable performance (MAE = 45.56 mV, R^2 = 0.447), demonstrating that composition-only features capture substantial HER activity trends. The full pipeline — including the literature mining toolkit, dataset curation scripts, graph construction code, and trained models — is publicly released to support reproducible HEA catalyst research.



\*\*Keywords:\*\* High-Entropy Alloys, Hydrogen Evolution Reaction, Graph Neural Networks, Materials Discovery, Natural Language Processing, Green Hydrogen



\---



\## 1. Introduction



\### 1.1 Motivation



Hydrogen produced via water electrolysis powered by renewable energy — commonly called \*\*green hydrogen\*\* — is a cornerstone of global decarbonization strategies. The Hydrogen Evolution Reaction (HER):



&#x20;   2H+ + 2e- -> H2    (acidic)

&#x20;   2H2O + 2e- -> H2 + 2OH-    (alkaline)



is the cathodic half-reaction in water electrolysis. Its efficiency is limited by the sluggish kinetics of the catalyst. Platinum-group metals (Pt, Ir) are the state-of-the-art HER catalysts, but their scarcity and high cost prohibit large-scale deployment.



\*\*High-Entropy Alloys (HEAs)\*\* — solid solutions containing five or more principal metallic elements in near-equimolar ratios — have emerged as promising alternatives. Their unique properties include:



\- \*\*Tunable composition:\*\* Continuous variation across a vast compositional space

\- \*\*Multi-site synergy:\*\* Different metal sites can catalyze different elementary steps

\- \*\*Enhanced stability:\*\* Entropy stabilization resists phase segregation



However, the HEA compositional space is astronomically large. Even with a modest set of 20 metals in 5-element alloys, the number of possible combinations exceeds \*\*15,504\*\*. Systematic experimental exploration is intractable.



\### 1.2 Contributions



We present \*\*CatalystGPT\*\*, a computational pipeline with the following contributions:



1\. \*\*An NLP-based literature mining pipeline\*\* that extracts catalyst compositions, overpotentials, and Tafel slopes from PDFs of scientific papers, using spaCy and pdfplumber.

2\. \*\*A composition-aware graph representation\*\* of HEA catalysts, encoding stoichiometric fractions and pairwise metal-metal interaction features.

3\. \*\*An Edge-conditioned Graph Neural Network (EdgeGNN)\*\* that uses `NNConv` layers to learn from both node (atomic) and edge (pairwise) features.

4\. \*\*Honest, reproducible evaluation\*\* using 5-fold cross-validation with nested early stopping on 173 curated catalysts, benchmarked against mean baseline and Random Forest.

5\. \*\*Full open-source release\*\* of code, datasets, and trained models.



\### 1.3 Key Finding



Our EdgeGNN achieves \*\*comparable performance to Random Forest\*\* (R^2 = 0.427 vs. 0.447) using only composition information, demonstrating that graph-based deep learning can capture HER activity trends from HEA composition alone.



\---



\## 2. Related Work



\### 2.1 Machine Learning for HER Catalysts



Recent work has applied ML to catalyst discovery across multiple material classes. Gradient boosting models (XGBoost, CatBoost) have been used to predict HER overpotential from elemental descriptors. Graph Neural Networks (GNNs) have shown particular promise for crystalline materials, where atomic connectivity is naturally represented as a graph.



\### 2.2 HEA Catalyst Datasets



The recent \*Scientific Data\* dataset by Gorsse et al. (2026) compiled \*\*180 multinary alloy HER catalysts\*\* from the literature with harmonized metadata — the first large-scale open dataset specifically for composition-only HEA modeling. We use this dataset as our primary training set.



\### 2.3 Literature Mining for Materials



NLP-based extraction from scientific literature has gained traction as a way to build large materials datasets automatically. Prior work has focused on named entity recognition of chemical formulas and property extraction from tables. Our pipeline builds on these ideas and additionally handles composition-weighted graph construction downstream.



\---



\## 3. Methodology



\### 3.1 Data Collection



\#### 3.1.1 Curated Public Dataset



We use the dataset from Gorsse et al. (2026), containing \*\*180 HEA HER catalysts\*\* with:

\- Normalized alloy formula

\- Atomic-percent composition for 18 retained elements

\- `onset\_potential` (mV) — our target variable

\- `tafel\_slope` (mV/dec) — secondary property



\#### 3.1.2 Custom NLP Extraction



We additionally downloaded \*\*75 scientific papers\*\* from arXiv, PubMed Central, MDPI, and Frontiers, and built an NLP extraction pipeline:



1\. \*\*Text extraction\*\* — `pdfplumber` extracts text and tables from PDFs

2\. \*\*Entity recognition\*\* — spaCy tokenization identifies catalyst formulas via regex

3\. \*\*Property extraction\*\* — regex patterns identify overpotential and Tafel slope values in sentences and tables

4\. \*\*Cleaning\*\* — filters remove outliers, deduplicate, and validate formulas



From 75 papers, we extracted \*\*23 clean HEA catalysts\*\* after aggressive filtering. While small, this dataset serves as a proof-of-concept for the pipeline and is included for reproducibility.



\### 3.2 Dataset Curation



Combining the two sources and filtering:

\- Require 3+ distinct metal elements

\- Onset potential in range \[5, 400] mV

\- Remove duplicates



Final dataset: \*\*173 unique HEA catalysts\*\*.



\### 3.3 Graph Representation



Each catalyst is represented as a \*\*fully-connected directed graph\*\*:



\*\*Nodes\*\* (one per unique metal element):

\- Atomic number (Z)

\- Atomic radius

\- Electronegativity

\- Group and period

\- Electron affinity

\- First ionization energy

\- Melting point

\- Density

\- \*\*Composition fraction\*\* (atomic % / 100)



\*\*Edges\*\* (all ordered pairs):

\- Electronegativity difference (|X\_i - X\_j|)

\- Atomic radius difference (|r\_i - r\_j|)

\- Composition-weighted average



\### 3.4 EdgeGNN Architecture



The model consists of:



\- \*\*3 NNConv layers\*\* with hidden dimension 64, using edge features to generate dynamic weight matrices

\- \*\*Batch normalization\*\* after each convolution

\- \*\*ReLU activation\*\* and dropout (0.2)

\- \*\*Dual global pooling\*\* (mean + max) concatenated into a 128-dim graph embedding

\- \*\*3-layer MLP head\*\* (128 -> 64 -> 32 -> 1)



Parameters: approximately 311,000.



\### 3.5 Training Protocol



\- \*\*5-fold cross-validation\*\* (outer split)

\- \*\*Nested inner validation\*\* (15% of training set) for early stopping

\- \*\*Normalization\*\* computed only from training fold statistics (no leakage)

\- \*\*SmoothL1Loss\*\* as the training objective

\- \*\*Adam optimizer\*\* with learning rate 3e-4

\- \*\*ReduceLROnPlateau\*\* scheduler

\- \*\*Gradient clipping\*\* at norm 1.0

\- \*\*Early stopping\*\* with patience of 60 epochs



\### 3.6 Baselines



\- \*\*Mean predictor:\*\* Predicts the training-set mean onset potential for every catalyst

\- \*\*Random Forest:\*\* 300 trees, max depth 10, trained on 90+ elemental descriptors (metal presence, mean/std/max/min of atomic properties)



\---



\## 4. Results



\### 4.1 Overall Performance



| Model | MAE (mV) | RMSE (mV) | R^2 |

|-------|----------|-----------|-----|

| Baseline (mean) | 79.36 | 95.63 | -0.011 |

| Random Forest | \*\*45.56\*\* | \*\*70.71\*\* | \*\*0.447\*\* |

| EdgeGNN (proposed) | 49.35 | 71.99 | 0.427 |



\*\*All three models were evaluated using identical 5-fold cross-validation splits.\*\*



\### 4.2 Per-Fold EdgeGNN Performance



| Fold | Test MAE (mV) |

|------|---------------|

| 1 | 35.51 |

| 2 | 58.86 |

| 3 | 41.92 |

| 4 | 60.00 |

| 5 | 50.81 |

| \*\*Mean +/- std\*\* | \*\*49.42 +/- 9.51\*\* |



Fold-to-fold variance (std = 9.51 mV) reflects the small dataset size.



\### 4.3 Comparison with Baselines



\- \*\*vs. Mean baseline:\*\* The EdgeGNN reduces MAE by \*\*37.8%\*\* (79.36 -> 49.35 mV).

\- \*\*vs. Random Forest:\*\* The EdgeGNN achieves 8.3% higher MAE but nearly identical R^2 (0.427 vs. 0.447). The two models perform comparably.



\### 4.4 Feature Importance



Random Forest feature importance analysis identified the most influential descriptors:

1\. Minimum ionization energy

2\. Minimum melting point

3\. Mean ionization energy

4\. Ionization energy standard deviation

5\. Electron affinity standard deviation



These results suggest that \*\*electronic structure heterogeneity\*\* — captured by dispersion of ionization energies and electron affinities — is a dominant driver of HER activity in HEAs.



\---



\## 5. Discussion



\### 5.1 Interpretation



Both Random Forest and EdgeGNN substantially outperform the trivial mean baseline, confirming that HEA composition carries \*\*predictive signal\*\* for HER activity. The near-equivalent performance of the two models suggests that:



1\. For datasets of this size (\~173 samples), \*\*simple models can match complex ones\*\*.

2\. The graph representation provides \*\*no additional information\*\* beyond what tabular compositional features capture.

3\. \*\*Compositional heterogeneity\*\* (e.g., ionization energy spread) is more predictive than pairwise metal interactions.



\### 5.2 Why the GNN Does Not Outperform RF



Three factors:



1\. \*\*Small dataset:\*\* 173 samples is 10-100x smaller than typical GNN training sets.

2\. \*\*Fully-connected graph:\*\* Without crystal structure, all metals are equally connected. This provides no connectivity information.

3\. \*\*Composition-only features:\*\* Crystal structure, morphology, and experimental conditions (pH, electrolyte) are not encoded.



\### 5.3 The NLP Pipeline



While our custom NLP extraction produced only 23 catalysts (vs. 180 in the curated dataset), the pipeline itself is a \*\*reproducible, extensible tool\*\* for automated materials literature mining. With improved extraction accuracy and a larger corpus, this approach can scale to thousands of papers.



\---



\## 6. Limitations



We explicitly acknowledge:



1\. \*\*Small dataset:\*\* 173 catalysts is insufficient for confident deep learning.

2\. \*\*Composition-only modeling:\*\* Real HER activity depends on crystal structure, morphology, and conditions.

3\. \*\*Onset potential proxy:\*\* We use `onset\_potential` (the potential at which HER begins) as the target — this differs from the more common `overpotential @ 10 mA/cm²`.

4\. \*\*Heterogeneous data sources:\*\* The curated dataset and our NLP-extracted dataset differ in measurement conditions.

5\. \*\*No experimental validation:\*\* All results are computational; experimental confirmation is future work.



\---



\## 7. Conclusion and Future Work



We presented \*\*CatalystGPT\*\*, an end-to-end pipeline for HEA catalyst discovery that combines NLP-based literature mining with a composition-aware Graph Neural Network. On 173 curated HEA catalysts, our EdgeGNN achieves \*\*R^2 = 0.427\*\*, competitive with Random Forest (R^2 = 0.447) and substantially better than the mean baseline.



\### Future Directions



1\. \*\*Scale the dataset\*\* to 1,000+ catalysts by improving NLP extraction and curating additional sources.

2\. \*\*Add structural features\*\* (crystal structure, coordination number, d-band center proxies).

3\. \*\*Multi-target learning\*\* (predict overpotential, Tafel slope, and stability simultaneously).

4\. \*\*Experimental validation\*\* of top-predicted catalysts.

5\. \*\*Deploy a web interface\*\* for the research community.



\---



\## References



1\. Gorsse, S., Tang, B., Tang, Y., Ma, M., \& Liu, Z. (2026). Curated dataset of multinary alloy HER catalysts for composition-only modelling with Magpie descriptors and GP baselines. \*Scientific Data\*.



2\. George, E. P., Raabe, D., \& Ritchie, R. O. (2019). High-entropy alloys. \*Nature Reviews Materials\*, 4(8), 515-534.



3\. Sanchez-Lengeling, B., \& Aspuru-Guzik, A. (2018). Inverse molecular design using machine learning. \*Science\*, 361(6400), 360-365.



4\. Zheng, Y., Jiao, Y., Qiao, S. Z., et al. (2016). Hydrogen evolution reaction in alkaline media. \*Angewandte Chemie\*, 55(52), 16081-16083.



5\. Xie, T., \& Grossman, J. C. (2018). Crystal Graph Convolutional Neural Networks for an accurate and interpretable prediction of material properties. \*Physical Review Letters\*, 120(14), 145301.



6\. Hu, W., et al. (2021). Open Graph Benchmark: Datasets for machine learning on graphs. \*NeurIPS\*.



7\. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., \& Dahl, G. E. (2017). Neural Message Passing for Quantum Chemistry. \*ICML\*.



8\. Belsky, A., Hellenbrandt, M., Karen, V. L., \& Luksch, P. (2002). New developments in the Inorganic Crystal Structure Database. \*Acta Crystallographica Section B\*, 58(3), 364-369.



\---



\## Appendix A: Reproducibility



All code is available at:

https://github.com/Harpreet29062005/CatalystGPT



Environment: Python 3.13, PyTorch 2.14, PyTorch Geometric 2.8, pymatgen 2026.9



To reproduce main results:



&#x20;   python src/gnn/build\_graphs\_v4.py

&#x20;   set PYTHONPATH=%CD%\\src\\gnn

&#x20;   python src/gnn/train\_v4.py

&#x20;   python src/gnn/make\_plots\_v2.py



\---



\## Appendix B: Dataset Statistics



\- Total catalysts: 173

\- Onset potential range: 5.00 - 385.00 mV

\- Onset potential mean: 103.62 mV

\- Onset potential std: 120.83 mV

\- Element count distribution:

&#x20; - 3 elements: 15 catalysts

&#x20; - 4 elements: 27 catalysts

&#x20; - 5 elements: 115 catalysts (true HEA)

&#x20; - 6 elements: 16 catalysts

&#x20; - 7+ elements: 4 catalysts

