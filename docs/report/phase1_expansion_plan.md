\# CatalystGPT — Phase 1: Dataset Expansion Plan



\*\*Author:\*\* Harpreet Thappa

\*\*Date:\*\* October 2026

\*\*Goal:\*\* Expand training dataset from 173 → 500+ HEA catalysts



\---



\## Motivation



Current model achieves R² = 0.427 on 173 catalysts — a proof-of-concept result. To publish and reach state-of-the-art performance, we need 3–10× more data. This document defines the expansion strategy.



\---



\## Data Sources — Complete Inventory



\### Priority 1: Catalysis-Hub (Start Here)



| Field | Value |

|-------|-------|

| \*\*URL\*\* | https://catalysis-hub.org |

| \*\*Content\*\* | DFT adsorption energies (H\*, OH\*, O\*) on metal surfaces |

| \*\*Size\*\* | 100,000+ reactions |

| \*\*License\*\* | Open (CC BY 4.0) |

| \*\*Access\*\* | REST API + Python SDK (`pip install cathub`) |

| \*\*Why\*\* | Direct H\* adsorption energies — the key HER descriptor |

| \*\*Expected yield\*\* | \~50–100 HEA-relevant entries after filtering |



\### Priority 2: Materials Project



| Field | Value |

|-------|-------|

| \*\*URL\*\* | https://materialsproject.org |

| \*\*Content\*\* | DFT-computed properties for 150,000+ materials |

| \*\*License\*\* | Academic (requires API key) |

| \*\*Access\*\* | `pip install mp-api` + API key (free for students) |

| \*\*Expected yield\*\* | \~100–200 alloys after HER-relevant filtering |

| \*\*Risk\*\* | API key approval takes 1–3 days |



\### Priority 3: Open Catalyst Project (OC20/OC22)



| Field | Value |

|-------|-------|

| \*\*URL\*\* | https://opencatalystproject.org |

| \*\*Content\*\* | 1.3M+ DFT relaxation trajectories (oc20), catalysis-focused |

| \*\*License\*\* | Open (CC BY 4.0) |

| \*\*Access\*\* | Direct download (\~100 GB total — download subsets only) |

| \*\*Expected yield\*\* | \~200–500 surface-adsorbate entries after filtering |

| \*\*Risk\*\* | Large dataset; filtering script needed |



\### Priority 4: NOMAD Repository



| Field | Value |

|-------|-------|

| \*\*URL\*\* | https://nomad-lab.eu |

| \*\*Content\*\* | Raw DFT data from published papers |

| \*\*License\*\* | Open |

| \*\*Access\*\* | REST API + Python SDK |

| \*\*Expected yield\*\* | \~50–100 entries |

| \*\*Risk\*\* | Unstructured; filtering required |



\### Priority 5: Custom NLP Expansion



| Field | Value |

|-------|-------|

| \*\*Method\*\* | Run improved NLP pipeline on 300+ new papers |

| \*\*Source\*\* | PubMed Central, MDPI, Frontiers (all open-access) |

| \*\*Expected yield\*\* | \~30–50 new catalysts |

| \*\*Risk\*\* | NLP extraction quality limited (\~30% precision) |



\---



\## Execution Schedule



\### Week 1: Catalysis-Hub + Setup

\- Day 1–2: Create accounts, register API keys (MP, NOMAD)

\- Day 3–4: Install `cathub`, `mp-api`, query Catalysis-Hub

\- Day 5: Filter to HEA-relevant entries, save `catalysis\_hub.csv`

\- \*\*Target:\*\* 50 additional catalysts



\### Week 2: Materials Project

\- Day 1–3: Query MP for HEA alloys with computed HER-relevant properties

\- Day 4: Filter, clean, save `mp\_data.csv`

\- Day 5: Merge with existing dataset

\- \*\*Target:\*\* 100+ additional catalysts



\### Week 3: Open Catalyst Project

\- Day 1–2: Download OC20 subset (only HEA-relevant)

\- Day 3–5: Filter, extract H\* adsorption energies

\- \*\*Target:\*\* 200+ additional catalysts



\### Week 4: NLP Expansion + Merge

\- Day 1–3: Run NLP pipeline on 300+ new papers

\- Day 4: Final merge, deduplication, quality check

\- Day 5: Retrain models with expanded dataset

\- \*\*Target:\*\* 500+ total unique catalysts



\---



\## Risk Mitigation



| Risk | Likelihood | Mitigation |

|------|-----------|------------|

| API access denied | Medium | Use Catalysis-Hub (no key) first |

| Data formats incompatible | High | Write adapters for each source |

| Units differ across sources | High | Document unit conversions carefully |

| Duplicate entries | Medium | Deduplicate by formula + conditions |

| Filtering removes too much | High | Start with permissive filters |

| Time overrun | High | Weekly checkpoint; drop low-priority sources |



\---



\## Success Criteria for Phase 1



By end of Week 4, we must have:



\- \[ ] 500+ unique HEA catalyst compositions in a unified CSV

\- \[ ] Each entry has at least overpotential or Tafel slope

\- \[ ] Documented source and license for every entry

\- \[ ] No duplicate compositions from different sources

\- \[ ] Full data pipeline committed to GitHub



\---



\## Deliverables



1\. `data/processed/expanded\_dataset.csv` — merged 500+ catalysts

2\. `src/data/` — folder with source-specific adapters

3\. `docs/report/data\_sources.md` — provenance documentation

4\. Updated model trained on expanded dataset

5\. New metrics comparison in README



\---



\## What This Enables



After Phase 1:

\- \*\*Better model performance\*\* — R² target: 0.6+

\- \*\*Publications\*\* — larger datasets are required by peer reviewers

\- \*\*Benchmarks\*\* — compare against published ML-for-HER papers

\- \*\*Foundation for Phase 2\*\* — DFT validation requires large training set



\---



\## Honest Limitations



Even with 500+ catalysts:

\- Dataset will still be small by ML standards (10,000+ ideal)

\- Composition-only features will remain

\- No experimental conditions (pH, electrolyte) modeled



\*\*500+ is a milestone, not a finish line. Phase 2–6 will continue from here.\*\*



\---



\## Checklist for Today (Day 1 of Week 1)



\- \[ ] Commit this planning document to GitHub

\- \[ ] Create Catalysis-Hub account

\- \[ ] Install `cathub` Python package

\- \[ ] Run a first test query

\- \[ ] Save sample data to `data/raw/catalysis\_hub\_sample.csv`



\*\*Next session:\*\* Filter Catalysis-Hub data and build the first adapter.

