# UAE Used-Car Market Analysis & Intelligent Valuation Engine

**Course:** ADY201m – Applied Data Analysis  
**Institution:** FPT University — Department of Artificial Intelligence  
**Dataset:** [UAE Used Car Prices & Features (10K+ Listings) — Kaggle](https://www.kaggle.com/datasets/alikalwar/uae-used-car-prices-and-features-10k-listings)  
**Database Engine:** SQLite 3.25+  
**Dashboard Framework:** Streamlit  

---

## Executive Summary

The UAE used-car marketplace spans economy commuter sedans (from ~8,000 AED) to hypercars (exceeding 14 million AED). This project delivers an **end-to-end analytical decision-support platform** covering:

1. **Relational Database Design** — 10,000 raw listings normalized into an 8-table, 3NF SQLite database with foreign keys and multi-index performance optimization.
2. **Advanced SQL Data Mining** — 12 complex business queries (Q1–Q12) demonstrating aggregations, multi-table joins, subqueries, CTEs, and window/analytic functions.
3. **Statistical Python Analysis** — Powertrain profiling, tier stratification, within-model depreciation econometrics, feature premium regression, and Kruskal-Wallis condition testing.
4. **Machine Learning Valuation** — Log-linear OLS (HC3 robust SE), Random Forest, and Gradient Boosting models with Duan smearing back-transformation and 5-fold cross-validation.
5. **Interactive Streamlit Dashboard** — Multi-page web application with live SQL exploration, filterable market analytics, ML-powered fair price prediction, deal arbitrage scanner, and dual AED/VND currency display.

**Best Model:** Gradient Boosting (GBDT) — holdout R² = 0.514, MAE = 133,735 AED, Duan smearing factor = 1.2454.

---

## Repository Structure

```text
used-car-project/
├── data/
│   ├── raw/
│   │   └── uae_used_cars_10k.csv        # Immutable raw source (10,000 listings, 12 columns)
│   ├── interim/
│   │   └── clean_listings.csv           # Cleaned staging dataset with parsed features
│   ├── processed/
│   │   ├── model_features.csv           # Engineered feature matrix (32 cols) for modeling
│   │   └── deal_scores.csv              # Valuation residuals + deal flags for all 10,000 rows
│   └── db/
│       └── uae_cars.sqlite              # Normalized 8-table relational SQLite database
│
├── sql/
│   ├── 01_schema.sql                    # DDL: 8 tables, indexes, v_listings view
│   ├── 02_load.sql                      # Dimension population and staging logic
│   ├── 03_validation.sql                # FK integrity checks (expect 0 violations)
│   └── 04_queries.sql                   # 12 business queries (Q1–Q12) with comments
│
├── notebooks/
│   ├── 01_build_db.ipynb                # Phase 1: DB build and verification
│   ├── 02_eda_cleaning.ipynb            # Phase 2: EDA, missingness, imputation, cleaning
│   ├── 03_analysis_viz.ipynb            # Phase 3: Statistical testing & 12-figure catalog
│   └── 04_modeling.ipynb                # Phase 4: Regression, ML, deal scoring
│
├── src/
│   ├── __init__.py
│   ├── config.py                        # All constants (seed=42, FX=7100 VND/AED, paths)
│   ├── db.py                            # SQLite connection manager & query utilities
│   ├── features.py                      # Description parser & feature engineering logic
│   ├── modeling.py                      # Model training, CV wrappers, smearing correction
│   ├── logger.py                        # Shared structured logging setup
│   ├── build_db.py                      # CLI: rebuilds SQLite from raw CSV
│   ├── run_phase2.py                    # CLI: EDA + preprocessing pipeline
│   ├── run_phase3.py                    # CLI: Python analysis + generates all figures
│   └── run_phase4.py                    # CLI: trains all models, scores deals, saves artifacts
│
├── app/
│   ├── Home.py                          # Streamlit entry point: market overview dashboard
│   ├── utils.py                         # Cached DB/model loaders, currency formatting, sidebar
│   └── pages/
│       ├── 1_Market_Explorer.py         # Multi-filter interactive visual analytics
│       ├── 2_Price_Predictor.py         # Fair price prediction engine (GBDT + Duan smearing)
│       ├── 3_Deal_Finder.py             # Deal arbitrage & anomaly detection
│       └── 4_SQL_Console.py             # Live SQL runner with 12 preset queries
│
├── models/
│   ├── price_model.joblib               # Serialized production GBDT pipeline
│   └── metadata.json                    # Feature list, all CV/test metrics, deal flags, smearing
│
├── figures/                             # 17 publication-ready figures (PNG, 150 DPI) + 2 HTML
├── screenshots/                         # SQL result screenshots (Q1–Q12) + dashboard (dash_01–05)
├── logs/                                # Timestamped execution logs for each pipeline phase
├── docs/
│   └── final-report-draft.md            # Full markdown draft of the report (all 4 sections)
│
├── evidence_ledger.xlsx                 # Claims C-01 to C-32: all marked Verified & Linked
├── Final_Project_Report_Template_EN_updated.docx  # Submission-ready report (0 [PENDING] tags)
├── requirements.txt                     # Pinned Python dependencies
└── README.md                            # This file
```

---

## Reproduction Guide (Step-by-Step)

> All commands are run from inside the `used-car-project/` directory.  
> Python 3.10+ required.

### Step 1 — Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### Step 2 — Build the SQLite Database

```bash
python3 src/build_db.py
```

Expected output: `DB build complete: 10,000 listings across 8 tables, 0 FK violations.`

Optional verification:
```bash
sqlite3 data/db/uae_cars.sqlite ".tables"
sqlite3 data/db/uae_cars.sqlite "SELECT COUNT(*) FROM listings;"
```

### Step 3 — Run EDA & Preprocessing (Phase 2)

```bash
python3 src/run_phase2.py
```

Outputs: `data/interim/clean_listings.csv`, `data/processed/model_features.csv`, figures fig_2_2 to fig_2_4, `logs/phase2_eda_preprocessing.log`

### Step 4 — Run Python Analysis & Visualizations (Phase 3)

```bash
python3 src/run_phase3.py
```

Outputs: 12 figures (`figures/fig_3_1_*.png` through `fig_3_12_*.png`), `logs/phase3_analysis.log`

### Step 5 — Train Models & Generate Deal Scores (Phase 4)

```bash
python3 src/run_phase4.py
```

Outputs:
- `models/price_model.joblib` — serialized GBDT production pipeline
- `models/metadata.json` — all metrics and deal distribution
- `data/processed/deal_scores.csv` — 10,000 listings with deal flags
- `figures/fig_4_1_*.png` through `fig_4_5_*.png`
- `logs/phase4_modeling.log`

Key results:
```
GBDT (5-fold CV):  R² = 0.5405, RMSE = 0.7687
GBDT (Test set):   R² = 0.5143, MAE  = 133,735 AED
Deal flags:        Undervalued=3,389 | Fair=2,973 | Overpriced=3,638
```

### Step 6 — Run All Notebooks (Alternative)

Execute notebooks in order using Jupyter:
```bash
jupyter notebook
# or: jupyter lab
```
Run all cells in: `01_build_db.ipynb` → `02_eda_cleaning.ipynb` → `03_analysis_viz.ipynb` → `04_modeling.ipynb`

### Step 7 — Launch the Streamlit Dashboard

```bash
streamlit run app/Home.py
```

Open browser at: **`http://localhost:8501`**

| Page | Description |
|---|---|
| **Home** | Market macro indicators, distribution charts |
| **Market Explorer** | Multi-filter analytics, price/mileage scatter, make treemap |
| **Price Predictor** | Enter vehicle specs → fair market value (AED + VND) |
| **Deal Finder** | Scan 10,000 listings for Undervalued / Overpriced deals |
| **SQL Console** | Run Q1–Q12 preset queries or custom SELECT against SQLite |

---

## SQL Query Catalog (Q1–12)

| ID | Business Question | SQL Techniques |
|---|---|---|
| Q1 | Overall market indicators (AED & VND) | `AVG`, `COUNT`, `CROSS JOIN` to FX rate |
| Q2 | High-volume makes average pricing | `GROUP BY`, `HAVING COUNT >= 30` |
| Q3 | Geographic market share & supply concentration | `SUM() OVER ()` window function |
| Q4 | Year-over-year price dynamics | CTE, `LAG()` window function |
| Q5 | Top 3 most expensive models per make | `ROW_NUMBER() OVER (PARTITION BY make)` |
| Q6 | Cumulative supply concentration / Pareto | Running `SUM() OVER ()` |
| Q7 | Overpriced inventory (>2× model benchmark) | Correlated subquery |
| Q8 | Optional feature price premiums | CTE pivot aggregations |
| Q9 | Vehicle condition impact vs. model baseline | Multi-join CTE, percent deviation |
| Q10 | Mileage bands across price quartiles | `NTILE(4)` window function |
| Q11 | 3-year moving average price trend | `AVG() OVER (ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)` |
| Q12 | Median asking price by body style | Window median percentile calculation |

---

## Model Performance Summary

| Model | CV R² | Test R² | Test MAE (AED) | Median APE |
|---|---|---|---|---|
| Baseline (Mean) | −0.000 | −0.000 | 232,808 | 140.7% |
| OLS Linear | 0.440 | 0.424 | 157,120 | 57.4% |
| Random Forest | 0.531 | 0.508 | 136,526 | 49.3% |
| **GBDT (best)** | **0.541** | **0.514** | **133,735** | **49.2%** |

Key OLS findings:
- Cylinder count: **+23.05% per cylinder** (p < 10⁻²⁰⁰)
- Porsche premium: **+212%** vs. Toyota baseline
- Condition stated: **no statistically significant price effect** (H=2.624, p=0.758)
- All 6 optional feature flags: within [−1.19%, +1.04%] — statistically noise

---

## Quality and Evidence Assurance

Every factual finding and metric cited in the final report is backed by a tangible artifact logged in `evidence_ledger.xlsx` (claims C-01 to C-32, all marked **Verified & Linked**).

Artifact categories:
- SQL query screenshots: `screenshots/Q1.png` – `Q12.png`
- Python analysis tables: Tables 3.3–3.8
- Statistical test outputs: Spearman/Pearson matrices, Kruskal-Wallis H-statistics
- Model training log: `logs/phase4_modeling.log`
- Figures: `figures/fig_2_*.png`, `fig_3_*.png`, `fig_4_*.png`

---

## Pre-Submission Checklist

- [ ] `Final_Project_Report_Template_EN_updated.docx` — 32 tables, 39 embedded images
- [ ] `docs/final-report-draft.md` — all relative image paths
- [ ] `evidence_ledger.xlsx` — all 32 claims marked Verified & Linked
- [ ] `models/price_model.joblib` — model file present
- [ ] `models/metadata.json` — metrics and smearing factor saved
- [ ] `data/processed/deal_scores.csv` — 10,000 rows with `deal_flag` column
- [ ] `logs/phase4_modeling.log` — training log confirms clean exit
- [ ] All 17 figures in `figures/` present
- [ ] All 17 screenshots in `screenshots/` present (Q1–Q12 + dash_01–05)
- [ ] No absolute machine paths in any `.py`, `.md`, `.json`, `.sql` file
- [ ] `requirements.txt` is current
- [ ] `venv/` excluded from submission

---

`.gitignore` recommended entries:
```
venv/
__pycache__/
*.pyc
*.pyo
.DS_Store
```

---

## Key Dependencies

```
pandas, numpy, scikit-learn, statsmodels
matplotlib, seaborn, plotly
streamlit, joblib, openpyxl
python-docx, scipy
```

See `requirements.txt` for pinned versions.

---

## Key Deliverables

| Deliverable | Location |
|---|---|
| Final Report (DOCX) | `Final_Project_Report_Template_EN_updated.docx` |
| Report Draft (Markdown) | `docs/final-report-draft.md` |
| Evidence Ledger | `evidence_ledger.xlsx` |
| SQLite Database | `data/db/uae_cars.sqlite` |
| SQL Query Suite (12 queries) | `sql/04_queries.sql` |
| Production ML Model | `models/price_model.joblib` |
| Model Metadata & Metrics | `models/metadata.json` |
| Deal Scores (all 10,000) | `data/processed/deal_scores.csv` |
| Streamlit Dashboard | `app/Home.py` |
| Notebooks (Phases 1–4) | `notebooks/01_build_db.ipynb` – `04_modeling.ipynb` |
| Training Execution Log | `logs/phase4_modeling.log` |
