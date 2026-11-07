# UAE Used Cars: Final Project Completion Plan

Prepared for the ADY201m Final Project Report (FPT University, AI Department), based on the guidelines file, the report template and `uae_used_cars_10k.csv`.

## 1. What the documents require

- **Deliverables:** a PDF report exported from the template, plus source code (`.ipynb`/`.py`, `.sql`) and the dataset files, or one public GitHub/Drive link.
- **Zero-credit rule:** any conclusion without direct evidence (chart, summary table, output screenshot, or the code that produced it) scores zero. Every claim needs its evidence next to it.
- **Template structure:**
  - I. Problem
  - II. Data collection, EDA and preprocessing
  - III. SQL, Python analysis and visualization
  - IV. Modeling, plus one dashboard tool with a critical review
- **Chart standards:** each chart needs a title, labeled axes with units, a legend and a harmonious palette. Under each chart goes a 3–5 sentence paragraph covering what the numbers show, why it matters, and how it links back to Chapter 1.
- **Dataset rule:** the template says the dataset should be relational (multiple tables with foreign keys) and have several thousand records. The recommended datasets in the guideline appendix are suggestions, so using your own is allowed.

## 2. What the data looks like

The data is 10,000 UAE used-car listings with 12 columns: Make, Model, Year, Price, Mileage, Body Type, Cylinders, Transmission, Fuel Type, Color, Location and Description.

| Finding | Detail | Consequence |
| --- | --- | --- |
| **Single flat table** | The template expects a relational schema. | Normalize it in SQL (see Phase 1). Say so honestly in the report, and ideally confirm with your instructor. |
| **Hidden structure in `Description`** | Each description follows "{Year} {Make} {Model} with {features}. Condition: {condition}." It holds 6 optional features and 6 condition categories. | Parse it into a many-to-many `car_features` table and a `conditions` table. These are your extra joins. |
| **Missing and dirty values** | `Cylinders` has 105 NaN and 50 "Unknown" (1.55% in total). `Location` has leading spaces, so 16 raw variants collapse to 8 locations (Al Ain is a city in Abu Dhabi emirate). There are no fully duplicated rows, but only 9,985 unique descriptions. | Clean in Phase 2 and quantify the missing values per column. Check the 15 repeated descriptions. |
| **Extreme price skew** | Median is 102,766 and mean is 245,235. Skew is 9.28, falling to 0.54 after a log transform. The IQR rule flags 1,100 rows (11%) above about 502,591. Maximum is 14.7M (McLaren P1). | These are genuine supercars, not errors. Preserve them and model `log(Price)`. |
| **Counterintuitive correlations** | Spearman correlation of Price with Year is about 0.00, and with Mileage about 0.01. | This is the main analytical story. Price here is driven by make and model, not by age or mileage, which suggests the data is synthetic. Report it as a finding and caveat it. |
| **Weak condition effect** | Median price barely varies across the 6 conditions (cars with "Accident history" are not cheaper). | Another non-obvious insight for Chapter III. |
| **Sparse categories** | 190 models have fewer than 5 listings, and 17 makes have fewer than 5. | Group rare categories into "Other" before encoding. |

**Baseline feasibility check** (predicting log price, 80/20 split, before any tuning):

- Linear regression without Make: R² ≈ 0.35.
- Linear regression with Make: R² ≈ 0.54.
- Random forest with Make: R² ≈ 0.61.

Make alone explains about 43% of the variance. A realistic target is R² of about 0.55–0.70, and the report should say so rather than overselling.

## 3. Proposed project framing (Chapter I)

- **Context:** the UAE used-car market, with its large luxury and supercar segment.
- **Problem:** listing prices are inconsistent, so buyers can't tell a fair deal from an overpriced one and dealers can't price inventory consistently.
- **General objective:** build an analytical system that explains and predicts used-car prices in the UAE.
- **Detailed objectives:**
  1. Quantify price drivers by make, body type, emirate and fuel type.
  2. Test whether age, mileage and condition really affect price.
  3. Build a regression model that predicts price.
  4. Flag under- and over-priced listings.
  5. Deliver an interactive dashboard.
- **Approach:** descriptive, then diagnostic, then predictive, each justified against the objectives.

## 4. Phased execution plan

### Phase 0: Setup

- Create the repo structure: `/data/raw`, `/data/clean`, `/sql`, `/notebooks`, `/app`, `/figures`, `/report`.
- Confirm the open questions in section 7.
- Fix a random seed and a reference year for car age, and document both.

### Phase 1: Relational database and SQL (III.1, II.1)

- Use SQLite (locked decision). It supports window functions from version 3.25.
- Normalize into a schema such as:
  - `makes(make_id PK, make_name)`
  - `models(model_id PK, make_id FK, model_name)`
  - `locations(location_id PK, emirate)`
  - `body_types`
  - `conditions`
  - `features`
  - `cars(car_id PK, model_id FK, location_id FK, body_type_id FK, condition_id FK, year, price, mileage, cylinders, transmission, fuel_type, color)`
  - `car_features(car_id FK, feature_id FK)`
- Draw the ER diagram (dbdiagram.io or draw.io) for the schema section.
- Write 8–10 queries, with a screenshot of each result and a sentence on its business meaning. Coverage should include:
  - **Filtering and grouping:** average price and listing count per make, using `HAVING count >= 30`.
  - **Window function with `LAG`:** year-over-year change in average price by model year.
  - **`ROW_NUMBER`/`RANK` with `PARTITION BY`:** top 3 models per make by average price.
  - **Running total and Pareto:** cumulative share of listings by make.
  - **Subquery or CTE:** listings priced more than 2× their model average, labeled "overpriced".
  - **`NTILE`:** price quartiles crossed with mileage bands.
  - **Moving average:** `ROWS BETWEEN` frame over model years.
  - **Joins:** feature premium (average price with vs. without each feature) and price by condition.
  - **Market share:** by emirate.

### Phase 2: EDA and preprocessing (II.2, II.3)

- **EDA (II.2):**
  - Report shape, dtypes and descriptive statistics for numeric and categorical columns.
  - Add a table of initial observations.
- **Missing values:** report % per column. Impute `Cylinders` by Make+Model mode, falling back to Body Type mode, then the global mode. Convert "Unknown" to NaN first.
- **Outliers:**
  - Use boxplots and the IQR rule, and report the 1,100 flagged rows.
  - Preserve them with a justification, and log-transform the target.
- **Type conversions:** `Cylinders` to numeric, trim `Location`, standardize text case.
- **Engineered features (list all of them):**
  - `car_age`
  - `mileage_per_year`
  - `log_price`
  - `n_features`
  - Six `has_<feature>` flags
  - `is_accident_history`
  - `region_group` (Dubai, Abu Dhabi, Northern Emirates)
  - `make_tier` (budget, mid, premium, ultra-luxury, based on make median price)
- **Encoding:**
  - One-hot for low-cardinality variables.
  - Target or frequency encoding for Make/Model, fit on the training set only to avoid leakage.

### Phase 3: Python analysis and visualization (III.2, III.3)

- **Pandas/NumPy analysis:**
  - Crosstabs such as Body Type × Fuel Type.
  - Multi-index groupby (Make × age bucket).
  - Pearson and Spearman correlation matrices.
  - A within-model analysis: price relative to the model median against age and mileage. This confirms that age and mileage don't matter even after controlling for model.
- **Charts (10–12), covering all four required archetypes:**
  - Bars: top makes and median price by body type.
  - Lines: median price and listing count by year.
  - Scatter: mileage vs. price on a log axis, colored by make tier.
  - Histograms: raw vs. log price.
  - Boxplots: price by body type or fuel.
  - Heatmap: correlations.
  - Plotly: an interactive treemap and an interactive scatter.
- Write a 3–5 sentence interpretation under each chart, answering the three required questions.

### Phase 4: Modeling (IV.1–IV.3)

- **Linear regression:** statsmodels OLS on `log_price`, so you can report R², coefficient p-values, MSE and RMSE. Check VIF and residual plots.
- **Comparison models:** random forest and XGBoost/Gradient Boosting, with 5-fold cross-validation and a mean-price baseline.
- **Metrics:** R² on the log scale, and RMSE, MAE and MAPE back-transformed to AED.
- **Business translation:**
  - Convert coefficients to percentage effects with exp(β)−1, for example the manual-transmission discount.
  - Use permutation importance.
  - Use residuals to build an "undervalued/overvalued listing" detector.
  - Propose concrete pricing and inventory actions.
- **Classification metrics:** the confusion matrix, precision and recall in the template apply only to classification, so they're not needed for a regression target. Optionally add a secondary "overpriced yes/no" classifier if you want that coverage.

### Phase 5: Dashboard and tool review (IV.4)

- **Tool:** Streamlit is the most practical choice, since it reuses your Python code.
- **Pages:** KPI overview, filterable market explorer, price predictor, and a deal finder.
- **Evidence:** screenshots, and a deployment link if you use Streamlit Community Cloud.
- **Critical review:** usability, strengths, limitations, and a comparison against custom code and against Power BI or Dash.

### Phase 6: Assembly and QA

- Fill every cover placeholder (title, instructor, class code, group number, members and IDs).
- Check that every claim in the report has evidence beside it.
- Check that every chart has the mandatory elements and its paragraph.
- Update the table of contents and page numbers, export to PDF, and re-read the PDF.
- Package the code, `.sql` files and CSVs (or the repo link), and run the notebook top to bottom on a clean kernel.

## 5. Evidence checklist per chapter

| Section | Required evidence |
| --- | --- |
| II.1 | ER diagram, source URL, full data dictionary table |
| II.2 | `.shape`, `.info()` and `.describe()` outputs |
| II.3 | Missing-value table, before/after boxplots, feature list, encoding code |
| III.1 | SQL code, output screenshot and interpretation for every query |
| III.2 | Crosstab and groupby tables, correlation matrix |
| III.3 | Every chart with its 3–5 sentence analysis |
| IV | Model code, metrics table, coefficient and importance tables, residual plots |
| IV.4 | Dashboard screenshots and the written review |

## 6. Suggested split for 3–4 members

- **Member A:** database design and SQL (Phase 1).
- **Member B:** EDA, preprocessing and the data dictionary (Phase 2).
- **Member C:** Python analysis and charts (Phase 3).
- **Member D:** modeling, dashboard and tool review (Phases 4–5).
- **Everyone:** write their own chapter sections, then the group leader integrates and runs the QA.

## 7. Decisions locked in, and what is still open

- **Source (locked):** [UAE Used Car Prices & Features (10K+ Listings) on Kaggle](https://www.kaggle.com/datasets/alikalwar/uae-used-car-prices-and-features-10k-listings), by alikalwar. Check the license on the dataset page before submitting.
- **Currency (locked):** prices are AED. Model and analyse in AED, then convert to VND for presentation at one frozen rate (details in Part B, section B0).
- **Database and dashboard (locked):** SQLite and Streamlit.
- **Still open:** whether the instructor accepts a single CSV normalized into a relational schema, plus group details and the deadline.

## Part B. Detailed blueprint

Part B turns sections 4 to 6 into numbered work orders with an output file and an acceptance check for each step, and it governs wherever it differs from Part A. Step IDs (P0-01, P1-03 and so on) are the shared names for tasks, branches and commit messages.

### B0. Project constants, folders and Phase 0 steps

Fix these values once in `src/config.py` and import them everywhere, so no notebook or dashboard page hard-codes them.

| Constant | Value | Reason |
| --- | --- | --- |
| `RANDOM_SEED` | 42 | Same splits and model results on every machine |
| `REFERENCE_YEAR` | 2025 | Assumed listing snapshot year (newest model year is 2024), so `car_age` is at least 1. State this as an assumption in Chapter II |
| Currency of record | AED | Source currency; all SQL, statistics and models use AED |
| `AED_TO_VND` | 7,100 per AED, frozen | Search results from July and August 2026 show roughly 7,113 to 7,158 mid-market. Check the rate on the day you freeze it, and store the value and its date |
| Where VND appears | Tables, charts and the dashboard only | The model fits `log(Price_AED)`, so a constant rate would only shift the intercept and leave R² unchanged |
| Approximate VND scale | Median about 730 million, mean about 1.74 billion, maximum about 104 billion | Use "million VND" or "billion VND" in axis labels |
| Split and validation | 80/20 train/test, 5-fold CV on train | One held-out test set used once, at the end |
| Database | SQLite 3.25 or later, file `data/db/uae_cars.sqlite` | Window functions need 3.25. Run `PRAGMA foreign_keys = ON` on every connection, because SQLite ignores foreign keys by default |
| Language and numbering | English; figures `Fig. 3.4`, tables `Table 2.1` | Chapter number, then sequence |

Caveat to print once in Chapter II: the VND figures are a unit conversion for readability. They are not estimates of Vietnamese market prices, which differ because of import taxes and local demand.

Target folder layout:

```text
uae-used-cars/
  data/raw/uae_used_cars_10k.csv      # never edited
  data/interim/clean_listings.csv
  data/processed/model_features.csv
  data/db/uae_cars.sqlite
  sql/01_schema.sql  02_load.sql  03_validation.sql  04_queries.sql
  notebooks/01_build_db.ipynb  02_eda_cleaning.ipynb
            03_analysis_viz.ipynb  04_modeling.ipynb
  src/config.py  db.py  features.py  modeling.py
  app/Home.py  app/pages/  app/utils.py
  models/price_model.joblib  models/metadata.json
  figures/  screenshots/  report/
  requirements.txt  README.md  evidence_ledger.xlsx
```

Phase 0 steps:

| ID | Step | Output | Acceptance check |
| --- | --- | --- | --- |
| P0-01 | Create the GitHub repo with the layout above and protect `main` | Repo link | All members can push to branches |
| P0-02 | Create a virtual environment and pin `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn` (1.3 or later), `statsmodels`, `xgboost`, `streamlit`, `joblib`, `openpyxl` | `requirements.txt` | `pip install -r requirements.txt` works on a clean machine |
| P0-03 | Write `config.py` with the constants above | `src/config.py` | Every notebook imports from it |
| P0-04 | Download the CSV from Kaggle, store it in `data/raw/`, record its SHA-256 and row count (10,000 rows, 12 columns) | Checksum in README | Re-hash matches |
| P0-05 | Add `nbstripout` so notebook outputs do not cause merge conflicts, and use one notebook per phase | Git hook | Two members can edit different notebooks without conflicts |
| P0-06 | Create `evidence_ledger.xlsx` with columns: claim, evidence ID (figure, table or screenshot), report page, owner | Ledger | Filled as work proceeds; checked in Phase 6 |
| P0-07 | Agree the figure style: one palette, font sizes, 150 dpi export, file names `fig_<chapter>_<n>_<slug>.png` | Style cell in `config.py` | All charts share one look |

### B1. Phase 1: SQLite database and SQL analysis (Report II.1 and III.1)

Goal: turn one flat CSV into an 8-table relational database plus a small FX table, then answer 12 business questions with SQL that shows filtering, grouping, joins, subqueries and window functions. The raw `Description` text is the key to a genuine many-to-many relationship, because every description holds 2 to 4 features and one condition.

Target schema (`sql/01_schema.sql`):

```sql
PRAGMA foreign_keys = ON;
CREATE TABLE makes      (make_id INTEGER PRIMARY KEY, make_name TEXT NOT NULL UNIQUE);
CREATE TABLE models     (model_id INTEGER PRIMARY KEY,
                         make_id INTEGER NOT NULL REFERENCES makes(make_id),
                         model_name TEXT NOT NULL, UNIQUE (make_id, model_name));
CREATE TABLE emirates   (emirate_id INTEGER PRIMARY KEY, emirate_name TEXT NOT NULL UNIQUE);
CREATE TABLE body_types (body_type_id INTEGER PRIMARY KEY, body_type_name TEXT NOT NULL UNIQUE);
CREATE TABLE conditions (condition_id INTEGER PRIMARY KEY, condition_name TEXT NOT NULL UNIQUE);
CREATE TABLE features   (feature_id INTEGER PRIMARY KEY, feature_name TEXT NOT NULL UNIQUE);
CREATE TABLE cars (
  car_id       INTEGER PRIMARY KEY,
  model_id     INTEGER NOT NULL REFERENCES models(model_id),
  emirate_id   INTEGER NOT NULL REFERENCES emirates(emirate_id),
  body_type_id INTEGER NOT NULL REFERENCES body_types(body_type_id),
  condition_id INTEGER NOT NULL REFERENCES conditions(condition_id),
  model_year   INTEGER NOT NULL CHECK (model_year BETWEEN 2000 AND 2025),
  price_aed    INTEGER NOT NULL CHECK (price_aed > 0),
  mileage_km   INTEGER NOT NULL CHECK (mileage_km >= 0),
  cylinders    INTEGER,
  transmission TEXT NOT NULL,
  fuel_type    TEXT NOT NULL,
  color        TEXT NOT NULL);
CREATE TABLE car_features (
  car_id INTEGER NOT NULL REFERENCES cars(car_id),
  feature_id INTEGER NOT NULL REFERENCES features(feature_id),
  PRIMARY KEY (car_id, feature_id));
CREATE TABLE fx_rates (rate_date TEXT PRIMARY KEY, aed_to_vnd REAL NOT NULL);
CREATE INDEX idx_cars_model ON cars(model_id);
CREATE INDEX idx_cars_emirate ON cars(emirate_id);
CREATE INDEX idx_cars_year ON cars(model_year);
```

The source has no listing ID, so `car_id` is a surrogate key equal to the CSV row number starting at 1. Mileage has no stated unit; the plan assumes kilometres and says so in the data dictionary.

Build and validation steps:

| ID | Step | Output | Acceptance check |
| --- | --- | --- | --- |
| P1-01 | Write and run `01_schema.sql`; draw the ER diagram in dbdiagram.io or draw.io | DB file, `fig_2_1_er_diagram.png` | `PRAGMA foreign_key_list(cars)` shows 4 foreign keys |
| P1-02 | Load the raw CSV unchanged into a staging table `stg_listings` with `car_id` added | 10,000 staging rows | Row count and price sum equal the CSV |
| P1-03 | Parse `Description` in Python with the pattern `  with (.*)\. Condition: (.*)\.$ `, because SQLite has no regex | Parsed feature list and condition per row | 0 parse failures (all 10,000 matched in profiling) |
| P1-04 | Clean in staging: trim `Location`, convert `Unknown` cylinders to NULL, cast cylinders to integer | Clean staging table | 155 NULL cylinders (105 blank plus 50 Unknown) |
| P1-05 | Fill the dimension tables with `INSERT OR IGNORE ... SELECT DISTINCT` | 6 dimension tables | 65 makes, 13 body types, 8 locations, 6 conditions, 6 features, 520 models (distinct make-model pairs) |
| P1-06 | Fill `cars` by joining staging to the dimensions on names | 10,000 rows | No NULL foreign keys |
| P1-07 | Fill `car_features` from the parsed lists | Link table | 29,883 rows, 2 to 4 per car |
| P1-08 | Insert the frozen rate into `fx_rates` with its date | 1 row | Matches `AED_TO_VND` in `config.py` |
| P1-09 | Run `03_validation.sql`: counts, NULL counts, `PRAGMA foreign_key_check` | Validation screenshot | `foreign_key_check` returns 0 rows |
| P1-10 | Commit `build_db.py` so anyone can rebuild the database from the raw CSV in one command | Script | Rebuild gives identical counts |

SQLite limits to plan around: there is no `MEDIAN()` function, no regex, and `RIGHT` or `FULL JOIN` only from version 3.39. Compute medians with `ROW_NUMBER()` and `COUNT()` windows, and cross-check the answer in pandas.

Query catalog (`sql/04_queries.sql`, each query starts with a comment holding its business question):

| ID | Business question | SQL techniques | Output used in report |
| --- | --- | --- | --- |
| Q1 | What are the market KPIs: listings, average, minimum and maximum price in AED and VND? | Aggregates, join to `fx_rates` | KPI table |
| Q2 | Which makes have 30 or more listings, and what do they cost on average? | `GROUP BY`, `HAVING`, 3-table join | Table, feeds a bar chart |
| Q3 | How are listings and average price split by emirate? | Window `SUM() OVER ()` for share | Table, feeds a chart |
| Q4 | How does average price move from one model year to the next? | CTE, `LAG()` | Year-over-year table |
| Q5 | What are the 3 most expensive models in each make (5 or more listings)? | `ROW_NUMBER() OVER (PARTITION BY ...)` | Ranking table |
| Q6 | How concentrated is the market: cumulative share of listings by make? | Running `SUM() OVER (ORDER BY ...)` | Pareto table |
| Q7 | Which makes have the most listings priced above twice their model average? | CTE with per-model average, join, `HAVING` | Overpricing table |
| Q8 | What price premium does each feature carry? | `car_features` join, `EXISTS` subquery, conditional average | Feature premium table |
| Q9 | Does condition change price once model is controlled for? | Price divided by model average, grouped by condition | Condition table |
| Q10 | How do mileage bands cross price quartiles? | `NTILE(4)`, `CASE`, conditional counts | Cross-tab |
| Q11 | What is the 3-year moving average of price by model year? | `AVG() OVER (ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)` | Smoothed trend |
| Q12 | What is the median price by body type? | `ROW_NUMBER()` and `COUNT()` windows | Median table |

Two patterns to reuse:

```sql
-- Q4: year-over-year change in average price
WITH yearly AS (
  SELECT model_year, COUNT(*) AS n, AVG(price_aed) AS avg_price
  FROM cars GROUP BY model_year)
SELECT model_year, n, ROUND(avg_price) AS avg_price_aed,
       ROUND(100.0 * (avg_price - LAG(avg_price) OVER (ORDER BY model_year))
             / LAG(avg_price) OVER (ORDER BY model_year), 1) AS yoy_pct
FROM yearly ORDER BY model_year;

-- Q7: listings priced above 2x their model average
WITH model_avg AS (
  SELECT model_id, AVG(price_aed) AS avg_price
  FROM cars GROUP BY model_id HAVING COUNT(*) >= 5)
SELECT mk.make_name, COUNT(*) AS overpriced_listings
FROM cars c
JOIN model_avg ma ON ma.model_id = c.model_id
JOIN models m ON m.model_id = c.model_id
JOIN makes mk ON mk.make_id = m.make_id
WHERE c.price_aed > 2 * ma.avg_price
GROUP BY mk.make_name ORDER BY overpriced_listings DESC LIMIT 10;
```

Per-query routine (repeat for Q1 to Q12):

1. Write the query and run it in DB Browser for SQLite or the VS Code SQLite viewer.
2. Screenshot the result grid at full resolution, showing the top 10 to 15 rows and the query text.
3. Write 2 to 3 sentences on what the metric means for pricing decisions. State the number, not just the trend.
4. Re-compute one figure from the result in pandas as a cross-check.
5. Add the claim and screenshot ID to the evidence ledger.

Exit criterion for Phase 1: validation passes, all 12 queries have a screenshot and a written meaning, and at least one query each demonstrates filtering, grouping, a multi-table join, a subquery and a window function.

### B2. Phase 2: EDA and preprocessing (Report II.2 and II.3)

Goal: describe the raw data with evidence, then clean it with every decision quantified, and end with one modelling table of 10,000 rows. Nothing is deleted without a count and a reason in the report.

EDA steps (`notebooks/02_eda_cleaning.ipynb`):

| ID | Step | Output | Acceptance check |
| --- | --- | --- | --- |
| P2-01 | Create the view `v_listings` in SQLite (all tables joined) and read it into pandas | Flat frame | 10,000 rows |
| P2-02 | Table inventory: rows, columns and size of every table | Table 2.1 | 10,000 rows in `cars`, 29,883 in `car_features` |
| P2-03 | Data dictionary: generate the skeleton from `PRAGMA table_info`, then write each meaning by hand (table, attribute, type, description, PK/FK flag) | Table 2.2 | Every attribute of every table is listed |
| P2-04 | Descriptive statistics: for numeric columns count, mean, median, std, min, max; for categorical columns unique count, mode, frequency | Tables 2.3 and 2.4 | Reproduces the reference values below |
| P2-05 | Dtype table: numeric, categorical, boolean or text for each attribute | Table 2.5 | Matches the schema |
| P2-06 | Two EDA visuals: price histogram raw vs log, and listing counts by body type | Fig. 2.2 and 2.3 | Both meet the chart standards |
| P2-07 | Write 6 to 8 initial observations, each pointing to an output | Observation list | Each has an evidence ID |

Reference values the EDA must reproduce: price mean 245,235 AED, median 102,766, standard deviation 470,977, minimum 7,183, maximum 14,686,975; model year 2005 to 2024 with roughly 500 listings per year; mileage 10,006 to 299,996; top make mercedes-benz with 1,486 listings; SUV 4,607 listings; Dubai 8,011 after trimming; automatic transmission 9,626; gasoline 9,714.

Observations to expect and verify: price is heavily right-skewed (skewness 9.28 raw, 0.54 after log); the model-year distribution is almost flat, which is unusual for real listings; mileage does not rise with age; Dubai holds about 80% of listings.

Preprocessing decisions (`02_eda_cleaning.ipynb`, saved to `data/interim/clean_listings.csv`):

| Issue | Size | Treatment | Justification |
| --- | --- | --- | --- |
| `Cylinders` missing | 105 blank (1.05%) plus 50 "Unknown" (0.50%) | Convert Unknown to NaN. Impute the mode within the same make and model, then within body type, then the global mode (6). Add flag `cylinders_imputed`. Cast to integer | Engine size is mostly a model attribute, so group modes are accurate, and only 1.55% of rows change |
| `Location` spellings | 16 raw values from stray leading spaces (" Dubai" 7,990 vs "Dubai" 21) | Strip whitespace, giving 8 locations | Same place written two ways would split every group-by |
| Al Ain | 47 rows | Keep as its own location in SQL, merge into Abu Dhabi in `region_group` | Al Ain is a city in the Emirate of Abu Dhabi |
| Price outliers (raw) | 1,100 rows (11.0%) above the IQR upper fence of 502,591 AED; none below | Keep all rows | They are genuine supercar and luxury listings (McLaren, Ferrari, Rolls-Royce), the segment the project studies |
| Price outliers (log scale) | 91 rows (0.9%) outside 5,116 to 2,275,954 AED | Keep, model `log(Price)`, then refit without these 91 as a sensitivity check | Shows whether extremes drive the result |
| Implausible usage | 1,268 rows (12.7%) imply more than 50,000 km per year (for example a 2024 car with 293,041 km) | Flag `implausible_usage`, keep rows, compare model results with and without them | The column is the dataset's own, so report the oddity rather than silently remove it |
| Duplicates | 0 fully duplicated rows; 15 rows share a description with another row | Inspect, keep (they differ in price or mileage) | Not true duplicates |
| Rare categories | 190 models and 17 makes with fewer than 5 listings; 4 rows with make `other-make`; 114 body type `Other`; 78 `Other Color` | Keep as valid values. In charts show the top 15 makes and group the rest as "Other". In models use smoothed target encoding | Avoids unstable dummy columns |
| Data types | `Cylinders` is text; `Description` holds structured text | Convert cylinders to integer, set categorical dtype for category columns, parse `Description` into features and condition | The file has no date strings, so the only date-like field is `Year` |

Engineered features (`src/features.py`, listed in full in Chapter II):

| Feature | Definition | Type | Model input |
| --- | --- | --- | --- |
| `car_age` | `REFERENCE_YEAR - Year`, minimum 1 | Numeric | Yes |
| `mileage_per_year` | `Mileage / car_age` | Numeric | Yes |
| `log_mileage` | `ln(Mileage)` | Numeric | Yes |
| `log_price` | `ln(Price_AED)` | Numeric | Target |
| `n_features` | Number of listed features (2 to 4) | Numeric | Yes |
| `has_sunroof`, `has_leather_seats`, `has_navigation`, `has_bluetooth`, `has_rear_camera`, `has_adaptive_cruise` | 1 if the feature appears in the description | Boolean | Yes |
| `condition` | One of 6 labels from the description | Categorical | Yes |
| `is_accident_history` | 1 if condition is "Accident history" | Boolean | Yes |
| `region_group` | Dubai, Abu Dhabi (with Al Ain), Sharjah, Other Northern Emirates | Categorical | Yes |
| `make_tier` | Four bands of make median price, computed on the training set only | Categorical | Yes, in the linear model |
| `cylinders_imputed`, `implausible_usage` | Flags from the table above | Boolean | Yes |
| `price_vnd` | `Price_AED * AED_TO_VND` | Numeric | No, display only |

Encoding plan:

- **One-hot:** transmission (2 values), fuel type (4), body type (13), region group (4), condition (6), colour (17 values; group colours with fewer than 100 listings as "Other colour").
- **Smoothed target encoding, cross-fitted:** make and the 520 make-model pairs (use `sklearn.preprocessing.TargetEncoder`). For the linear model, use dummies for the 20 largest makes plus "Other make" instead, so coefficients stay readable.
- **Numeric:** cylinders stay numeric; standardize only for Ridge.
- **Leakage rules:** split train and test first. Fit imputers, encoders, scalers and `make_tier` inside a scikit-learn `Pipeline` on the training data only. Never feed `Description` or any price-derived column to a model.

Exit criterion: `clean_listings.csv` and `model_features.csv` saved with 10,000 rows, a before-and-after missing-value table, raw-vs-log price boxplots, the full feature table, and every preprocessing claim in the evidence ledger.

### B3. Phase 3: Python analysis and visualization (Report III.2 and III.3)

Goal: go beyond averages to findings a reader could not get from a glance at the data, and show each one with a compliant chart and a written interpretation. Work in `notebooks/03_analysis_viz.ipynb`.

Analysis steps:

| ID | Step | Output | Acceptance check |
| --- | --- | --- | --- |
| P3-01 | Load `clean_listings.csv` and assert 10,000 rows | Loaded frame | Assertion passes |
| P3-02 | Cross-tabs: body type by fuel type (counts and row %), and region group by make tier | Table 3.1 | Row percentages sum to 100 |
| P3-03 | Multi-index group-by on make tier and age bucket (1 to 5, 6 to 10, 11 to 15, 16 to 20 years): count, median price, median mileage | Table 3.2 | Counts sum to 10,000 |
| P3-04 | Pearson and Spearman correlation matrices for `log_price`, `car_age`, `Mileage`, `mileage_per_year`, cylinders, `n_features`, with p-values for the Spearman pairs | Table 3.3, Fig. 3.6 | Price-year and price-mileage Spearman values are near 0, as in profiling |
| P3-05 | Within-model test: divide each price by the median price of its make-model (models with 5 or more listings), then regress that ratio on age and mileage | Table 3.4 | Slopes and confidence intervals reported |
| P3-06 | Feature premium: median price with vs without each of the 6 features, plus the within-model version | Table 3.5, Fig. 3.9 | Agrees with SQL query Q8 |
| P3-07 | Condition effect: Kruskal-Wallis test of the price ratio across the 6 conditions | Table 3.6, Fig. 3.8 | p-value and effect size reported |
| P3-08 | Concentration: share of listings vs share of total market value for the top makes | Table 3.7 | Agrees with SQL query Q6 |
| P3-09 | Write 5 or more non-obvious insights, each with its number, evidence ID and business implication | Insight table | Every insight cites a figure or table |

Insight candidates to test, not to assume: age and mileage explain almost none of the price once make and model are known; value is concentrated in a few luxury makes; features and condition carry no clear premium; price rises with cylinder count. Reference medians already seen: electric 181,353 AED, hybrid 172,234, diesel 146,025, gasoline 100,686; manual 73,182 vs automatic 104,485. Since make is a confounder, report these only alongside the within-model comparison.

Chart catalog (12 figures; archetypes cover bar, line, scatter, histogram, boxplot, heatmap and two interactive Plotly charts):

| ID | Archetype | Library | What it plots | Question or hypothesis |
| --- | --- | --- | --- | --- |
| Fig. 3.1 | Horizontal bar | Seaborn | Listings per make, top 15 | How concentrated is supply? |
| Fig. 3.2 | Bar | Matplotlib | Median price per make, top 15 by listings | Which brands set the price level? |
| Fig. 3.3 | Boxplot | Seaborn | `log_price` by body type | Which body types are most expensive and most variable? |
| Fig. 3.4 | Line | Matplotlib | Median price and listing count by model year | Do newer cars cost more? |
| Fig. 3.5 | Scatter, log y-axis | Seaborn | Mileage vs price, coloured by make tier, with a LOWESS line | Does mileage lower price? |
| Fig. 3.6 | Heatmap | Seaborn | Spearman correlation matrix | Which numeric variables move together? |
| Fig. 3.7 | Grouped bar | Matplotlib | Listings and median price by region group | Is Dubai priced differently? |
| Fig. 3.8 | Boxplot | Seaborn | Price ratio to model median by condition | Does condition move price? |
| Fig. 3.9 | Bar with error bars | Matplotlib | Within-model feature premium | Do features pay for themselves? |
| Fig. 3.10 | Treemap, interactive | Plotly | Make then model, sized by listings, coloured by median price | Where is the value? |
| Fig. 3.11 | Scatter, interactive | Plotly | Car age vs price ratio, hover shows make and model | Which listings are outliers for their model? |
| Fig. 3.12 | Heatmap | Seaborn | Median price by make tier and age bucket (from P3-03) | Does age matter within a tier? |

Chart standards (check each figure against this list before it enters the report):

- The title states the finding, for example "Median price barely changes with model year".
- Both axes carry labels with units. Price axes show AED, with a secondary axis in million VND built from `AED_TO_VND` (`secondary_yaxis` with a function pair in Matplotlib).
- A legend appears whenever colour or marker carries meaning. Use one colourblind-safe palette, and keep the same colour for the same make tier in every figure.
- Fonts at least 10 pt, 150 dpi export, and a source note: "Source: Kaggle UAE used car listings, n = 10,000".
- Plotly figures are exported as PNG for the PDF with `kaleido` and kept as HTML in the repo; add `kaleido` to `requirements.txt`.

Interpretation paragraph (3 to 5 sentences, placed directly under each chart):

1. State the exact numbers or pattern the chart shows.
2. Say what is surprising or notable about it.
3. Say why it matters for pricing, buying or inventory decisions.
4. Link it to a numbered objective from section 3 of Part A.
5. Optionally add one caveat, such as the synthetic-looking year distribution.

Exit criterion: 12 compliant figures each with its paragraph, 7 analysis tables, at least 5 insights in the insight table, and all claims logged in the evidence ledger.

### B4. Phase 4: Regression and machine learning (Report IV.1 to IV.3)

Goal: predict `log(Price_AED)` with one interpretable model (linear regression, for coefficients and p-values) and two stronger models (random forest and gradient boosting, for accuracy), then turn the results into actions. Work in `notebooks/04_modeling.ipynb` with shared code in `src/modeling.py`.

The interpretable model:

```latex
\ln(\mathrm{Price}_i) = \beta_0 + \beta_1\,\mathrm{age}_i + \beta_2\,\ln(\mathrm{mileage}_i) + \beta_3\,\mathrm{cylinders}_i + \sum_k \gamma_k\,\mathrm{category}_{ik} + \varepsilon_i
```

A coefficient `β` means a `(e^β − 1) × 100` percent change in price for one unit, other variables held fixed.

Why each feature group is in the model (this is the justification the template asks for):

| Feature group | Reason for inclusion | Expected direction |
| --- | --- | --- |
| Make (top 20 as dummies, rest as "Other make"; reference `toyota`) | Brand explained about 43% of log-price variance in profiling | Large, brand-specific |
| `car_age` | Depreciation | Negative, probably weak in this data |
| `log_mileage` | Wear and tear | Negative, probably weak |
| Cylinders | Proxy for engine size and segment | Positive |
| Transmission, fuel type, body type (reference `SUV`) | Market segment | Mixed |
| `region_group` (reference Dubai) | Local demand | Small |
| Condition (reference "No damage") and the 6 `has_` flags | Quality signals | Small or zero |

Steps:

| ID | Step | Output | Acceptance check |
| --- | --- | --- | --- |
| P4-01 | Split 80/20 with `random_state=42`, stratified on deciles of `log_price`; save the test row IDs | Split files | Train 8,000 and test 2,000 rows with similar price distributions |
| P4-02 | Baseline M0: predict the training mean of `log_price` | Baseline metrics | R² close to 0 on test |
| P4-03 | M1: OLS in statsmodels with robust (HC3) standard errors; check VIF below 10 for numeric terms, residual-vs-fitted, Q-Q plot and Breusch-Pagan test | Coefficient table with p-values and 95% intervals | Diagnostics shown as figures |
| P4-04 | Optional M1b: Ridge with `RidgeCV` to compare with OLS | Coefficient comparison | Same sign pattern as OLS |
| P4-05 | M2: random forest, tuned by `RandomizedSearchCV` (20 draws, 5-fold CV, RMSE on log scale) over depth and leaf size | Best parameters, CV score | CV score logged |
| P4-06 | M3: XGBoost (or `HistGradientBoostingRegressor` if installation fails), tuned over trees, learning rate, depth and subsample | Best parameters, CV score | CV score logged |
| P4-07 | Compare M0 to M3 by 5-fold CV mean and standard deviation of R² and RMSE | Table 4.1 | One model chosen for prediction by lowest CV RMSE; M1 kept for interpretation |
| P4-08 | Score the held-out test set once with every model | Table 4.2 | No tuning after this step |
| P4-09 | Back-transform to AED with a smearing factor (mean of `exp(residual)` on the training set) and report RMSE and MAE in AED and in million VND, plus MAPE for each price band: under 100,000, 100,000 to 500,000, above 500,000 AED | Table 4.3 | Band errors reported, because AED errors are dominated by supercars |
| P4-10 | Diagnostics: predicted vs actual on log axes, residuals by make tier | Fig. 4.1 to 4.3 | No visible bias by tier, or bias discussed |
| P4-11 | Sensitivity: refit without the 91 log-scale outliers and without `implausible_usage` rows | Table 4.4 | Change in R² and key coefficients stated |
| P4-12 | Interpretation: OLS coefficients as percent effects, permutation importance on the test set (10 repeats), partial dependence of age and mileage | Tables 4.5 and 4.6, Fig. 4.4 and 4.5 | Top 10 effects described in words |
| P4-13 | Out-of-fold predictions for all 10,000 rows with `cross_val_predict`; residual = actual minus predicted log price | `deal_scores.csv` | No row is scored by a model that saw it |
| P4-14 | Flag listings above `+0.25` log points (about 28% over prediction) as overpriced and below `-0.25` (about 22% under) as undervalued; adjust the cut-off to the residual standard deviation and report the share flagged | Table 4.7 | Share flagged between about 5% and 25% |
| P4-15 | Refit the chosen model on all 10,000 rows and save `models/price_model.joblib` and `metadata.json` (features, categories, smearing factor, metrics, library versions, FX rate and date) | Model files | The app loads them without retraining |

Expected range, to confirm and not to promise: profiling gave R² of about 0.54 for linear regression with make and about 0.61 for a random forest on `log_price`. State the final numbers honestly. A moderate R² with a clear explanation (price is set by make and model, while age, mileage and condition add little) is a valid result.

Classification metrics: a confusion matrix needs a class label. The overpriced flag is defined from the model's own residuals, so scoring it as a classifier would be circular. Add classification only if the instructor requires it, using a non-circular target such as a three-class `price_band` (lower, middle and upper third of price) with logistic regression, and report the confusion matrix, accuracy, precision, recall and F1.

Business translation (Table 4.8, each row backed by a coefficient, importance score or flag count):

| Finding to write up | Action | Who uses it |
| --- | --- | --- |
| Make and model set the price level | Start pricing from the model benchmark, then adjust | Dealers, marketplaces |
| Age and mileage have small effects in this data | Do not rely on age or kilometre rules of thumb here, and state the data caveat | Analysts |
| Overpriced listings exceed predicted price by a stated margin | Show a "priced above market" badge and suggest a price range | Marketplaces |
| Undervalued listings sit below predicted price | Rank them first for buyers and for resellers | Buyers, resellers |
| Prediction error is highest in the supercar band | Use manual appraisal above 500,000 AED | Dealers |

Exit criterion: all models evaluated on one untouched test set, Tables 4.1 to 4.8 and Figures 4.1 to 4.5 produced, the saved model reloads and predicts, and each business claim cites a model output.

### B5. Phase 5: Streamlit dashboard and tool review (Report IV.4)

Goal: one dashboard that reads the SQLite database and the saved model, so the report can show SQL, analysis and prediction working together, plus an honest written review of Streamlit.

Architecture:

- Market pages read `data/db/uae_cars.sqlite` through a read-only connection (`file:...?mode=ro`) and cache results with `st.cache_data`.
- The predictor loads `models/price_model.joblib` and `metadata.json` once with `st.cache_resource`.
- Constants (FX rate, reference year) come from `src/config.py`. The sidebar lets the user switch between AED and VND and edit the rate, which defaults to 7,100 and shows its date. The choice is held in `st.session_state` so every page agrees.
- Shared filters in the sidebar: make, body type, region group, model-year range and price range.

Pages:

| Page (file) | Purpose | Main components | Data source |
| --- | --- | --- | --- |
| Home (`app/Home.py`) | Market overview | KPI cards (listings, median and mean price, share in Dubai, top make), 2 charts, a data-caveat note | SQL aggregates |
| Market Explorer (`pages/1_Market_Explorer.py`) | Filterable analysis | Price distribution, median price by make, mileage-vs-price scatter, model-year trend, make treemap, CSV download of the filtered rows | SQL plus pandas |
| Price Predictor (`pages/2_Price_Predictor.py`) | Estimate a fair price | Inputs: make, then model (list depends on make), year, mileage, cylinders, body type, transmission, fuel, location, features, condition. Output: predicted price in AED and VND, a range from the 10th and 90th percentile of out-of-fold residuals, and the model's median for comparison | Saved model |
| Deal Finder (`pages/3_Deal_Finder.py`) | Find under- and over-priced listings | Table with predicted price, gap and flag, filter by flag, predicted-vs-asking scatter | `deal_scores.csv` |
| SQL Console (`pages/4_SQL_Console.py`) | Show the database working | The 12 saved queries as presets, plus a box that accepts only `SELECT`, with a 500-row limit | SQLite, read-only |

Build steps:

| ID | Step | Acceptance check |
| --- | --- | --- |
| P5-01 | Create `app/utils.py` (connection, cached loaders, AED and VND formatter) | Loaders return 10,000 rows |
| P5-02 | Build the sidebar with currency, rate and filters | Changing the toggle updates all pages |
| P5-03 | Build Home and check each KPI against SQL query Q1 | Numbers match exactly |
| P5-04 | Build Market Explorer | Filters change every chart and the row count |
| P5-05 | Build Price Predictor with validation (year 2005 to 2024, mileage 0 to 300,000, make-model pairs from the data only); unseen pairs fall back to the make level with a visible warning | Prediction for a known listing is close to its out-of-fold prediction |
| P5-06 | Build Deal Finder | Flag counts equal Table 4.7 |
| P5-07 | Build SQL Console; reject any statement not starting with `SELECT` and any containing `;` followed by more text | An `UPDATE` attempt is refused |
| P5-08 | Smoke-test every page with default and extreme filters (one make, one emirate, empty result) | No exceptions, friendly empty-state messages |
| P5-09 | Take 6 to 8 screenshots at about 1440 px width: Home, Explorer with a filter, Predictor with a result, Deal Finder, SQL Console | Files in `screenshots/` |
| P5-10 | Deploy to Streamlit Community Cloud from the GitHub repo (entry file `app/Home.py`; commit the database and model files if they are small); keep a screen recording as backup | Public link works from a private browser window |
| P5-11 | Add run instructions to the README: `pip install -r requirements.txt` then `streamlit run app/Home.py` | A teammate runs it from a clean clone |

Tool review (IV.4.2). Keep a short tool log while building (minutes to the first working page, state or caching bugs met, features missed, deployment friction), then write the review from the log so it is evidence and not opinion:

- **Usability:** learning curve for people who already know pandas, how fast pages reached a working state, quality of error messages.
- **Strengths:** pure Python, no front-end code, widgets and caching built in, one command to run.
- **Limitations:** the script reruns on every interaction, so slow code needs caching; layout control is limited; `session_state` takes care; no built-in user accounts; free hosting can put idle apps to sleep. Check the current Community Cloud limits before quoting any.
- **Comparative assessment:** a short table against alternatives.

| Criterion | Streamlit | Power BI | Dash | Custom web stack |
| --- | --- | --- | --- | --- |
| Time to first dashboard | Hours | Hours, if the data is already tabular | A day | Days |
| Reuse of the Python model | Direct | Needs a script or export step | Direct | Direct, via an API |
| Layout and interaction control | Moderate | High for BI visuals | High | Full |
| Skills needed | Python | DAX and the GUI | Python and callbacks | Python plus web development |
| Best fit | Fast analytical apps | Business reporting | Complex interactive analytics | Production products |

Exit criterion: the app runs from a clean clone, Home KPIs match SQL, 6 to 8 screenshots and a deployment link (or recording) exist, and the written review cites the tool log.

### B6. Phase 6: Report assembly, quality check and submission

Goal: write the report inside the official template, keep its headings unchanged, and make sure every conclusion sits next to its evidence. The guidelines give zero credit to any conclusion without evidence, so this phase is a verification pass as much as a writing pass.

Section-by-section map (template heading, what to write, evidence to place, where it comes from):

| Template section | What to write | Evidence to place | Source artifact |
| --- | --- | --- | --- |
| I.1 Problem Statement | UAE used-car market context, the pricing problem, why it matters | Q1 KPI table, Fig. 3.1 | Part A section 3, Q1 |
| I.2 Project Objectives | One general objective and 5 detailed objectives | None needed, but each objective is later cited by number | Part A section 3 |
| I.3 Analytical Approach | Descriptive, diagnostic and predictive analysis, each justified against the objectives | None needed | Part A section 3 |
| II.1 Data Collection | Kaggle URL, licence note, normalization method, schema, data dictionary | ER diagram (Fig. 2.1), DDL screenshot, Table 2.2 | P1-01 to P1-10, P2-03 |
| II.2 EDA | Table inventory, dtypes, descriptive statistics, first observations | Tables 2.1, 2.3 to 2.5, Fig. 2.2 and 2.3 | P2-01 to P2-07 |
| II.3 Preprocessing | Missing values, outliers, type conversions, engineered features, encoding | Before-and-after missing table, boxplots, feature table, encoding code snippet | `02_eda_cleaning.ipynb` |
| III.1 SQL analysis | 12 queries, each with code, output screenshot and business meaning | Q1 to Q12 screenshots | `sql/04_queries.sql` |
| III.2 Python analysis | Cross-tabs, multi-index group-by, correlations, within-model test, insights | Tables 3.1 to 3.7, insight table | `03_analysis_viz.ipynb` |
| III.3.1 Libraries | Which library for which purpose: Matplotlib and Seaborn for static, Plotly for interactive | Short code snippet | Notebook 03 |
| III.3.2 Catalog | The 12 charts with the compliance checklist | Fig. 3.1 to 3.12 | Notebook 03 |
| III.3.3 Interpretation | A 3 to 5 sentence paragraph under every chart | The paragraphs | Notebook 03 |
| IV.1 Model building | Features and their justification, model definitions | Feature-justification table, pipeline code snippet | Notebook 04 |
| IV.2 Model evaluation | R², p-values, MSE and RMSE, band errors, diagnostics | Tables 4.1 to 4.4, Fig. 4.1 to 4.3 | Notebook 04 |
| IV.3 Business value | Coefficients as percent effects, actions | Tables 4.5 to 4.8, Fig. 4.4 and 4.5 | Notebook 04 |
| IV.4.1 Dashboard | Architecture, pages, link | 6 to 8 screenshots | Phase 5 |
| IV.4.2 Tool review | Usability, strengths, limitations, comparison | Comparison table, tool log summary | Phase 5 |

Place limitations where they belong rather than in a new section: the synthetic-looking year and mileage patterns and the single-source data go in II.2 and IV.3, so the template's headings stay intact.

Code snippet rule: show only the lines that produce the stated result (about 5 to 20 lines), in monospace, with a caption, immediately before the output. Full code lives in the repo.

Quality checklist (tick before export):

- [ ] Cover page filled: title, instructor, class code, group number, 3 to 4 members with student IDs, leader marked
- [ ] Every claim has a row in `evidence_ledger.xlsx` with a figure, table, screenshot or code reference
- [ ] Every chart has a title, labelled axes with units, a legend where needed, and its 3 to 5 sentence paragraph
- [ ] SQL section shows filtering, grouping, aggregation, joins, a subquery and window functions, each with code, screenshot and meaning
- [ ] Preprocessing shows missing-value percentages, outlier method, type conversions, the full engineered-feature list and the encoding
- [ ] Regression results show R², coefficient p-values, MSE or RMSE, and the features are justified
- [ ] The same number is identical wherever it appears; spot-check 10 values against the notebooks
- [ ] One AED-to-VND rate is stated once with its date, and the "not a Vietnam price" caveat appears once
- [ ] Notebooks run top to bottom on a clean kernel and reproduce every figure
- [ ] Table of contents refreshed and page numbers correct; headings match the template
- [ ] PDF exported, images legible at 100% zoom, no broken fonts
- [ ] Dataset cited with URL and licence checked; course rules on AI-assisted work checked and followed

Submission package:

1. One PDF exported from the template. Suggested name `ADY201m_Group<X>_FinalReport.pdf`, unless the instructor sets a pattern.
2. Code and data, either as a ZIP or as one public repository link: notebooks, `.sql` files, raw and cleaned CSVs, the SQLite file, `app/`, `models/`, `requirements.txt` and `README.md`.
3. The dashboard link, or the screen recording if hosting fails.

The guidelines set no page limit. Expect roughly 35 to 45 pages with all figures and screenshots, and ask the instructor whether a limit applies.

### B7. Roles, schedule, gates and risks

The schedule below uses working days from the start date, because no deadline is known yet. Stretch or compress it once the deadline is fixed, but keep the order and the gates.

Roles for 4 members (merge B and C for a group of 3):

| Member | Owns | Also writes |
| --- | --- | --- |
| A | Phase 1 database and SQL, then the SQL Console and Home pages of the app | Report II.1 and III.1 |
| B | Phase 2 EDA and preprocessing, data dictionary | Report II.2 and II.3 |
| C | Phase 3 analysis and charts | Report III.2 and III.3 |
| D | Phase 4 modelling, then Predictor and Deal Finder pages | Report IV.1 to IV.3 |
| Group leader (one of the four) | Repo, evidence ledger, integration, PDF export, submission | Report I and IV.4 |

Schedule:

| Days | Work | Owner | Depends on | Gate |
| --- | --- | --- | --- | --- |
| 1 | Phase 0 setup, agree constants and chart style | All | None | Everyone can run the repo |
| 2 to 3 | Build and validate the database (P1-01 to P1-10); start raw-data EDA (P2-01 to P2-05) | A, B | Phase 0 | Validation passes |
| 3 to 5 | SQL queries Q1 to Q12 with screenshots; cleaning and feature engineering (P2-06, preprocessing, features) | A, B | Database built | **G1:** `model_features.csv` and the database are frozen |
| 6 to 8 | Analysis and charts (P3); split, baselines, OLS, tuned models, test scoring (P4-01 to P4-08) | C, D | G1 | **G2:** test metrics frozen, no more tuning |
| 9 to 10 | Back-transform, diagnostics, interpretation, deal scores, saved model (P4-09 to P4-15); finish charts | D, C | G2 | Model reloads and predicts |
| 8 to 11 | App skeleton, Home, Explorer, SQL Console (can start once G1 passes) | A | G1 | Home KPIs match Q1 |
| 11 to 12 | Predictor and Deal Finder pages, screenshots, deployment | D, A | Saved model | **G3:** app runs from a clean clone |
| 10 to 13 | Write report sections as each phase closes; integrate into the template | All, leader | Each phase | All sections drafted |
| 13 to 14 | Evidence audit, checklist, PDF export, package, submit with buffer | Leader | G3 | **G4:** checklist fully ticked |

Risks and mitigations:

| Risk | Likelihood and impact | Mitigation |
| --- | --- | --- |
| The instructor wants a dataset that is relational at the source | Medium, high | Ask now. Fallback: add a second table the team builds and cites, such as make to country of origin or brand group, and join it |
| The data looks synthetic: model years are almost evenly spread and neither age nor mileage tracks price | High, medium | Report it as a finding with the within-model test as evidence. Do not present it as real-market depreciation, and caveat it in II.2 and IV.3 |
| R² is only moderate | High, low | Report the real numbers, add band-level errors, and explain what drives price |
| SQLite has no `MEDIAN` or regex | Certain, low | Use window functions for medians and parse text in Python |
| A claim reaches the report without evidence | Medium, high | Evidence ledger, per-chapter checklist, final audit in Phase 6 |
| Leakage through encoders or `make_tier` | Medium, high | Fit everything inside pipelines on training data only; review at G2 |
| Plotly PNG export fails | Medium, low | Install `kaleido` on day 1; fall back to a screenshot of the HTML chart |
| Free hosting limits or sleep | Medium, medium | Keep the repo small, keep a screen recording as backup |
| Notebook merge conflicts | Medium, low | One notebook per phase and `nbstripout` |
| Exchange rate moves during the project | Low, low | One frozen rate with its date, stated once |

### B8. Sources and as-of notes

- Dataset: [UAE Used Car Prices & Features (10K+ Listings) on Kaggle](https://www.kaggle.com/datasets/alikalwar/uae-used-car-prices-and-features-10k-listings). The page was opened, but only its title and summary were readable, so the licence and column notes are not confirmed here. Read them on the page before submitting.
- Every data figure in this plan (row counts, medians, missing values, outlier counts, baseline R²) comes from profiling the uploaded CSV, `uae_used_cars_10k.csv`, on 1 November 2026. Re-run these in your own notebooks, because the report must show them as outputs.
- AED to VND: approximate figures from search-result snippets, not opened pages: about 7,113 on 23 August 2026 ([currencylive.com](https://currencylive.com/exchange-rate/aed-to-vnd-exchange-rate-today/)) and about 7,158 on 10 July 2026 ([clock7.com](https://www.clock7.com/currency/aed-to-vnd/)). The plan uses 7,100 as a round constant. Check a live rate on the day you freeze it.
