# EUcohesion

## Goal
Estimate whether **EU Cohesion Policy (ERDF)** reduced **regional development gaps** across EU **NUTS2** regions by increasing **GDP per capita growth** and accelerating **convergence**.

---

## Research question (problematic)
1) **Causal impact**
- What is the causal effect of **ERDF expenditure intensity** on subsequent **GDP per capita growth** at the **NUTS2** level?

2) **Convergence**
- To what extent did ERDF spending during the **2014–2020** programming period contribute to:
  - **beta convergence** (poorer regions catch up faster), and
  - **sigma convergence** (dispersion of GDP per capita across regions declines),

with outcomes tracked into the **post-2020 years** (after 2020, subject to data availability).

---

## Programming period
**Multiannual Financial Framework (MFF): 2014–2020**

---

## Data sources
Main hub (DG REGIO research datasets):
`https://ec.europa.eu/regional_policy/policy/evaluations/data-for-research_en`

### A) Treatment: ERDF / cohesion funding intensity
1) ESIF 2014–2020 Finance Implementation Details (99js-gm52)  
`https://cohesiondata.ec.europa.eu/2014-2020-Finances/ESIF-2014-2020-Finance-Implementation-Details/99js-gm52/about_data`

2) Historic EU payments – annual timeseries, regionalised & modelled (NUTS2) (tc55-7ysv)  
`https://cohesiondata.ec.europa.eu/Other/Historic-EU-payments-annual-timeseries-regionalise/tc55-7ysv/about_data`

### B) Outcomes: regional economic performance
3) Eurostat NUTS2 regional GDP — `nama_10r_2gdp`  
`https://ec.europa.eu/eurostat/databrowser/product/view/nama_10r_2gdp`

4) Eurostat NUTS2 population — `demo_r_d2jan`  
`https://ec.europa.eu/eurostat/databrowser/product/page/demo_r_d2jan`

### C) Policy classification / eligibility (harmonisation + heterogeneity)
5) Structural Funds regional categories (less developed / transition / more developed)  
`https://ec.europa.eu/eurostat/documents/3859598/9397402/KS-GQ-18-007-EN-N.pdf`

### D) Controls (time-varying, NUTS2)
6) Labour market / cycle
- Unemployment rate — `tgs00010`  
  `https://ec.europa.eu/eurostat/databrowser/view/tgs00010/default/table?lang=en`
- Employment rate — `tgs00007`  
  `https://ec.europa.eu/eurostat/databrowser/view/tgs00007/default/table?lang=en`

7) Human capital
- Tertiary educational attainment (25–64) — `tgs00109`  
  `https://ec.europa.eu/eurostat/databrowser/view/tgs00109/default/table?lang=en`

8) Innovation
- Intramural R&D expenditure (GERD) by NUTS2 — `tgs00042`  
  `https://ec.europa.eu/eurostat/databrowser/view/tgs00042/default/table?lang=en`

9) Economic structure / real activity proxy
- GVA by NUTS2 — `nama_10r_2gvagr`  
  `https://ec.europa.eu/eurostat/databrowser/view/nama_10r_2gvagr/default/table?lang=en`

### E) Geographic consistency (NUTS changes)
10) NUTS revision / regional consistency reference  
`https://ec.europa.eu/eurostat/documents/3859598/15193590/KS-GQ-22-010-EN-N.pdf`

---

## Local data downloaded 

### Cohesion / ERDF funding
- `data/raw/cohesion_esif_2014_2020_finance_implementation_99js-gm52.csv`
- `data/raw/cohesion_historic_eu_payments_nuts2_timeseries_tc55-7ysv.csv`

### Eurostat outcomes + scaling
- `data/raw/eurostat_nama_10r_2gdp.csv`
- `data/raw/eurostat_demo_r_d2jan.csv`

### Eurostat controls
- `data/raw/eurostat_tgs00010_unemployment_rate.csv`
- `data/raw/eurostat_tgs00007_employment_rate.csv`
- `data/raw/eurostat_tgs00109_tertiary_education_25_64.csv`
- `data/raw/eurostat_tgs00042_gerd_rd_expenditure.csv`
- `data/raw/eurostat_nama_10r_2gvagr_gva.csv`

### Reference PDFs 
- `data/raw/ref_structural_funds_regional_categories_KS-GQ-18-007-EN-N.pdf`
- `data/raw/ref_nuts_revision_consistency_KS-GQ-22-010-EN-N.pdf`

---

## Methodology: build the analysis dataset (NUTS2 region-year panel)

### 0) Fixed conventions (set in `src/config.py`)
Define once and reuse everywhere:
- Geographic unit: **NUTS2**
- Time unit: **year**
- Analysis window: include **pre-2014** (for pre-trends) and **post-2020** (for outcomes), subject to availability.
- Reference NUTS vintage: choose one and map all sources to it (using the NUTS reference PDF).
- Treatment: **ERDF payments/expenditure per capita** (prefer disbursed amounts over allocations).
- Outcome (choose a primary + keep alternatives as robustness):
  - Primary A (gaps/convergence): **GDP per capita in PPS (level)**, use logs.
  - Primary B (growth impact): **real GDP per capita growth** (log-difference).

### 1) Create the panel backbone (skeleton)
1. Build a master list of NUTS2 regions for the chosen reference vintage.
2. Create a region-year grid: all `nuts2_id × year`.
3. Add:
   - `nuts2_id` (string)
   - `country` (first two letters of `nuts2_id`)
   - `year` (int)

Output: `data/interim/panel_skeleton.csv`

### 2) Harmonize NUTS codes across datasets
1. Identify the NUTS vintage per source.
2. Map all region codes to the chosen reference NUTS vintage.
3. If splits/merges occur:
   - Monetary totals: allocate/aggregate using **population weights** (document the rule).
   - GDP and population: recompute per-capita *after* mapping.

Output: consistent `nuts2_id` across all intermediate datasets.

### 3) Construct treatment: ERDF expenditure intensity
Inputs:
- `cohesion_esif_2014_2020_finance_implementation_99js-gm52.csv`
- `cohesion_historic_eu_payments_nuts2_timeseries_tc55-7ysv.csv`
- `eurostat_demo_r_d2jan.csv`

Steps:
1. Filter funding data to **Fund = ERDF** (baseline spec).
2. Choose financial concept (baseline): **payments/expenditure** (money disbursed).
3. Aggregate to `nuts2_id-year` (sum EUR).
4. Create per-capita intensity:
   - `erdf_eur_pc = erdf_eur / population`
5. Create lags (baseline):
   - `erdf_eur_pc_l1`, `erdf_eur_pc_l2`, `erdf_eur_pc_l3`
6. Optional dynamic summaries:
   - `erdf_eur_pc_l1_l3_sum` (cumulative lagged exposure)

Output: `data/interim/treatment_erdf.csv`

### 4) Construct outcomes: GDP per capita and growth
Inputs:
- `eurostat_nama_10r_2gdp.csv`
- `eurostat_demo_r_d2jan.csv`

Steps:
1. Compute GDP per capita series consistent with the main outcome choice.
2. Recommended variables:
   - `gdp_pc_pps` (level, if available) and `log_gdp_pc_pps`
   - `gdp_pc_real` (real/volume) and `gdp_pc_real_growth = 100 * (log(gdp_pc_real) - log(gdp_pc_real_l1))`

Output: `data/interim/outcomes_gdp.csv`

### 5) Build time-varying controls (NUTS2-year)
Inputs:
- unemployment, employment, education, R&D, GVA

Steps:
1. Clean to `nuts2_id-year` format.
2. Harmonize NUTS codes (step 2).
3. Standardize names (suggested):
   - `unemp_rate`
   - `emp_rate`
   - `tertiary_share_25_64`
   - `rd_gerd_eur` (or per capita variant if chosen)
   - `gva_real` (or sector structure proxy if available)

Output: `data/interim/controls.csv`

### 6) Add eligibility / region category
Input:
- `ref_structural_funds_regional_categories_KS-GQ-18-007-EN-N.pdf`

Steps:
1. Create a mapping `nuts2_id → category_2014_2020` in:
   - `{less_developed, transition, more_developed}`
2. Store it as a clean lookup table.

Output: `data/interim/eligibility_categories.csv`

### 7) Merge into the master analysis panel
Merge order (recommended):
1. `panel_skeleton`
2. population
3. outcomes
4. treatment (+ lags)
5. controls
6. eligibility category

Hard checks after each merge:
- uniqueness of `(nuts2_id, year)`
- row count should remain stable
- log merge match rates

Output (analysis-ready):
- `data/processed/panel_master.parquet`
- (optional) `data/processed/panel_master.csv`

### 8) Data QA (required)
Generate and save:
- missingness report (by variable, year)
- outlier flags for `erdf_eur_pc` and GDP variables
- basic descriptive plots (distributions and trends)
- sanity checks: population > 0, non-negative € flows, etc.

Outputs:
- `outputs/tables/data_quality_summary.csv`
- `outputs/figures/` (png/pdf)

### 9) Convergence datasets (derived)
- **Sigma convergence** time series:
  - `sigma_log_gdp = std_dev_across_regions(log_gdp_pc_pps)` by year
  - Output: `data/processed/sigma_convergence.csv`
- **Beta convergence** regression-ready variables:
  - growth, initial income, interactions
  - Output: contained in `panel_master`

---

## Code structure
```text
eu-cohesion-policy/
  README.md
  environment.yml          # or requirements.txt
  .gitignore
  data/
    raw/                   # downloaded originals (or omitted if too big)
    interim/
    processed/
  notebooks/
    01_report.ipynb        # narrative + figures + results
  src/
    __init__.py
    config.py              # paths, constants, NUTS mapping choices
    ingest.py              # load raw files
    clean.py               # harmonize regions, standardize units
    features.py            # treatment intensity, outcomes, controls, lags
    models.py              # panel FE / event study
    viz.py                 # standard plots
    utils.py
  scripts/
    run_pipeline.py        # end-to-end
    build_dataset.py
    run_models.py
  outputs/
    figures/
    tables/
  tests/                   # optional (strong signal)