# EUcohesion

## Goal
Estimate whether EU Cohesion Policy (ERDF) reduced regional development gaps across EU NUTS2 regions by increasing GDP per capita growth and accelerating convergence.

## Reproducible Workflow
From project root:

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
python scripts/run_models.py
```

Make targets:

```bash
make pipeline   # build processed datasets + QA
make models     # run econometric analysis + figures + notebook
make report     # alias for make models
```

## Data Build Inputs (Raw)
The dataset build stage (`scripts/run_pipeline.py`) expects these exact files in `data/raw/`:
- `cohesion_esif_2014_2020_finance_implementation_99js-gm52.csv`
- `cohesion_historic_eu_payments_nuts2_timeseries_tc55-7ysv.csv`
- `eurostat_nama_10r_2gdp.csv`
- `eurostat_demo_r_d2jan.csv`
- `eurostat_tgs00010_unemployment_rate.csv`
- `eurostat_tgs00007_employment_rate.csv`
- `eurostat_tgs00109_tertiary_education_25_64.csv`
- `eurostat_tgs00042_gerd_rd_expenditure.csv`
- `eurostat_nama_10r_2gvagr_gva.csv`
- `ref_structural_funds_regional_categories_KS-GQ-18-007-EN-N.pdf`
- `ref_nuts_revision_consistency_KS-GQ-22-010-EN-N.pdf`

If any are missing, the build fails with an explicit list of missing filenames.

## Processed Inputs Used by Analysis
The analysis stage (`scripts/run_models.py`) uses only:
- `data/processed/panel_master.parquet`
- `data/processed/sigma_convergence.csv`

If required columns are missing in these files, `run_models.py` fails with a clear error naming missing columns and indicating they should come from `scripts/build_dataset.py` (`src/pipeline.py`).

## Implemented Dataset Outputs
### Processed
- `data/processed/panel_master.parquet`
- `data/processed/panel_master.csv`
- `data/processed/sigma_convergence.csv`

### Interim
- `data/interim/panel_skeleton.csv`
- `data/interim/treatment_erdf.csv`
- `data/interim/outcomes_gdp.csv`
- `data/interim/controls.csv`
- `data/interim/eligibility_categories.csv`

### Data QA (build stage)
- `outputs/tables/data_quality_summary.csv`
- `outputs/tables/missingness_by_variable_year.csv`
- `outputs/tables/outliers_erdf_pc.csv`
- `outputs/figures/missingness_heatmap.png`
- `outputs/figures/erdf_eur_pc_distribution.png`
- `outputs/figures/gdp_pc_distribution.png`
- `outputs/figures/gdp_pc_trend_treated_vs_untreated_quantiles.png`

## `panel_master` Schema
Current columns in `data/processed/panel_master.parquet`:
- `nuts2_id`, `country`, `year`
- `population`
- `erdf_eur`, `erdf_eur_pc`, `erdf_eur_pc_l1`, `erdf_eur_pc_l2`, `erdf_eur_pc_l3`
- `gdp_mio_eur`, `gdp_eur`, `gdp_pc`, `log_gdp_pc`, `gdp_pc_growth`
- `unemp_rate`, `emp_rate`, `tertiary_share_25_64`, `rd_gerd`, `gva`
- `category_2014_2020`

Additional schema/missingness snapshots from analysis stage:
- `outputs/tables/panel_master_schema.csv`
- `outputs/tables/panel_master_overview.csv`
- `outputs/tables/panel_master_key_missingness.csv`

## Analysis Layer
Implemented in:
- `src/models.py` (estimation + robustness + report notebook generation)
- `src/viz.py` (analysis plots)
- `scripts/run_models.py` (single-command runner)

### Controls Handling
Requested controls: `unemp_rate`, `emp_rate`, `tertiary_share_25_64`, `rd_gerd`, `gva`.
If any are absent, they are omitted with a warning (current panel contains all five).

### Fixed Effects and Clustering
- Region FE always: `C(nuts2_id)`
- Time FE: `C(year)` or `C(country_year)` depending on model
- Baseline SEs: cluster by `nuts2_id`
- Two-way clustering robustness implemented: `nuts2_id` + `country`

## Model Specifications (A-E)
Using outcome `gdp_pc_growth`:

1. Model A (baseline TWFE):
`gdp_pc_growth ~ erdf_eur_pc_l1 + controls + region_FE + year_FE`

2. Model B (national-shock control):
`gdp_pc_growth ~ erdf_eur_pc_l1 + controls + region_FE + country_year_FE`

3. Model C (distributed lags):
`gdp_pc_growth ~ erdf_eur_pc_l1 + erdf_eur_pc_l2 + erdf_eur_pc_l3 + controls + region_FE + year_FE`

4. Model D (beta convergence baseline):
`gdp_pc_growth ~ log_gdp_pc_l1 + region_FE + year_FE`

5. Model E (beta + ERDF interaction):
`gdp_pc_growth ~ log_gdp_pc_l1 + erdf_eur_pc_l1 + erdf_eur_pc_l1*log_gdp_pc_l1 + controls + region_FE + year_FE`

6. Model E (country-year FE variant):
Same as Model E but with `country_year_FE` instead of `year_FE`.

### Additional Implemented Checks
- Dynamic response plot from lag coefficients (`l0-l3` where available).
- Placebo leads-lags falsification with leads `f1`, `f2`.
- Outlier robustness: drop global top 1% of `erdf_eur_pc`.
- Balanced panel robustness: regions complete for all baseline variables in window `2017-2023`.
- Scaling robustness: `erdf_k_eur_pc_l1 = erdf_eur_pc_l1 / 1000`.

## Analysis Outputs
### Tables
- `outputs/tables/twfe_main_results.csv`
- `outputs/tables/dl_lags_results.csv`
- `outputs/tables/dynamic_lag_response.csv`
- `outputs/tables/leads_lags_results.csv`
- `outputs/tables/beta_convergence_results.csv`
- `outputs/tables/robustness_outliers.csv`
- `outputs/tables/robustness_balanced_panel.csv`
- `outputs/tables/robustness_scaling.csv`
- `outputs/tables/model_comparison_summary.csv`

### Figures
- `outputs/figures/dynamic_lag_response.png`
- `outputs/figures/leads_lags_placebo.png`
- `outputs/figures/sigma_convergence.png`
- `outputs/figures/beta_convergence_partial.png`

### Report Notebook
- `notebooks/01_report.ipynb` (generated and executed top-to-bottom by `scripts/run_models.py`)

## Results (Current Run)
Values below come from the current generated CSV outputs.

### Main causal models
- Model A (`erdf_eur_pc_l1`): coef `0.0021`, SE `0.0037`, p `0.571`.
- Model B (`erdf_eur_pc_l1`): coef `0.0036`, SE `0.0042`, p `0.383`.
- Two-way cluster versions remain statistically insignificant:
  - Model A: SE `0.0049`, p `0.669`.
  - Model B: SE `0.0057`, p `0.521`.

### Distributed lags (Model C)
- `erdf_eur_pc_l1`: `0.0099` (p `0.059`)
- `erdf_eur_pc_l2`: `0.0140` (p `0.0056`)
- `erdf_eur_pc_l3`: `0.0056` (p `0.140`)

### Placebo leads-lags
- Leads are not statistically significant:
  - `f2`: p `0.436`
  - `f1`: p `0.823`
- Lag coefficients are positive and significant in that specification:
  - `l1`: p `0.0085`
  - `l2`: p `0.0022`
  - `l3`: p `0.0273`

### Convergence
- Sigma convergence series available for `2016-2023`.
- `sigma_log_gdp` falls from `0.726` (2016) to `0.627` (2023).
- Beta convergence (Model D): `log_gdp_pc_l1 = -33.50` (p `< 1e-39`), consistent with convergence.
- In Model E, ERDF main effect and ERDF×income interaction are not statistically significant:
  - `erdf_eur_pc_l1`: p `0.340`
  - `erdf_eur_pc_l1:log_gdp_pc_l1`: p `0.352`

### Robustness
- Outlier exclusion (global p99 threshold `264.11`) leaves A/B conclusions unchanged.
- Balanced panel (`2017-2023`, `115` regions) leaves A/B conclusions unchanged.
- Scaling to EUR 1,000 per capita only rescales coefficients, not significance.

## Assumptions and Limitations / TODO
- GDP is currently nominal (`MIO_EUR`); interpretation is limited for real growth comparisons.
  - Future improvement: switch to PPS/real-deflated measure in dataset build.
- `category_2014_2020` is currently placeholder `NaN`.
  - Heterogeneity by less-developed/transition/more-developed category is not yet estimated.
- Coverage overlap for fully controlled growth regressions is concentrated in `2017-2023` due missingness.

## Determinism and Offline Behavior
- No automatic internet downloads in build or analysis stages.
- Outputs are deterministic given fixed processed inputs.
- Scripts run end-to-end from project root with one command each.
