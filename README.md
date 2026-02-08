# EUcohesion

## Goal
Estimate whether EU Cohesion Policy (ERDF) reduced regional development gaps across EU NUTS2 regions by increasing GDP per capita growth and accelerating convergence.

## V3 Status
V3 is implemented and reproducible end-to-end.

- Data build now outputs policy-rule artifacts for identification (`running_variable_eligibility.csv`, `erdf_cumulative_exposure.csv`).
- Models stage runs TWFE benchmarks plus RD and IV identification around eligibility.
- `outputs/report.html` is a standalone, interactive, styled report (single local HTML file).

## Reproducible Workflow
From project root:

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
python scripts/run_models.py
```

Make targets:

```bash
make pipeline   # V3 data build (+ cached downloads if missing)
make models     # V3 TWFE + RD + IV + robustness + notebook + report.html
make report     # alias for make models
```

After first successful run, downloaded V2/V3 raw caches are stored in `data/raw/` and reruns are offline.

## Raw Inputs
### Existing raw inputs
- `data/raw/cohesion_esif_2014_2020_finance_implementation_99js-gm52.csv`
- `data/raw/cohesion_historic_eu_payments_nuts2_timeseries_tc55-7ysv.csv`
- `data/raw/eurostat_nama_10r_2gdp.csv`
- `data/raw/eurostat_demo_r_d2jan.csv`
- `data/raw/eurostat_tgs00010_unemployment_rate.csv`
- `data/raw/eurostat_tgs00007_employment_rate.csv`
- `data/raw/eurostat_tgs00109_tertiary_education_25_64.csv`
- `data/raw/eurostat_tgs00042_gerd_rd_expenditure.csv`
- `data/raw/eurostat_nama_10r_2gvagr_gva.csv`
- `data/raw/ref_structural_funds_regional_categories_KS-GQ-18-007-EN-N.pdf`
- `data/raw/ref_nuts_revision_consistency_KS-GQ-22-010-EN-N.pdf`

### Cached Eurostat downloads (auto-fetched once if missing)
- `data/raw/eurostat_gdp_pc_pps_nuts2.csv` (`nama_10r_2gdp`, `unit=PPS_EU27_2020_HAB`)
- `data/raw/eurostat_gdp_real_nuts2.csv` (`nama_10r_2gvagr`, `na_item=B1GQ`, `unit=I15`)
- `data/raw/eurostat_gdp_pc_pps_rel_eu_nuts2.csv` (`nama_10r_2gdp`, `unit=PPS_HAB_EU27_2020`)
- `data/raw/eligibility_categories_2014_2020.csv` (reconstructed from PPS relative-to-EU thresholds)

If files are missing and cannot be prepared, the build fails with explicit missing filenames.

## Data Build (V3)
Implemented in:
- `src/ingest.py`
- `src/pipeline.py`

### Processed outputs
- `data/processed/panel_master.parquet`
- `data/processed/panel_master.csv`
- `data/processed/sigma_convergence.csv`
- `data/processed/running_variable_eligibility.csv`
- `data/processed/erdf_cumulative_exposure.csv`

### Interim outputs
- `data/interim/panel_skeleton.csv`
- `data/interim/treatment_erdf.csv`
- `data/interim/outcomes_gdp.csv`
- `data/interim/controls.csv`
- `data/interim/eligibility_categories.csv`

### QA outputs
Tables:
- `outputs/tables/data_quality_summary.csv`
- `outputs/tables/missingness_by_variable_year.csv`
- `outputs/tables/outliers_erdf_pc.csv`

Figures:
- `outputs/figures/missingness_heatmap.png`
- `outputs/figures/erdf_eur_pc_distribution.png`
- `outputs/figures/gdp_pc_distribution.png`
- `outputs/figures/gdp_pc_pps_distribution.png`
- `outputs/figures/gdp_pc_real_distribution.png`
- `outputs/figures/gdp_pc_trend_treated_vs_untreated_quantiles.png`
- `outputs/figures/growth_nominal_pps_real_trends.png`

## Panel Schema (current)
`data/processed/panel_master.parquet` includes:
- IDs/keys: `nuts2_id`, `country`, `year`
- Population/treatment: `population`, `erdf_eur`, `erdf_eur_pc`, `erdf_eur_pc_l1..l3`
- Nominal outcome: `gdp_pc`, `log_gdp_pc`, `gdp_pc_growth`
- PPS outcome: `gdp_pc_pps`, `log_gdp_pc_pps`, `gdp_pc_pps_growth`
- Real outcome: `gdp_volume_index`, `gdp_pc_real`, `log_gdp_pc_real`, `gdp_pc_real_growth`
- Controls: `unemp_rate`, `emp_rate`, `tertiary_share_25_64`, `rd_gerd`, `gva`
- Eligibility: `category_2014_2020`, `r_value`, `eligible_lt75`, `ref_years_used`
- Program exposure: `erdf_eur_pc_cum_2014_2020`, `erdf_eur_pc_cum_2015_2020`

## V3 Identification Strategy
Implemented in:
- `src/models.py`
- `src/viz.py`
- `scripts/run_models.py`

### Running variable and policy rule
For region `i`, running variable:

`r_i = (GDPpc_PPS_i / EU_avg_GDPpc_PPS) * 100`

(using reference average years 2007–2009; fallback up to 2013).

Threshold for less-developed eligibility: `r_i < 75`.

Saved in `data/processed/running_variable_eligibility.csv` with:
- `nuts2_id`, `country`, `r_value`, `eligible_lt75`, `category_2014_2020`, `ref_years_used`

### RD
Local-linear RD (triangular kernel), bandwidth grid `{5, 7.5, 10, 12.5, 15}` around cutoff 75.

- Main windows: `2016-2020` and `2021-2023` average growth
- Placebo: pre-period `2010-2013` average growth
- Sharp RD: treatment jump at `eligible_lt75`
- Fuzzy RD: treatment = `erdf_eur_pc_cum_2014_2020`, instrumented by `eligible_lt75`

### IV (headline estimator)
Headline estimator is panel IV (2SLS) for `gdp_pc_real_growth`:

- Endogenous regressor: `erdf_eur_pc_l1`
- Instrument:

`Z_it = eligible_lt75_i * eu_erdf_intensity_t`

where `eu_erdf_intensity_t` is EU-wide annual mean ERDF per-capita intensity.

- FE: `region FE + year FE` (country×year FE attempted; skipped when rank-deficient)
- SE: clustered by region (plus two-way cluster robustness where feasible)
- First-stage diagnostics reported.

### TWFE kept as benchmark
- Model A: `outcome ~ erdf_eur_pc_l1 + controls + region_FE + year_FE`
- Model B: `outcome ~ erdf_eur_pc_l1 + controls + region_FE + country×year_FE`
- Model C: `outcome ~ erdf_eur_pc_l1 + erdf_eur_pc_l2 + erdf_eur_pc_l3 + controls + region_FE + year_FE`

## V3 Analysis Outputs
### Tables
- `outputs/tables/twfe_main_results_v3.csv`
- `outputs/tables/dl_lags_results_v3.csv`
- `outputs/tables/dynamic_lag_response_v3.csv`
- `outputs/tables/leads_lags_results_v3.csv`
- `outputs/tables/beta_convergence_results_v3.csv`
- `outputs/tables/heterogeneity_by_category_v3.csv`
- `outputs/tables/robustness_outliers_v3.csv`
- `outputs/tables/robustness_balanced_panel_v3.csv`
- `outputs/tables/robustness_scaling_v3.csv`
- `outputs/tables/rd_main_results_v3.csv`
- `outputs/tables/rd_placebo_pretrend_v3.csv`
- `outputs/tables/rd_bandwidth_sensitivity_v3.csv`
- `outputs/tables/iv_2sls_results_v3.csv`
- `outputs/tables/iv_first_stage_v3.csv`
- `outputs/tables/model_comparison_summary_v3.csv`
- `outputs/tables/panel_master_schema.csv`
- `outputs/tables/panel_master_overview.csv`
- `outputs/tables/panel_master_key_missingness.csv`

### Figures
- `outputs/figures/sigma_convergence_v3.png`
- `outputs/figures/dynamic_lag_response_v3.png`
- `outputs/figures/leads_lags_placebo_v3.png`
- `outputs/figures/beta_convergence_partial_v3.png`
- `outputs/figures/rd_binned_scatter_v3.png`
- `outputs/figures/rd_bandwidth_sensitivity_v3.png`

### Report deliverables
- `outputs/report.html` (standalone, interactive, no server)
- `outputs/tables/report_inputs_manifest.json`
- `notebooks/01_report.ipynb` (generated + executed)

## V3 Results (latest run)
Headline outcome: `gdp_pc_real_growth`.

### TWFE benchmark
- Model A (`erdf_eur_pc_l1`): coef `0.00568`, SE `0.00302`, p `0.060`.
- Model B (`erdf_eur_pc_l1`): coef `0.00336`, SE `0.00289`, p `0.244`.
- Model C lags:
  - `l1`: `0.011999` (p `0.013`)
  - `l2`: `0.007897` (p `0.0366`)
  - `l3`: `-0.003085` (p `0.247`)

### IV (headline)
- Panel IV (`region FE + year FE`, cluster by region):
  - coef on `erdf_eur_pc_l1`: `0.00423`
  - SE `0.05907`, p `0.943`
  - first-stage F `1.54` (weak instrument warning)

### RD
At bandwidth 10 around the 75 threshold:
- Real growth, post `2016-2020`:
  - Sharp RD jump: `-0.159` (p `0.908`)
  - Fuzzy RD: `-0.00030` (p `0.905`, first-stage F `1.79`)
- Real growth, post `2021-2023`:
  - Sharp RD jump: `2.420` (p `0.013`)
  - Fuzzy RD: `0.00464` (p `0.256`)
- Placebo pretrend (`2010-2013`, sharp RD): `-1.942` (p `0.429`)

### Convergence and heterogeneity
- Sigma convergence (`2000` → `2023`):
  - PPS sigma: `0.534` → `0.391`
  - Real sigma: `0.714` → `0.657`
- Beta convergence Model D:
  - Real: `log_gdp_pc_real_l1 = -5.267` (p `2.14e-18`)
  - PPS: `log_gdp_pc_pps_l1 = -7.725` (p `2.80e-70`)
- Category subset (Model E, real, ERDF term):
  - `less_developed`: `-0.0358` (p `0.083`)
  - `transition`: `0.1412` (p `0.036`)
  - `more_developed`: `-0.0246` (p `0.706`)

### Robustness (headline)
- Outliers excluded (global p99 `264.11`): Model A coef `0.00632` (p `0.037`), Model B p `0.275`.
- Balanced panel (`2017-2023`, `115` regions): Model A p `0.070`, Model B p `0.370`.
- Scaling (`EUR 1,000` per-capita): Model A coef `5.680` (p `0.060`).

## Limitations / TODO
- Real GDP per capita remains a volume-index anchored reconstruction, not direct chain-linked real EUR levels.
- Eligibility/running-variable coverage is partial vs full panel due NUTS coverage differences (`279` mapped regions vs `398` panel regions).
- IV first-stage is weak in the current specification; causal magnitude should be interpreted cautiously.
- Country×year FE IV variants are rank-deficient on current overlap and are skipped with warnings.

## Determinism and Offline Behavior
- Build stage downloads only missing cache files and writes them to `data/raw/`.
- Models stage does not download data.
- Given fixed raw/processed inputs, outputs are deterministic.
