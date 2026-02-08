# EUcohesion

## Goal
Estimate whether EU Cohesion Policy (ERDF) reduced regional development gaps across EU NUTS2 regions by increasing GDP per capita growth and accelerating convergence.

## V3.1 Status
V3.1 is implemented and reproducible end-to-end.

- Added explicit RD/IV viability diagnostics and transparent estimator selection.
- Upgraded identification layer with candidate instruments and cross-sectional IV fallback.
- Rebuilt standalone `outputs/report.html` as a tabbed, interactive, no-scroll report.

## Reproducible Workflow
From project root:

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
python scripts/run_models.py
```

Make targets:

```bash
make pipeline
make models
make report
```

`make report` is an alias of `make models`.

## Raw Inputs
### Existing raw files
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

### Cached downloads (fetched once if missing)
- `data/raw/eurostat_gdp_pc_pps_nuts2.csv`
- `data/raw/eurostat_gdp_real_nuts2.csv`
- `data/raw/eurostat_gdp_pc_pps_rel_eu_nuts2.csv`
- `data/raw/eligibility_categories_2014_2020.csv`

After first successful run, these caches are reused offline.

## Build Outputs
### Processed
- `data/processed/panel_master.parquet`
- `data/processed/panel_master.csv`
- `data/processed/sigma_convergence.csv`
- `data/processed/running_variable_eligibility.csv`
- `data/processed/erdf_cumulative_exposure.csv`

### Interim
- `data/interim/panel_skeleton.csv`
- `data/interim/treatment_erdf.csv`
- `data/interim/outcomes_gdp.csv`
- `data/interim/controls.csv`
- `data/interim/eligibility_categories.csv`

### Build QA
- `outputs/tables/data_quality_summary.csv`
- `outputs/tables/missingness_by_variable_year.csv`
- `outputs/tables/outliers_erdf_pc.csv`
- `outputs/figures/missingness_heatmap.png`
- `outputs/figures/erdf_eur_pc_distribution.png`
- `outputs/figures/gdp_pc_distribution.png`
- `outputs/figures/gdp_pc_pps_distribution.png`
- `outputs/figures/gdp_pc_real_distribution.png`
- `outputs/figures/gdp_pc_trend_treated_vs_untreated_quantiles.png`
- `outputs/figures/growth_nominal_pps_real_trends.png`

## Identification (V3.1)
Implemented in:
- `src/models.py`
- `src/viz.py`
- `scripts/run_models.py`

### Running variable and thresholds
Running variable:

`r_i = (GDPpc_PPS_i / EU_avg_GDPpc_PPS) * 100`

- `eligible_lt75 = 1[r_i < 75]`
- `eligible_lt90 = 1[r_i < 90]`

Saved in `data/processed/running_variable_eligibility.csv`.

### RD diagnostics and use
- RD first-stage test: discontinuity in cumulative ERDF at 75 cutoff.
- Output: `outputs/tables/rd_first_stage_funding_jump_v31.csv`, `outputs/figures/rd_first_stage_funding_jump_v31.png`.
- Result: first-stage F is weak across bandwidths (all < 2), so fuzzy RD is treated as not viable for causal ERDF LATE.
- RD retained as sharp-RD eligibility ITT evidence.

### IV candidate set
Candidates tested (with explicit first-stage table):
1. `Z1_it = eligible_lt75_i × post2014_t`
2. `Z2_it = eligible_lt75_i × EU_mean_ERDF_t`
3. `Z3_it = eligible_lt75_i × country_mean_ERDF_t (leave-one-out)`
4. `Z4_i = eligible_lt75_i` for cumulative exposure (cross-section)
5. `Z5_it = eligible_lt90_i × EU_mean_ERDF_t`

Diagnostics output:
- `outputs/tables/iv_first_stage_candidates_v31.csv`

### Headline estimator selection rule
Estimator is chosen transparently from diagnostics:
- prefer IV candidate with strongest IV first-stage for headline window,
- require strong first-stage (F >= 10),
- otherwise fallback to strict TWFE benchmark.

## V3.1 Analysis Outputs
### Core tables
- `outputs/tables/twfe_main_results_v31.csv`
- `outputs/tables/dl_lags_results_v31.csv`
- `outputs/tables/dynamic_lag_response_v31.csv`
- `outputs/tables/leads_lags_results_v31.csv`
- `outputs/tables/beta_convergence_results_v31.csv`
- `outputs/tables/heterogeneity_by_category_v31.csv`
- `outputs/tables/robustness_outliers_v31.csv`
- `outputs/tables/robustness_balanced_panel_v31.csv`
- `outputs/tables/robustness_scaling_v31.csv`

### RD/IV diagnostics and results
- `outputs/tables/rd_first_stage_funding_jump_v31.csv`
- `outputs/tables/rd_outcome_sharp_results_v31.csv`
- `outputs/tables/rd_bandwidth_sensitivity_v31.csv`
- `outputs/tables/rd_placebo_pretrend_v31.csv`
- `outputs/tables/iv_first_stage_candidates_v31.csv`
- `outputs/tables/iv_2sls_results_v31.csv`
- `outputs/tables/iv_cross_section_results_v31.csv`
- `outputs/tables/model_comparison_summary_v31.csv`

### Figures
- `outputs/figures/dynamic_lag_response_v31.png`
- `outputs/figures/leads_lags_placebo_v31.png`
- `outputs/figures/sigma_convergence_v31.png`
- `outputs/figures/beta_convergence_partial_v31.png`
- `outputs/figures/rd_first_stage_funding_jump_v31.png`
- `outputs/figures/rd_outcome_binned_scatter_v31.png`
- `outputs/figures/rd_bandwidth_sensitivity_v31.png`
- `outputs/figures/iv_first_stage_scatter_v31.png`

### Report deliverables
- `outputs/report.html` (single-file standalone, tabbed, interactive)
- `outputs/tables/report_inputs_manifest.json`
- `notebooks/01_report.ipynb` (generated and executed)

## V3.1 Results (latest run)
Headline outcome: `gdp_pc_real_growth`.

### Headline estimator (selected)
Selected: cross-section IV with cumulative exposure (`erdf_eur_pc_cum_2014_2020`) instrumented by `eligible_lt90` (`Z4_90`), with country FE + pre controls.

- Headline window `2016-2020`: coef `-0.00363`, SE `0.00213`, p `0.088`, first-stage F `15.65`.
- Longer-run window `2021-2023`: coef `-0.00509`, SE `0.00280`, p `0.069`, first-stage F `15.65`.

### RD viability diagnostics
Funding-jump first-stage around 75 cutoff is weak across bandwidths:
- F-stat range ~`1.08` to `1.69` (bw `5` to `20`), p-values > `0.19` throughout.
- Conclusion: fuzzy RD not used as a causal ERDF LATE estimator in V3.1.

### Sharp RD (eligibility ITT)
At bandwidth `10`:
- Real growth, post `2016-2020`: coef `-0.159`, p `0.908`.
- Real growth, post `2021-2023`: coef `2.420`, p `0.013`.
- Placebo pretrend (`2010-2013`): coef `-1.942`, p `0.429`.

### TWFE benchmark (kept)
- Model A (`erdf_eur_pc_l1`): coef `0.00568`, p `0.060`.
- Model B (`erdf_eur_pc_l1`): coef `0.00336`, p `0.244`.
- Model C lags:
  - `l1`: `0.011999` (p `0.013`)
  - `l2`: `0.007897` (p `0.0366`)
  - `l3`: `-0.003085` (p `0.247`)

### Convergence and robustness highlights
- Sigma (`2000` -> `2023`):
  - PPS: `0.534` -> `0.391`
  - Real: `0.714` -> `0.657`
- Beta convergence Model D:
  - Real `log_gdp_pc_real_l1`: `-5.267` (p `2.14e-18`)
  - PPS `log_gdp_pc_pps_l1`: `-7.725` (p `2.80e-70`)
- Robustness (headline TWFE): conclusions broadly unchanged across outlier exclusion, balanced panel, and scaling checks.

## Limitations / Guardrails
- Real GDP per capita remains reconstructed from regional volume indices (not direct chain-linked EUR levels).
- Eligibility/running-variable coverage is partial relative to full panel due NUTS coverage differences.
- RD fuzzy design is weak in this data; sharp RD should be interpreted as eligibility ITT.
- IV claims rely on policy-threshold exclusion assumptions; interpret with caution.

## Determinism and Offline Behavior
- Build stage only downloads missing cached raws and stores them in `data/raw/`.
- Models stage does not download data.
- Outputs are deterministic given fixed inputs and scripts.
