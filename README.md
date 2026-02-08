# EUcohesion

## Goal
Estimate whether EU Cohesion Policy (ERDF) reduced regional development gaps across EU NUTS2 regions by increasing GDP per capita growth and accelerating convergence.

## V2 Status
V2 is implemented and reproducible end-to-end.

- Build stage adds PPS and real outcomes, non-empty eligibility categories, updated sigma convergence.
- Models stage runs multi-outcome causal/convergence specs and robustness.
- Standalone local report is generated at `outputs/report.html`.

## Reproducible Workflow
From project root:

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
python scripts/run_models.py
```

Make targets:

```bash
make pipeline   # V2 data build (downloads missing V2 raw caches once)
make models     # V2 estimation + robustness + notebook + HTML report
make report     # alias for make models
```

After first successful run, all V2 downloaded files are cached in `data/raw/` and the project runs fully offline.

## Raw Inputs
### Existing raw inputs (V1)
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

### New V2 cached raw inputs (downloaded from official Eurostat API)
- `data/raw/eurostat_gdp_pc_pps_nuts2.csv`
  - Source: `nama_10r_2gdp`, `freq=A`, `unit=PPS_EU27_2020_HAB`
- `data/raw/eurostat_gdp_real_nuts2.csv`
  - Source: `nama_10r_2gvagr`, `freq=A`, `na_item=B1GQ`, `unit=I15`
- `data/raw/eligibility_categories_2014_2020.csv`
  - Reconstructed (Option 2) from Eurostat `nama_10r_2gdp`, `unit=PPS_HAB_EU27_2020`

If required files are missing and cannot be fetched, the build fails with explicit missing filenames.

## V2 Data Construction
Implemented in:
- `src/ingest.py` (official Eurostat download + cache)
- `src/pipeline.py` (harmonization, features, QA, processed outputs)

### New outcomes in `panel_master`
In addition to nominal variables, V2 adds:
- `gdp_pc_pps`, `log_gdp_pc_pps`, `gdp_pc_pps_growth`
- `gdp_pc_real`, `log_gdp_pc_real`, `gdp_pc_real_growth`

Nominal variables retained:
- `gdp_pc`, `log_gdp_pc`, `gdp_pc_growth`

### Real GDP construction method
NUTS2 real output is built from volume index (`gdp_volume_index`, 2015=100, `nama_10r_2gvagr`, `B1GQ`) and anchored to each region’s nominal GDP per capita reference level (prefer year 2015; fallback earliest available year with both series):
- `gdp_pc_real_t = gdp_pc_nominal_ref * (volume_index_t / volume_index_ref)`

### Eligibility categories method (implemented)
`category_2014_2020` is reconstructed from GDP pc in PPS relative to EU average (`PPS_HAB_EU27_2020`):
- reference average years: `2007-2009`
- fallback if reference missing: average up to `2013`
- thresholds:
  - `<75`: `less_developed`
  - `75-<90`: `transition`
  - `>=90`: `more_developed`

Output mapping file schema:
- `nuts2_id`, `category_2014_2020`

### Sigma convergence (V2)
`data/processed/sigma_convergence.csv` now contains:
- `year`
- `sigma_log_gdp_pps`
- `sigma_log_gdp_real`
- `sigma_log_gdp_nominal` (kept as optional robustness)

### Build outputs
Processed:
- `data/processed/panel_master.parquet`
- `data/processed/panel_master.csv`
- `data/processed/sigma_convergence.csv`

Interim:
- `data/interim/panel_skeleton.csv`
- `data/interim/treatment_erdf.csv`
- `data/interim/outcomes_gdp.csv`
- `data/interim/controls.csv`
- `data/interim/eligibility_categories.csv`

QA tables:
- `outputs/tables/data_quality_summary.csv`
- `outputs/tables/missingness_by_variable_year.csv`
- `outputs/tables/outliers_erdf_pc.csv`

QA figures:
- `outputs/figures/missingness_heatmap.png`
- `outputs/figures/erdf_eur_pc_distribution.png`
- `outputs/figures/gdp_pc_distribution.png`
- `outputs/figures/gdp_pc_pps_distribution.png`
- `outputs/figures/gdp_pc_real_distribution.png`
- `outputs/figures/gdp_pc_trend_treated_vs_untreated_quantiles.png`
- `outputs/figures/growth_nominal_pps_real_trends.png`

## V2 Analysis Layer
Implemented in:
- `src/models.py`
- `src/viz.py`
- `scripts/run_models.py`

### Outcome strategy
- Headline outcome: `gdp_pc_real_growth`
- Secondary outcomes: `gdp_pc_pps_growth`, `gdp_pc_growth` (nominal)

### Main model specs
- Model A: `outcome ~ erdf_eur_pc_l1 + controls + region_FE + year_FE`
- Model B: `outcome ~ erdf_eur_pc_l1 + controls + region_FE + country×year_FE`
- Model C: `outcome ~ erdf_eur_pc_l1 + erdf_eur_pc_l2 + erdf_eur_pc_l3 + controls + region_FE + year_FE`

### Convergence specs
- Model D (beta): `outcome ~ log_outcome_l1 + region_FE + year_FE`
- Model E: `outcome ~ log_outcome_l1 + erdf_eur_pc_l1 + erdf_eur_pc_l1*log_outcome_l1 + controls + region_FE + year_FE`
- Model E variant: same with `country×year_FE`

### Heterogeneity
Implemented via separate category subset regressions (acceptable alternative to interaction-heavy full model):
- `Model E (category subset)` for `less_developed`, `transition`, `more_developed`.

### Placebo
Headline outcome placebo leads/lags included:
- leads `erdf_eur_pc_f2`, `erdf_eur_pc_f1`
- lags `erdf_eur_pc_l1..l3`

### Robustness
- Outlier exclusion (drop global top 1% `erdf_eur_pc`)
- Balanced panel (`2017-2023` complete sample)
- Scaling (`erdf_k_eur_pc_l1 = erdf_eur_pc_l1/1000`)

## V2 Analysis Outputs
### Tables
- `outputs/tables/twfe_main_results_v2.csv`
- `outputs/tables/dl_lags_results_v2.csv`
- `outputs/tables/model_comparison_summary_v2.csv`
- `outputs/tables/beta_convergence_results_v2.csv`
- `outputs/tables/heterogeneity_by_category_v2.csv`
- `outputs/tables/leads_lags_results_v2.csv`
- `outputs/tables/dynamic_lag_response_v2.csv`
- `outputs/tables/robustness_outliers_v2.csv`
- `outputs/tables/robustness_balanced_panel_v2.csv`
- `outputs/tables/robustness_scaling_v2.csv`
- `outputs/tables/panel_master_schema.csv`
- `outputs/tables/panel_master_overview.csv`
- `outputs/tables/panel_master_key_missingness.csv`

### Figures
- `outputs/figures/sigma_convergence_v2.png`
- `outputs/figures/dynamic_lag_response_v2.png`
- `outputs/figures/leads_lags_placebo_v2.png`
- `outputs/figures/beta_convergence_partial_v2.png`

### Report deliverables
- `outputs/report.html` (standalone local file, no server required)
- `outputs/tables/report_inputs_manifest.json`
- `notebooks/01_report.ipynb` (generated + executed top-to-bottom)

## V2 Results (current run)
Headline outcome: `gdp_pc_real_growth`.

Main causal effects:
- Model A (`erdf_eur_pc_l1`): coef `0.00568`, SE `0.00303`, p `0.060`.
- Model B (`erdf_eur_pc_l1`): coef `0.00336`, SE `0.00289`, p `0.244`.
- Model C distributed lags (headline):
  - `l1`: `0.011999` (p `0.013`)
  - `l2`: `0.007897` (p `0.037`)
  - `l3`: `-0.003085` (p `0.247`)

Placebo leads/lags (headline):
- lead `f2`: p `0.079`
- lead `f1`: p `0.382`
- lag `l1`: p `0.005`
- lag `l2`: p `0.025`

Beta convergence:
- Real Model D: `log_gdp_pc_real_l1 = -5.267` (p `2.14e-18`) supports beta convergence.
- PPS Model D: `log_gdp_pc_pps_l1 = -7.725` (p `2.80e-70`) supports beta convergence.

ERDF interaction in Model E:
- Real: `erdf_eur_pc_l1` p `0.589`, interaction p `0.555`.
- PPS: `erdf_eur_pc_l1` p `0.857`, interaction p `0.829`.

Heterogeneity (category subset Model E, ERDF term):
- Real outcome:
  - `less_developed`: `-0.0358` (p `0.083`)
  - `transition`: `0.1412` (p `0.036`)
  - `more_developed`: `-0.0246` (p `0.706`)

Robustness (headline):
- Outliers excluded (`p99=264.11`): Model A coef `0.00632` (p `0.037`), Model B p `0.275`.
- Balanced panel (`2017-2023`, `115` regions): Model A p `0.070`, Model B p `0.370`.
- Scaled treatment (€1,000 pc): Model A coef `5.680` (p `0.060`), Model B p `0.244`.

Sigma convergence (V2):
- Common coverage years: `2000-2023`.
- `sigma_log_gdp_pps`: `0.534` → `0.391` (decline).
- `sigma_log_gdp_real`: `0.714` → `0.657` (decline).

## Limitations / TODO
- Real GDP per capita uses volume-index reconstruction (2015 anchor), not direct chain-linked EUR levels.
- Category coverage is incomplete for all panel rows due NUTS revision/code coverage differences.
- Observational panel FE estimates remain subject to residual confounding.

## Determinism and Offline Behavior
- No automatic internet downloads during models stage.
- Build stage downloads only missing V2 caches once, then reuses local `data/raw/` files.
- Outputs are deterministic given fixed raw/processed inputs and scripted execution.
