# EUcohesion

## Goal
Estimate whether EU Cohesion Policy (ERDF) reduced regional development gaps across EU NUTS2 regions by increasing GDP per capita growth and accelerating convergence.

## Implemented Data Pipeline
The repository now includes an end-to-end Python pipeline that:
1. Builds the master analysis panel.
2. Produces data QA tables and figures.
3. Produces sigma-convergence outputs.

## Required Raw Inputs
The pipeline expects these exact files in `data/raw/`:
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

If any file is missing, the build fails immediately with an explicit error listing every missing filename.

## How To Run
From project root:

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
```

Equivalent Make target:

```bash
make pipeline
```

Direct dataset build script:

```bash
python scripts/build_dataset.py
```

## Output Artifacts
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

### QA Tables
- `outputs/tables/data_quality_summary.csv`
- `outputs/tables/missingness_by_variable_year.csv`
- `outputs/tables/outliers_erdf_pc.csv`

### QA Figures
- `outputs/figures/missingness_heatmap.png`
- `outputs/figures/erdf_eur_pc_distribution.png`
- `outputs/figures/gdp_pc_distribution.png`
- `outputs/figures/gdp_pc_trend_treated_vs_untreated_quantiles.png`

## `panel_master` Schema
Final columns in `data/processed/panel_master.parquet`:
- `nuts2_id`
- `country`
- `year`
- `population`
- `erdf_eur`
- `erdf_eur_pc`
- `erdf_eur_pc_l1`
- `erdf_eur_pc_l2`
- `erdf_eur_pc_l3`
- `gdp_mio_eur`
- `gdp_eur`
- `gdp_pc`
- `log_gdp_pc`
- `gdp_pc_growth`
- `unemp_rate`
- `emp_rate`
- `tertiary_share_25_64`
- `rd_gerd`
- `gva`
- `category_2014_2020`

## Transformation Notes / Assumptions
### Keys
- Standard keys are `nuts2_id` (string) and `year` (int).
- `nuts2_id` is harmonized from source columns with robust inference + fallback config in `src/config.py`.
- The panel skeleton is built from the union of observed NUTS2 regions and the full inferred year range.

### Population
- Source: `eurostat_demo_r_d2jan.csv`
- Filters: annual frequency, `unit=NR`, `sex=T`, `age=TOTAL`
- Output variable: `population`

### ERDF Treatment
- Main treatment source: `cohesion_historic_eu_payments_nuts2_timeseries_tc55-7ysv.csv`
- Filter: `Fund == ERDF`
- Amount used: `EU_Payment_annual` (fallback to `Modelled_annual_expenditure` if needed)
- Aggregation: sum at `nuts2_id`-`year`
- Variables:
  - `erdf_eur`
  - `erdf_eur_pc = erdf_eur / population`
  - lags `erdf_eur_pc_l1`, `erdf_eur_pc_l2`, `erdf_eur_pc_l3`

Note on ESIF finance file (`99js-gm52`): it is validated as required input, but it does not currently yield a reliable direct NUTS2 key in this pipeline, so treatment construction relies on the historic regionalized NUTS2 payments dataset.

### Outcomes
- GDP source: `eurostat_nama_10r_2gdp.csv`
- GDP measure used: `unit=MIO_EUR` (current market prices)
- Construction:
  - `gdp_mio_eur` from Eurostat
  - `gdp_eur = gdp_mio_eur * 1,000,000`
  - `gdp_pc = gdp_eur / population`
  - `log_gdp_pc = log(gdp_pc)`
  - `gdp_pc_growth = 100 * (log_gdp_pc - log_gdp_pc_l1)`

### Controls (raw units retained)
- `unemp_rate` from `tgs00010` (`PC`, %)
- `emp_rate` from `tgs00007` (`PC`, %)
- `tertiary_share_25_64` from `tgs00109` (`PC`, %)
- `rd_gerd` from `tgs00042` (`PC_GDP`, % of GDP)
- `gva` from `nama_10r_2gvagr` (`I15`, index 2015=100)

### Eligibility Categories
- `category_2014_2020` is currently a placeholder (`NaN`) for all regions.
- `data/interim/eligibility_categories.csv` is generated and ready for manual population.
- TODO to complete:
  1. Build a clean lookup table from `ref_structural_funds_regional_categories_KS-GQ-18-007-EN-N.pdf`.
  2. Map each `nuts2_id` to one of `{less_developed, transition, more_developed}`.
  3. Re-run pipeline.

## Convergence Output
- `data/processed/sigma_convergence.csv`
- Definition: `sigma_log_gdp = std(log_gdp_pc)` across regions by year.

## Data QA Outputs
- `outputs/tables/data_quality_summary.csv`: variable-level `min`, `median`, `max`, `mean`, `std`, missingness stats.
- `outputs/tables/missingness_by_variable_year.csv`: missing count/share by variable and year.
- `outputs/tables/outliers_erdf_pc.csv`: top 1% `erdf_eur_pc` observations by year.
- Figures in `outputs/figures/`:
  - missingness heatmap
  - ERDF per-capita distribution
  - GDP per-capita distribution
  - treated vs untreated quantile trend plot

## Reproducibility / Determinism
- Pipeline is fully scripted and runnable end-to-end from project root.
- Outputs are written in deterministic sorted key order (`nuts2_id`, `year`).
- No automatic raw-data downloads are performed.

## Implementation Files
- `src/config.py`: paths, required filenames, fallback column config, dataset filters.
- `src/pipeline.py`: ingestion, harmonization, feature construction, QA, output writing.
- `scripts/build_dataset.py`: dataset build entrypoint.
- `scripts/run_pipeline.py`: top-level pipeline runner (currently calls dataset build stage).
