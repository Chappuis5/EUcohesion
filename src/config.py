from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUTS_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

REQUIRED_RAW_FILES = [
    "cohesion_esif_2014_2020_finance_implementation_99js-gm52.csv",
    "cohesion_historic_eu_payments_nuts2_timeseries_tc55-7ysv.csv",
    "eurostat_nama_10r_2gdp.csv",
    "eurostat_demo_r_d2jan.csv",
    "eurostat_tgs00010_unemployment_rate.csv",
    "eurostat_tgs00007_employment_rate.csv",
    "eurostat_tgs00109_tertiary_education_25_64.csv",
    "eurostat_tgs00042_gerd_rd_expenditure.csv",
    "eurostat_nama_10r_2gvagr_gva.csv",
    "ref_structural_funds_regional_categories_KS-GQ-18-007-EN-N.pdf",
    "ref_nuts_revision_consistency_KS-GQ-22-010-EN-N.pdf",
    # V2 cached files (downloaded once and then reused offline).
    "eurostat_gdp_pc_pps_nuts2.csv",
    "eurostat_gdp_real_nuts2.csv",
    "eurostat_gdp_pc_pps_rel_eu_nuts2.csv",
    "eligibility_categories_2014_2020.csv",
]

RAW_FILES = {
    "esif_finance": "cohesion_esif_2014_2020_finance_implementation_99js-gm52.csv",
    "historic_payments": "cohesion_historic_eu_payments_nuts2_timeseries_tc55-7ysv.csv",
    "population": "eurostat_demo_r_d2jan.csv",
    "gdp": "eurostat_nama_10r_2gdp.csv",
    "gdp_pc_pps": "eurostat_gdp_pc_pps_nuts2.csv",
    "gdp_real": "eurostat_gdp_real_nuts2.csv",
    "gdp_pc_pps_rel_eu": "eurostat_gdp_pc_pps_rel_eu_nuts2.csv",
    "eligibility_categories": "eligibility_categories_2014_2020.csv",
    "unemployment": "eurostat_tgs00010_unemployment_rate.csv",
    "employment": "eurostat_tgs00007_employment_rate.csv",
    "tertiary": "eurostat_tgs00109_tertiary_education_25_64.csv",
    "rd_gerd": "eurostat_tgs00042_gerd_rd_expenditure.csv",
    "gva": "eurostat_nama_10r_2gvagr_gva.csv",
}

V2_DOWNLOAD_TARGETS = [
    RAW_FILES["gdp_pc_pps"],
    RAW_FILES["gdp_real"],
    RAW_FILES["gdp_pc_pps_rel_eu"],
    RAW_FILES["eligibility_categories"],
]

# Fallback column configuration after header normalization in src/pipeline.py.
COLUMN_FALLBACKS = {
    RAW_FILES["historic_payments"]: {
        "nuts2": "nuts2_id",
        "year": "year",
        "value_candidates": ["eu_payment_annual", "modelled_annual_expenditure"],
        "fund": "fund",
    },
    RAW_FILES["population"]: {
        "nuts2": "geo",
        "year": "time_period",
        "value_candidates": ["obs_value"],
    },
    RAW_FILES["gdp"]: {
        "nuts2": "geo",
        "year": "time_period",
        "value_candidates": ["obs_value"],
    },
    RAW_FILES["gdp_pc_pps"]: {
        "nuts2": "geo",
        "year": "time",
        "value_candidates": ["obs_value"],
    },
    RAW_FILES["gdp_real"]: {
        "nuts2": "geo",
        "year": "time",
        "value_candidates": ["obs_value", "gdp_volume_index"],
    },
    RAW_FILES["gdp_pc_pps_rel_eu"]: {
        "nuts2": "geo",
        "year": "time",
        "value_candidates": ["obs_value"],
    },
    RAW_FILES["eligibility_categories"]: {
        "nuts2": "nuts2_id",
        "category": "category_2014_2020",
    },
    RAW_FILES["unemployment"]: {
        "nuts2": "geo",
        "year": "time_period",
        "value_candidates": ["obs_value"],
    },
    RAW_FILES["employment"]: {
        "nuts2": "geo",
        "year": "time_period",
        "value_candidates": ["obs_value"],
    },
    RAW_FILES["tertiary"]: {
        "nuts2": "geo",
        "year": "time_period",
        "value_candidates": ["obs_value"],
    },
    RAW_FILES["rd_gerd"]: {
        "nuts2": "geo",
        "year": "time_period",
        "value_candidates": ["obs_value"],
    },
    RAW_FILES["gva"]: {
        "nuts2": "geo",
        "year": "time_period",
        "value_candidates": ["obs_value"],
    },
}

EUROSTAT_FILTERS = {
    RAW_FILES["population"]: {"freq": "A", "unit": "NR", "sex": "T", "age": "TOTAL"},
    RAW_FILES["gdp"]: {"freq": "A", "unit": "MIO_EUR"},
    RAW_FILES["gdp_pc_pps"]: {"freq": "A", "unit": "PPS_EU27_2020_HAB"},
    RAW_FILES["gdp_real"]: {"freq": "A", "na_item": "B1GQ", "unit": "I15"},
    RAW_FILES["gdp_pc_pps_rel_eu"]: {"freq": "A", "unit": "PPS_HAB_EU27_2020"},
    RAW_FILES["unemployment"]: {
        "freq": "A",
        "unit": "PC",
        "sex": "T",
        "age": "Y15-74",
        "isced11": "TOTAL",
    },
    RAW_FILES["employment"]: {"freq": "A", "unit": "PC", "sex": "T", "age": "Y15-64"},
    RAW_FILES["tertiary"]: {
        "freq": "A",
        "unit": "PC",
        "sex": "T",
        "age": "Y25-64",
        "isced11": "ED5-8",
    },
    RAW_FILES["rd_gerd"]: {"freq": "A", "unit": "PC_GDP", "sectperf": "TOTAL"},
    RAW_FILES["gva"]: {"freq": "A", "na_item": "B1G", "unit": "I15"},
}

NUTS2_REGEX = r"^[A-Z]{2}[A-Z0-9]{2}$"
UNKNOWN_NUTS2_SUFFIXES = ("ZZ", "XX")

PANEL_MASTER_PARQUET = DATA_PROCESSED_DIR / "panel_master.parquet"
PANEL_MASTER_CSV = DATA_PROCESSED_DIR / "panel_master.csv"
SIGMA_CONVERGENCE_CSV = DATA_PROCESSED_DIR / "sigma_convergence.csv"
RUNNING_VARIABLE_ELIGIBILITY_CSV = DATA_PROCESSED_DIR / "running_variable_eligibility.csv"
ERDF_CUMULATIVE_EXPOSURE_CSV = DATA_PROCESSED_DIR / "erdf_cumulative_exposure.csv"

INTERIM_FILES = {
    "panel_skeleton": DATA_INTERIM_DIR / "panel_skeleton.csv",
    "treatment_erdf": DATA_INTERIM_DIR / "treatment_erdf.csv",
    "outcomes_gdp": DATA_INTERIM_DIR / "outcomes_gdp.csv",
    "controls": DATA_INTERIM_DIR / "controls.csv",
    "eligibility_categories": DATA_INTERIM_DIR / "eligibility_categories.csv",
}

QA_TABLES = {
    "summary": OUTPUTS_TABLES_DIR / "data_quality_summary.csv",
    "missingness": OUTPUTS_TABLES_DIR / "missingness_by_variable_year.csv",
    "outliers": OUTPUTS_TABLES_DIR / "outliers_erdf_pc.csv",
}

QA_FIGURES = {
    "missingness_heatmap": OUTPUTS_FIGURES_DIR / "missingness_heatmap.png",
    "erdf_distribution": OUTPUTS_FIGURES_DIR / "erdf_eur_pc_distribution.png",
    "gdp_distribution": OUTPUTS_FIGURES_DIR / "gdp_pc_distribution.png",
    "gdp_pps_distribution": OUTPUTS_FIGURES_DIR / "gdp_pc_pps_distribution.png",
    "gdp_real_distribution": OUTPUTS_FIGURES_DIR / "gdp_pc_real_distribution.png",
    "treated_vs_untreated_trends": OUTPUTS_FIGURES_DIR / "gdp_pc_trend_treated_vs_untreated_quantiles.png",
    "growth_comparison_trend": OUTPUTS_FIGURES_DIR / "growth_nominal_pps_real_trends.png",
}
