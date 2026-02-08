from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nbformat
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from nbclient import NotebookClient
from plotly.io import to_html

from src import config
from src import viz

LOGGER = logging.getLogger(__name__)

REQUESTED_CONTROLS = [
    "unemp_rate",
    "emp_rate",
    "tertiary_share_25_64",
    "rd_gerd",
    "gva",
]

GROWTH_OUTCOME_CANDIDATES = [
    "gdp_pc_real_growth",
    "gdp_pc_pps_growth",
    "gdp_pc_growth",
]

LAG_LOG_BY_OUTCOME = {
    "gdp_pc_real_growth": "log_gdp_pc_real_l1",
    "gdp_pc_pps_growth": "log_gdp_pc_pps_l1",
    "gdp_pc_growth": "log_gdp_pc_l1",
}

OUTPUT_TABLES_DIR = config.OUTPUTS_TABLES_DIR
OUTPUT_FIGURES_DIR = config.OUTPUTS_FIGURES_DIR
OUTPUTS_DIR = config.OUTPUTS_DIR
NOTEBOOK_PATH = config.PROJECT_ROOT / "notebooks" / "01_report.ipynb"
REPORT_HTML_PATH = OUTPUTS_DIR / "report.html"
REPORT_MANIFEST_PATH = OUTPUT_TABLES_DIR / "report_inputs_manifest.json"

BASE_TABLE_PATHS = {
    "panel_schema": OUTPUT_TABLES_DIR / "panel_master_schema.csv",
    "panel_overview": OUTPUT_TABLES_DIR / "panel_master_overview.csv",
    "panel_key_missingness": OUTPUT_TABLES_DIR / "panel_master_key_missingness.csv",
}

TABLE_PATHS_V2 = {
    "twfe_main": OUTPUT_TABLES_DIR / "twfe_main_results_v2.csv",
    "dl_lags": OUTPUT_TABLES_DIR / "dl_lags_results_v2.csv",
    "dynamic_lag": OUTPUT_TABLES_DIR / "dynamic_lag_response_v2.csv",
    "leads_lags": OUTPUT_TABLES_DIR / "leads_lags_results_v2.csv",
    "beta": OUTPUT_TABLES_DIR / "beta_convergence_results_v2.csv",
    "heterogeneity": OUTPUT_TABLES_DIR / "heterogeneity_by_category_v2.csv",
    "robust_outliers": OUTPUT_TABLES_DIR / "robustness_outliers_v2.csv",
    "robust_balanced": OUTPUT_TABLES_DIR / "robustness_balanced_panel_v2.csv",
    "robust_scaling": OUTPUT_TABLES_DIR / "robustness_scaling_v2.csv",
    "model_summary": OUTPUT_TABLES_DIR / "model_comparison_summary_v2.csv",
}

FIGURE_PATHS_V2 = {
    "dynamic_lag": OUTPUT_FIGURES_DIR / "dynamic_lag_response_v2.png",
    "leads_lags": OUTPUT_FIGURES_DIR / "leads_lags_placebo_v2.png",
    "sigma": OUTPUT_FIGURES_DIR / "sigma_convergence_v2.png",
    "beta_partial": OUTPUT_FIGURES_DIR / "beta_convergence_partial_v2.png",
}

TABLE_PATHS_V3 = {
    "twfe_main": OUTPUT_TABLES_DIR / "twfe_main_results_v3.csv",
    "dl_lags": OUTPUT_TABLES_DIR / "dl_lags_results_v3.csv",
    "dynamic_lag": OUTPUT_TABLES_DIR / "dynamic_lag_response_v3.csv",
    "leads_lags": OUTPUT_TABLES_DIR / "leads_lags_results_v3.csv",
    "beta": OUTPUT_TABLES_DIR / "beta_convergence_results_v3.csv",
    "heterogeneity": OUTPUT_TABLES_DIR / "heterogeneity_by_category_v3.csv",
    "robust_outliers": OUTPUT_TABLES_DIR / "robustness_outliers_v3.csv",
    "robust_balanced": OUTPUT_TABLES_DIR / "robustness_balanced_panel_v3.csv",
    "robust_scaling": OUTPUT_TABLES_DIR / "robustness_scaling_v3.csv",
    "rd_main": OUTPUT_TABLES_DIR / "rd_main_results_v3.csv",
    "rd_placebo": OUTPUT_TABLES_DIR / "rd_placebo_pretrend_v3.csv",
    "iv_2sls": OUTPUT_TABLES_DIR / "iv_2sls_results_v3.csv",
    "iv_first_stage": OUTPUT_TABLES_DIR / "iv_first_stage_v3.csv",
    "model_summary": OUTPUT_TABLES_DIR / "model_comparison_summary_v3.csv",
}

FIGURE_PATHS_V3 = {
    "dynamic_lag": OUTPUT_FIGURES_DIR / "dynamic_lag_response_v3.png",
    "leads_lags": OUTPUT_FIGURES_DIR / "leads_lags_placebo_v3.png",
    "sigma": OUTPUT_FIGURES_DIR / "sigma_convergence_v3.png",
    "beta_partial": OUTPUT_FIGURES_DIR / "beta_convergence_partial_v3.png",
    "rd_binned": OUTPUT_FIGURES_DIR / "rd_binned_scatter_v3.png",
    "rd_bandwidth": OUTPUT_FIGURES_DIR / "rd_bandwidth_sensitivity_v3.png",
}

TABLE_PATHS_V31 = {
    "twfe_main": OUTPUT_TABLES_DIR / "twfe_main_results_v31.csv",
    "dl_lags": OUTPUT_TABLES_DIR / "dl_lags_results_v31.csv",
    "dynamic_lag": OUTPUT_TABLES_DIR / "dynamic_lag_response_v31.csv",
    "leads_lags": OUTPUT_TABLES_DIR / "leads_lags_results_v31.csv",
    "beta": OUTPUT_TABLES_DIR / "beta_convergence_results_v31.csv",
    "heterogeneity": OUTPUT_TABLES_DIR / "heterogeneity_by_category_v31.csv",
    "robust_outliers": OUTPUT_TABLES_DIR / "robustness_outliers_v31.csv",
    "robust_balanced": OUTPUT_TABLES_DIR / "robustness_balanced_panel_v31.csv",
    "robust_scaling": OUTPUT_TABLES_DIR / "robustness_scaling_v31.csv",
    "rd_first_stage_funding_jump": OUTPUT_TABLES_DIR / "rd_first_stage_funding_jump_v31.csv",
    "rd_outcome_sharp": OUTPUT_TABLES_DIR / "rd_outcome_sharp_results_v31.csv",
    "rd_bandwidth_sensitivity": OUTPUT_TABLES_DIR / "rd_bandwidth_sensitivity_v31.csv",
    "rd_placebo_pretrend": OUTPUT_TABLES_DIR / "rd_placebo_pretrend_v31.csv",
    "iv_first_stage_candidates": OUTPUT_TABLES_DIR / "iv_first_stage_candidates_v31.csv",
    "iv_panel_results": OUTPUT_TABLES_DIR / "iv_2sls_results_v31.csv",
    "iv_cross_section_results": OUTPUT_TABLES_DIR / "iv_cross_section_results_v31.csv",
    "model_summary": OUTPUT_TABLES_DIR / "model_comparison_summary_v31.csv",
}

FIGURE_PATHS_V31 = {
    "dynamic_lag": OUTPUT_FIGURES_DIR / "dynamic_lag_response_v31.png",
    "leads_lags": OUTPUT_FIGURES_DIR / "leads_lags_placebo_v31.png",
    "sigma": OUTPUT_FIGURES_DIR / "sigma_convergence_v31.png",
    "beta_partial": OUTPUT_FIGURES_DIR / "beta_convergence_partial_v31.png",
    "rd_first_stage_funding_jump": OUTPUT_FIGURES_DIR / "rd_first_stage_funding_jump_v31.png",
    "rd_outcome_binned_scatter": OUTPUT_FIGURES_DIR / "rd_outcome_binned_scatter_v31.png",
    "rd_bandwidth_sensitivity": OUTPUT_FIGURES_DIR / "rd_bandwidth_sensitivity_v31.png",
    "iv_first_stage_scatter": OUTPUT_FIGURES_DIR / "iv_first_stage_scatter_v31.png",
}

VALID_CATEGORIES = ["less_developed", "transition", "more_developed"]


@dataclass
class RegressionRun:
    model_name: str
    outcome: str
    formula: str
    fe_type: str
    clustering: str
    result: object
    sample_df: pd.DataFrame
    controls_used: List[str]

    @property
    def n_obs(self) -> int:
        return int(self.sample_df.shape[0])

    @property
    def n_regions(self) -> int:
        return int(self.sample_df["nuts2_id"].nunique())

    @property
    def year_min(self) -> int:
        return int(self.sample_df["year"].min())

    @property
    def year_max(self) -> int:
        return int(self.sample_df["year"].max())


@dataclass
class RDEstimate:
    estimator: str
    outcome: str
    window: str
    bandwidth: float
    coef: float
    std_err: float
    t_stat: float
    p_value: float
    ci_95_lower: float
    ci_95_upper: float
    n_obs: int
    n_left: int
    n_right: int
    cutoff: float
    kernel: str
    polynomial_order: int
    treatment_var: str
    first_stage_f_stat: float = np.nan


def _ensure_output_dirs() -> None:
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.10g")


def _validate_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    dataset_path: Path,
    source_hint: str,
) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Missing required columns in {dataset_path}: {missing_text}. "
            f"These should come from {source_hint}."
        )


def load_analysis_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    panel_path = config.PANEL_MASTER_PARQUET
    sigma_path = config.SIGMA_CONVERGENCE_CSV

    if not panel_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {panel_path}. Run scripts/build_dataset.py to generate it."
        )
    if not sigma_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {sigma_path}. Run scripts/build_dataset.py to generate it."
        )

    panel = pd.read_parquet(panel_path)
    sigma = pd.read_csv(sigma_path)

    panel_required = [
        "nuts2_id",
        "country",
        "year",
        "erdf_eur_pc",
        "erdf_eur_pc_l1",
        "erdf_eur_pc_l2",
        "erdf_eur_pc_l3",
        "gdp_pc_growth",
        "gdp_pc_pps_growth",
        "gdp_pc_real_growth",
        "log_gdp_pc",
        "log_gdp_pc_pps",
        "log_gdp_pc_real",
        "category_2014_2020",
    ]

    sigma_required = ["year", "sigma_log_gdp_pps", "sigma_log_gdp_real"]

    _validate_columns(
        panel,
        panel_required,
        panel_path,
        "data processing pipeline output in scripts/build_dataset.py (src/pipeline.py)",
    )
    _validate_columns(
        sigma,
        sigma_required,
        sigma_path,
        "data processing pipeline output in scripts/build_dataset.py (src/pipeline.py)",
    )

    panel = panel.copy()
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel = panel[panel["year"].notna()].copy()
    panel["year"] = panel["year"].astype(int)
    panel = panel.sort_values(["nuts2_id", "year"]).reset_index(drop=True)

    sigma = sigma.copy()
    sigma["year"] = pd.to_numeric(sigma["year"], errors="coerce")
    sigma = sigma[sigma["year"].notna()].copy()
    sigma["year"] = sigma["year"].astype(int)

    for column in ["sigma_log_gdp_pps", "sigma_log_gdp_real", "sigma_log_gdp_nominal"]:
        if column in sigma.columns:
            sigma[column] = pd.to_numeric(sigma[column], errors="coerce")

    sigma = sigma.sort_values("year").reset_index(drop=True)
    return panel, sigma


def load_identification_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    running_path = config.RUNNING_VARIABLE_ELIGIBILITY_CSV
    exposure_path = config.ERDF_CUMULATIVE_EXPOSURE_CSV

    if not running_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {running_path}. "
            "Run scripts/run_pipeline.py to generate V3 running-variable outputs."
        )
    if not exposure_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {exposure_path}. "
            "Run scripts/run_pipeline.py to generate V3 cumulative exposure outputs."
        )

    running = pd.read_csv(running_path)
    exposure = pd.read_csv(exposure_path)

    _validate_columns(
        running,
        ["nuts2_id", "country", "r_value", "eligible_lt75", "category_2014_2020", "ref_years_used"],
        running_path,
        "scripts/build_dataset.py (src/pipeline.py)",
    )
    _validate_columns(
        exposure,
        ["nuts2_id", "country", "erdf_eur_pc_cum_2014_2020", "erdf_eur_pc_cum_2015_2020"],
        exposure_path,
        "scripts/build_dataset.py (src/pipeline.py)",
    )

    running["nuts2_id"] = running["nuts2_id"].astype("string").str.strip().str.upper()
    running["country"] = running["country"].astype("string").str.strip().str.upper()
    running["r_value"] = pd.to_numeric(running["r_value"], errors="coerce")
    running["eligible_lt75"] = pd.to_numeric(running["eligible_lt75"], errors="coerce")
    running = running.dropna(subset=["nuts2_id", "r_value", "eligible_lt75"]).copy()
    running["eligible_lt75"] = (running["eligible_lt75"] > 0).astype(int)
    running = running.drop_duplicates(subset=["nuts2_id"], keep="first").reset_index(drop=True)

    exposure["nuts2_id"] = exposure["nuts2_id"].astype("string").str.strip().str.upper()
    exposure["country"] = exposure["country"].astype("string").str.strip().str.upper()
    exposure["erdf_eur_pc_cum_2014_2020"] = pd.to_numeric(
        exposure["erdf_eur_pc_cum_2014_2020"], errors="coerce"
    )
    exposure["erdf_eur_pc_cum_2015_2020"] = pd.to_numeric(
        exposure["erdf_eur_pc_cum_2015_2020"], errors="coerce"
    )
    exposure = exposure.drop_duplicates(subset=["nuts2_id"], keep="first").reset_index(drop=True)

    return running, exposure


def get_available_controls(panel: pd.DataFrame) -> List[str]:
    available = [column for column in REQUESTED_CONTROLS if column in panel.columns]
    missing = sorted(set(REQUESTED_CONTROLS) - set(available))
    if missing:
        LOGGER.warning("Controls not found and omitted from regressions: %s", ", ".join(missing))
    return available


def _choose_headline_outcome(panel: pd.DataFrame) -> str:
    if "gdp_pc_real_growth" in panel.columns and panel["gdp_pc_real_growth"].notna().any():
        return "gdp_pc_real_growth"
    if "gdp_pc_pps_growth" in panel.columns and panel["gdp_pc_pps_growth"].notna().any():
        return "gdp_pc_pps_growth"
    raise ValueError(
        "Neither gdp_pc_real_growth nor gdp_pc_pps_growth has usable data. "
        "These should come from V2 pipeline outputs in panel_master.parquet."
    )


def _available_growth_outcomes(panel: pd.DataFrame, headline: str) -> List[str]:
    outcomes: List[str] = [headline]
    for candidate in ["gdp_pc_pps_growth", "gdp_pc_growth", "gdp_pc_real_growth"]:
        if candidate in panel.columns and panel[candidate].notna().any() and candidate not in outcomes:
            outcomes.append(candidate)
    return outcomes


def prepare_analysis_panel(panel: pd.DataFrame) -> pd.DataFrame:
    prepared = panel.copy().sort_values(["nuts2_id", "year"]).reset_index(drop=True)
    prepared["country_year"] = prepared["country"].astype(str) + "_" + prepared["year"].astype(str)

    numeric_candidates = [
        "erdf_eur_pc",
        "erdf_eur_pc_l1",
        "erdf_eur_pc_l2",
        "erdf_eur_pc_l3",
        "erdf_eur_pc_cum_2014_2020",
        "erdf_eur_pc_cum_2015_2020",
        "r_value",
        "eligible_lt75",
        "gdp_pc_growth",
        "gdp_pc_pps_growth",
        "gdp_pc_real_growth",
        "log_gdp_pc",
        "log_gdp_pc_pps",
        "log_gdp_pc_real",
        *REQUESTED_CONTROLS,
    ]

    for column in numeric_candidates:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared["log_gdp_pc_l1"] = prepared.groupby("nuts2_id", sort=False)["log_gdp_pc"].shift(1)
    prepared["log_gdp_pc_pps_l1"] = prepared.groupby("nuts2_id", sort=False)["log_gdp_pc_pps"].shift(1)
    prepared["log_gdp_pc_real_l1"] = prepared.groupby("nuts2_id", sort=False)["log_gdp_pc_real"].shift(1)

    prepared["erdf_eur_pc_f1"] = prepared.groupby("nuts2_id", sort=False)["erdf_eur_pc"].shift(-1)
    prepared["erdf_eur_pc_f2"] = prepared.groupby("nuts2_id", sort=False)["erdf_eur_pc"].shift(-2)
    prepared["erdf_k_eur_pc_l1"] = prepared["erdf_eur_pc_l1"] / 1000.0

    prepared["category_2014_2020"] = (
        prepared["category_2014_2020"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z]+", "_", regex=True)
        .str.strip("_")
    )

    if "eligible_lt75" in prepared.columns:
        prepared["eligible_lt75"] = (pd.to_numeric(prepared["eligible_lt75"], errors="coerce") > 0).astype("Int64")

    return prepared


def save_panel_schema_and_overview(panel: pd.DataFrame, controls_used: Sequence[str]) -> None:
    schema = pd.DataFrame(
        {
            "column": panel.columns,
            "dtype": [str(dtype) for dtype in panel.dtypes],
            "missing_pct": [float(panel[column].isna().mean()) for column in panel.columns],
        }
    ).sort_values("column")
    _write_csv(schema, BASE_TABLE_PATHS["panel_schema"])

    key_variables = [
        "erdf_eur_pc",
        "erdf_eur_pc_l1",
        "erdf_eur_pc_l2",
        "erdf_eur_pc_l3",
        "gdp_pc_growth",
        "gdp_pc_pps_growth",
        "gdp_pc_real_growth",
        *list(controls_used),
    ]
    key_variables = [variable for variable in key_variables if variable in panel.columns]

    key_missingness = pd.DataFrame(
        {
            "variable": key_variables,
            "missing_rate": [float(panel[column].isna().mean()) for column in key_variables],
        }
    ).sort_values("variable")
    _write_csv(key_missingness, BASE_TABLE_PATHS["panel_key_missingness"])

    overview = pd.DataFrame(
        [
            {"metric": "min_year", "value": int(panel["year"].min())},
            {"metric": "max_year", "value": int(panel["year"].max())},
            {"metric": "n_regions", "value": int(panel["nuts2_id"].nunique())},
            {"metric": "n_rows", "value": int(panel.shape[0])},
            {
                "metric": "headline_outcome",
                "value": _choose_headline_outcome(panel),
            },
        ]
    )
    _write_csv(overview, BASE_TABLE_PATHS["panel_overview"])


def _build_formula(outcome: str, regressors: Sequence[str], controls: Sequence[str], fe_type: str) -> str:
    rhs_terms = list(regressors) + list(controls) + ["C(nuts2_id)"]
    if fe_type == "year":
        rhs_terms.append("C(year)")
    elif fe_type == "country_year":
        rhs_terms.append("C(country_year)")
    else:
        raise ValueError(f"Unsupported fe_type: {fe_type}")
    return f"{outcome} ~ {' + '.join(rhs_terms)}"


def _fit_ols_with_clustering(
    model_name: str,
    outcome: str,
    formula: str,
    sample_df: pd.DataFrame,
    fe_type: str,
    clustering: str,
    controls_used: Sequence[str],
) -> RegressionRun:
    if sample_df.empty:
        raise ValueError(f"Model {model_name} ({outcome}) has empty sample after dropping missing rows.")

    if clustering == "nuts2":
        groups = sample_df["nuts2_id"]
        cov_kwds = {"groups": groups}
    elif clustering == "nuts2_country":
        nuts2_codes = pd.Categorical(sample_df["nuts2_id"]).codes
        country_codes = pd.Categorical(sample_df["country"]).codes
        groups = np.column_stack([nuts2_codes, country_codes])
        cov_kwds = {"groups": groups}
    else:
        raise ValueError(f"Unsupported clustering option: {clustering}")

    model = smf.ols(formula=formula, data=sample_df)
    result = model.fit(cov_type="cluster", cov_kwds=cov_kwds)

    return RegressionRun(
        model_name=model_name,
        outcome=outcome,
        formula=formula,
        fe_type=fe_type,
        clustering=clustering,
        result=result,
        sample_df=sample_df,
        controls_used=list(controls_used),
    )


def run_fe_model(
    panel: pd.DataFrame,
    model_name: str,
    outcome: str,
    regressors: Sequence[str],
    controls: Sequence[str],
    fe_type: str,
    clustering: str,
) -> RegressionRun:
    regressor_columns: List[str] = []
    for regressor in regressors:
        tokens = [token for token in regressor.replace("*", ":").split(":") if token]
        regressor_columns.extend(tokens or [regressor])

    required_columns = ["nuts2_id", "country", "year", outcome, *regressor_columns, *controls]
    required_columns = list(dict.fromkeys(required_columns))
    if fe_type == "country_year":
        required_columns.append("country_year")

    missing = sorted(set(required_columns) - set(panel.columns))
    if missing:
        raise ValueError(
            f"Model {model_name} ({outcome}) is missing required columns: {', '.join(missing)}. "
            "Check V2 panel_master output columns from src/pipeline.py."
        )

    sample_df = panel[required_columns].dropna().copy()
    formula = _build_formula(outcome=outcome, regressors=regressors, controls=controls, fe_type=fe_type)

    return _fit_ols_with_clustering(
        model_name=model_name,
        outcome=outcome,
        formula=formula,
        sample_df=sample_df,
        fe_type=fe_type,
        clustering=clustering,
        controls_used=controls,
    )


def run_fe_model_safe(**kwargs) -> Optional[RegressionRun]:
    try:
        return run_fe_model(**kwargs)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Skipping model %s (%s): %s", kwargs.get("model_name"), kwargs.get("outcome"), exc)
        return None


def run_to_table(run: RegressionRun, keep_fe_terms: bool = False) -> pd.DataFrame:
    params = run.result.params
    bse = run.result.bse
    tvals = run.result.tvalues
    pvals = run.result.pvalues
    conf = run.result.conf_int(alpha=0.05)

    table = pd.DataFrame(
        {
            "term": params.index,
            "coef": params.values,
            "std_err": bse.values,
            "t_stat": tvals.values,
            "p_value": pvals.values,
            "ci_95_lower": conf.iloc[:, 0].values,
            "ci_95_upper": conf.iloc[:, 1].values,
        }
    )

    if not keep_fe_terms:
        fe_prefixes = ("C(nuts2_id)", "C(year)", "C(country_year)", "Intercept")
        table = table[~table["term"].str.startswith(fe_prefixes)].copy()

    table["outcome"] = run.outcome
    table["model"] = run.model_name
    table["n_obs"] = run.n_obs
    table["n_regions"] = run.n_regions
    table["fe_type"] = run.fe_type
    table["clustering"] = run.clustering
    table["sample_year_min"] = run.year_min
    table["sample_year_max"] = run.year_max
    table["controls_used"] = ",".join(run.controls_used)

    ordered_columns = [
        "outcome",
        "model",
        "term",
        "coef",
        "std_err",
        "t_stat",
        "p_value",
        "ci_95_lower",
        "ci_95_upper",
        "n_obs",
        "n_regions",
        "fe_type",
        "clustering",
        "sample_year_min",
        "sample_year_max",
        "controls_used",
    ]

    return table[ordered_columns].reset_index(drop=True)


def extract_terms(run: RegressionRun, term_horizons: Dict[str, int]) -> pd.DataFrame:
    base = run_to_table(run, keep_fe_terms=False)
    rows: List[Dict[str, object]] = []
    for term, horizon in term_horizons.items():
        row = base[base["term"] == term]
        if row.empty:
            continue
        row_dict = row.iloc[0].to_dict()
        row_dict["horizon"] = horizon
        rows.append(row_dict)

    if not rows:
        return pd.DataFrame(
            columns=[
                "outcome",
                "model",
                "term",
                "horizon",
                "coef",
                "std_err",
                "t_stat",
                "p_value",
                "ci_95_lower",
                "ci_95_upper",
                "n_obs",
                "n_regions",
                "fe_type",
                "clustering",
                "sample_year_min",
                "sample_year_max",
                "controls_used",
            ]
        )

    result = pd.DataFrame(rows)
    return result[
        [
            "outcome",
            "model",
            "term",
            "horizon",
            "coef",
            "std_err",
            "t_stat",
            "p_value",
            "ci_95_lower",
            "ci_95_upper",
            "n_obs",
            "n_regions",
            "fe_type",
            "clustering",
            "sample_year_min",
            "sample_year_max",
            "controls_used",
        ]
    ].sort_values("horizon").reset_index(drop=True)


def _extract_key_row(run: RegressionRun, term: str) -> Optional[Dict[str, object]]:
    table = run_to_table(run, keep_fe_terms=False)
    row = table[table["term"] == term]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def _build_model_summary(
    runs: Sequence[RegressionRun],
    key_term_lookup: Dict[str, str],
    outlier_threshold: Optional[float] = None,
    balanced_window: Optional[Tuple[int, int]] = None,
    balanced_regions: Optional[int] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run in runs:
        key_term = key_term_lookup.get(run.model_name)
        if key_term is None:
            continue
        key_row = _extract_key_row(run, key_term)
        if key_row is None and run.model_name == "Model D":
            for alternative in ["log_gdp_pc_real_l1", "log_gdp_pc_pps_l1", "log_gdp_pc_l1"]:
                key_row = _extract_key_row(run, alternative)
                if key_row is not None:
                    key_term = alternative
                    break
        if key_row is None:
            continue
        rows.append(
            {
                "outcome": run.outcome,
                "model": run.model_name,
                "key_term": key_term,
                "coef": key_row["coef"],
                "std_err": key_row["std_err"],
                "p_value": key_row["p_value"],
                "ci_95_lower": key_row["ci_95_lower"],
                "ci_95_upper": key_row["ci_95_upper"],
                "n_obs": run.n_obs,
                "n_regions": run.n_regions,
                "fe_type": run.fe_type,
                "clustering": run.clustering,
                "sample_year_min": run.year_min,
                "sample_year_max": run.year_max,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary = summary.sort_values(["outcome", "model"]).reset_index(drop=True)
    summary["outlier_rule"] = "drop global top 1% erdf_eur_pc"
    summary["outlier_threshold"] = outlier_threshold

    if balanced_window is not None:
        summary["balanced_window_start"] = balanced_window[0]
        summary["balanced_window_end"] = balanced_window[1]
    else:
        summary["balanced_window_start"] = np.nan
        summary["balanced_window_end"] = np.nan

    summary["balanced_regions"] = balanced_regions
    return summary


def _get_interaction_name(run: RegressionRun, a: str, b: str) -> Optional[str]:
    terms = set(run.result.params.index.tolist())
    for candidate in [f"{a}:{b}", f"{b}:{a}"]:
        if candidate in terms:
            return candidate
    return None


def build_beta_marginal_effect(run: RegressionRun, lag_log_column: str) -> pd.DataFrame:
    interaction_term = _get_interaction_name(run, "erdf_eur_pc_l1", lag_log_column)
    if interaction_term is None:
        return pd.DataFrame(columns=["percentile", "x_value", "marginal_effect", "ci_lower", "ci_upper"])

    params = run.result.params
    cov = run.result.cov_params()

    if "erdf_eur_pc_l1" not in params or interaction_term not in params:
        return pd.DataFrame(columns=["percentile", "x_value", "marginal_effect", "ci_lower", "ci_upper"])

    beta_erdf = params["erdf_eur_pc_l1"]
    beta_inter = params[interaction_term]

    var_erdf = cov.loc["erdf_eur_pc_l1", "erdf_eur_pc_l1"]
    var_inter = cov.loc[interaction_term, interaction_term]
    covar = cov.loc["erdf_eur_pc_l1", interaction_term]

    x = run.sample_df[lag_log_column].dropna()
    if x.empty:
        return pd.DataFrame(columns=["percentile", "x_value", "marginal_effect", "ci_lower", "ci_upper"])

    percentiles = np.arange(5, 100, 5)
    x_values = np.percentile(x, percentiles)

    rows = []
    for pct, x_val in zip(percentiles, x_values):
        marginal = beta_erdf + beta_inter * x_val
        variance = var_erdf + (x_val**2) * var_inter + 2 * x_val * covar
        std_err = float(np.sqrt(max(variance, 0.0)))
        rows.append(
            {
                "percentile": int(pct),
                "x_value": float(x_val),
                "marginal_effect": float(marginal),
                "ci_lower": float(marginal - 1.96 * std_err),
                "ci_upper": float(marginal + 1.96 * std_err),
            }
        )

    return pd.DataFrame(rows)


def run_core_models_by_outcome(
    panel: pd.DataFrame,
    outcomes: Sequence[str],
    controls: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[RegressionRun]]:
    twfe_rows: List[pd.DataFrame] = []
    dl_rows: List[pd.DataFrame] = []
    runs: List[RegressionRun] = []

    for outcome in outcomes:
        model_a = run_fe_model(
            panel=panel,
            model_name="Model A",
            outcome=outcome,
            regressors=["erdf_eur_pc_l1"],
            controls=controls,
            fe_type="year",
            clustering="nuts2",
        )
        model_b = run_fe_model(
            panel=panel,
            model_name="Model B",
            outcome=outcome,
            regressors=["erdf_eur_pc_l1"],
            controls=controls,
            fe_type="country_year",
            clustering="nuts2",
        )
        model_a_2way = run_fe_model_safe(
            panel=panel,
            model_name="Model A (two-way cluster)",
            outcome=outcome,
            regressors=["erdf_eur_pc_l1"],
            controls=controls,
            fe_type="year",
            clustering="nuts2_country",
        )
        model_b_2way = run_fe_model_safe(
            panel=panel,
            model_name="Model B (two-way cluster)",
            outcome=outcome,
            regressors=["erdf_eur_pc_l1"],
            controls=controls,
            fe_type="country_year",
            clustering="nuts2_country",
        )

        model_c = run_fe_model(
            panel=panel,
            model_name="Model C",
            outcome=outcome,
            regressors=["erdf_eur_pc_l1", "erdf_eur_pc_l2", "erdf_eur_pc_l3"],
            controls=controls,
            fe_type="year",
            clustering="nuts2",
        )

        twfe_rows.extend([run_to_table(model_a), run_to_table(model_b)])
        dl_rows.append(run_to_table(model_c))
        runs.extend([model_a, model_b, model_c])

        if model_a_2way is not None:
            twfe_rows.append(run_to_table(model_a_2way))
            runs.append(model_a_2way)
        if model_b_2way is not None:
            twfe_rows.append(run_to_table(model_b_2way))
            runs.append(model_b_2way)

    twfe_table = pd.concat(twfe_rows, ignore_index=True) if twfe_rows else pd.DataFrame()
    dl_table = pd.concat(dl_rows, ignore_index=True) if dl_rows else pd.DataFrame()
    return twfe_table, dl_table, runs


def run_dynamic_and_placebo(
    panel: pd.DataFrame,
    headline_outcome: str,
    controls: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[RegressionRun]]:
    runs: List[RegressionRun] = []

    model_dynamic = run_fe_model(
        panel=panel,
        model_name="Model C (headline with l0)",
        outcome=headline_outcome,
        regressors=["erdf_eur_pc", "erdf_eur_pc_l1", "erdf_eur_pc_l2", "erdf_eur_pc_l3"],
        controls=controls,
        fe_type="year",
        clustering="nuts2",
    )
    runs.append(model_dynamic)

    dynamic_terms = {
        "erdf_eur_pc": 0,
        "erdf_eur_pc_l1": 1,
        "erdf_eur_pc_l2": 2,
        "erdf_eur_pc_l3": 3,
    }
    dynamic_table = extract_terms(model_dynamic, dynamic_terms)

    model_placebo = run_fe_model(
        panel=panel,
        model_name="Model Placebo Leads-Lags",
        outcome=headline_outcome,
        regressors=[
            "erdf_eur_pc_f2",
            "erdf_eur_pc_f1",
            "erdf_eur_pc",
            "erdf_eur_pc_l1",
            "erdf_eur_pc_l2",
            "erdf_eur_pc_l3",
        ],
        controls=controls,
        fe_type="year",
        clustering="nuts2",
    )
    runs.append(model_placebo)

    placebo_terms = {
        "erdf_eur_pc_f2": -2,
        "erdf_eur_pc_f1": -1,
        "erdf_eur_pc": 0,
        "erdf_eur_pc_l1": 1,
        "erdf_eur_pc_l2": 2,
        "erdf_eur_pc_l3": 3,
    }
    placebo_table = extract_terms(model_placebo, placebo_terms)

    return dynamic_table, placebo_table, runs


def run_beta_and_heterogeneity(
    panel: pd.DataFrame,
    controls: Sequence[str],
    headline_outcome: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], List[RegressionRun]]:
    beta_outcomes = [
        outcome
        for outcome in ["gdp_pc_real_growth", "gdp_pc_pps_growth"]
        if outcome in panel.columns and panel[outcome].notna().any()
    ]

    beta_rows: List[pd.DataFrame] = []
    heterogeneity_rows: List[pd.DataFrame] = []
    runs: List[RegressionRun] = []

    beta_partial: Optional[pd.DataFrame] = None

    for outcome in beta_outcomes:
        lag_log = LAG_LOG_BY_OUTCOME[outcome]

        model_d = run_fe_model(
            panel=panel,
            model_name="Model D",
            outcome=outcome,
            regressors=[lag_log],
            controls=[],
            fe_type="year",
            clustering="nuts2",
        )
        model_e = run_fe_model(
            panel=panel,
            model_name="Model E",
            outcome=outcome,
            regressors=[lag_log, "erdf_eur_pc_l1", f"erdf_eur_pc_l1:{lag_log}"],
            controls=controls,
            fe_type="year",
            clustering="nuts2",
        )
        model_e_cy = run_fe_model(
            panel=panel,
            model_name="Model E (country-year FE)",
            outcome=outcome,
            regressors=[lag_log, "erdf_eur_pc_l1", f"erdf_eur_pc_l1:{lag_log}"],
            controls=controls,
            fe_type="country_year",
            clustering="nuts2",
        )

        beta_rows.extend([run_to_table(model_d), run_to_table(model_e), run_to_table(model_e_cy)])
        runs.extend([model_d, model_e, model_e_cy])

        if outcome == headline_outcome:
            beta_partial = build_beta_marginal_effect(model_e, lag_log)

        for category in VALID_CATEGORIES:
            subset = panel[panel["category_2014_2020"] == category].copy()
            if subset.empty:
                continue
            category_run = run_fe_model_safe(
                panel=subset,
                model_name="Model E (category subset)",
                outcome=outcome,
                regressors=[lag_log, "erdf_eur_pc_l1", f"erdf_eur_pc_l1:{lag_log}"],
                controls=controls,
                fe_type="year",
                clustering="nuts2",
            )
            if category_run is None:
                continue

            rows = run_to_table(category_run)
            rows["category_2014_2020"] = category
            heterogeneity_rows.append(rows)
            runs.append(category_run)

    beta_table = pd.concat(beta_rows, ignore_index=True) if beta_rows else pd.DataFrame()
    heterogeneity_table = (
        pd.concat(heterogeneity_rows, ignore_index=True) if heterogeneity_rows else pd.DataFrame()
    )

    return beta_table, heterogeneity_table, beta_partial, runs


def run_robustness(
    panel: pd.DataFrame,
    controls: Sequence[str],
    headline_outcome: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, Tuple[int, int], int, List[RegressionRun]]:
    runs: List[RegressionRun] = []

    # 1) Outliers: global top 1% erdf_eur_pc.
    p99 = float(panel["erdf_eur_pc"].dropna().quantile(0.99))
    outlier_subset = panel[(panel["erdf_eur_pc"].isna()) | (panel["erdf_eur_pc"] <= p99)].copy()

    outlier_a = run_fe_model(
        panel=outlier_subset,
        model_name="Model A (outliers excluded)",
        outcome=headline_outcome,
        regressors=["erdf_eur_pc_l1"],
        controls=controls,
        fe_type="year",
        clustering="nuts2",
    )
    outlier_b = run_fe_model(
        panel=outlier_subset,
        model_name="Model B (outliers excluded)",
        outcome=headline_outcome,
        regressors=["erdf_eur_pc_l1"],
        controls=controls,
        fe_type="country_year",
        clustering="nuts2",
    )

    outlier_table = pd.concat([run_to_table(outlier_a), run_to_table(outlier_b)], ignore_index=True)
    outlier_table["outlier_rule"] = "drop global top 1% erdf_eur_pc"
    outlier_table["erdf_eur_pc_global_p99"] = p99
    runs.extend([outlier_a, outlier_b])

    # 2) Balanced panel.
    required = [headline_outcome, "erdf_eur_pc_l1", *controls]
    candidates = panel[required + ["year"]].dropna()
    years = sorted(candidates["year"].unique().tolist())
    if not years:
        raise ValueError("No complete rows available to build balanced panel robustness sample.")

    start_year = max(2017, int(min(years)))
    end_year = min(2023, int(max(years)))
    window_years = list(range(start_year, end_year + 1))

    balanced_candidate = panel[panel["year"].isin(window_years)].copy()
    n_years = len(window_years)

    region_complete = balanced_candidate.groupby("nuts2_id").apply(
        lambda g: len(g) == n_years and g[required].notna().all().all()
    )
    balanced_regions = region_complete[region_complete].index.tolist()

    balanced_subset = balanced_candidate[balanced_candidate["nuts2_id"].isin(balanced_regions)].copy()
    if balanced_subset.empty:
        raise ValueError(
            "Balanced panel robustness sample is empty for window "
            f"{start_year}-{end_year}."
        )

    balanced_a = run_fe_model(
        panel=balanced_subset,
        model_name="Model A (balanced panel)",
        outcome=headline_outcome,
        regressors=["erdf_eur_pc_l1"],
        controls=controls,
        fe_type="year",
        clustering="nuts2",
    )
    balanced_b = run_fe_model(
        panel=balanced_subset,
        model_name="Model B (balanced panel)",
        outcome=headline_outcome,
        regressors=["erdf_eur_pc_l1"],
        controls=controls,
        fe_type="country_year",
        clustering="nuts2",
    )

    balanced_table = pd.concat([run_to_table(balanced_a), run_to_table(balanced_b)], ignore_index=True)
    balanced_table["balanced_window_start"] = start_year
    balanced_table["balanced_window_end"] = end_year
    balanced_table["balanced_regions"] = len(balanced_regions)
    runs.extend([balanced_a, balanced_b])

    # 3) Scaling.
    scaling_a = run_fe_model(
        panel=panel,
        model_name="Model A (scaled treatment)",
        outcome=headline_outcome,
        regressors=["erdf_k_eur_pc_l1"],
        controls=controls,
        fe_type="year",
        clustering="nuts2",
    )
    scaling_b = run_fe_model(
        panel=panel,
        model_name="Model B (scaled treatment)",
        outcome=headline_outcome,
        regressors=["erdf_k_eur_pc_l1"],
        controls=controls,
        fe_type="country_year",
        clustering="nuts2",
    )

    scaling_table = pd.concat([run_to_table(scaling_a), run_to_table(scaling_b)], ignore_index=True)
    scaling_table["treatment_scale"] = "EUR 1,000 per capita"
    runs.extend([scaling_a, scaling_b])

    return (
        outlier_table,
        balanced_table,
        scaling_table,
        p99,
        (start_year, end_year),
        len(balanced_regions),
        runs,
    )


def _extract_first_stage_f_stat(first_stage_diagnostics: object) -> float:
    if first_stage_diagnostics is None:
        return float("nan")

    if isinstance(first_stage_diagnostics, pd.DataFrame):
        if first_stage_diagnostics.empty:
            return float("nan")
        row = first_stage_diagnostics.iloc[0]
        for candidate in ["f.stat", "f_stat", "f", "partial_f_stat"]:
            if candidate in row.index:
                value = pd.to_numeric(row[candidate], errors="coerce")
                if pd.notna(value):
                    return float(value)
        for value in row.values.tolist():
            numeric = pd.to_numeric(value, errors="coerce")
            if pd.notna(numeric):
                return float(numeric)
        return float("nan")

    try:
        numeric = pd.to_numeric(first_stage_diagnostics, errors="coerce")
        if pd.notna(numeric):
            return float(numeric)
    except Exception:  # noqa: BLE001
        return float("nan")
    return float("nan")


def _rd_sample(
    df: pd.DataFrame,
    outcome: str,
    treatment: Optional[str],
    bandwidth: float,
    cutoff: float,
) -> pd.DataFrame:
    required = ["r_value", "eligible_lt75", outcome]
    if treatment is not None:
        required.append(treatment)

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"RD sample is missing required columns: {', '.join(missing)}. "
            "These should come from panel_master and running_variable_eligibility outputs."
        )

    sample = df[required].dropna().copy()
    sample["distance"] = sample["r_value"] - cutoff
    sample = sample[sample["distance"].abs() <= bandwidth].copy()
    if sample.empty:
        raise ValueError(
            f"RD sample is empty for outcome={outcome}, bandwidth={bandwidth}."
        )

    sample["w_tri"] = 1.0 - (sample["distance"].abs() / bandwidth)
    sample = sample[sample["w_tri"] > 0].copy()

    n_left = int((sample["distance"] < 0).sum())
    n_right = int((sample["distance"] >= 0).sum())
    if n_left < 10 or n_right < 10:
        raise ValueError(
            f"Insufficient RD support around cutoff for outcome={outcome}, bandwidth={bandwidth}. "
            f"left={n_left}, right={n_right}"
        )

    return sample


def estimate_sharp_rd(
    region_df: pd.DataFrame,
    outcome: str,
    window: str,
    bandwidth: float,
    cutoff: float = 75.0,
) -> RDEstimate:
    sample = _rd_sample(region_df, outcome=outcome, treatment=None, bandwidth=bandwidth, cutoff=cutoff)
    sample["running_c"] = sample["distance"]

    model = smf.wls(
        f"{outcome} ~ eligible_lt75 + running_c + eligible_lt75:running_c",
        data=sample,
        weights=sample["w_tri"],
    )
    fit = model.fit(cov_type="HC1")

    coef = float(fit.params["eligible_lt75"])
    std_err = float(fit.bse["eligible_lt75"])
    t_stat = float(fit.tvalues["eligible_lt75"])
    p_value = float(fit.pvalues["eligible_lt75"])
    ci = fit.conf_int(alpha=0.05).loc["eligible_lt75"]

    return RDEstimate(
        estimator="sharp_rd",
        outcome=outcome,
        window=window,
        bandwidth=float(bandwidth),
        coef=coef,
        std_err=std_err,
        t_stat=t_stat,
        p_value=p_value,
        ci_95_lower=float(ci.iloc[0]),
        ci_95_upper=float(ci.iloc[1]),
        n_obs=int(sample.shape[0]),
        n_left=int((sample["distance"] < 0).sum()),
        n_right=int((sample["distance"] >= 0).sum()),
        cutoff=cutoff,
        kernel="triangular",
        polynomial_order=1,
        treatment_var="eligible_lt75",
    )


def estimate_fuzzy_rd(
    region_df: pd.DataFrame,
    outcome: str,
    treatment_var: str,
    window: str,
    bandwidth: float,
    cutoff: float = 75.0,
) -> RDEstimate:
    sample = _rd_sample(
        region_df,
        outcome=outcome,
        treatment=treatment_var,
        bandwidth=bandwidth,
        cutoff=cutoff,
    )
    sample["running_c"] = sample["distance"]
    sample["eligible_running"] = sample["eligible_lt75"] * sample["running_c"]

    exog = pd.DataFrame(
        {
            "const": 1.0,
            "running_c": sample["running_c"],
            "eligible_running": sample["eligible_running"],
        }
    )
    endog = sample[[treatment_var]]
    instruments = sample[["eligible_lt75"]]
    dependent = sample[outcome]
    weights = sample["w_tri"]

    model = IV2SLS(
        dependent=dependent,
        exog=exog,
        endog=endog,
        instruments=instruments,
        weights=weights,
    )
    fit = model.fit(cov_type="robust")

    coef = float(fit.params[treatment_var])
    std_err = float(fit.std_errors[treatment_var])
    t_stat = float(fit.tstats[treatment_var])
    p_value = float(fit.pvalues[treatment_var])
    ci = fit.conf_int(level=0.95).loc[treatment_var]

    first_stage_f = _extract_first_stage_f_stat(getattr(fit.first_stage, "diagnostics", None))

    return RDEstimate(
        estimator="fuzzy_rd",
        outcome=outcome,
        window=window,
        bandwidth=float(bandwidth),
        coef=coef,
        std_err=std_err,
        t_stat=t_stat,
        p_value=p_value,
        ci_95_lower=float(ci.iloc[0]),
        ci_95_upper=float(ci.iloc[1]),
        n_obs=int(sample.shape[0]),
        n_left=int((sample["distance"] < 0).sum()),
        n_right=int((sample["distance"] >= 0).sum()),
        cutoff=cutoff,
        kernel="triangular",
        polynomial_order=1,
        treatment_var=treatment_var,
        first_stage_f_stat=first_stage_f,
    )


def _rd_rows_to_frame(rows: Sequence[RDEstimate]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "estimator",
                "outcome",
                "window",
                "bandwidth",
                "coef",
                "std_err",
                "t_stat",
                "p_value",
                "ci_95_lower",
                "ci_95_upper",
                "n_obs",
                "n_left",
                "n_right",
                "cutoff",
                "kernel",
                "polynomial_order",
                "treatment_var",
                "first_stage_f_stat",
            ]
        )
    df = pd.DataFrame([row.__dict__ for row in rows])
    return df.sort_values(["outcome", "window", "estimator", "bandwidth"]).reset_index(drop=True)


def build_region_level_identification_frame(
    panel: pd.DataFrame,
    running: pd.DataFrame,
    exposure: pd.DataFrame,
    controls: Sequence[str],
    outcomes: Sequence[str],
) -> pd.DataFrame:
    base = panel[["nuts2_id", "country", "year", *list(controls), *list(outcomes)]].copy()
    base = base.sort_values(["nuts2_id", "year"]).reset_index(drop=True)

    region = (
        base[["nuts2_id", "country"]]
        .drop_duplicates(subset=["nuts2_id"], keep="first")
        .sort_values("nuts2_id")
        .reset_index(drop=True)
    )

    for outcome in outcomes:
        for label, start, end in [
            ("post_2016_2020", 2016, 2020),
            ("post_2021_2023", 2021, 2023),
            ("pre_2010_2013", 2010, 2013),
        ]:
            series = (
                base[base["year"].between(start, end)]
                .groupby("nuts2_id", as_index=True)[outcome]
                .mean()
                .rename(f"{outcome}_{label}")
            )
            region = region.merge(series, left_on="nuts2_id", right_index=True, how="left")

    for control in controls:
        series = (
            base[base["year"].between(2010, 2013)]
            .groupby("nuts2_id", as_index=True)[control]
            .mean()
            .rename(f"{control}_pre_2010_2013")
        )
        region = region.merge(series, left_on="nuts2_id", right_index=True, how="left")

    region = region.merge(
        running[
            [
                "nuts2_id",
                "country",
                "r_value",
                "eligible_lt75",
                "category_2014_2020",
                "ref_years_used",
            ]
        ],
        on=["nuts2_id", "country"],
        how="left",
        validate="one_to_one",
    )
    region = region.merge(
        exposure,
        on=["nuts2_id", "country"],
        how="left",
        validate="one_to_one",
    )
    return region.sort_values("nuts2_id").reset_index(drop=True)


def run_rd_analysis(
    panel: pd.DataFrame,
    running: pd.DataFrame,
    exposure: pd.DataFrame,
    headline_outcome: str,
    pps_outcome: Optional[str],
    controls: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = [headline_outcome]
    if pps_outcome and pps_outcome not in outcomes:
        outcomes.append(pps_outcome)

    region_df = build_region_level_identification_frame(
        panel=panel,
        running=running,
        exposure=exposure,
        controls=controls,
        outcomes=outcomes,
    )

    baseline_bandwidth = 10.0
    bandwidth_grid = [5.0, 7.5, 10.0, 12.5, 15.0]
    rd_rows_main: List[RDEstimate] = []
    rd_rows_sensitivity: List[RDEstimate] = []

    for outcome in outcomes:
        for window in ["post_2016_2020", "post_2021_2023"]:
            outcome_col = f"{outcome}_{window}"
            if outcome_col not in region_df.columns:
                continue

            for bw in bandwidth_grid:
                try:
                    sharp = estimate_sharp_rd(
                        region_df,
                        outcome=outcome_col,
                        window=window,
                        bandwidth=bw,
                    )
                    rd_rows_sensitivity.append(sharp)
                    if np.isclose(bw, baseline_bandwidth):
                        rd_rows_main.append(sharp)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(
                        "RD sharp estimate skipped for %s (%s, bw=%s): %s",
                        outcome,
                        window,
                        bw,
                        exc,
                    )

            try:
                fuzzy = estimate_fuzzy_rd(
                    region_df,
                    outcome=outcome_col,
                    treatment_var="erdf_eur_pc_cum_2014_2020",
                    window=window,
                    bandwidth=baseline_bandwidth,
                )
                rd_rows_main.append(fuzzy)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "RD fuzzy estimate skipped for %s (%s): %s",
                    outcome,
                    window,
                    exc,
                )

    rd_main = _rd_rows_to_frame(rd_rows_main)
    rd_sensitivity = _rd_rows_to_frame(rd_rows_sensitivity)

    placebo_rows: List[RDEstimate] = []
    placebo_outcome_col = f"{headline_outcome}_pre_2010_2013"
    if placebo_outcome_col in region_df.columns:
        try:
            placebo = estimate_sharp_rd(
                region_df,
                outcome=placebo_outcome_col,
                window="pre_2010_2013",
                bandwidth=baseline_bandwidth,
            )
            placebo_rows.append(placebo)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("RD placebo estimate skipped: %s", exc)

    rd_placebo = _rd_rows_to_frame(placebo_rows)
    return region_df, rd_main, rd_sensitivity, rd_placebo


def run_iv_2sls_panel(
    panel: pd.DataFrame,
    headline_outcome: str,
    pps_outcome: Optional[str],
    controls: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = [headline_outcome]
    if pps_outcome and pps_outcome not in outcomes:
        outcomes.append(pps_outcome)

    required = [
        "nuts2_id",
        "country",
        "year",
        "country_year",
        "eligible_lt75",
        "erdf_eur_pc",
        "erdf_eur_pc_l1",
        *controls,
    ]
    for outcome in outcomes:
        required.append(outcome)
    missing = sorted(set(required) - set(panel.columns))
    if missing:
        raise ValueError(
            "IV analysis missing required panel columns: "
            f"{', '.join(missing)}. These should come from scripts/build_dataset.py (src/pipeline.py)."
        )

    work = panel[required].copy()

    # Time-varying policy intensity instrument:
    # eligible_i × EU-wide annual ERDF intensity_t.
    yearly_intensity = work.groupby("year", as_index=True)["erdf_eur_pc"].mean()
    work["eu_erdf_intensity_t"] = work["year"].map(yearly_intensity)
    work["z_eligible_intensity"] = work["eligible_lt75"] * work["eu_erdf_intensity_t"]

    iv_rows: List[Dict[str, object]] = []
    first_stage_rows: List[Dict[str, object]] = []

    for outcome in outcomes:
        for fe_label, fe_expr in [("year", "C(year)"), ("country_year", "C(country_year)")]:
            sample_cols = [
                "nuts2_id",
                "country",
                "year",
                "country_year",
                outcome,
                "erdf_eur_pc_l1",
                "z_eligible_intensity",
                *controls,
            ]
            sample = work[sample_cols].dropna().copy()
            if sample.empty:
                LOGGER.warning("IV sample empty for outcome=%s, fe=%s", outcome, fe_label)
                continue
            if sample["z_eligible_intensity"].std(ddof=0) == 0:
                LOGGER.warning(
                    "IV sample has zero instrument variance for outcome=%s, fe=%s",
                    outcome,
                    fe_label,
                )
                continue

            control_terms = " + ".join(controls) if controls else ""
            if control_terms:
                control_terms = f"{control_terms} + "

            formula = (
                f"{outcome} ~ 1 + {control_terms}C(nuts2_id) + {fe_expr} "
                "[erdf_eur_pc_l1 ~ z_eligible_intensity]"
            )

            clusters_nuts2 = pd.DataFrame(
                {"nuts2_cluster": pd.Categorical(sample["nuts2_id"]).codes},
                index=sample.index,
            )
            clusters_two_way = pd.DataFrame(
                {
                    "nuts2_cluster": pd.Categorical(sample["nuts2_id"]).codes,
                    "country_cluster": pd.Categorical(sample["country"]).codes,
                },
                index=sample.index,
            )

            for clustering, clusters in [
                ("nuts2", clusters_nuts2),
                ("nuts2_country", clusters_two_way),
            ]:
                try:
                    iv_model = IV2SLS.from_formula(formula=formula, data=sample)
                    iv_fit = iv_model.fit(cov_type="clustered", clusters=clusters)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(
                        "Skipping IV model (%s, %s, %s): %s",
                        outcome,
                        fe_label,
                        clustering,
                        exc,
                    )
                    continue

                if "erdf_eur_pc_l1" not in iv_fit.params.index:
                    LOGGER.warning(
                        "Skipping IV model (%s, %s, %s): treatment coefficient missing.",
                        outcome,
                        fe_label,
                        clustering,
                    )
                    continue

                ci = iv_fit.conf_int(level=0.95).loc["erdf_eur_pc_l1"]
                first_stage_f = _extract_first_stage_f_stat(
                    getattr(iv_fit.first_stage, "diagnostics", None)
                )

                iv_rows.append(
                    {
                        "outcome": outcome,
                        "model": "IV 2SLS (panel)",
                        "fe_type": fe_label,
                        "clustering": clustering,
                        "term": "erdf_eur_pc_l1",
                        "coef": float(iv_fit.params["erdf_eur_pc_l1"]),
                        "std_err": float(iv_fit.std_errors["erdf_eur_pc_l1"]),
                        "t_stat": float(iv_fit.tstats["erdf_eur_pc_l1"]),
                        "p_value": float(iv_fit.pvalues["erdf_eur_pc_l1"]),
                        "ci_95_lower": float(ci.iloc[0]),
                        "ci_95_upper": float(ci.iloc[1]),
                        "first_stage_f_stat": first_stage_f,
                        "n_obs": int(iv_fit.nobs),
                        "n_regions": int(sample["nuts2_id"].nunique()),
                        "sample_year_min": int(sample["year"].min()),
                        "sample_year_max": int(sample["year"].max()),
                        "instrument": "eligible_lt75 × eu_erdf_intensity_t",
                        "formula": formula,
                    }
                )

                fs_formula = (
                    f"erdf_eur_pc_l1 ~ z_eligible_intensity + {control_terms}C(nuts2_id) + {fe_expr}"
                )
                fs_fit = smf.ols(fs_formula, data=sample).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": sample["nuts2_id"]},
                )
                if "z_eligible_intensity" in fs_fit.params.index:
                    fs_ci = fs_fit.conf_int(alpha=0.05).loc["z_eligible_intensity"]
                    first_stage_rows.append(
                        {
                            "outcome": outcome,
                            "model": "IV first stage",
                            "fe_type": fe_label,
                            "term": "z_eligible_intensity",
                            "coef": float(fs_fit.params["z_eligible_intensity"]),
                            "std_err": float(fs_fit.bse["z_eligible_intensity"]),
                            "t_stat": float(fs_fit.tvalues["z_eligible_intensity"]),
                            "p_value": float(fs_fit.pvalues["z_eligible_intensity"]),
                            "ci_95_lower": float(fs_ci.iloc[0]),
                            "ci_95_upper": float(fs_ci.iloc[1]),
                            "n_obs": int(sample.shape[0]),
                            "n_regions": int(sample["nuts2_id"].nunique()),
                            "first_stage_f_stat": first_stage_f,
                        }
                    )

    iv_table = pd.DataFrame(iv_rows)
    first_stage_table = pd.DataFrame(first_stage_rows)

    if iv_table.empty:
        iv_table = pd.DataFrame(
            columns=[
                "outcome",
                "model",
                "fe_type",
                "clustering",
                "term",
                "coef",
                "std_err",
                "t_stat",
                "p_value",
                "ci_95_lower",
                "ci_95_upper",
                "first_stage_f_stat",
                "n_obs",
                "n_regions",
                "sample_year_min",
                "sample_year_max",
                "instrument",
                "formula",
            ]
        )
    if first_stage_table.empty:
        first_stage_table = pd.DataFrame(
            columns=[
                "outcome",
                "model",
                "fe_type",
                "term",
                "coef",
                "std_err",
                "t_stat",
                "p_value",
                "ci_95_lower",
                "ci_95_upper",
                "n_obs",
                "n_regions",
                "first_stage_f_stat",
            ]
        )

    if not iv_table.empty:
        iv_table = iv_table.sort_values(["outcome", "fe_type", "clustering"]).reset_index(drop=True)
    if not first_stage_table.empty:
        first_stage_table = first_stage_table.sort_values(["outcome", "fe_type"]).reset_index(drop=True)

    return iv_table, first_stage_table


def build_model_comparison_v3(
    headline_outcome: str,
    twfe_main: pd.DataFrame,
    dl_lags: pd.DataFrame,
    rd_main: pd.DataFrame,
    iv_table: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    twfe_subset = twfe_main[
        (twfe_main["outcome"] == headline_outcome)
        & (twfe_main["term"] == "erdf_eur_pc_l1")
        & (twfe_main["model"].isin(["Model A", "Model B"]))
    ].copy()
    if not twfe_subset.empty:
        twfe_subset["estimator_family"] = "TWFE"
        twfe_subset["window"] = "panel"
        rows.append(
            twfe_subset[
                [
                    "estimator_family",
                    "model",
                    "outcome",
                    "term",
                    "coef",
                    "std_err",
                    "p_value",
                    "ci_95_lower",
                    "ci_95_upper",
                    "n_obs",
                    "n_regions",
                    "fe_type",
                    "clustering",
                    "window",
                ]
            ]
        )

    dl_subset = dl_lags[
        (dl_lags["outcome"] == headline_outcome)
        & (dl_lags["model"] == "Model C")
        & (dl_lags["term"] == "erdf_eur_pc_l1")
    ].copy()
    if not dl_subset.empty:
        dl_subset["estimator_family"] = "TWFE-dynamic"
        dl_subset["window"] = "panel"
        rows.append(
            dl_subset[
                [
                    "estimator_family",
                    "model",
                    "outcome",
                    "term",
                    "coef",
                    "std_err",
                    "p_value",
                    "ci_95_lower",
                    "ci_95_upper",
                    "n_obs",
                    "n_regions",
                    "fe_type",
                    "clustering",
                    "window",
                ]
            ]
        )

    if not rd_main.empty:
        rd_subset = rd_main[
            (rd_main["outcome"] == f"{headline_outcome}_post_2016_2020")
            & (rd_main["bandwidth"] == 10.0)
        ].copy()
        if not rd_subset.empty:
            rd_subset["estimator_family"] = "RD"
            rd_subset["model"] = rd_subset["estimator"] + " (bw=10)"
            rd_subset["term"] = rd_subset["treatment_var"]
            rd_subset["fe_type"] = "local_linear"
            rd_subset["clustering"] = "HC1"
            rows.append(
                rd_subset[
                    [
                        "estimator_family",
                        "model",
                        "outcome",
                        "term",
                        "coef",
                        "std_err",
                        "p_value",
                        "ci_95_lower",
                        "ci_95_upper",
                        "n_obs",
                        "n_left",
                        "n_right",
                        "fe_type",
                        "clustering",
                        "window",
                    ]
                ].rename(columns={"n_left": "n_regions", "n_right": "n_regions_right"})
            )

    if not iv_table.empty:
        iv_subset = iv_table[
            (iv_table["outcome"] == headline_outcome)
            & (iv_table["clustering"] == "nuts2")
        ].copy()
        if not iv_subset.empty:
            iv_subset["estimator_family"] = "IV"
            iv_subset["window"] = "panel_post2014_instrument"
            rows.append(
                iv_subset[
                    [
                        "estimator_family",
                        "model",
                        "outcome",
                        "term",
                        "coef",
                        "std_err",
                        "p_value",
                        "ci_95_lower",
                        "ci_95_upper",
                        "n_obs",
                        "n_regions",
                        "fe_type",
                        "clustering",
                        "window",
                        "first_stage_f_stat",
                    ]
                ]
            )

    if not rows:
        return pd.DataFrame()

    comparison = pd.concat(rows, ignore_index=True, sort=False)
    return comparison.sort_values(["estimator_family", "model"]).reset_index(drop=True)


def create_html_report(
    headline_outcome: str,
    overview: pd.DataFrame,
    missingness: pd.DataFrame,
    headline_models: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    beta_summary: pd.DataFrame,
) -> Dict[str, List[str]]:
    figure_files = {
        "sigma": FIGURE_PATHS_V2["sigma"],
        "dynamic": FIGURE_PATHS_V2["dynamic_lag"],
        "placebo": FIGURE_PATHS_V2["leads_lags"],
        "beta_partial": FIGURE_PATHS_V2["beta_partial"],
        "growth_compare": config.QA_FIGURES["growth_comparison_trend"],
    }

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>EU Cohesion V2 Results</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2, h3 {{ margin-top: 28px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; background: #f9fafb; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; border-radius: 6px; }}
    .note {{ font-size: 0.9rem; color: #4b5563; }}
  </style>
</head>
<body>
  <h1>EU Cohesion V2 Empirical Results</h1>
  <p class=\"note\">Headline outcome: <strong>{headline_outcome}</strong></p>

  <h2>1) Dataset Overview</h2>
  {overview.to_html(index=False)}
  <h3>Key Missingness</h3>
  {missingness.sort_values('missing_rate', ascending=False).to_html(index=False)}

  <h2>2) Headline Causal Models (A/B/C)</h2>
  {headline_models.to_html(index=False)}

  <h2>3) Robustness Comparison</h2>
  {robustness_summary.to_html(index=False)}

  <h2>4) Beta and Sigma Convergence</h2>
  {beta_summary.to_html(index=False)}
  <div class=\"grid\">
    <div class=\"card\">
      <h3>Sigma Convergence</h3>
      <img src=\"figures/{figure_files['sigma'].name}\" alt=\"sigma convergence\" />
    </div>
    <div class=\"card\">
      <h3>Beta Marginal Effect</h3>
      <img src=\"figures/{figure_files['beta_partial'].name}\" alt=\"beta marginal effect\" />
    </div>
  </div>

  <h2>5) Placebo Leads/Lags</h2>
  <img src=\"figures/{figure_files['placebo'].name}\" alt=\"placebo leads lags\" />

  <h2>6) Key Figures</h2>
  <div class=\"grid\">
    <div class=\"card\">
      <h3>Dynamic Lag Response</h3>
      <img src=\"figures/{figure_files['dynamic'].name}\" alt=\"dynamic lag response\" />
    </div>
    <div class=\"card\">
      <h3>Nominal vs PPS vs Real Growth</h3>
      <img src=\"figures/{figure_files['growth_compare'].name}\" alt=\"growth comparison\" />
    </div>
  </div>

  <p class=\"note\">Generated by scripts/run_models.py from cached processed inputs.</p>
</body>
</html>
"""

    REPORT_HTML_PATH.write_text(html, encoding="utf-8")

    manifest = {
        "tables": [
            str(BASE_TABLE_PATHS["panel_overview"]),
            str(BASE_TABLE_PATHS["panel_key_missingness"]),
            str(TABLE_PATHS_V2["twfe_main"]),
            str(TABLE_PATHS_V2["dl_lags"]),
            str(TABLE_PATHS_V2["model_summary"]),
            str(TABLE_PATHS_V2["beta"]),
            str(TABLE_PATHS_V2["leads_lags"]),
            str(TABLE_PATHS_V2["robust_outliers"]),
            str(TABLE_PATHS_V2["robust_balanced"]),
            str(TABLE_PATHS_V2["robust_scaling"]),
        ],
        "figures": [str(path) for path in figure_files.values()],
        "report_html": str(REPORT_HTML_PATH),
    }

    REPORT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _style_table_html(df: pd.DataFrame, max_rows: int = 18) -> str:
    if df.empty:
        return "<p class='muted'>No rows available for this section.</p>"
    preview = df.head(max_rows).copy()
    return preview.to_html(index=False, classes="styled-table", border=0)


def _plotly_block(fig: go.Figure, include_js: bool) -> str:
    return to_html(
        fig,
        full_html=False,
        include_plotlyjs="inline" if include_js else False,
        config={"responsive": True, "displayModeBar": False},
    )


def create_html_report_v3(
    headline_outcome: str,
    overview: pd.DataFrame,
    key_missingness: pd.DataFrame,
    model_comparison: pd.DataFrame,
    rd_main: pd.DataFrame,
    rd_sensitivity: pd.DataFrame,
    rd_placebo: pd.DataFrame,
    iv_results: pd.DataFrame,
    iv_first_stage: pd.DataFrame,
    robustness: pd.DataFrame,
    sigma: pd.DataFrame,
    leads_lags: pd.DataFrame,
) -> Dict[str, List[str]]:
    kpi_n_regions = int(overview.loc[overview["metric"] == "n_regions", "value"].iloc[0])
    kpi_min_year = int(overview.loc[overview["metric"] == "min_year", "value"].iloc[0])
    kpi_max_year = int(overview.loc[overview["metric"] == "max_year", "value"].iloc[0])

    headline_iv = iv_results[
        (iv_results["outcome"] == headline_outcome)
        & (iv_results["fe_type"] == "year")
        & (iv_results["clustering"] == "nuts2")
    ]
    if headline_iv.empty:
        headline_iv = iv_results.head(1)

    if not headline_iv.empty:
        h_row = headline_iv.iloc[0]
        kpi_headline_coef = float(h_row["coef"])
        kpi_headline_p = float(h_row["p_value"])
        kpi_headline_f = float(h_row["first_stage_f_stat"])
    else:
        kpi_headline_coef = np.nan
        kpi_headline_p = np.nan
        kpi_headline_f = np.nan

    # Interactive charts
    include_js = True
    chart_blocks: Dict[str, str] = {}

    sigma_plot_df = sigma.dropna(subset=["year"]).copy().sort_values("year")
    fig_sigma = go.Figure()
    for column, label, color in [
        ("sigma_log_gdp_real", "Sigma real", "#0f766e"),
        ("sigma_log_gdp_pps", "Sigma PPS", "#c2410c"),
        ("sigma_log_gdp_nominal", "Sigma nominal", "#334155"),
    ]:
        if column in sigma_plot_df.columns:
            series = pd.to_numeric(sigma_plot_df[column], errors="coerce")
            if series.notna().any():
                fig_sigma.add_trace(
                    go.Scatter(
                        x=sigma_plot_df["year"],
                        y=series,
                        mode="lines+markers",
                        name=label,
                        line={"width": 2, "color": color},
                    )
                )
    fig_sigma.add_vrect(
        x0=2014,
        x1=2020,
        fillcolor="rgba(15,118,110,0.12)",
        line_width=0,
        annotation_text="2014-2020",
        annotation_position="top left",
    )
    fig_sigma.update_layout(
        title="Sigma convergence",
        xaxis_title="Year",
        yaxis_title="Sigma(log GDP per capita)",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        height=380,
    )
    chart_blocks["sigma"] = _plotly_block(fig_sigma, include_js=include_js)
    include_js = False

    if not model_comparison.empty:
        comp = model_comparison.copy()
        comp["label"] = comp["estimator_family"].astype(str) + " | " + comp["model"].astype(str)
        comp["err"] = 1.96 * pd.to_numeric(comp["std_err"], errors="coerce")
        fig_comp = px.scatter(
            comp,
            x="coef",
            y="label",
            color="estimator_family",
            error_x="err",
            hover_data=["p_value", "outcome", "window"],
            color_discrete_sequence=["#0f766e", "#c2410c", "#334155", "#be123c"],
        )
        fig_comp.add_vline(x=0.0, line_dash="dash", line_color="#64748b")
        fig_comp.update_layout(
            title="Model comparison (headline outcome)",
            xaxis_title="Coefficient estimate",
            yaxis_title="Estimator/model",
            template="plotly_white",
            margin={"l": 20, "r": 20, "t": 50, "b": 40},
            height=420,
            legend_title_text="Estimator",
        )
        chart_blocks["comparison"] = _plotly_block(fig_comp, include_js=include_js)
    else:
        chart_blocks["comparison"] = "<p class='muted'>Model comparison unavailable.</p>"

    if not rd_sensitivity.empty:
        rd_plot = rd_sensitivity.copy()
        rd_plot["label"] = rd_plot["outcome"].astype(str) + " | " + rd_plot["window"].astype(str)
        rd_plot["err"] = 1.96 * pd.to_numeric(rd_plot["std_err"], errors="coerce")
        fig_rd_bw = px.line(
            rd_plot,
            x="bandwidth",
            y="coef",
            color="label",
            markers=True,
            error_y="err",
            color_discrete_sequence=px.colors.qualitative.Dark24,
        )
        fig_rd_bw.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
        fig_rd_bw.update_layout(
            title="RD bandwidth sensitivity",
            xaxis_title="Bandwidth",
            yaxis_title="Estimated jump at 75 threshold",
            template="plotly_white",
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
            height=380,
        )
        chart_blocks["rd_bw"] = _plotly_block(fig_rd_bw, include_js=include_js)
    else:
        chart_blocks["rd_bw"] = "<p class='muted'>RD sensitivity unavailable.</p>"

    if not iv_first_stage.empty:
        fs = iv_first_stage.copy()
        fs["label"] = fs["outcome"] + " | " + fs["fe_type"]
        fs["err"] = 1.96 * pd.to_numeric(fs["std_err"], errors="coerce")
        fig_fs = px.bar(
            fs,
            x="label",
            y="coef",
            error_y="err",
            color="outcome",
            color_discrete_sequence=["#0f766e", "#c2410c", "#334155"],
        )
        fig_fs.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
        fig_fs.update_layout(
            title="First-stage coefficient on eligibility × post",
            xaxis_title="Outcome and FE",
            yaxis_title="First-stage coefficient",
            template="plotly_white",
            margin={"l": 40, "r": 20, "t": 50, "b": 60},
            height=380,
            legend_title_text="Outcome",
        )
        chart_blocks["first_stage"] = _plotly_block(fig_fs, include_js=include_js)
    else:
        chart_blocks["first_stage"] = "<p class='muted'>First-stage diagnostics unavailable.</p>"

    if not leads_lags.empty:
        ll = leads_lags.copy().sort_values("horizon")
        fig_ll = go.Figure()
        fig_ll.add_trace(
            go.Scatter(
                x=ll["horizon"],
                y=ll["coef"],
                mode="lines+markers",
                name="Leads/lags",
                line={"color": "#be123c", "width": 2},
                error_y={
                    "type": "data",
                    "array": 1.96 * pd.to_numeric(ll["std_err"], errors="coerce"),
                    "visible": True,
                },
            )
        )
        fig_ll.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
        fig_ll.add_vline(x=0.0, line_dash="dot", line_color="#64748b")
        fig_ll.update_layout(
            title="Placebo leads/lags",
            xaxis_title="Horizon (negative = lead)",
            yaxis_title="Coefficient",
            template="plotly_white",
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
            height=360,
        )
        chart_blocks["placebo"] = _plotly_block(fig_ll, include_js=include_js)
    else:
        chart_blocks["placebo"] = "<p class='muted'>Leads/lags placebo unavailable.</p>"

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EU Cohesion V3 Causal Report</title>
  <style>
    :root {{
      --ink: #0f172a;
      --muted: #475569;
      --bg: #f8fafc;
      --card: #ffffff;
      --line: #e2e8f0;
      --accent: #0f766e;
      --accent-2: #c2410c;
      --accent-3: #be123c;
      --shadow: 0 12px 30px rgba(2, 6, 23, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: radial-gradient(circle at 12% 0%, #d1fae5 0%, #f8fafc 35%) no-repeat;
      font-family: "Avenir Next", "Trebuchet MS", "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    .shell {{
      max-width: 1260px;
      margin: 0 auto;
      padding: 24px 24px 48px;
    }}
    .hero {{
      background: linear-gradient(120deg, #0f766e, #14532d);
      color: #f8fafc;
      border-radius: 18px;
      padding: 28px 30px;
      box-shadow: var(--shadow);
      margin-bottom: 18px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif;
      font-size: 2rem;
      letter-spacing: 0.2px;
    }}
    .hero p {{ margin: 6px 0; opacity: 0.95; }}
    .topnav {{
      position: sticky;
      top: 0;
      z-index: 50;
      background: rgba(248, 250, 252, 0.95);
      backdrop-filter: blur(8px);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px 12px;
      margin-bottom: 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .topnav a {{
      color: var(--muted);
      text-decoration: none;
      font-size: 0.92rem;
      padding: 4px 8px;
      border-radius: 8px;
    }}
    .topnav a:hover {{ background: #e2e8f0; color: var(--ink); }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(5, minmax(150px, 1fr));
      gap: 12px;
      margin: 14px 0 22px;
    }}
    .kpi {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      box-shadow: var(--shadow);
    }}
    .kpi .label {{ color: var(--muted); font-size: 0.84rem; margin-bottom: 6px; }}
    .kpi .value {{ font-size: 1.2rem; font-weight: 700; }}
    .section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: var(--shadow);
      padding: 18px;
      margin-bottom: 16px;
    }}
    .section h2 {{
      margin: 0 0 12px;
      font-size: 1.2rem;
      font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif;
    }}
    .muted {{ color: var(--muted); font-size: 0.9rem; }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    .styled-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 0.85rem;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
    }}
    .styled-table th, .styled-table td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 9px;
      text-align: left;
      vertical-align: top;
    }}
    .styled-table th {{
      background: #ecfeff;
      font-weight: 600;
    }}
    .img-card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
      background: #ffffff;
    }}
    .img-card img {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid var(--line);
    }}
    @media (max-width: 980px) {{
      .kpis {{ grid-template-columns: repeat(2, minmax(130px, 1fr)); }}
      .grid-2 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header class="hero">
      <h1>EU Cohesion Policy V3: Causal Identification Report</h1>
      <p>Headline estimator: panel IV (eligibility × post instrument) for <strong>{headline_outcome}</strong>.</p>
      <p>Benchmarks: TWFE models A/B/C, RD around the 75 threshold, placebo leads/lags, and convergence diagnostics.</p>
    </header>

    <nav class="topnav">
      <a href="#dataset">Dataset</a>
      <a href="#headline">Headline Results</a>
      <a href="#rd">RD</a>
      <a href="#iv">IV Diagnostics</a>
      <a href="#convergence">Convergence</a>
      <a href="#robustness">Robustness</a>
      <a href="#limits">Limitations</a>
    </nav>

    <section class="kpis">
      <div class="kpi"><div class="label">Regions</div><div class="value">{kpi_n_regions}</div></div>
      <div class="kpi"><div class="label">Year range</div><div class="value">{kpi_min_year}-{kpi_max_year}</div></div>
      <div class="kpi"><div class="label">Headline IV coef</div><div class="value">{kpi_headline_coef:.4f}</div></div>
      <div class="kpi"><div class="label">Headline IV p-value</div><div class="value">{kpi_headline_p:.4f}</div></div>
      <div class="kpi"><div class="label">First-stage F</div><div class="value">{kpi_headline_f:.2f}</div></div>
    </section>

    <section class="section" id="dataset">
      <h2>Dataset Coverage and Missingness</h2>
      <div class="grid-2">
        <div>{_style_table_html(overview, max_rows=10)}</div>
        <div>{_style_table_html(key_missingness.sort_values("missing_rate", ascending=False), max_rows=15)}</div>
      </div>
    </section>

    <section class="section" id="headline">
      <h2>Headline Causal Results</h2>
      <p class="muted">IV and RD are foregrounded; TWFE is retained as benchmark.</p>
      {chart_blocks["comparison"]}
      <div class="grid-2">
        <div>{_style_table_html(model_comparison, max_rows=18)}</div>
        <div>{_style_table_html(iv_results, max_rows=18)}</div>
      </div>
    </section>

    <section class="section" id="rd">
      <h2>Regression Discontinuity Around 75% Eligibility Cutoff</h2>
      {chart_blocks["rd_bw"]}
      <div class="grid-2">
        <div>{_style_table_html(rd_main, max_rows=16)}</div>
        <div>{_style_table_html(rd_placebo, max_rows=12)}</div>
      </div>
      <div class="grid-2" style="margin-top: 12px;">
        <div class="img-card">
          <p class="muted">Binned scatter (PNG)</p>
          <img src="figures/{FIGURE_PATHS_V3['rd_binned'].name}" alt="RD binned scatter" />
        </div>
        <div class="img-card">
          <p class="muted">Bandwidth sensitivity (PNG)</p>
          <img src="figures/{FIGURE_PATHS_V3['rd_bandwidth'].name}" alt="RD bandwidth sensitivity" />
        </div>
      </div>
    </section>

    <section class="section" id="iv">
      <h2>IV Diagnostics</h2>
      {chart_blocks["first_stage"]}
      <div class="grid-2">
        <div>{_style_table_html(iv_first_stage, max_rows=16)}</div>
        <div>{_style_table_html(iv_results, max_rows=16)}</div>
      </div>
    </section>

    <section class="section" id="convergence">
      <h2>Convergence Evidence</h2>
      {chart_blocks["sigma"]}
      <div class="grid-2">
        <div class="img-card">
          <p class="muted">Placebo leads/lags (interactive)</p>
          {chart_blocks["placebo"]}
        </div>
        <div class="img-card">
          <p class="muted">Placebo leads/lags (PNG fallback)</p>
          <img src="figures/{FIGURE_PATHS_V3['leads_lags'].name}" alt="Leads lags placebo" />
        </div>
      </div>
    </section>

    <section class="section" id="robustness">
      <h2>Robustness Overview</h2>
      {_style_table_html(robustness, max_rows=18)}
    </section>

    <section class="section" id="limits">
      <h2>Limitations</h2>
      <ul>
        <li>Real GDP per capita is reconstructed from regional volume indices anchored to nominal levels.</li>
        <li>Eligibility mapping is rule-based and may not align perfectly with all NUTS revision edge cases.</li>
        <li>RD sample size around threshold is finite; IV assumptions still require caution.</li>
      </ul>
      <p class="muted">Generated by scripts/run_models.py. This file is standalone and can be opened directly.</p>
    </section>
  </div>
</body>
</html>
"""

    REPORT_HTML_PATH.write_text(html, encoding="utf-8")

    manifest = {
        "tables": [
            str(BASE_TABLE_PATHS["panel_overview"]),
            str(BASE_TABLE_PATHS["panel_key_missingness"]),
            str(TABLE_PATHS_V3["twfe_main"]),
            str(TABLE_PATHS_V3["dl_lags"]),
            str(TABLE_PATHS_V3["rd_main"]),
            str(TABLE_PATHS_V3["rd_placebo"]),
            str(TABLE_PATHS_V3["iv_2sls"]),
            str(TABLE_PATHS_V3["iv_first_stage"]),
            str(TABLE_PATHS_V3["model_summary"]),
            str(TABLE_PATHS_V3["robust_outliers"]),
            str(TABLE_PATHS_V3["robust_balanced"]),
            str(TABLE_PATHS_V3["robust_scaling"]),
        ],
        "figures": [
            str(FIGURE_PATHS_V3["rd_binned"]),
            str(FIGURE_PATHS_V3["rd_bandwidth"]),
            str(FIGURE_PATHS_V3["leads_lags"]),
            str(FIGURE_PATHS_V3["sigma"]),
            str(FIGURE_PATHS_V3["dynamic_lag"]),
            str(FIGURE_PATHS_V3["beta_partial"]),
        ],
        "report_html": str(REPORT_HTML_PATH),
    }
    REPORT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _build_leave_one_out_country_mean(panel: pd.DataFrame, value_column: str) -> pd.Series:
    grouped = panel.groupby(["country", "year"], sort=False)[value_column]
    sum_by_group = grouped.transform("sum")
    count_by_group = grouped.transform("count")
    own = panel[value_column]
    leave_out = np.where(
        count_by_group > 1,
        (sum_by_group - own) / (count_by_group - 1),
        np.nan,
    )
    return pd.Series(leave_out, index=panel.index)


def _within_group_variance(series: pd.Series, group: pd.Series) -> float:
    try:
        within_std = series.groupby(group, sort=False).std()
        return float(within_std.fillna(0.0).sum())
    except Exception:  # noqa: BLE001
        return 0.0


def _first_stage_ols_diagnostic(
    data: pd.DataFrame,
    endog: str,
    instrument: str,
    controls: Sequence[str],
    fe_terms: Sequence[str],
    cluster_col: Optional[str],
    candidate_id: str,
    candidate_label: str,
    sample_type: str,
    spec_name: str,
    require_within_variation_by: Optional[str] = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "candidate_id": candidate_id,
        "candidate_label": candidate_label,
        "sample_type": sample_type,
        "spec_name": spec_name,
        "endogenous_var": endog,
        "instrument_var": instrument,
        "fe_spec": " + ".join(fe_terms) if fe_terms else "none",
        "controls_used": ",".join(controls),
        "n_obs": 0,
        "n_regions": 0,
        "coef": np.nan,
        "std_err": np.nan,
        "t_stat": np.nan,
        "p_value": np.nan,
        "f_stat": np.nan,
        "partial_r2": np.nan,
        "status": "failed",
        "status_reason": "",
    }

    required = [endog, instrument, *controls]
    if "nuts2_id" in data.columns:
        required.append("nuts2_id")
    if "country" in data.columns:
        required.append("country")
    if sample_type == "panel":
        required.append("year")
    missing = sorted(set(required) - set(data.columns))
    if missing:
        row["status_reason"] = f"missing columns: {', '.join(missing)}"
        return row

    sample = data[required].dropna().copy()
    row["n_obs"] = int(sample.shape[0])
    row["n_regions"] = int(sample["nuts2_id"].nunique()) if "nuts2_id" in sample.columns else np.nan

    if sample.empty:
        row["status_reason"] = "empty sample after dropping missing values"
        return row

    if pd.to_numeric(sample[instrument], errors="coerce").std(ddof=0) == 0:
        row["status_reason"] = "instrument has zero variance in estimation sample"
        return row

    if require_within_variation_by is not None:
        within_var = _within_group_variance(
            pd.to_numeric(sample[instrument], errors="coerce"),
            sample[require_within_variation_by],
        )
        if within_var == 0:
            row["status_reason"] = (
                f"instrument has no within-{require_within_variation_by} variation under FE"
            )
            return row

    rhs_full = [instrument, *controls, *fe_terms]
    rhs_restricted = [*controls, *fe_terms]
    formula_full = f"{endog} ~ {' + '.join(rhs_full) if rhs_full else '1'}"
    formula_restricted = f"{endog} ~ {' + '.join(rhs_restricted) if rhs_restricted else '1'}"

    try:
        if cluster_col is None:
            fit_full = smf.ols(formula=formula_full, data=sample).fit()
        else:
            fit_full = smf.ols(formula=formula_full, data=sample).fit(
                cov_type="cluster",
                cov_kwds={"groups": sample[cluster_col]},
            )
        fit_restricted = smf.ols(formula=formula_restricted, data=sample).fit()
    except Exception as exc:  # noqa: BLE001
        row["status_reason"] = f"fit failed: {exc}"
        return row

    if instrument not in fit_full.params.index:
        row["status_reason"] = "instrument dropped due rank deficiency"
        return row

    coef = float(fit_full.params[instrument])
    std_err = float(fit_full.bse[instrument])
    t_stat = float(fit_full.tvalues[instrument])
    p_value = float(fit_full.pvalues[instrument])
    f_stat = float(t_stat**2)

    if (1.0 - fit_restricted.rsquared) > 1e-9:
        partial_r2 = float((fit_full.rsquared - fit_restricted.rsquared) / (1.0 - fit_restricted.rsquared))
    else:
        partial_r2 = np.nan

    row.update(
        {
            "coef": coef,
            "std_err": std_err,
            "t_stat": t_stat,
            "p_value": p_value,
            "f_stat": f_stat,
            "partial_r2": partial_r2,
            "status": "ok",
            "status_reason": "",
        }
    )
    return row


def build_iv_first_stage_candidates_v31(
    panel: pd.DataFrame,
    region_df: pd.DataFrame,
    controls: Sequence[str],
) -> pd.DataFrame:
    work = panel.copy()
    work["eligible_lt75"] = pd.to_numeric(work["eligible_lt75"], errors="coerce")
    work["eligible_lt90"] = np.where(pd.to_numeric(work["r_value"], errors="coerce") < 90.0, 1.0, 0.0)
    work["post2014"] = (work["year"] >= 2014).astype(float)
    work["eu_mean_erdf_t"] = work.groupby("year", sort=False)["erdf_eur_pc"].transform("mean")
    work["country_mean_erdf_t_loo"] = _build_leave_one_out_country_mean(work, "erdf_eur_pc")

    work["z1_post75"] = work["eligible_lt75"] * work["post2014"]
    work["z2_eu75"] = work["eligible_lt75"] * work["eu_mean_erdf_t"]
    work["z3_country75"] = work["eligible_lt75"] * work["country_mean_erdf_t_loo"]
    work["z5_eu90"] = work["eligible_lt90"] * work["eu_mean_erdf_t"]

    rows: List[Dict[str, object]] = []
    panel_candidates = [
        ("Z1", "eligible_lt75 × post2014", "z1_post75"),
        ("Z2", "eligible_lt75 × eu_mean_erdf_t", "z2_eu75"),
        ("Z3", "eligible_lt75 × country_mean_erdf_t_loo", "z3_country75"),
        ("Z5", "eligible_lt90 × eu_mean_erdf_t", "z5_eu90"),
    ]
    for candidate_id, label, instrument in panel_candidates:
        rows.append(
            _first_stage_ols_diagnostic(
                data=work,
                endog="erdf_eur_pc_l1",
                instrument=instrument,
                controls=controls,
                fe_terms=["C(nuts2_id)", "C(year)"],
                cluster_col="nuts2_id",
                candidate_id=candidate_id,
                candidate_label=label,
                sample_type="panel",
                spec_name="region_FE + year_FE",
                require_within_variation_by="nuts2_id",
            )
        )

    # Region-level (cross-sectional) candidates for cumulative exposure.
    region = region_df.copy()
    region["eligible_lt75"] = pd.to_numeric(region["eligible_lt75"], errors="coerce")
    region["eligible_lt90"] = np.where(pd.to_numeric(region["r_value"], errors="coerce") < 90.0, 1.0, 0.0)

    cross_controls = [
        "log_gdp_pc_real_pre_2010_2013",
        "unemp_rate_pre_2010_2013",
        "emp_rate_pre_2010_2013",
    ]
    available_cross_controls = [column for column in cross_controls if column in region.columns]

    for endog in ["erdf_eur_pc_cum_2014_2020", "erdf_eur_pc_cum_2015_2020"]:
        if endog not in region.columns:
            continue
        rows.append(
            _first_stage_ols_diagnostic(
                data=region,
                endog=endog,
                instrument="eligible_lt75",
                controls=available_cross_controls,
                fe_terms=["C(country)"],
                cluster_col=None,
                candidate_id="Z4_75",
                candidate_label="eligible_lt75 (cross-section cumulative exposure)",
                sample_type="cross_section",
                spec_name="country_FE + pre_controls",
                require_within_variation_by=None,
            )
        )
        rows.append(
            _first_stage_ols_diagnostic(
                data=region,
                endog=endog,
                instrument="eligible_lt90",
                controls=available_cross_controls,
                fe_terms=["C(country)"],
                cluster_col=None,
                candidate_id="Z4_90",
                candidate_label="eligible_lt90 (cross-section cumulative exposure)",
                sample_type="cross_section",
                spec_name="country_FE + pre_controls",
                require_within_variation_by=None,
            )
        )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates

    candidates = candidates.sort_values(
        ["status", "f_stat", "sample_type", "candidate_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    return candidates


def _select_headline_iv_candidate(candidates: pd.DataFrame) -> Optional[Dict[str, object]]:
    if candidates.empty:
        return None

    valid = candidates[(candidates["status"] == "ok") & pd.to_numeric(candidates["f_stat"], errors="coerce").notna()].copy()
    if valid.empty:
        return None

    valid["f_stat"] = pd.to_numeric(valid["f_stat"], errors="coerce")
    # Prefer stronger first stages; break ties by sample type (cross-section cumulative is less rank-sensitive here).
    valid["sample_priority"] = np.where(valid["sample_type"] == "cross_section", 0, 1)
    best = valid.sort_values(["f_stat", "sample_priority"], ascending=[False, True]).iloc[0]
    return best.to_dict()


def run_panel_iv_with_instrument_v31(
    panel: pd.DataFrame,
    outcome: str,
    instrument_var: str,
    controls: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "nuts2_id",
        "country",
        "year",
        outcome,
        "erdf_eur_pc_l1",
        instrument_var,
        *controls,
    ]
    missing = sorted(set(required) - set(panel.columns))
    if missing:
        raise ValueError(
            f"Panel IV missing columns: {', '.join(missing)}. "
            "These should come from processed panel and constructed instrument features."
        )

    sample = panel[required].dropna().copy()
    if sample.empty:
        raise ValueError("Panel IV sample is empty after dropping missing values.")

    formula = (
        f"{outcome} ~ 1 + {' + '.join(controls + ['C(nuts2_id)', 'C(year)'])} "
        f"[erdf_eur_pc_l1 ~ {instrument_var}]"
    )

    iv_fit = IV2SLS.from_formula(formula=formula, data=sample).fit(
        cov_type="clustered",
        clusters=pd.DataFrame({"nuts2_cluster": pd.Categorical(sample["nuts2_id"]).codes}, index=sample.index),
    )

    ci = iv_fit.conf_int(level=0.95).loc["erdf_eur_pc_l1"]
    first_stage_f = _extract_first_stage_f_stat(getattr(iv_fit.first_stage, "diagnostics", None))

    iv_table = pd.DataFrame(
        [
            {
                "outcome": outcome,
                "model": "IV 2SLS panel (selected candidate)",
                "sample_type": "panel",
                "term": "erdf_eur_pc_l1",
                "instrument": instrument_var,
                "coef": float(iv_fit.params["erdf_eur_pc_l1"]),
                "std_err": float(iv_fit.std_errors["erdf_eur_pc_l1"]),
                "t_stat": float(iv_fit.tstats["erdf_eur_pc_l1"]),
                "p_value": float(iv_fit.pvalues["erdf_eur_pc_l1"]),
                "ci_95_lower": float(ci.iloc[0]),
                "ci_95_upper": float(ci.iloc[1]),
                "first_stage_f_stat": first_stage_f,
                "n_obs": int(iv_fit.nobs),
                "n_regions": int(sample["nuts2_id"].nunique()),
                "sample_year_min": int(sample["year"].min()),
                "sample_year_max": int(sample["year"].max()),
                "controls_used": ",".join(controls),
                "fe_spec": "region_FE + year_FE",
            }
        ]
    )

    fs_diag = getattr(iv_fit.first_stage, "diagnostics", None)
    first_stage_table = pd.DataFrame()
    if isinstance(fs_diag, pd.DataFrame) and not fs_diag.empty:
        fs_row = fs_diag.iloc[0]
        first_stage_table = pd.DataFrame(
            [
                {
                    "endogenous_var": "erdf_eur_pc_l1",
                    "instrument_var": instrument_var,
                    "rsquared": float(pd.to_numeric(fs_row.get("rsquared"), errors="coerce")),
                    "partial_r2": float(pd.to_numeric(fs_row.get("partial.rsquared"), errors="coerce")),
                    "f_stat": float(pd.to_numeric(fs_row.get("f.stat"), errors="coerce")),
                    "f_pvalue": float(pd.to_numeric(fs_row.get("f.pval"), errors="coerce")),
                    "n_obs": int(sample.shape[0]),
                    "sample_type": "panel",
                    "fe_spec": "region_FE + year_FE",
                }
            ]
        )

    return iv_table, first_stage_table


def run_cross_section_iv_v31(
    region_df: pd.DataFrame,
    instrument_var: str,
    endog_var: str,
    controls: Sequence[str],
    headline_outcome: str,
    pps_outcome: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = region_df.copy()
    if instrument_var == "eligible_lt90" and "eligible_lt90" not in work.columns:
        work["eligible_lt90"] = np.where(pd.to_numeric(work["r_value"], errors="coerce") < 90.0, 1.0, 0.0)

    outcomes = [
        (headline_outcome, f"{headline_outcome}_post_2016_2020"),
        (headline_outcome, f"{headline_outcome}_post_2021_2023"),
    ]
    if pps_outcome is not None:
        outcomes.extend(
            [
                (pps_outcome, f"{pps_outcome}_post_2016_2020"),
                (pps_outcome, f"{pps_outcome}_post_2021_2023"),
            ]
        )

    results_rows: List[Dict[str, object]] = []
    first_stage_rows: List[Dict[str, object]] = []
    scatter_source: Optional[pd.DataFrame] = None

    for outcome_label, outcome_col in outcomes:
        if outcome_col not in region_df.columns:
            continue

        required = ["nuts2_id", "country", outcome_col, endog_var, instrument_var, *controls]
        sample = work[required].dropna().copy()
        if sample.empty:
            continue

        rhs_controls = [*controls, "C(country)"]
        rhs = " + ".join(rhs_controls)
        formula = (
            f"{outcome_col} ~ 1 + {rhs} "
            f"[{endog_var} ~ {instrument_var}]"
        )

        try:
            iv_fit = IV2SLS.from_formula(formula=formula, data=sample).fit(cov_type="robust")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Cross-section IV failed for %s: %s", outcome_col, exc)
            continue

        ci = iv_fit.conf_int(level=0.95).loc[endog_var]
        first_stage_f = _extract_first_stage_f_stat(getattr(iv_fit.first_stage, "diagnostics", None))

        results_rows.append(
            {
                "outcome": outcome_label,
                "window": outcome_col.replace(f"{outcome_label}_", ""),
                "model": "IV 2SLS cross-section (selected candidate)",
                "sample_type": "cross_section",
                "term": endog_var,
                "instrument": instrument_var,
                "coef": float(iv_fit.params[endog_var]),
                "std_err": float(iv_fit.std_errors[endog_var]),
                "t_stat": float(iv_fit.tstats[endog_var]),
                "p_value": float(iv_fit.pvalues[endog_var]),
                "ci_95_lower": float(ci.iloc[0]),
                "ci_95_upper": float(ci.iloc[1]),
                "first_stage_f_stat": first_stage_f,
                "n_obs": int(iv_fit.nobs),
                "n_regions": int(sample["nuts2_id"].nunique()),
                "controls_used": ",".join(controls),
                "fe_spec": "country_FE",
            }
        )

        fs_diag = getattr(iv_fit.first_stage, "diagnostics", None)
        if isinstance(fs_diag, pd.DataFrame) and not fs_diag.empty:
            fs_row = fs_diag.iloc[0]
            first_stage_rows.append(
                {
                    "outcome": outcome_label,
                    "window": outcome_col.replace(f"{outcome_label}_", ""),
                    "endogenous_var": endog_var,
                    "instrument_var": instrument_var,
                    "rsquared": float(pd.to_numeric(fs_row.get("rsquared"), errors="coerce")),
                    "partial_r2": float(pd.to_numeric(fs_row.get("partial.rsquared"), errors="coerce")),
                    "f_stat": float(pd.to_numeric(fs_row.get("f.stat"), errors="coerce")),
                    "f_pvalue": float(pd.to_numeric(fs_row.get("f.pval"), errors="coerce")),
                    "n_obs": int(sample.shape[0]),
                    "sample_type": "cross_section",
                    "fe_spec": "country_FE",
                }
            )

        if scatter_source is None and outcome_label == headline_outcome and outcome_col.endswith("post_2016_2020"):
            try:
                fs_fit = smf.ols(
                    f"{endog_var} ~ {instrument_var} + {' + '.join(rhs_controls)}",
                    data=sample,
                ).fit()
                scatter_source = pd.DataFrame(
                    {
                        "actual_exposure": sample[endog_var].astype(float),
                        "fitted_exposure": fs_fit.fittedvalues.astype(float),
                    }
                )
            except Exception:  # noqa: BLE001
                scatter_source = pd.DataFrame(columns=["actual_exposure", "fitted_exposure"])

    return pd.DataFrame(results_rows), pd.DataFrame(first_stage_rows), (
        scatter_source
        if scatter_source is not None
        else pd.DataFrame(columns=["actual_exposure", "fitted_exposure"])
    )


def run_rd_funding_jump_v31(
    region_df: pd.DataFrame,
    bandwidths: Sequence[float],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    treatment_sd = float(pd.to_numeric(region_df["erdf_eur_pc_cum_2014_2020"], errors="coerce").std())

    for bw in bandwidths:
        try:
            est = estimate_sharp_rd(
                region_df=region_df,
                outcome="erdf_eur_pc_cum_2014_2020",
                window="funding_jump",
                bandwidth=float(bw),
            )
            f_stat = float(est.t_stat**2)
            rows.append(
                {
                    "bandwidth": float(bw),
                    "jump_coef": est.coef,
                    "std_err": est.std_err,
                    "t_stat": est.t_stat,
                    "p_value": est.p_value,
                    "ci_95_lower": est.ci_95_lower,
                    "ci_95_upper": est.ci_95_upper,
                    "n_obs": est.n_obs,
                    "n_left": est.n_left,
                    "n_right": est.n_right,
                    "first_stage_f_stat": f_stat,
                    "jump_over_sd": est.coef / treatment_sd if treatment_sd > 0 else np.nan,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "bandwidth": float(bw),
                    "jump_coef": np.nan,
                    "std_err": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "ci_95_lower": np.nan,
                    "ci_95_upper": np.nan,
                    "n_obs": 0,
                    "n_left": 0,
                    "n_right": 0,
                    "first_stage_f_stat": np.nan,
                    "jump_over_sd": np.nan,
                    "error": str(exc),
                }
            )

    table = pd.DataFrame(rows).sort_values("bandwidth").reset_index(drop=True)
    pvals = pd.to_numeric(table["p_value"], errors="coerce")
    econ = pd.to_numeric(table["jump_over_sd"], errors="coerce").abs()
    fstats = pd.to_numeric(table["first_stage_f_stat"], errors="coerce")

    near_zero = bool((pvals > 0.10).fillna(True).all() and (econ < 0.20).fillna(True).all())
    weak_f = bool((fstats < 10).fillna(True).all())
    table["fuzzy_rd_viable"] = not (near_zero or weak_f)
    table["viability_reason"] = np.where(
        table["fuzzy_rd_viable"],
        "funding jump strong enough for fuzzy RD",
        np.where(
            near_zero,
            "funding jump statistically/economically near zero across bandwidths",
            "funding jump imprecise/weak (first-stage F below 10)",
        ),
    )
    return table


def run_sharp_rd_outcomes_v31(
    region_df: pd.DataFrame,
    headline_outcome: str,
    pps_outcome: Optional[str],
    bandwidths: Sequence[float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = [headline_outcome]
    if pps_outcome is not None and pps_outcome not in outcomes:
        outcomes.append(pps_outcome)

    baseline_bw = 10.0
    baseline_rows: List[Dict[str, object]] = []
    sensitivity_rows: List[Dict[str, object]] = []
    placebo_rows: List[Dict[str, object]] = []

    for outcome in outcomes:
        for window in ["post_2016_2020", "post_2021_2023"]:
            outcome_col = f"{outcome}_{window}"
            if outcome_col not in region_df.columns:
                continue
            for bw in bandwidths:
                try:
                    est = estimate_sharp_rd(
                        region_df=region_df,
                        outcome=outcome_col,
                        window=window,
                        bandwidth=float(bw),
                    )
                    row = {
                        "outcome": outcome,
                        "window": window,
                        "bandwidth": float(bw),
                        "coef": est.coef,
                        "std_err": est.std_err,
                        "t_stat": est.t_stat,
                        "p_value": est.p_value,
                        "ci_95_lower": est.ci_95_lower,
                        "ci_95_upper": est.ci_95_upper,
                        "n_obs": est.n_obs,
                        "n_left": est.n_left,
                        "n_right": est.n_right,
                        "interpretation": "eligibility ITT effect (sharp RD at 75 cutoff)",
                    }
                    sensitivity_rows.append(row)
                    if np.isclose(float(bw), baseline_bw):
                        baseline_rows.append(row)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(
                        "Sharp RD skipped for outcome=%s, window=%s, bw=%s: %s",
                        outcome,
                        window,
                        bw,
                        exc,
                    )

    placebo_col = f"{headline_outcome}_pre_2010_2013"
    if placebo_col in region_df.columns:
        try:
            placebo = estimate_sharp_rd(
                region_df=region_df,
                outcome=placebo_col,
                window="pre_2010_2013",
                bandwidth=baseline_bw,
            )
            placebo_rows.append(
                {
                    "outcome": headline_outcome,
                    "window": "pre_2010_2013",
                    "bandwidth": baseline_bw,
                    "coef": placebo.coef,
                    "std_err": placebo.std_err,
                    "t_stat": placebo.t_stat,
                    "p_value": placebo.p_value,
                    "ci_95_lower": placebo.ci_95_lower,
                    "ci_95_upper": placebo.ci_95_upper,
                    "n_obs": placebo.n_obs,
                    "n_left": placebo.n_left,
                    "n_right": placebo.n_right,
                    "interpretation": "placebo pretrend jump should be near zero",
                }
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Sharp RD placebo failed: %s", exc)

    baseline_table = pd.DataFrame(baseline_rows).sort_values(["outcome", "window"]).reset_index(drop=True)
    sensitivity_table = pd.DataFrame(sensitivity_rows).sort_values(
        ["outcome", "window", "bandwidth"]
    ).reset_index(drop=True)
    placebo_table = pd.DataFrame(placebo_rows)

    return baseline_table, sensitivity_table, placebo_table


def _plot_first_stage_scatter_png(first_stage_scatter: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6))
    if first_stage_scatter.empty:
        ax.text(0.5, 0.5, "No first-stage scatter data", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = pd.to_numeric(first_stage_scatter["fitted_exposure"], errors="coerce")
        y = pd.to_numeric(first_stage_scatter["actual_exposure"], errors="coerce")
        ax.scatter(x, y, alpha=0.75, color="#0f766e", edgecolor="white", linewidth=0.4)
        if x.notna().sum() >= 3 and y.notna().sum() >= 3:
            coeff = np.polyfit(x.fillna(x.mean()), y.fillna(y.mean()), deg=1)
            grid = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(grid, coeff[0] * grid + coeff[1], color="#c2410c", linewidth=2)
        ax.set_xlabel("First-stage fitted cumulative ERDF per capita")
        ax.set_ylabel("Observed cumulative ERDF per capita")
        ax.grid(alpha=0.25)
    ax.set_title("IV first stage: observed vs fitted exposure")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _get_git_commit_short() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _table_widget_html(table_id: str, df: pd.DataFrame, max_rows: int = 250) -> str:
    if df.empty:
        return "<div class='empty-table'>No rows available.</div>"

    preview = df.head(max_rows).copy()
    table_html = preview.to_html(
        index=False,
        classes="dyn-table",
        border=0,
        table_id=table_id,
    )
    return (
        f"<div class='table-card'>"
        f"<div class='table-toolbar'>"
        f"<input id='{table_id}_search' class='table-search' type='text' placeholder='Search rows...' />"
        f"<span class='table-note'>Rows shown: {len(preview)} / {len(df)}</span>"
        f"</div>{table_html}</div>"
    )


def create_html_report_v31(
    headline_outcome: str,
    preferred_estimator_label: str,
    preferred_estimator_note: str,
    overview: pd.DataFrame,
    panel: pd.DataFrame,
    missingness: pd.DataFrame,
    twfe_main: pd.DataFrame,
    dl_lags: pd.DataFrame,
    dynamic_lag: pd.DataFrame,
    rd_funding_jump: pd.DataFrame,
    rd_outcome_sharp: pd.DataFrame,
    rd_bandwidth: pd.DataFrame,
    rd_placebo: pd.DataFrame,
    iv_candidates: pd.DataFrame,
    iv_panel: pd.DataFrame,
    iv_cross: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    sigma: pd.DataFrame,
    beta: pd.DataFrame,
    model_comparison: pd.DataFrame,
    first_stage_scatter: pd.DataFrame,
) -> Dict[str, List[str]]:
    run_ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    git_hash = _get_git_commit_short()

    n_regions = int(panel["nuts2_id"].nunique())
    min_year = int(panel["year"].min())
    max_year = int(panel["year"].max())

    iv_headline_df = iv_cross[
        (iv_cross["outcome"] == headline_outcome)
        & (iv_cross["window"] == "post_2016_2020")
    ]
    if iv_headline_df.empty and not iv_cross.empty:
        iv_headline_df = iv_cross.head(1)
    if iv_headline_df.empty and not iv_panel.empty:
        iv_headline_df = iv_panel.head(1)

    if not iv_headline_df.empty:
        iv_headline = iv_headline_df.iloc[0]
        kpi_headline_coef = float(iv_headline["coef"])
        kpi_headline_p = float(iv_headline["p_value"])
        kpi_headline_f = float(iv_headline["first_stage_f_stat"])
    else:
        kpi_headline_coef = np.nan
        kpi_headline_p = np.nan
        kpi_headline_f = np.nan

    weak_iv_warning = bool(np.isnan(kpi_headline_f) or kpi_headline_f < 10.0)

    sample_by_year = (
        panel.groupby("year", as_index=False)
        .agg(
            n_rows=("nuts2_id", "size"),
            n_regions=("nuts2_id", "nunique"),
            headline_non_missing=(headline_outcome, lambda s: int(s.notna().sum())),
        )
        .sort_values("year")
    )
    sample_by_country = (
        panel.groupby("country", as_index=False)
        .agg(
            n_regions=("nuts2_id", "nunique"),
            n_rows=("nuts2_id", "size"),
            headline_non_missing=(headline_outcome, lambda s: int(s.notna().sum())),
        )
        .sort_values("n_regions", ascending=False)
    )

    twfe_headline = twfe_main[
        (twfe_main["outcome"] == headline_outcome)
        & (twfe_main["model"].isin(["Model A", "Model B"]))
    ].copy()
    dl_headline = dl_lags[
        (dl_lags["outcome"] == headline_outcome)
        & (dl_lags["model"] == "Model C")
    ].copy()

    # Interactive figures
    include_js = True
    charts: Dict[str, str] = {}

    fig_coverage = go.Figure()
    fig_coverage.add_trace(
        go.Bar(
            x=sample_by_year["year"],
            y=sample_by_year["n_regions"],
            name="Regions",
            marker_color="#0f766e",
        )
    )
    fig_coverage.add_trace(
        go.Scatter(
            x=sample_by_year["year"],
            y=sample_by_year["headline_non_missing"],
            name=f"Non-missing {headline_outcome}",
            mode="lines+markers",
            yaxis="y2",
            line={"color": "#c2410c", "width": 2},
        )
    )
    fig_coverage.update_layout(
        title="Coverage by year",
        template="plotly_white",
        margin={"l": 40, "r": 40, "t": 45, "b": 40},
        yaxis={"title": "Regions"},
        yaxis2={"title": "Rows with headline outcome", "overlaying": "y", "side": "right"},
        height=300,
        legend={"orientation": "h"},
    )
    charts["coverage"] = _plotly_block(fig_coverage, include_js=include_js)
    include_js = False

    key_vars = [
        "erdf_eur_pc",
        "erdf_eur_pc_l1",
        "gdp_pc_real_growth",
        "gdp_pc_pps_growth",
        "unemp_rate",
        "emp_rate",
    ]
    heat = missingness[missingness["variable"].isin(key_vars)].copy()
    if not heat.empty:
        heat_pivot = heat.pivot(index="variable", columns="year", values="missing_share").sort_index()
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=heat_pivot.values,
                x=heat_pivot.columns.astype(int),
                y=heat_pivot.index.tolist(),
                colorscale="YlOrRd",
                zmin=0,
                zmax=1,
                colorbar={"title": "Missing share"},
            )
        )
        fig_heat.update_layout(
            title="Missingness heatmap (key variables)",
            template="plotly_white",
            margin={"l": 40, "r": 20, "t": 45, "b": 40},
            height=320,
        )
        charts["missing_heat"] = _plotly_block(fig_heat, include_js=include_js)
    else:
        charts["missing_heat"] = "<div class='empty-chart'>Missingness heatmap unavailable.</div>"

    fig_dynamic = go.Figure()
    if not dynamic_lag.empty:
        dyn = dynamic_lag.sort_values("horizon")
        fig_dynamic.add_trace(
            go.Scatter(
                x=dyn["horizon"],
                y=dyn["coef"],
                mode="lines+markers",
                error_y={"type": "data", "array": 1.96 * pd.to_numeric(dyn["std_err"], errors="coerce")},
                line={"color": "#0f766e", "width": 2},
                name="Lag response",
            )
        )
    fig_dynamic.add_hline(y=0, line_dash="dash", line_color="#64748b")
    fig_dynamic.update_layout(
        title="Dynamic lag response",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 45, "b": 40},
        height=300,
        xaxis_title="Lag horizon",
        yaxis_title="Coefficient",
    )
    charts["dynamic"] = _plotly_block(fig_dynamic, include_js=include_js)

    fig_rd_fs = go.Figure()
    if not rd_funding_jump.empty:
        fs = rd_funding_jump.sort_values("bandwidth")
        fig_rd_fs.add_trace(
            go.Scatter(
                x=fs["bandwidth"],
                y=fs["jump_coef"],
                mode="lines+markers",
                error_y={"type": "data", "array": 1.96 * pd.to_numeric(fs["std_err"], errors="coerce")},
                name="Funding jump",
                line={"color": "#c2410c", "width": 2},
            )
        )
    fig_rd_fs.add_hline(y=0, line_dash="dash", line_color="#64748b")
    fig_rd_fs.update_layout(
        title="RD first stage: funding discontinuity at 75 cutoff",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 45, "b": 40},
        height=300,
        xaxis_title="Bandwidth",
        yaxis_title="Jump in cumulative ERDF pc",
    )
    charts["rd_funding"] = _plotly_block(fig_rd_fs, include_js=include_js)

    fig_rd_outcome = go.Figure()
    if not rd_bandwidth.empty:
        plot_df = rd_bandwidth[
            (rd_bandwidth["outcome"] == headline_outcome)
            & (rd_bandwidth["window"] == "post_2016_2020")
        ].copy()
        if not plot_df.empty:
            fig_rd_outcome.add_trace(
                go.Scatter(
                    x=plot_df["bandwidth"],
                    y=plot_df["coef"],
                    mode="lines+markers",
                    error_y={"type": "data", "array": 1.96 * pd.to_numeric(plot_df["std_err"], errors="coerce")},
                    line={"color": "#0f766e", "width": 2},
                    name="Eligibility ITT",
                )
            )
    fig_rd_outcome.add_hline(y=0, line_dash="dash", line_color="#64748b")
    fig_rd_outcome.update_layout(
        title="RD outcome sensitivity (sharp RD ITT)",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 45, "b": 40},
        height=300,
        xaxis_title="Bandwidth",
        yaxis_title=f"{headline_outcome} jump",
    )
    charts["rd_outcome"] = _plotly_block(fig_rd_outcome, include_js=include_js)

    fig_iv_scatter = go.Figure()
    if not first_stage_scatter.empty:
        fig_iv_scatter.add_trace(
            go.Scatter(
                x=first_stage_scatter["fitted_exposure"],
                y=first_stage_scatter["actual_exposure"],
                mode="markers",
                marker={"color": "#0f766e", "size": 7, "opacity": 0.7},
                name="Regions",
            )
        )
        x = pd.to_numeric(first_stage_scatter["fitted_exposure"], errors="coerce")
        y = pd.to_numeric(first_stage_scatter["actual_exposure"], errors="coerce")
        if x.notna().sum() >= 3 and y.notna().sum() >= 3:
            coeff = np.polyfit(x.fillna(x.mean()), y.fillna(y.mean()), 1)
            grid = np.linspace(float(x.min()), float(x.max()), 120)
            fig_iv_scatter.add_trace(
                go.Scatter(
                    x=grid,
                    y=coeff[0] * grid + coeff[1],
                    mode="lines",
                    line={"color": "#c2410c", "width": 2},
                    name="Linear fit",
                )
            )
    fig_iv_scatter.update_layout(
        title="IV first stage: observed vs fitted cumulative exposure",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 45, "b": 40},
        height=300,
        xaxis_title="First-stage fitted exposure",
        yaxis_title="Observed cumulative exposure",
    )
    charts["iv_scatter"] = _plotly_block(fig_iv_scatter, include_js=include_js)

    fig_sigma = go.Figure()
    for col, label, color in [
        ("sigma_log_gdp_real", "Sigma real", "#0f766e"),
        ("sigma_log_gdp_pps", "Sigma PPS", "#c2410c"),
        ("sigma_log_gdp_nominal", "Sigma nominal", "#334155"),
    ]:
        if col in sigma.columns and pd.to_numeric(sigma[col], errors="coerce").notna().any():
            fig_sigma.add_trace(
                go.Scatter(
                    x=sigma["year"],
                    y=pd.to_numeric(sigma[col], errors="coerce"),
                    mode="lines+markers",
                    name=label,
                    line={"width": 2, "color": color},
                )
            )
    fig_sigma.add_vrect(
        x0=2014, x1=2020, fillcolor="rgba(15,118,110,0.12)", line_width=0, annotation_text="2014-2020"
    )
    fig_sigma.update_layout(
        title="Sigma convergence",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 45, "b": 40},
        height=300,
    )
    charts["sigma"] = _plotly_block(fig_sigma, include_js=include_js)

    # Table blocks
    table_overview = _table_widget_html("tbl_overview", overview)
    table_year = _table_widget_html("tbl_year", sample_by_year)
    table_country = _table_widget_html("tbl_country", sample_by_country)
    table_twfe = _table_widget_html("tbl_twfe", twfe_headline)
    table_dl = _table_widget_html("tbl_dl", dl_headline)
    table_rd_fs = _table_widget_html("tbl_rd_fs", rd_funding_jump)
    table_rd_main = _table_widget_html("tbl_rd_main", rd_outcome_sharp)
    table_rd_placebo = _table_widget_html("tbl_rd_placebo", rd_placebo)
    table_iv_candidates = _table_widget_html("tbl_iv_candidates", iv_candidates)
    table_iv_panel = _table_widget_html("tbl_iv_panel", iv_panel)
    table_iv_cross = _table_widget_html("tbl_iv_cross", iv_cross)
    table_robust = _table_widget_html("tbl_robust", robustness_summary)
    table_beta = _table_widget_html("tbl_beta", beta)
    table_model_comp = _table_widget_html("tbl_model_comp", model_comparison)

    weak_banner = (
        "<div class='warn-banner'>Weak-instrument warning: selected first-stage F is below 10.</div>"
        if weak_iv_warning
        else "<div class='ok-banner'>Selected instrument passes the F ≥ 10 threshold.</div>"
    )

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EU Cohesion V3.1 Report</title>
  <style>
    :root {{
      --bg: #f4f7f6;
      --card: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --line: #dbe4e6;
      --brand: #0f766e;
      --brand-2: #c2410c;
      --warn: #9f1239;
      --shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; margin: 0; overflow: hidden; font-family: "Avenir Next", "Trebuchet MS", "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }}
    .app {{ display: grid; grid-template-rows: auto auto 1fr; height: 100%; }}
    .header {{
      background: linear-gradient(130deg, #0f766e, #14532d);
      color: #ecfeff;
      padding: 12px 20px;
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      box-shadow: var(--shadow);
    }}
    .title {{ font-size: 1.2rem; font-weight: 700; }}
    .meta {{ font-size: 0.86rem; opacity: 0.95; }}
    .tabs {{
      display: flex;
      gap: 8px;
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
      background: #f8fafc;
      flex-wrap: wrap;
    }}
    .tab-btn {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--muted);
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 0.85rem;
      cursor: pointer;
    }}
    .tab-btn.active {{ background: var(--brand); border-color: var(--brand); color: #fff; }}
    .content {{ position: relative; height: 100%; }}
    .tab-panel {{
      display: none;
      position: absolute;
      inset: 0;
      overflow: auto;
      padding: 14px;
    }}
    .tab-panel.active {{ display: block; }}
    .grid {{ display: grid; gap: 12px; }}
    .grid-2 {{ grid-template-columns: 1fr 1fr; }}
    .grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: var(--shadow);
      padding: 12px;
    }}
    .card h3 {{ margin: 0 0 8px; font-size: 0.98rem; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)); gap: 10px; }}
    .kpi {{ background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 10px; }}
    .kpi .label {{ font-size: 0.78rem; color: var(--muted); margin-bottom: 4px; }}
    .kpi .value {{ font-size: 1.06rem; font-weight: 700; }}
    .muted {{ color: var(--muted); font-size: 0.9rem; }}
    .warn-banner, .ok-banner {{ border-radius: 10px; padding: 8px 10px; font-size: 0.88rem; margin-bottom: 8px; }}
    .warn-banner {{ background: #fff1f2; color: var(--warn); border: 1px solid #fecdd3; }}
    .ok-banner {{ background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }}
    .table-card {{ border: 1px solid var(--line); border-radius: 10px; overflow: hidden; background: #fff; }}
    .table-toolbar {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; padding: 6px 8px; border-bottom: 1px solid var(--line); background: #f8fafc; }}
    .table-search {{ width: 220px; max-width: 100%; border: 1px solid var(--line); border-radius: 8px; padding: 6px 8px; font-size: 0.82rem; }}
    .table-note {{ font-size: 0.76rem; color: var(--muted); }}
    table.dyn-table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
    table.dyn-table th, table.dyn-table td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; }}
    table.dyn-table th {{ position: sticky; top: 0; background: #eefaf8; cursor: pointer; user-select: none; z-index: 1; }}
    .empty-table, .empty-chart {{ color: var(--muted); padding: 12px; border: 1px dashed var(--line); border-radius: 8px; background: #fff; }}
    .spec-box {{ background: #f8fafc; border: 1px solid var(--line); border-radius: 10px; padding: 10px; font-size: 0.86rem; }}
    details {{ border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; background: #fff; }}
    details + details {{ margin-top: 8px; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    @media (max-width: 1200px) {{
      .grid-2, .grid-3, .kpi-grid {{ grid-template-columns: 1fr; }}
      .table-search {{ width: 140px; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <header class="header">
      <div class="title">EU Cohesion Policy V3.1 Identification Report</div>
      <div class="meta">Run: {run_ts} | Commit: {git_hash}</div>
    </header>
    <nav class="tabs">
      <button class="tab-btn active" data-tab="overview">Overview</button>
      <button class="tab-btn" data-tab="data">Data & Coverage</button>
      <button class="tab-btn" data-tab="twfe">TWFE</button>
      <button class="tab-btn" data-tab="rd">RD</button>
      <button class="tab-btn" data-tab="iv">IV</button>
      <button class="tab-btn" data-tab="robustness">Robustness</button>
      <button class="tab-btn" data-tab="convergence">Convergence</button>
      <button class="tab-btn" data-tab="limits">Limitations & Next steps</button>
    </nav>
    <main class="content">
      <section class="tab-panel active" id="tab-overview">
        <div class="grid">
          <div class="card">
            <h3>Executive Summary</h3>
            <p class="muted">
              Preferred estimator: <strong>{preferred_estimator_label}</strong>. {preferred_estimator_note}
              Headline outcome is <strong>{headline_outcome}</strong>.
            </p>
            <div class="kpi-grid">
              <div class="kpi"><div class="label">Regions</div><div class="value">{n_regions}</div></div>
              <div class="kpi"><div class="label">Year span</div><div class="value">{min_year}-{max_year}</div></div>
              <div class="kpi"><div class="label">Headline coef</div><div class="value">{kpi_headline_coef:.4f}</div></div>
              <div class="kpi"><div class="label">Headline p-value</div><div class="value">{kpi_headline_p:.4f}</div></div>
              <div class="kpi"><div class="label">First-stage F</div><div class="value">{kpi_headline_f:.2f}</div></div>
            </div>
          </div>
          <div class="card">
            <h3>Trust Guide</h3>
            <ul>
              <li>Trust most: diagnostics-backed estimator with strongest first-stage and transparent assumptions.</li>
              <li>Use RD sharp estimates as eligibility ITT unless funding first-stage is strong enough for fuzzy RD.</li>
              <li>Treat TWFE as benchmark sensitivity, not standalone causal proof.</li>
            </ul>
            {weak_banner}
          </div>
          <div class="card">
            <h3>Model Comparison</h3>
            {table_model_comp}
          </div>
        </div>
      </section>

      <section class="tab-panel" id="tab-data">
        <div class="grid grid-2">
          <div class="card"><h3>Coverage by Year</h3>{charts["coverage"]}</div>
          <div class="card"><h3>Missingness Heatmap</h3>{charts["missing_heat"]}</div>
        </div>
        <div class="grid grid-2" style="margin-top: 10px;">
          <div class="card"><h3>Sample by Year</h3>{table_year}</div>
          <div class="card"><h3>Sample by Country</h3>{table_country}</div>
        </div>
        <div class="grid grid-2" style="margin-top: 10px;">
          <div class="card"><h3>Overview Table</h3>{table_overview}</div>
          <div class="card">
            <h3>Key Variable Definitions</h3>
            <div class="spec-box">
              <p><strong>Running variable:</strong> r = GDPpc_PPS / EU average × 100 (reference years 2007-2009; fallback <=2013).</p>
              <p><strong>Eligibility:</strong> eligible_lt75 = 1[r &lt; 75], eligible_lt90 = 1[r &lt; 90].</p>
              <p><strong>Exposure:</strong> erdf_eur_pc_l1 and cumulative erdf_eur_pc_cum_2014_2020.</p>
              <p><strong>Headline outcome:</strong> gdp_pc_real_growth.</p>
            </div>
          </div>
        </div>
      </section>

      <section class="tab-panel" id="tab-twfe">
        <div class="grid grid-2">
          <div class="card">
            <h3>Specification</h3>
            <div class="spec-box">
              <p>Model A: outcome ~ erdf_l1 + controls + region FE + year FE</p>
              <p>Model B: outcome ~ erdf_l1 + controls + region FE + country×year FE</p>
              <p>Model C: outcome ~ erdf_l1 + erdf_l2 + erdf_l3 + controls + region FE + year FE</p>
            </div>
          </div>
          <div class="card"><h3>Dynamic Lag Plot</h3>{charts["dynamic"]}</div>
        </div>
        <div class="grid grid-2" style="margin-top: 10px;">
          <div class="card"><h3>TWFE Main</h3>{table_twfe}</div>
          <div class="card"><h3>Distributed Lags</h3>{table_dl}</div>
        </div>
      </section>

      <section class="tab-panel" id="tab-rd">
        <div class="grid grid-2">
          <div class="card"><h3>Funding Discontinuity (First Stage)</h3>{charts["rd_funding"]}</div>
          <div class="card"><h3>Outcome RD Sensitivity</h3>{charts["rd_outcome"]}</div>
        </div>
        <div class="grid grid-3" style="margin-top: 10px;">
          <div class="card"><h3>Funding Jump Table</h3>{table_rd_fs}</div>
          <div class="card"><h3>Sharp RD Outcomes</h3>{table_rd_main}</div>
          <div class="card"><h3>Placebo Pretrend</h3>{table_rd_placebo}</div>
        </div>
        <div class="grid grid-2" style="margin-top: 10px;">
          <div class="card"><img src="figures/{FIGURE_PATHS_V31["rd_first_stage_funding_jump"].name}" alt="rd funding jump png" style="width:100%; border-radius:8px; border:1px solid var(--line);" /></div>
          <div class="card"><img src="figures/{FIGURE_PATHS_V31["rd_outcome_binned_scatter"].name}" alt="rd outcome png" style="width:100%; border-radius:8px; border:1px solid var(--line);" /></div>
        </div>
      </section>

      <section class="tab-panel" id="tab-iv">
        {weak_banner}
        <div class="grid grid-2">
          <div class="card"><h3>Instrument Candidate Leaderboard</h3>{table_iv_candidates}</div>
          <div class="card"><h3>First-stage Scatter</h3>{charts["iv_scatter"]}</div>
        </div>
        <div class="grid grid-2" style="margin-top: 10px;">
          <div class="card"><h3>Panel IV Results</h3>{table_iv_panel}</div>
          <div class="card"><h3>Cross-section IV Results</h3>{table_iv_cross}</div>
        </div>
        <div class="card" style="margin-top: 10px;">
          <img src="figures/{FIGURE_PATHS_V31["iv_first_stage_scatter"].name}" alt="iv first stage scatter png" style="width:100%; border-radius:8px; border:1px solid var(--line);" />
        </div>
      </section>

      <section class="tab-panel" id="tab-robustness">
        <div class="grid grid-2">
          <div class="card"><h3>Robustness Table</h3>{table_robust}</div>
          <div class="card">
            <h3>Outcome Robustness (Real vs PPS)</h3>
            { _table_widget_html("tbl_twfe_outcomes", twfe_main[(twfe_main["term"]=="erdf_eur_pc_l1") & (twfe_main["model"].isin(["Model A","Model B"]))][["outcome","model","coef","std_err","p_value","n_obs","n_regions"]]) }
          </div>
        </div>
      </section>

      <section class="tab-panel" id="tab-convergence">
        <div class="grid grid-2">
          <div class="card"><h3>Sigma Convergence</h3>{charts["sigma"]}</div>
          <div class="card"><h3>Beta Convergence</h3>{table_beta}</div>
        </div>
      </section>

      <section class="tab-panel" id="tab-limits">
        <div class="grid">
          <div class="card">
            <h3>Limitations</h3>
            <ul>
              <li>Real GDP per capita is reconstructed from volume indices, not direct chain-linked real EUR levels.</li>
              <li>Eligibility and running-variable coverage is partial because of NUTS coverage differences.</li>
              <li>Interpret RD as eligibility ITT unless funding first-stage is strong.</li>
              <li>Interpret IV as causal only when first-stage diagnostics are strong and stable.</li>
            </ul>
          </div>
          <div class="card">
            <h3>Interpretation Guardrails</h3>
            <details open>
              <summary>Supported claims</summary>
              <p>Associations and policy-rule-based estimates under the maintained exclusion assumptions and observed first-stage strength diagnostics.</p>
            </details>
            <details>
              <summary>Unsupported claims</summary>
              <p>Universal structural treatment effects across all regions/times, especially when first-stage strength falls below conventional thresholds.</p>
            </details>
            <details>
              <summary>Next steps</summary>
              <p>Strengthen external instrument design (national envelopes/programmatic shocks), harmonize category coverage under NUTS revisions, and test alternative deflators/PPS-based real outcomes.</p>
            </details>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    (function() {{
      const tabButtons = Array.from(document.querySelectorAll('.tab-btn'));
      const panels = Array.from(document.querySelectorAll('.tab-panel'));
      tabButtons.forEach(btn => {{
        btn.addEventListener('click', () => {{
          const target = btn.getAttribute('data-tab');
          tabButtons.forEach(b => b.classList.toggle('active', b === btn));
          panels.forEach(p => p.classList.toggle('active', p.id === 'tab-' + target));
        }});
      }});

      function sortTable(table, colIdx, asc) {{
        const tbody = table.tBodies[0];
        const rows = Array.from(tbody.rows);
        rows.sort((a, b) => {{
          const va = (a.cells[colIdx]?.innerText || '').trim();
          const vb = (b.cells[colIdx]?.innerText || '').trim();
          const na = Number(va); const nb = Number(vb);
          const bothNum = !Number.isNaN(na) && !Number.isNaN(nb);
          if (bothNum) return asc ? na - nb : nb - na;
          return asc ? va.localeCompare(vb) : vb.localeCompare(va);
        }});
        rows.forEach(r => tbody.appendChild(r));
      }}

      Array.from(document.querySelectorAll('table.dyn-table')).forEach(table => {{
        const headers = Array.from(table.tHead?.rows?.[0]?.cells || []);
        headers.forEach((th, idx) => {{
          let asc = true;
          th.addEventListener('click', () => {{
            sortTable(table, idx, asc);
            asc = !asc;
          }});
        }});
      }});

      Array.from(document.querySelectorAll('input.table-search')).forEach(input => {{
        const tableId = input.id.replace('_search', '');
        const table = document.getElementById(tableId);
        if (!table || !table.tBodies.length) return;
        input.addEventListener('input', () => {{
          const q = input.value.toLowerCase();
          Array.from(table.tBodies[0].rows).forEach(row => {{
            const text = row.innerText.toLowerCase();
            row.style.display = text.includes(q) ? '' : 'none';
          }});
        }});
      }});
    }})();
  </script>
</body>
</html>
"""

    REPORT_HTML_PATH.write_text(html, encoding="utf-8")

    manifest = {
        "tables": [
            str(BASE_TABLE_PATHS["panel_overview"]),
            str(BASE_TABLE_PATHS["panel_key_missingness"]),
            str(TABLE_PATHS_V31["twfe_main"]),
            str(TABLE_PATHS_V31["dl_lags"]),
            str(TABLE_PATHS_V31["rd_first_stage_funding_jump"]),
            str(TABLE_PATHS_V31["rd_outcome_sharp"]),
            str(TABLE_PATHS_V31["rd_bandwidth_sensitivity"]),
            str(TABLE_PATHS_V31["rd_placebo_pretrend"]),
            str(TABLE_PATHS_V31["iv_first_stage_candidates"]),
            str(TABLE_PATHS_V31["iv_panel_results"]),
            str(TABLE_PATHS_V31["iv_cross_section_results"]),
            str(TABLE_PATHS_V31["model_summary"]),
            str(TABLE_PATHS_V31["robust_outliers"]),
            str(TABLE_PATHS_V31["robust_balanced"]),
            str(TABLE_PATHS_V31["robust_scaling"]),
            str(TABLE_PATHS_V31["beta"]),
        ],
        "figures": [
            str(FIGURE_PATHS_V31["dynamic_lag"]),
            str(FIGURE_PATHS_V31["leads_lags"]),
            str(FIGURE_PATHS_V31["sigma"]),
            str(FIGURE_PATHS_V31["beta_partial"]),
            str(FIGURE_PATHS_V31["rd_first_stage_funding_jump"]),
            str(FIGURE_PATHS_V31["rd_outcome_binned_scatter"]),
            str(FIGURE_PATHS_V31["rd_bandwidth_sensitivity"]),
            str(FIGURE_PATHS_V31["iv_first_stage_scatter"]),
        ],
        "report_html": str(REPORT_HTML_PATH),
    }
    REPORT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def create_report_notebook(headline_outcome: str) -> None:
    nb = nbformat.v4.new_notebook()

    cells = [
        nbformat.v4.new_markdown_cell(
            "# EU Cohesion V3.1 Report\n"
            f"Headline outcome: **{headline_outcome}**"
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n"
            "root = Path('..').resolve()\n"
            "overview = pd.read_csv(root / 'outputs/tables/panel_master_overview.csv')\n"
            "missingness = pd.read_csv(root / 'outputs/tables/panel_master_key_missingness.csv')\n"
            "twfe = pd.read_csv(root / 'outputs/tables/twfe_main_results_v31.csv')\n"
            "dl = pd.read_csv(root / 'outputs/tables/dl_lags_results_v31.csv')\n"
            "summary = pd.read_csv(root / 'outputs/tables/model_comparison_summary_v31.csv')\n"
            "beta = pd.read_csv(root / 'outputs/tables/beta_convergence_results_v31.csv')\n"
            "display(overview)\n"
            "display(missingness.sort_values('missing_rate', ascending=False))"
        ),
        nbformat.v4.new_markdown_cell("## Core Models"),
        nbformat.v4.new_code_cell("display(twfe)\ndisplay(dl)\ndisplay(summary)"),
        nbformat.v4.new_markdown_cell("## Beta + Heterogeneity"),
        nbformat.v4.new_code_cell(
            "hetero = pd.read_csv(root / 'outputs/tables/heterogeneity_by_category_v31.csv')\n"
            "rd = pd.read_csv(root / 'outputs/tables/rd_outcome_sharp_results_v31.csv')\n"
            "iv_panel = pd.read_csv(root / 'outputs/tables/iv_2sls_results_v31.csv')\n"
            "iv_cross = pd.read_csv(root / 'outputs/tables/iv_cross_section_results_v31.csv')\n"
            "display(beta)\n"
            "display(rd)\n"
            "display(iv_panel)\n"
            "display(iv_cross)\n"
            "display(hetero)"
        ),
        nbformat.v4.new_markdown_cell("## Figures"),
        nbformat.v4.new_code_cell(
            "display(Image(filename=str(root / 'outputs/figures/sigma_convergence_v31.png')))\n"
            "display(Image(filename=str(root / 'outputs/figures/dynamic_lag_response_v31.png')))\n"
            "display(Image(filename=str(root / 'outputs/figures/leads_lags_placebo_v31.png')))\n"
            "display(Image(filename=str(root / 'outputs/figures/beta_convergence_partial_v31.png')))\n"
            "display(Image(filename=str(root / 'outputs/figures/rd_first_stage_funding_jump_v31.png')))\n"
            "display(Image(filename=str(root / 'outputs/figures/rd_outcome_binned_scatter_v31.png')))\n"
            "display(Image(filename=str(root / 'outputs/figures/rd_bandwidth_sensitivity_v31.png')))\n"
            "display(Image(filename=str(root / 'outputs/figures/iv_first_stage_scatter_v31.png')))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Limitations\n"
            "- Real GDP per capita is reconstructed from NUTS2 volume index (2015=100) anchored to nominal per-capita levels.\n"
            "- Eligibility categories are reconstructed from 2007-2009 PPS relative-to-EU thresholds.\n"
            "- Category mapping is not available for all region-year rows due NUTS/code coverage changes."
        ),
    ]

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }

    nbformat.write(nb, NOTEBOOK_PATH)

    with NOTEBOOK_PATH.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    NotebookClient(notebook, timeout=1200, kernel_name="python3").execute(cwd=str(NOTEBOOK_PATH.parent))

    with NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)


def run_models_pipeline() -> Dict[str, Path]:
    _ensure_output_dirs()

    panel, sigma = load_analysis_inputs()
    running, exposure = load_identification_inputs()
    controls = get_available_controls(panel)
    panel = prepare_analysis_panel(panel)

    if "r_value" not in panel.columns or panel["r_value"].isna().all():
        panel = panel.merge(
            running[["nuts2_id", "r_value", "eligible_lt75", "ref_years_used"]],
            on="nuts2_id",
            how="left",
            validate="many_to_one",
        )
    if "erdf_eur_pc_cum_2014_2020" not in panel.columns or panel["erdf_eur_pc_cum_2014_2020"].isna().all():
        panel = panel.merge(
            exposure,
            on=["nuts2_id", "country"],
            how="left",
            validate="many_to_one",
        )

    _validate_columns(
        panel,
        ["r_value", "eligible_lt75", "erdf_eur_pc_cum_2014_2020", "erdf_eur_pc_cum_2015_2020"],
        config.PANEL_MASTER_PARQUET,
        "scripts/build_dataset.py (src/pipeline.py)",
    )

    if panel["category_2014_2020"].notna().sum() == 0:
        raise ValueError(
            "category_2014_2020 is all missing in panel_master.parquet. "
            "This should come from eligibility_categories_2014_2020.csv in scripts/build_dataset.py."
        )

    save_panel_schema_and_overview(panel, controls_used=controls)

    headline_outcome = _choose_headline_outcome(panel)
    outcomes = _available_growth_outcomes(panel, headline_outcome)
    pps_outcome = (
        "gdp_pc_pps_growth"
        if "gdp_pc_pps_growth" in outcomes and headline_outcome != "gdp_pc_pps_growth"
        else None
    )
    LOGGER.info("Headline outcome: %s", headline_outcome)
    LOGGER.info("Outcomes estimated: %s", ", ".join(outcomes))

    # TWFE/Event-study benchmark layer.
    twfe_main, dl_lags, core_runs = run_core_models_by_outcome(panel, outcomes, controls)
    _write_csv(twfe_main, TABLE_PATHS_V31["twfe_main"])
    _write_csv(dl_lags, TABLE_PATHS_V31["dl_lags"])

    dynamic_table, placebo_table, dynamic_runs = run_dynamic_and_placebo(
        panel=panel,
        headline_outcome=headline_outcome,
        controls=controls,
    )
    _write_csv(dynamic_table, TABLE_PATHS_V31["dynamic_lag"])
    _write_csv(placebo_table, TABLE_PATHS_V31["leads_lags"])

    viz.plot_dynamic_lag_response(
        dynamic_table,
        FIGURE_PATHS_V31["dynamic_lag"],
        title=f"Dynamic lag response ({headline_outcome})",
    )
    viz.plot_leads_lags_placebo(
        placebo_table,
        FIGURE_PATHS_V31["leads_lags"],
        title=f"Placebo leads/lags ({headline_outcome})",
    )
    viz.plot_sigma_convergence_multi(sigma, FIGURE_PATHS_V31["sigma"])

    beta_table, heterogeneity_table, beta_partial, beta_runs = run_beta_and_heterogeneity(
        panel=panel,
        controls=controls,
        headline_outcome=headline_outcome,
    )
    _write_csv(beta_table, TABLE_PATHS_V31["beta"])
    _write_csv(heterogeneity_table, TABLE_PATHS_V31["heterogeneity"])
    if beta_partial is None:
        beta_partial = pd.DataFrame(columns=["percentile", "x_value", "marginal_effect", "ci_lower", "ci_upper"])
    viz.plot_beta_marginal_effect(
        beta_partial,
        FIGURE_PATHS_V31["beta_partial"],
        title=f"Marginal ERDF effect by initial income ({headline_outcome})",
    )

    (
        robust_outliers,
        robust_balanced,
        robust_scaling,
        outlier_threshold,
        balanced_window,
        balanced_regions,
        robustness_runs,
    ) = run_robustness(panel, controls, headline_outcome)
    _write_csv(robust_outliers, TABLE_PATHS_V31["robust_outliers"])
    _write_csv(robust_balanced, TABLE_PATHS_V31["robust_balanced"])
    _write_csv(robust_scaling, TABLE_PATHS_V31["robust_scaling"])

    # Region-level frame for RD/IV diagnostics and cross-sectional IV.
    region_outcomes = [headline_outcome]
    if pps_outcome is not None:
        region_outcomes.append(pps_outcome)
    region_outcomes.extend([col for col in ["log_gdp_pc_real", "log_gdp_pc_pps"] if col in panel.columns])
    region_ident_df = build_region_level_identification_frame(
        panel=panel,
        running=running,
        exposure=exposure,
        controls=controls,
        outcomes=region_outcomes,
    )

    rd_bandwidths = [5.0, 7.5, 10.0, 12.5, 15.0, 20.0]

    # RD first-stage funding discontinuity diagnostics.
    rd_funding_jump = run_rd_funding_jump_v31(region_ident_df, bandwidths=rd_bandwidths)
    _write_csv(rd_funding_jump, TABLE_PATHS_V31["rd_first_stage_funding_jump"])
    viz.plot_rd_binned_scatter(
        region_ident_df,
        outcome_col="erdf_eur_pc_cum_2014_2020",
        output_path=FIGURE_PATHS_V31["rd_first_stage_funding_jump"],
        cutoff=75.0,
        bandwidth=20.0,
        title="RD first stage: cumulative ERDF per capita at 75 cutoff",
    )

    # Sharp RD outcome effects (eligibility ITT) + placebo.
    rd_outcome_sharp, rd_bandwidth, rd_placebo = run_sharp_rd_outcomes_v31(
        region_df=region_ident_df,
        headline_outcome=headline_outcome,
        pps_outcome=pps_outcome,
        bandwidths=rd_bandwidths,
    )
    _write_csv(rd_outcome_sharp, TABLE_PATHS_V31["rd_outcome_sharp"])
    _write_csv(rd_bandwidth, TABLE_PATHS_V31["rd_bandwidth_sensitivity"])
    _write_csv(rd_placebo, TABLE_PATHS_V31["rd_placebo_pretrend"])

    rd_outcome_plot_col = f"{headline_outcome}_post_2016_2020"
    if rd_outcome_plot_col in region_ident_df.columns:
        viz.plot_rd_binned_scatter(
            region_ident_df,
            outcome_col=rd_outcome_plot_col,
            output_path=FIGURE_PATHS_V31["rd_outcome_binned_scatter"],
            cutoff=75.0,
            bandwidth=20.0,
            title=f"Sharp RD eligibility effect ({rd_outcome_plot_col})",
        )
    else:
        viz.plot_rd_binned_scatter(
            region_ident_df,
            outcome_col="erdf_eur_pc_cum_2014_2020",
            output_path=FIGURE_PATHS_V31["rd_outcome_binned_scatter"],
            cutoff=75.0,
            bandwidth=20.0,
            title="Sharp RD outcome plot fallback",
        )
    viz.plot_rd_bandwidth_sensitivity(
        rd_bandwidth,
        output_path=FIGURE_PATHS_V31["rd_bandwidth_sensitivity"],
        title="Sharp RD outcome bandwidth sensitivity",
    )

    # IV first-stage candidate diagnostics.
    iv_candidates = build_iv_first_stage_candidates_v31(
        panel=panel,
        region_df=region_ident_df,
        controls=controls,
    )
    if "iv_first_stage_f_headline" not in iv_candidates.columns:
        iv_candidates["iv_first_stage_f_headline"] = np.nan

    panel_iv_data = panel.copy()
    panel_iv_data["eligible_lt75"] = pd.to_numeric(panel_iv_data["eligible_lt75"], errors="coerce")
    panel_iv_data["eligible_lt90"] = np.where(pd.to_numeric(panel_iv_data["r_value"], errors="coerce") < 90.0, 1.0, 0.0)
    panel_iv_data["post2014"] = (panel_iv_data["year"] >= 2014).astype(float)
    panel_iv_data["eu_mean_erdf_t"] = panel_iv_data.groupby("year", sort=False)["erdf_eur_pc"].transform("mean")
    panel_iv_data["country_mean_erdf_t_loo"] = _build_leave_one_out_country_mean(panel_iv_data, "erdf_eur_pc")
    panel_iv_data["z1_post75"] = panel_iv_data["eligible_lt75"] * panel_iv_data["post2014"]
    panel_iv_data["z2_eu75"] = panel_iv_data["eligible_lt75"] * panel_iv_data["eu_mean_erdf_t"]
    panel_iv_data["z3_country75"] = panel_iv_data["eligible_lt75"] * panel_iv_data["country_mean_erdf_t_loo"]
    panel_iv_data["z5_eu90"] = panel_iv_data["eligible_lt90"] * panel_iv_data["eu_mean_erdf_t"]

    # Panel IV result (best panel candidate if available).
    iv_panel_results = pd.DataFrame(
        columns=[
            "outcome",
            "model",
            "sample_type",
            "term",
            "instrument",
            "coef",
            "std_err",
            "t_stat",
            "p_value",
            "ci_95_lower",
            "ci_95_upper",
            "first_stage_f_stat",
            "n_obs",
            "n_regions",
            "sample_year_min",
            "sample_year_max",
            "controls_used",
            "fe_spec",
        ]
    )
    panel_ok = iv_candidates[
        (iv_candidates["sample_type"] == "panel") & (iv_candidates["status"] == "ok")
    ].copy()
    if not panel_ok.empty:
        panel_ok["f_stat"] = pd.to_numeric(panel_ok["f_stat"], errors="coerce")
        best_panel = panel_ok.sort_values("f_stat", ascending=False).iloc[0]
        try:
            iv_panel_results, _ = run_panel_iv_with_instrument_v31(
                panel=panel_iv_data,
                outcome=headline_outcome,
                instrument_var=str(best_panel["instrument_var"]),
                controls=controls,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Panel IV (selected candidate) failed: %s", exc)
    _write_csv(iv_panel_results, TABLE_PATHS_V31["iv_panel_results"])

    # Cross-sectional IV result (best cross-section candidate, often stronger).
    iv_cross_results = pd.DataFrame(
        columns=[
            "outcome",
            "window",
            "model",
            "sample_type",
            "term",
            "instrument",
            "coef",
            "std_err",
            "t_stat",
            "p_value",
            "ci_95_lower",
            "ci_95_upper",
            "first_stage_f_stat",
            "n_obs",
            "n_regions",
            "controls_used",
            "fe_spec",
        ]
    )
    first_stage_scatter = pd.DataFrame(columns=["actual_exposure", "fitted_exposure"])
    selected_cross_candidate: Optional[Dict[str, object]] = None
    selected_cross_first_stage_f = np.nan
    cross_ok = iv_candidates[
        (iv_candidates["sample_type"] == "cross_section") & (iv_candidates["status"] == "ok")
    ].copy()
    if not cross_ok.empty:
        candidate_runs: List[Tuple[float, Dict[str, object], pd.DataFrame, pd.DataFrame]] = []
        for idx, candidate in cross_ok.iterrows():
            controls_text = str(candidate.get("controls_used", ""))
            cross_controls = [token for token in controls_text.split(",") if token]
            try:
                candidate_results, _, candidate_scatter = run_cross_section_iv_v31(
                    region_df=region_ident_df,
                    instrument_var=str(candidate["instrument_var"]),
                    endog_var=str(candidate["endogenous_var"]),
                    controls=cross_controls,
                    headline_outcome=headline_outcome,
                    pps_outcome=pps_outcome,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Cross-sectional IV candidate failed (%s): %s", candidate["candidate_id"], exc)
                continue

            if candidate_results.empty:
                continue

            headline_row = candidate_results[
                (candidate_results["outcome"] == headline_outcome)
                & (candidate_results["window"] == "post_2016_2020")
            ]
            if headline_row.empty:
                headline_row = candidate_results.head(1)
            first_f = float(pd.to_numeric(headline_row.iloc[0]["first_stage_f_stat"], errors="coerce"))
            iv_candidates.loc[idx, "iv_first_stage_f_headline"] = first_f
            candidate_runs.append((first_f, candidate.to_dict(), candidate_results, candidate_scatter))

        if candidate_runs:
            candidate_runs.sort(key=lambda item: item[0], reverse=True)
            selected_cross_first_stage_f, selected_cross_candidate, iv_cross_results, first_stage_scatter = candidate_runs[0]
            LOGGER.info(
                "Selected cross-section IV candidate %s with first-stage F=%.2f",
                selected_cross_candidate.get("candidate_id"),
                selected_cross_first_stage_f,
            )
    _write_csv(iv_candidates, TABLE_PATHS_V31["iv_first_stage_candidates"])
    _write_csv(iv_cross_results, TABLE_PATHS_V31["iv_cross_section_results"])
    _plot_first_stage_scatter_png(first_stage_scatter, FIGURE_PATHS_V31["iv_first_stage_scatter"])

    # Decision: choose headline estimator transparently from diagnostics.
    panel_first_stage_f = np.nan
    if not iv_panel_results.empty:
        panel_first_stage_f = float(pd.to_numeric(iv_panel_results.iloc[0]["first_stage_f_stat"], errors="coerce"))

    if selected_cross_candidate is not None and selected_cross_first_stage_f >= 10.0:
        preferred_estimator_label = (
            f"cross_section IV ({selected_cross_candidate['candidate_id']})"
        )
        preferred_estimator_note = (
            "Headline estimator selected from candidate instruments using IV first-stage strength "
            f"(headline first-stage F={selected_cross_first_stage_f:.2f})."
        )
    elif pd.notna(panel_first_stage_f) and panel_first_stage_f >= 10.0:
        preferred_estimator_label = "panel IV (selected candidate)"
        preferred_estimator_note = (
            f"Cross-section IV candidates are weaker; panel IV retained with first-stage F={panel_first_stage_f:.2f}."
        )
    else:
        preferred_estimator_label = "Strict TWFE benchmark (fallback)"
        preferred_estimator_note = (
            "RD/IV first-stage diagnostics do not meet conventional strength thresholds; "
            "causal interpretation is limited and benchmark TWFE/event-study is emphasized."
        )

    # Unified comparison table (TWFE + sharp RD ITT + IV where available).
    comparison_rows: List[Dict[str, object]] = []
    twfe_subset = twfe_main[
        (twfe_main["outcome"] == headline_outcome)
        & (twfe_main["term"] == "erdf_eur_pc_l1")
        & (twfe_main["model"].isin(["Model A", "Model B"]))
    ]
    for _, row in twfe_subset.iterrows():
        comparison_rows.append(
            {
                "estimator_family": "TWFE",
                "model": row["model"],
                "outcome": headline_outcome,
                "window": "panel",
                "term": row["term"],
                "coef": row["coef"],
                "std_err": row["std_err"],
                "p_value": row["p_value"],
                "ci_95_lower": row["ci_95_lower"],
                "ci_95_upper": row["ci_95_upper"],
                "n_obs": row["n_obs"],
                "n_regions": row["n_regions"],
                "first_stage_f_stat": np.nan,
                "is_headline_estimator": preferred_estimator_label.startswith("Strict TWFE"),
            }
        )
    dl_subset = dl_lags[
        (dl_lags["outcome"] == headline_outcome)
        & (dl_lags["model"] == "Model C")
        & (dl_lags["term"] == "erdf_eur_pc_l1")
    ]
    for _, row in dl_subset.iterrows():
        comparison_rows.append(
            {
                "estimator_family": "TWFE-dynamic",
                "model": "Model C (l1 term)",
                "outcome": headline_outcome,
                "window": "panel",
                "term": row["term"],
                "coef": row["coef"],
                "std_err": row["std_err"],
                "p_value": row["p_value"],
                "ci_95_lower": row["ci_95_lower"],
                "ci_95_upper": row["ci_95_upper"],
                "n_obs": row["n_obs"],
                "n_regions": row["n_regions"],
                "first_stage_f_stat": np.nan,
                "is_headline_estimator": False,
            }
        )
    rd_subset = rd_outcome_sharp[
        (rd_outcome_sharp["outcome"] == headline_outcome)
        & (rd_outcome_sharp["window"] == "post_2016_2020")
    ]
    for _, row in rd_subset.iterrows():
        comparison_rows.append(
            {
                "estimator_family": "RD (sharp ITT)",
                "model": f"Sharp RD bw={row['bandwidth']}",
                "outcome": headline_outcome,
                "window": row["window"],
                "term": "eligible_lt75",
                "coef": row["coef"],
                "std_err": row["std_err"],
                "p_value": row["p_value"],
                "ci_95_lower": row["ci_95_lower"],
                "ci_95_upper": row["ci_95_upper"],
                "n_obs": row["n_obs"],
                "n_regions": row["n_left"] + row["n_right"],
                "first_stage_f_stat": np.nan,
                "is_headline_estimator": preferred_estimator_label.startswith("RD"),
            }
        )
    if not iv_cross_results.empty:
        iv_head = iv_cross_results[
            (iv_cross_results["outcome"] == headline_outcome)
            & (iv_cross_results["window"] == "post_2016_2020")
        ]
        if iv_head.empty:
            iv_head = iv_cross_results.head(1)
        for _, row in iv_head.iterrows():
            comparison_rows.append(
                {
                    "estimator_family": "IV (cross-section)",
                    "model": row["model"],
                    "outcome": row["outcome"],
                    "window": row["window"],
                    "term": row["term"],
                    "coef": row["coef"],
                    "std_err": row["std_err"],
                    "p_value": row["p_value"],
                    "ci_95_lower": row["ci_95_lower"],
                    "ci_95_upper": row["ci_95_upper"],
                    "n_obs": row["n_obs"],
                    "n_regions": row["n_regions"],
                    "first_stage_f_stat": row["first_stage_f_stat"],
                    "is_headline_estimator": preferred_estimator_label.startswith("cross_section IV"),
                }
            )
    if not iv_panel_results.empty:
        row = iv_panel_results.iloc[0]
        comparison_rows.append(
            {
                "estimator_family": "IV (panel)",
                "model": row["model"],
                "outcome": row["outcome"],
                "window": "panel",
                "term": row["term"],
                "coef": row["coef"],
                "std_err": row["std_err"],
                "p_value": row["p_value"],
                "ci_95_lower": row["ci_95_lower"],
                "ci_95_upper": row["ci_95_upper"],
                "n_obs": row["n_obs"],
                "n_regions": row["n_regions"],
                "first_stage_f_stat": row["first_stage_f_stat"],
                "is_headline_estimator": preferred_estimator_label.startswith("panel IV"),
            }
        )

    model_comparison = pd.DataFrame(comparison_rows).sort_values(
        ["is_headline_estimator", "estimator_family", "model"],
        ascending=[False, True, True],
    )
    _write_csv(model_comparison, TABLE_PATHS_V31["model_summary"])

    # Base summaries used by report.
    overview = pd.read_csv(BASE_TABLE_PATHS["panel_overview"])
    try:
        missingness = pd.read_csv(config.QA_TABLES["missingness"])
    except Exception:  # noqa: BLE001
        variables = [col for col in panel.columns if col not in ["nuts2_id", "country", "year"]]
        frames: List[pd.DataFrame] = []
        for variable in variables:
            grouped = panel.groupby("year", sort=True)[variable].agg(
                n_obs="size",
                n_missing=lambda series: int(series.isna().sum()),
            )
            grouped = grouped.reset_index()
            grouped["variable"] = variable
            grouped["missing_share"] = grouped["n_missing"] / grouped["n_obs"]
            frames.append(grouped[["year", "variable", "n_obs", "n_missing", "missing_share"]])
        missingness = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["year", "variable", "n_obs", "n_missing", "missing_share"]
        )

    robustness_summary = pd.concat(
        [
            robust_outliers[robust_outliers["term"] == "erdf_eur_pc_l1"],
            robust_balanced[robust_balanced["term"] == "erdf_eur_pc_l1"],
            robust_scaling[robust_scaling["term"] == "erdf_k_eur_pc_l1"],
        ],
        ignore_index=True,
    )

    create_html_report_v31(
        headline_outcome=headline_outcome,
        preferred_estimator_label=preferred_estimator_label,
        preferred_estimator_note=preferred_estimator_note,
        overview=overview,
        panel=panel,
        missingness=missingness,
        twfe_main=twfe_main,
        dl_lags=dl_lags,
        dynamic_lag=dynamic_table,
        rd_funding_jump=rd_funding_jump,
        rd_outcome_sharp=rd_outcome_sharp,
        rd_bandwidth=rd_bandwidth,
        rd_placebo=rd_placebo,
        iv_candidates=iv_candidates,
        iv_panel=iv_panel_results,
        iv_cross=iv_cross_results,
        robustness_summary=robustness_summary,
        sigma=sigma,
        beta=beta_table,
        model_comparison=model_comparison,
        first_stage_scatter=first_stage_scatter,
    )

    create_report_notebook(headline_outcome)

    LOGGER.info("V3.1 analysis completed. Diagnostics, selected identification, and tabbed report written under outputs/.")

    outputs: Dict[str, Path] = {}
    outputs.update({f"table_base_{name}": path for name, path in BASE_TABLE_PATHS.items()})
    outputs.update({f"table_v31_{name}": path for name, path in TABLE_PATHS_V31.items()})
    outputs.update({f"figure_v31_{name}": path for name, path in FIGURE_PATHS_V31.items()})
    outputs["report_html"] = REPORT_HTML_PATH
    outputs["report_manifest"] = REPORT_MANIFEST_PATH
    outputs["report_notebook"] = NOTEBOOK_PATH
    return outputs
