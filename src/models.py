from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nbformat
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from nbclient import NotebookClient

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


def create_report_notebook(headline_outcome: str) -> None:
    nb = nbformat.v4.new_notebook()

    cells = [
        nbformat.v4.new_markdown_cell(
            "# EU Cohesion V2 Report\n"
            f"Headline outcome: **{headline_outcome}**"
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n"
            "root = Path('..').resolve()\n"
            "overview = pd.read_csv(root / 'outputs/tables/panel_master_overview.csv')\n"
            "missingness = pd.read_csv(root / 'outputs/tables/panel_master_key_missingness.csv')\n"
            "twfe = pd.read_csv(root / 'outputs/tables/twfe_main_results_v2.csv')\n"
            "dl = pd.read_csv(root / 'outputs/tables/dl_lags_results_v2.csv')\n"
            "summary = pd.read_csv(root / 'outputs/tables/model_comparison_summary_v2.csv')\n"
            "beta = pd.read_csv(root / 'outputs/tables/beta_convergence_results_v2.csv')\n"
            "display(overview)\n"
            "display(missingness.sort_values('missing_rate', ascending=False))"
        ),
        nbformat.v4.new_markdown_cell("## Core Models"),
        nbformat.v4.new_code_cell("display(twfe)\ndisplay(dl)\ndisplay(summary)"),
        nbformat.v4.new_markdown_cell("## Beta + Heterogeneity"),
        nbformat.v4.new_code_cell(
            "hetero = pd.read_csv(root / 'outputs/tables/heterogeneity_by_category_v2.csv')\n"
            "display(beta)\n"
            "display(hetero)"
        ),
        nbformat.v4.new_markdown_cell("## Figures"),
        nbformat.v4.new_code_cell(
            "display(Image(filename=str(root / 'outputs/figures/sigma_convergence_v2.png')))\n"
            "display(Image(filename=str(root / 'outputs/figures/dynamic_lag_response_v2.png')))\n"
            "display(Image(filename=str(root / 'outputs/figures/leads_lags_placebo_v2.png')))\n"
            "display(Image(filename=str(root / 'outputs/figures/beta_convergence_partial_v2.png')))"
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
    controls = get_available_controls(panel)
    panel = prepare_analysis_panel(panel)

    if panel["category_2014_2020"].notna().sum() == 0:
        raise ValueError(
            "category_2014_2020 is all missing in panel_master.parquet. "
            "This should come from eligibility_categories_2014_2020.csv in the V2 build pipeline."
        )

    save_panel_schema_and_overview(panel, controls_used=controls)

    headline_outcome = _choose_headline_outcome(panel)
    outcomes = _available_growth_outcomes(panel, headline_outcome)

    LOGGER.info("Headline outcome: %s", headline_outcome)
    LOGGER.info("Outcomes estimated: %s", ", ".join(outcomes))

    twfe_main, dl_lags, core_runs = run_core_models_by_outcome(panel, outcomes, controls)
    _write_csv(twfe_main, TABLE_PATHS_V2["twfe_main"])
    _write_csv(dl_lags, TABLE_PATHS_V2["dl_lags"])

    dynamic_table, placebo_table, dynamic_runs = run_dynamic_and_placebo(
        panel=panel,
        headline_outcome=headline_outcome,
        controls=controls,
    )
    _write_csv(dynamic_table, TABLE_PATHS_V2["dynamic_lag"])
    _write_csv(placebo_table, TABLE_PATHS_V2["leads_lags"])

    viz.plot_dynamic_lag_response(
        dynamic_table,
        FIGURE_PATHS_V2["dynamic_lag"],
        title=f"Dynamic lag response ({headline_outcome})",
    )
    viz.plot_leads_lags_placebo(
        placebo_table,
        FIGURE_PATHS_V2["leads_lags"],
        title=f"Placebo leads/lags ({headline_outcome})",
    )
    viz.plot_sigma_convergence_multi(sigma, FIGURE_PATHS_V2["sigma"])

    beta_table, heterogeneity_table, beta_partial, beta_runs = run_beta_and_heterogeneity(
        panel=panel,
        controls=controls,
        headline_outcome=headline_outcome,
    )
    _write_csv(beta_table, TABLE_PATHS_V2["beta"])
    _write_csv(heterogeneity_table, TABLE_PATHS_V2["heterogeneity"])

    if beta_partial is None:
        beta_partial = pd.DataFrame(columns=["percentile", "x_value", "marginal_effect", "ci_lower", "ci_upper"])
    viz.plot_beta_marginal_effect(
        beta_partial,
        FIGURE_PATHS_V2["beta_partial"],
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

    _write_csv(robust_outliers, TABLE_PATHS_V2["robust_outliers"])
    _write_csv(robust_balanced, TABLE_PATHS_V2["robust_balanced"])
    _write_csv(robust_scaling, TABLE_PATHS_V2["robust_scaling"])

    all_runs = [*core_runs, *dynamic_runs, *beta_runs, *robustness_runs]

    key_term_lookup = {
        "Model A": "erdf_eur_pc_l1",
        "Model B": "erdf_eur_pc_l1",
        "Model A (two-way cluster)": "erdf_eur_pc_l1",
        "Model B (two-way cluster)": "erdf_eur_pc_l1",
        "Model C": "erdf_eur_pc_l1",
        "Model D": "log_gdp_pc_real_l1",
        "Model E": "erdf_eur_pc_l1",
        "Model E (country-year FE)": "erdf_eur_pc_l1",
        "Model A (outliers excluded)": "erdf_eur_pc_l1",
        "Model B (outliers excluded)": "erdf_eur_pc_l1",
        "Model A (balanced panel)": "erdf_eur_pc_l1",
        "Model B (balanced panel)": "erdf_eur_pc_l1",
        "Model A (scaled treatment)": "erdf_k_eur_pc_l1",
        "Model B (scaled treatment)": "erdf_k_eur_pc_l1",
    }

    model_summary = _build_model_summary(
        runs=all_runs,
        key_term_lookup=key_term_lookup,
        outlier_threshold=outlier_threshold,
        balanced_window=balanced_window,
        balanced_regions=balanced_regions,
    )
    _write_csv(model_summary, TABLE_PATHS_V2["model_summary"])

    overview = pd.read_csv(BASE_TABLE_PATHS["panel_overview"])
    missingness = pd.read_csv(BASE_TABLE_PATHS["panel_key_missingness"])

    headline_core = pd.concat(
        [
            twfe_main[(twfe_main["outcome"] == headline_outcome) & (twfe_main["model"].isin(["Model A", "Model B"]))],
            dl_lags[(dl_lags["outcome"] == headline_outcome) & (dl_lags["model"] == "Model C")],
        ],
        ignore_index=True,
    )

    robustness_summary = model_summary[
        model_summary["model"].isin(
            [
                "Model A (outliers excluded)",
                "Model B (outliers excluded)",
                "Model A (balanced panel)",
                "Model B (balanced panel)",
                "Model A (scaled treatment)",
                "Model B (scaled treatment)",
            ]
        )
    ].copy()

    beta_summary = beta_table[
        beta_table["model"].isin(["Model D", "Model E", "Model E (country-year FE)"])
    ].copy()

    create_html_report(
        headline_outcome=headline_outcome,
        overview=overview,
        missingness=missingness,
        headline_models=headline_core,
        robustness_summary=robustness_summary,
        beta_summary=beta_summary,
    )

    create_report_notebook(headline_outcome)

    LOGGER.info("V2 analysis completed. Tables and figures written under outputs/.")

    outputs: Dict[str, Path] = {}
    outputs.update({f"table_base_{name}": path for name, path in BASE_TABLE_PATHS.items()})
    outputs.update({f"table_v2_{name}": path for name, path in TABLE_PATHS_V2.items()})
    outputs.update({f"figure_v2_{name}": path for name, path in FIGURE_PATHS_V2.items()})
    outputs["report_html"] = REPORT_HTML_PATH
    outputs["report_manifest"] = REPORT_MANIFEST_PATH
    outputs["report_notebook"] = NOTEBOOK_PATH
    return outputs
