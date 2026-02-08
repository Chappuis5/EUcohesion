from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

PANEL_REQUIRED_COLUMNS = [
    "nuts2_id",
    "country",
    "year",
    "gdp_pc_growth",
    "gdp_pc",
    "log_gdp_pc",
    "erdf_eur_pc",
    "erdf_eur_pc_l1",
    "erdf_eur_pc_l2",
    "erdf_eur_pc_l3",
]

SIGMA_REQUIRED_COLUMNS = ["year", "sigma_log_gdp"]

OUTPUT_TABLES_DIR = config.OUTPUTS_TABLES_DIR
OUTPUT_FIGURES_DIR = config.OUTPUTS_FIGURES_DIR
NOTEBOOK_PATH = config.PROJECT_ROOT / "notebooks" / "01_report.ipynb"

TABLE_PATHS = {
    "panel_schema": OUTPUT_TABLES_DIR / "panel_master_schema.csv",
    "panel_overview": OUTPUT_TABLES_DIR / "panel_master_overview.csv",
    "panel_key_missingness": OUTPUT_TABLES_DIR / "panel_master_key_missingness.csv",
    "twfe_main": OUTPUT_TABLES_DIR / "twfe_main_results.csv",
    "dl_lags": OUTPUT_TABLES_DIR / "dl_lags_results.csv",
    "dynamic_lag": OUTPUT_TABLES_DIR / "dynamic_lag_response.csv",
    "leads_lags": OUTPUT_TABLES_DIR / "leads_lags_results.csv",
    "beta": OUTPUT_TABLES_DIR / "beta_convergence_results.csv",
    "robust_outliers": OUTPUT_TABLES_DIR / "robustness_outliers.csv",
    "robust_balanced": OUTPUT_TABLES_DIR / "robustness_balanced_panel.csv",
    "robust_scaling": OUTPUT_TABLES_DIR / "robustness_scaling.csv",
    "model_summary": OUTPUT_TABLES_DIR / "model_comparison_summary.csv",
}

FIGURE_PATHS = {
    "dynamic_lag": OUTPUT_FIGURES_DIR / "dynamic_lag_response.png",
    "leads_lags": OUTPUT_FIGURES_DIR / "leads_lags_placebo.png",
    "sigma": OUTPUT_FIGURES_DIR / "sigma_convergence.png",
    "beta_partial": OUTPUT_FIGURES_DIR / "beta_convergence_partial.png",
}


@dataclass
class RegressionRun:
    model_name: str
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
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.10g")


def _validate_columns(df: pd.DataFrame, required: Sequence[str], dataset_path: Path, source_hint: str) -> None:
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

    _validate_columns(
        panel,
        PANEL_REQUIRED_COLUMNS,
        panel_path,
        "data processing pipeline output in scripts/build_dataset.py (src/pipeline.py)",
    )
    _validate_columns(
        sigma,
        SIGMA_REQUIRED_COLUMNS,
        sigma_path,
        "data processing pipeline output in scripts/build_dataset.py (src/pipeline.py)",
    )

    panel = panel.copy()
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel = panel[panel["year"].notna()].copy()
    panel["year"] = panel["year"].astype(int)
    panel = panel.sort_values(["nuts2_id", "year"]).reset_index(drop=True)

    sigma = sigma.copy()
    sigma["year"] = pd.to_numeric(sigma["year"], errors="coerce").astype("Int64")
    sigma["sigma_log_gdp"] = pd.to_numeric(sigma["sigma_log_gdp"], errors="coerce")
    sigma = sigma[sigma["year"].notna()].copy()
    sigma["year"] = sigma["year"].astype(int)
    sigma = sigma.sort_values("year").reset_index(drop=True)

    return panel, sigma


def get_available_controls(panel: pd.DataFrame) -> List[str]:
    available = [column for column in REQUESTED_CONTROLS if column in panel.columns]
    missing = sorted(set(REQUESTED_CONTROLS) - set(available))
    if missing:
        LOGGER.warning("Controls not found and omitted from regressions: %s", ", ".join(missing))
    return available


def prepare_analysis_panel(panel: pd.DataFrame) -> pd.DataFrame:
    prepared = panel.copy().sort_values(["nuts2_id", "year"]).reset_index(drop=True)
    prepared["country_year"] = prepared["country"].astype(str) + "_" + prepared["year"].astype(str)

    numeric_candidates = [
        "gdp_pc_growth",
        "gdp_pc",
        "log_gdp_pc",
        "erdf_eur_pc",
        "erdf_eur_pc_l1",
        "erdf_eur_pc_l2",
        "erdf_eur_pc_l3",
        *REQUESTED_CONTROLS,
    ]
    for column in numeric_candidates:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared["log_gdp_pc_l1"] = prepared.groupby("nuts2_id", sort=False)["log_gdp_pc"].shift(1)

    if "erdf_eur_pc" in prepared.columns:
        prepared["erdf_eur_pc_f1"] = prepared.groupby("nuts2_id", sort=False)["erdf_eur_pc"].shift(-1)
        prepared["erdf_eur_pc_f2"] = prepared.groupby("nuts2_id", sort=False)["erdf_eur_pc"].shift(-2)

    prepared["erdf_k_eur_pc_l1"] = prepared["erdf_eur_pc_l1"] / 1000.0

    return prepared


def save_panel_schema_and_overview(panel: pd.DataFrame, controls_used: Sequence[str]) -> None:
    schema = pd.DataFrame(
        {
            "column": panel.columns,
            "dtype": [str(dtype) for dtype in panel.dtypes],
            "missing_pct": [float(panel[column].isna().mean()) for column in panel.columns],
        }
    ).sort_values("column")
    _write_csv(schema, TABLE_PATHS["panel_schema"])

    key_variables = [
        "erdf_eur_pc",
        "erdf_eur_pc_l1",
        "erdf_eur_pc_l2",
        "erdf_eur_pc_l3",
        "gdp_pc",
        "gdp_pc_growth",
        *list(controls_used),
    ]
    key_variables = [variable for variable in key_variables if variable in panel.columns]

    key_missingness = pd.DataFrame(
        {
            "variable": key_variables,
            "missing_rate": [float(panel[column].isna().mean()) for column in key_variables],
        }
    ).sort_values("variable")
    _write_csv(key_missingness, TABLE_PATHS["panel_key_missingness"])

    min_year = int(panel["year"].min())
    max_year = int(panel["year"].max())
    n_regions = int(panel["nuts2_id"].nunique())

    overview = pd.DataFrame(
        [
            {"metric": "min_year", "value": min_year},
            {"metric": "max_year", "value": max_year},
            {"metric": "n_regions", "value": n_regions},
            {"metric": "n_rows", "value": int(panel.shape[0])},
        ]
    )
    _write_csv(overview, TABLE_PATHS["panel_overview"])

    LOGGER.info("Panel coverage: %s to %s", min_year, max_year)
    LOGGER.info("Number of regions: %s", n_regions)
    LOGGER.info("Key-variable missingness saved to %s", TABLE_PATHS["panel_key_missingness"])


def _build_formula(
    outcome: str,
    regressors: Sequence[str],
    controls: Sequence[str],
    fe_type: str,
) -> str:
    terms = list(regressors) + list(controls)
    terms.append("C(nuts2_id)")

    if fe_type == "year":
        terms.append("C(year)")
    elif fe_type == "country_year":
        terms.append("C(country_year)")
    else:
        raise ValueError(f"Unsupported fe_type: {fe_type}")

    rhs = " + ".join(terms)
    return f"{outcome} ~ {rhs}"


def _fit_ols_with_clustering(
    model_name: str,
    formula: str,
    sample_df: pd.DataFrame,
    fe_type: str,
    clustering: str,
    controls_used: Sequence[str],
) -> RegressionRun:
    if sample_df.empty:
        raise ValueError(f"Model {model_name} has empty sample after dropping missing rows.")

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
        formula=formula,
        fe_type=fe_type,
        clustering=clustering,
        result=result,
        sample_df=sample_df,
        controls_used=list(controls_used),
    )


def run_twfe_model(
    panel: pd.DataFrame,
    model_name: str,
    outcome: str,
    regressors: Sequence[str],
    controls: Sequence[str],
    fe_type: str,
    clustering: str,
) -> RegressionRun:
    required_regressor_columns: List[str] = []
    for regressor in regressors:
        tokens = [token for token in regressor.replace("*", ":").split(":") if token]
        if not tokens:
            tokens = [regressor]
        required_regressor_columns.extend(tokens)

    required = ["nuts2_id", "country", "year", outcome, *required_regressor_columns, *controls]
    required = list(dict.fromkeys(required))
    if fe_type == "country_year":
        required.append("country_year")

    missing = sorted(set(required) - set(panel.columns))
    if missing:
        raise ValueError(
            f"Model {model_name} is missing required columns: {', '.join(missing)}. "
            "Check data/processed/panel_master.parquet and src/pipeline.py outputs."
        )

    sample_df = panel[required].dropna().copy()
    formula = _build_formula(outcome=outcome, regressors=regressors, controls=controls, fe_type=fe_type)

    return _fit_ols_with_clustering(
        model_name=model_name,
        formula=formula,
        sample_df=sample_df,
        fe_type=fe_type,
        clustering=clustering,
        controls_used=controls,
    )


def _result_to_long_table(run: RegressionRun, keep_fe_terms: bool = False) -> pd.DataFrame:
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

    table["model"] = run.model_name
    table["n_obs"] = run.n_obs
    table["n_regions"] = run.n_regions
    table["fe_type"] = run.fe_type
    table["clustering"] = run.clustering
    table["sample_year_min"] = run.year_min
    table["sample_year_max"] = run.year_max
    table["controls_used"] = ",".join(run.controls_used)

    ordered = [
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

    return table[ordered].reset_index(drop=True)


def _extract_terms(run: RegressionRun, requested_terms: Dict[str, int]) -> pd.DataFrame:
    base = _result_to_long_table(run, keep_fe_terms=False)
    records = []

    for term, horizon in requested_terms.items():
        term_row = base[base["term"] == term]
        if term_row.empty:
            continue
        row = term_row.iloc[0].to_dict()
        row["horizon"] = horizon
        records.append(row)

    if not records:
        return pd.DataFrame(
            columns=[
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

    extracted = pd.DataFrame(records)
    columns = [
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
    return extracted[columns].sort_values("horizon").reset_index(drop=True)


def _extract_single_coefficient(
    run: RegressionRun,
    term: str,
) -> Optional[Dict[str, object]]:
    table = _result_to_long_table(run, keep_fe_terms=False)
    row = table[table["term"] == term]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def _build_model_comparison_summary(runs: Sequence[RegressionRun]) -> pd.DataFrame:
    key_terms_by_model = {
        "Model A": "erdf_eur_pc_l1",
        "Model B": "erdf_eur_pc_l1",
        "Model A (two-way cluster)": "erdf_eur_pc_l1",
        "Model B (two-way cluster)": "erdf_eur_pc_l1",
        "Model C": "erdf_eur_pc_l1",
        "Model D": "log_gdp_pc_l1",
        "Model E": "erdf_eur_pc_l1",
        "Model E (country-year FE)": "erdf_eur_pc_l1",
        "Model A (outliers excluded)": "erdf_eur_pc_l1",
        "Model B (outliers excluded)": "erdf_eur_pc_l1",
        "Model A (balanced panel)": "erdf_eur_pc_l1",
        "Model B (balanced panel)": "erdf_eur_pc_l1",
        "Model A (scaled treatment)": "erdf_k_eur_pc_l1",
        "Model B (scaled treatment)": "erdf_k_eur_pc_l1",
    }

    rows: List[Dict[str, object]] = []

    for run in runs:
        key_term = key_terms_by_model.get(run.model_name)
        if key_term is None:
            continue
        coefficient = _extract_single_coefficient(run, key_term)
        if coefficient is None:
            continue

        rows.append(
            {
                "model": run.model_name,
                "key_term": key_term,
                "coef": coefficient["coef"],
                "std_err": coefficient["std_err"],
                "p_value": coefficient["p_value"],
                "ci_95_lower": coefficient["ci_95_lower"],
                "ci_95_upper": coefficient["ci_95_upper"],
                "n_obs": run.n_obs,
                "n_regions": run.n_regions,
                "fe_type": run.fe_type,
                "clustering": run.clustering,
                "sample_year_min": run.year_min,
                "sample_year_max": run.year_max,
            }
        )

    summary = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    return summary


def _get_interaction_term_name(run: RegressionRun, left: str, right: str) -> Optional[str]:
    terms = set(run.result.params.index.tolist())
    candidates = [f"{left}:{right}", f"{right}:{left}"]
    for candidate in candidates:
        if candidate in terms:
            return candidate
    return None


def _build_beta_marginal_effect(run: RegressionRun, sample_df: pd.DataFrame) -> pd.DataFrame:
    interaction_term = _get_interaction_term_name(run, "erdf_eur_pc_l1", "log_gdp_pc_l1")
    if interaction_term is None:
        return pd.DataFrame(columns=["percentile", "log_gdp_pc_l1", "marginal_effect", "ci_lower", "ci_upper"])

    params = run.result.params
    cov = run.result.cov_params()

    beta_treat = params.get("erdf_eur_pc_l1")
    beta_inter = params.get(interaction_term)

    if beta_treat is None or beta_inter is None:
        return pd.DataFrame(columns=["percentile", "log_gdp_pc_l1", "marginal_effect", "ci_lower", "ci_upper"])

    var_treat = cov.loc["erdf_eur_pc_l1", "erdf_eur_pc_l1"]
    var_inter = cov.loc[interaction_term, interaction_term]
    covar = cov.loc["erdf_eur_pc_l1", interaction_term]

    percentiles = np.arange(5, 100, 5)
    x_values = np.percentile(sample_df["log_gdp_pc_l1"].dropna(), percentiles)

    records = []
    for pct, x_val in zip(percentiles, x_values):
        marginal = beta_treat + beta_inter * x_val
        variance = var_treat + (x_val**2) * var_inter + 2 * x_val * covar
        std_err = float(np.sqrt(max(variance, 0.0)))
        ci_low = marginal - 1.96 * std_err
        ci_high = marginal + 1.96 * std_err
        records.append(
            {
                "percentile": int(pct),
                "log_gdp_pc_l1": float(x_val),
                "marginal_effect": float(marginal),
                "ci_lower": float(ci_low),
                "ci_upper": float(ci_high),
            }
        )

    return pd.DataFrame(records)


def _run_placebo_leads_lags(
    panel: pd.DataFrame,
    controls: Sequence[str],
) -> Tuple[Optional[RegressionRun], pd.DataFrame]:
    required = ["erdf_eur_pc_f1", "erdf_eur_pc_f2", "erdf_eur_pc", "erdf_eur_pc_l1", "erdf_eur_pc_l2", "erdf_eur_pc_l3"]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        LOGGER.warning("Skipping placebo leads/lags model; missing columns: %s", ", ".join(missing))
        empty = pd.DataFrame(
            columns=[
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
        return None, empty

    run = run_twfe_model(
        panel=panel,
        model_name="Model Placebo Leads-Lags",
        outcome="gdp_pc_growth",
        regressors=["erdf_eur_pc_f2", "erdf_eur_pc_f1", "erdf_eur_pc", "erdf_eur_pc_l1", "erdf_eur_pc_l2", "erdf_eur_pc_l3"],
        controls=controls,
        fe_type="year",
        clustering="nuts2",
    )

    leads_lags_terms = {
        "erdf_eur_pc_f2": -2,
        "erdf_eur_pc_f1": -1,
        "erdf_eur_pc": 0,
        "erdf_eur_pc_l1": 1,
        "erdf_eur_pc_l2": 2,
        "erdf_eur_pc_l3": 3,
    }
    effects = _extract_terms(run, requested_terms=leads_lags_terms)
    return run, effects


def _run_outlier_robustness(panel: pd.DataFrame, controls: Sequence[str]) -> Tuple[pd.DataFrame, float, List[RegressionRun]]:
    p99 = float(panel["erdf_eur_pc"].dropna().quantile(0.99))
    subset = panel[(panel["erdf_eur_pc"].isna()) | (panel["erdf_eur_pc"] <= p99)].copy()

    run_a = run_twfe_model(
        panel=subset,
        model_name="Model A (outliers excluded)",
        outcome="gdp_pc_growth",
        regressors=["erdf_eur_pc_l1"],
        controls=controls,
        fe_type="year",
        clustering="nuts2",
    )
    run_b = run_twfe_model(
        panel=subset,
        model_name="Model B (outliers excluded)",
        outcome="gdp_pc_growth",
        regressors=["erdf_eur_pc_l1"],
        controls=controls,
        fe_type="country_year",
        clustering="nuts2",
    )

    table = pd.concat([_result_to_long_table(run_a), _result_to_long_table(run_b)], ignore_index=True)
    table["outlier_rule"] = "drop global top 1% erdf_eur_pc"
    table["erdf_eur_pc_global_p99"] = p99
    return table, p99, [run_a, run_b]


def _run_balanced_panel_robustness(panel: pd.DataFrame, controls: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, int], List[RegressionRun]]:
    required = ["gdp_pc_growth", "erdf_eur_pc_l1", *controls]

    candidates = panel[required + ["year"]].dropna()
    years = sorted(candidates["year"].unique().tolist())
    if not years:
        raise ValueError("No complete rows available to build balanced panel robustness sample.")

    year_start = max(2017, int(min(years)))
    year_end = min(2023, int(max(years)))
    window_years = list(range(year_start, year_end + 1))

    subset = panel[panel["year"].isin(window_years)].copy()
    n_years = len(window_years)

    region_complete = subset.groupby("nuts2_id").apply(
        lambda g: len(g) == n_years and g[required].notna().all().all()
    )
    balanced_regions = region_complete[region_complete].index.tolist()

    balanced_panel = subset[subset["nuts2_id"].isin(balanced_regions)].copy()

    if balanced_panel.empty:
        raise ValueError(
            "Balanced panel robustness sample is empty for window "
            f"{year_start}-{year_end}."
        )

    run_a = run_twfe_model(
        panel=balanced_panel,
        model_name="Model A (balanced panel)",
        outcome="gdp_pc_growth",
        regressors=["erdf_eur_pc_l1"],
        controls=controls,
        fe_type="year",
        clustering="nuts2",
    )
    run_b = run_twfe_model(
        panel=balanced_panel,
        model_name="Model B (balanced panel)",
        outcome="gdp_pc_growth",
        regressors=["erdf_eur_pc_l1"],
        controls=controls,
        fe_type="country_year",
        clustering="nuts2",
    )

    table = pd.concat([_result_to_long_table(run_a), _result_to_long_table(run_b)], ignore_index=True)
    table["balanced_window_start"] = year_start
    table["balanced_window_end"] = year_end
    table["balanced_regions"] = len(balanced_regions)

    metadata = {
        "balanced_window_start": year_start,
        "balanced_window_end": year_end,
        "balanced_regions": len(balanced_regions),
    }

    return table, metadata, [run_a, run_b]


def _run_scaling_robustness(panel: pd.DataFrame, controls: Sequence[str]) -> Tuple[pd.DataFrame, List[RegressionRun]]:
    run_a = run_twfe_model(
        panel=panel,
        model_name="Model A (scaled treatment)",
        outcome="gdp_pc_growth",
        regressors=["erdf_k_eur_pc_l1"],
        controls=controls,
        fe_type="year",
        clustering="nuts2",
    )
    run_b = run_twfe_model(
        panel=panel,
        model_name="Model B (scaled treatment)",
        outcome="gdp_pc_growth",
        regressors=["erdf_k_eur_pc_l1"],
        controls=controls,
        fe_type="country_year",
        clustering="nuts2",
    )

    table = pd.concat([_result_to_long_table(run_a), _result_to_long_table(run_b)], ignore_index=True)
    table["treatment_scale"] = "EUR 1,000 per capita"
    return table, [run_a, run_b]


def _create_report_notebook() -> None:
    nb = nbformat.v4.new_notebook()

    cells = []
    cells.append(
        nbformat.v4.new_markdown_cell(
            "# EU Cohesion Analysis Report\n"
            "Offline analysis based on processed panel outputs only."
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import display, Markdown, Image\n"
            "\n"
            "root = Path('..').resolve()\n"
            "panel = pd.read_parquet(root / 'data/processed/panel_master.parquet')\n"
            "sigma = pd.read_csv(root / 'data/processed/sigma_convergence.csv')\n"
            "overview = pd.read_csv(root / 'outputs/tables/panel_master_overview.csv')\n"
            "missingness = pd.read_csv(root / 'outputs/tables/panel_master_key_missingness.csv')\n"
            "twfe = pd.read_csv(root / 'outputs/tables/twfe_main_results.csv')\n"
            "lags = pd.read_csv(root / 'outputs/tables/dl_lags_results.csv')\n"
            "beta = pd.read_csv(root / 'outputs/tables/beta_convergence_results.csv')\n"
            "summary = pd.read_csv(root / 'outputs/tables/model_comparison_summary.csv')\n"
            "display(Markdown('## Dataset Coverage'))\n"
            "display(overview)\n"
            "display(Markdown('## Key Missingness Rates'))\n"
            "display(missingness.sort_values('missing_rate', ascending=False))"
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "## Main Causal Results\n"
            "Model A/B estimate GDP per capita growth on lagged ERDF treatment with region FE and either year FE or country-year FE."
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "display(twfe)\n"
            "display(lags)\n"
            "display(summary)"
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "## Convergence Results\n"
            "- Beta convergence: negative `log_gdp_pc_l1` supports convergence.\n"
            "- Interaction term in Model E indicates whether ERDF effects vary by initial income."
        )
    )

    cells.append(nbformat.v4.new_code_cell("display(beta)\n" "display(sigma.head())"))

    cells.append(
        nbformat.v4.new_markdown_cell(
            "## Figures\n"
            "### Dynamic Lag Response"
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "display(Image(filename=str(root / 'outputs/figures/dynamic_lag_response.png')))\n"
            "display(Image(filename=str(root / 'outputs/figures/leads_lags_placebo.png')))\n"
            "display(Image(filename=str(root / 'outputs/figures/sigma_convergence.png')))\n"
            "display(Image(filename=str(root / 'outputs/figures/beta_convergence_partial.png')))"
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "## Interpretation and Limitations\n"
            "- GDP is currently based on nominal `MIO_EUR`; effect sizes should be interpreted cautiously.\n"
            "- `category_2014_2020` is currently missing, so category-based heterogeneity is not estimated.\n"
            "- Results are observational TWFE estimates with region and time fixed effects; remaining confounding may persist."
        )
    )

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

    client = NotebookClient(notebook, timeout=1200, kernel_name="python3")
    client.execute(cwd=str(NOTEBOOK_PATH.parent))

    with NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)


def run_models_pipeline() -> Dict[str, Path]:
    _ensure_output_dirs()

    panel, sigma = load_analysis_inputs()
    controls = get_available_controls(panel)
    panel = prepare_analysis_panel(panel)

    save_panel_schema_and_overview(panel, controls_used=controls)

    all_runs: List[RegressionRun] = []

    model_a = run_twfe_model(
        panel=panel,
        model_name="Model A",
        outcome="gdp_pc_growth",
        regressors=["erdf_eur_pc_l1"],
        controls=controls,
        fe_type="year",
        clustering="nuts2",
    )
    model_b = run_twfe_model(
        panel=panel,
        model_name="Model B",
        outcome="gdp_pc_growth",
        regressors=["erdf_eur_pc_l1"],
        controls=controls,
        fe_type="country_year",
        clustering="nuts2",
    )
    model_a_2way = run_twfe_model(
        panel=panel,
        model_name="Model A (two-way cluster)",
        outcome="gdp_pc_growth",
        regressors=["erdf_eur_pc_l1"],
        controls=controls,
        fe_type="year",
        clustering="nuts2_country",
    )
    model_b_2way = run_twfe_model(
        panel=panel,
        model_name="Model B (two-way cluster)",
        outcome="gdp_pc_growth",
        regressors=["erdf_eur_pc_l1"],
        controls=controls,
        fe_type="country_year",
        clustering="nuts2_country",
    )

    twfe_main = pd.concat(
        [
            _result_to_long_table(model_a),
            _result_to_long_table(model_b),
            _result_to_long_table(model_a_2way),
            _result_to_long_table(model_b_2way),
        ],
        ignore_index=True,
    )
    _write_csv(twfe_main, TABLE_PATHS["twfe_main"])
    all_runs.extend([model_a, model_b, model_a_2way, model_b_2way])

    model_c = run_twfe_model(
        panel=panel,
        model_name="Model C",
        outcome="gdp_pc_growth",
        regressors=["erdf_eur_pc_l1", "erdf_eur_pc_l2", "erdf_eur_pc_l3"],
        controls=controls,
        fe_type="year",
        clustering="nuts2",
    )
    dl_results = _result_to_long_table(model_c)
    _write_csv(dl_results, TABLE_PATHS["dl_lags"])
    all_runs.append(model_c)

    dynamic_regressors = ["erdf_eur_pc_l1", "erdf_eur_pc_l2", "erdf_eur_pc_l3"]
    dynamic_horizons = {"erdf_eur_pc_l1": 1, "erdf_eur_pc_l2": 2, "erdf_eur_pc_l3": 3}
    dynamic_model_name = "Model C"

    if "erdf_eur_pc" in panel.columns:
        model_c_l0 = run_twfe_model(
            panel=panel,
            model_name="Model C (with l0)",
            outcome="gdp_pc_growth",
            regressors=["erdf_eur_pc", "erdf_eur_pc_l1", "erdf_eur_pc_l2", "erdf_eur_pc_l3"],
            controls=controls,
            fe_type="year",
            clustering="nuts2",
        )
        dynamic_model_name = model_c_l0.model_name
        dynamic_horizons = {
            "erdf_eur_pc": 0,
            "erdf_eur_pc_l1": 1,
            "erdf_eur_pc_l2": 2,
            "erdf_eur_pc_l3": 3,
        }
        dynamic_effects = _extract_terms(model_c_l0, dynamic_horizons)
    else:
        dynamic_effects = _extract_terms(model_c, dynamic_horizons)

    _write_csv(dynamic_effects, TABLE_PATHS["dynamic_lag"])
    viz.plot_dynamic_lag_response(dynamic_effects, FIGURE_PATHS["dynamic_lag"], title=f"{dynamic_model_name}: dynamic lag response")

    placebo_run, placebo_effects = _run_placebo_leads_lags(panel, controls)
    _write_csv(placebo_effects, TABLE_PATHS["leads_lags"])
    viz.plot_leads_lags_placebo(placebo_effects, FIGURE_PATHS["leads_lags"])
    if placebo_run is not None:
        all_runs.append(placebo_run)

    viz.plot_sigma_convergence(sigma, FIGURE_PATHS["sigma"])

    model_d = run_twfe_model(
        panel=panel,
        model_name="Model D",
        outcome="gdp_pc_growth",
        regressors=["log_gdp_pc_l1"],
        controls=[],
        fe_type="year",
        clustering="nuts2",
    )

    model_e = run_twfe_model(
        panel=panel,
        model_name="Model E",
        outcome="gdp_pc_growth",
        regressors=["log_gdp_pc_l1", "erdf_eur_pc_l1", "erdf_eur_pc_l1:log_gdp_pc_l1"],
        controls=controls,
        fe_type="year",
        clustering="nuts2",
    )

    model_e_cy = run_twfe_model(
        panel=panel,
        model_name="Model E (country-year FE)",
        outcome="gdp_pc_growth",
        regressors=["log_gdp_pc_l1", "erdf_eur_pc_l1", "erdf_eur_pc_l1:log_gdp_pc_l1"],
        controls=controls,
        fe_type="country_year",
        clustering="nuts2",
    )

    beta_results = pd.concat(
        [
            _result_to_long_table(model_d),
            _result_to_long_table(model_e),
            _result_to_long_table(model_e_cy),
        ],
        ignore_index=True,
    )
    _write_csv(beta_results, TABLE_PATHS["beta"])
    all_runs.extend([model_d, model_e, model_e_cy])

    beta_marginal = _build_beta_marginal_effect(model_e, model_e.sample_df)
    viz.plot_beta_marginal_effect(beta_marginal, FIGURE_PATHS["beta_partial"])

    outlier_table, outlier_threshold, outlier_runs = _run_outlier_robustness(panel, controls)
    _write_csv(outlier_table, TABLE_PATHS["robust_outliers"])
    all_runs.extend(outlier_runs)

    balanced_table, balanced_meta, balanced_runs = _run_balanced_panel_robustness(panel, controls)
    _write_csv(balanced_table, TABLE_PATHS["robust_balanced"])
    all_runs.extend(balanced_runs)

    scaling_table, scaling_runs = _run_scaling_robustness(panel, controls)
    _write_csv(scaling_table, TABLE_PATHS["robust_scaling"])
    all_runs.extend(scaling_runs)

    comparison = _build_model_comparison_summary(all_runs)
    comparison["outlier_rule"] = "drop global top 1% erdf_eur_pc"
    comparison["outlier_threshold"] = outlier_threshold
    comparison["balanced_window_start"] = balanced_meta["balanced_window_start"]
    comparison["balanced_window_end"] = balanced_meta["balanced_window_end"]
    comparison["balanced_regions"] = balanced_meta["balanced_regions"]
    _write_csv(comparison, TABLE_PATHS["model_summary"])

    _create_report_notebook()

    LOGGER.info("Analysis pipeline completed. Outputs written under %s and %s", OUTPUT_TABLES_DIR, OUTPUT_FIGURES_DIR)

    output_paths: Dict[str, Path] = {}
    output_paths.update({f"table_{name}": path for name, path in TABLE_PATHS.items()})
    output_paths.update({f"figure_{name}": path for name, path in FIGURE_PATHS.items()})
    output_paths["report_notebook"] = NOTEBOOK_PATH
    return output_paths
