#!/usr/bin/env python3

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUTPUTS_FIGURES = PROJECT_ROOT / "outputs" / "figures"

REPORT_DIR = PROJECT_ROOT / "report"
REPORT_TABLES = REPORT_DIR / "tables"
REPORT_FIGURES = REPORT_DIR / "figures"


CSV_REQUIRED = [
    "panel_master_overview.csv",
    "twfe_main_results_v31.csv",
    "dl_lags_results_v31.csv",
    "rd_first_stage_funding_jump_v31.csv",
    "rd_outcome_sharp_results_v31.csv",
    "rd_placebo_pretrend_v31.csv",
    "iv_first_stage_candidates_v31.csv",
    "iv_cross_section_results_v31.csv",
    "iv_2sls_results_v31.csv",
    "model_comparison_summary_v31.csv",
    "beta_convergence_results_v31.csv",
]

FIG_REQUIRED = [
    "dynamic_lag_response_v31.png",
    "rd_first_stage_funding_jump_v31.png",
    "rd_outcome_binned_scatter_v31.png",
    "iv_first_stage_scatter_v31.png",
    "sigma_convergence_v31.png",
]


def _ensure_dirs() -> None:
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)


def _check_inputs_exist() -> None:
    missing: List[str] = []
    for name in CSV_REQUIRED:
        if not (OUTPUTS_TABLES / name).exists():
            missing.append(str(OUTPUTS_TABLES / name))
    for name in FIG_REQUIRED:
        if not (OUTPUTS_FIGURES / name).exists():
            missing.append(str(OUTPUTS_FIGURES / name))

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required report inputs. Run `python scripts/run_models.py` first.\n"
            f"{formatted}"
        )


def _copy_inputs() -> None:
    for name in CSV_REQUIRED:
        shutil.copy2(OUTPUTS_TABLES / name, REPORT_TABLES / name)
    for name in FIG_REQUIRED:
        shutil.copy2(OUTPUTS_FIGURES / name, REPORT_FIGURES / name)


def _fmt_numeric(df: pd.DataFrame, columns: List[str], decimals: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(
                lambda x: f"{x:.{decimals}f}" if pd.notna(x) else ""
            )
    return out


def _latex_escape_value(x: object) -> object:
    if not isinstance(x, str):
        return x
    return (
        x.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def _write_tabular(df: pd.DataFrame, out_path: Path) -> None:
    safe = df.copy()
    for col in safe.columns:
        safe[col] = safe[col].map(_latex_escape_value)

    colspec = "l" * len(safe.columns)
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    header = " & ".join(str(c) for c in safe.columns) + r" \\"
    lines.append(header)
    lines.append("\\midrule")

    for row in safe.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        lines.append(" & ".join(values) + r" \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORT_TABLES / name)


def _build_table_snippets() -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}

    overview = _load_csv("panel_master_overview.csv")
    overview = overview[overview["metric"].isin(["min_year", "max_year", "n_regions", "n_rows", "unique_keys"])].copy()
    overview["metric"] = overview["metric"].replace(
        {
            "min_year": "Min year",
            "max_year": "Max year",
            "n_regions": "Regions",
            "n_rows": "Rows",
            "unique_keys": "Unique (nuts2, year)",
        }
    )
    _write_tabular(overview, REPORT_TABLES / "overview_metrics.tex")
    outputs["overview_metrics"] = REPORT_TABLES / "overview_metrics.tex"

    twfe = _load_csv("twfe_main_results_v31.csv")
    twfe = twfe[
        (twfe["outcome"] == "gdp_pc_real_growth")
        & (twfe["model"].isin(["Model A", "Model B"]))
        & (twfe["term"] == "erdf_eur_pc_l1")
    ][["model", "coef", "std_err", "p_value", "n_obs", "n_regions", "clustering"]]
    twfe = twfe.rename(
        columns={
            "model": "Model",
            "coef": "Coef",
            "std_err": "SE",
            "p_value": "p",
            "n_obs": "N obs",
            "n_regions": "N regions",
            "clustering": "SE cluster",
        }
    )
    twfe = _fmt_numeric(twfe, ["Coef", "SE", "p"], decimals=4)
    _write_tabular(twfe, REPORT_TABLES / "twfe_main.tex")
    outputs["twfe_main"] = REPORT_TABLES / "twfe_main.tex"

    dl = _load_csv("dl_lags_results_v31.csv")
    dl = dl[
        (dl["outcome"] == "gdp_pc_real_growth")
        & (dl["model"] == "Model C")
        & (dl["term"].isin(["erdf_eur_pc_l1", "erdf_eur_pc_l2", "erdf_eur_pc_l3"]))
    ][["term", "coef", "std_err", "p_value", "n_obs"]]
    dl["term"] = dl["term"].replace(
        {
            "erdf_eur_pc_l1": "ERDF pc lag 1",
            "erdf_eur_pc_l2": "ERDF pc lag 2",
            "erdf_eur_pc_l3": "ERDF pc lag 3",
        }
    )
    dl = dl.rename(columns={"term": "Term", "coef": "Coef", "std_err": "SE", "p_value": "p", "n_obs": "N obs"})
    dl = _fmt_numeric(dl, ["Coef", "SE", "p"], decimals=4)
    _write_tabular(dl, REPORT_TABLES / "distributed_lags.tex")
    outputs["distributed_lags"] = REPORT_TABLES / "distributed_lags.tex"

    rd_fs = _load_csv("rd_first_stage_funding_jump_v31.csv")
    rd_fs = rd_fs[["bandwidth", "jump_coef", "std_err", "p_value", "first_stage_f_stat", "fuzzy_rd_viable"]]
    rd_fs = rd_fs.rename(
        columns={
            "bandwidth": "BW",
            "jump_coef": "Jump",
            "std_err": "SE",
            "p_value": "p",
            "first_stage_f_stat": "F",
            "fuzzy_rd_viable": "Fuzzy viable",
        }
    )
    rd_fs = _fmt_numeric(rd_fs, ["BW", "Jump", "SE", "p", "F"], decimals=3)
    _write_tabular(rd_fs, REPORT_TABLES / "rd_first_stage.tex")
    outputs["rd_first_stage"] = REPORT_TABLES / "rd_first_stage.tex"

    rd_out = _load_csv("rd_outcome_sharp_results_v31.csv")
    rd_out = rd_out[
        rd_out["outcome"].isin(["gdp_pc_real_growth", "gdp_pc_pps_growth"])
    ][["outcome", "window", "bandwidth", "coef", "std_err", "p_value", "n_obs"]]
    rd_out["outcome"] = rd_out["outcome"].replace(
        {"gdp_pc_real_growth": "Real growth", "gdp_pc_pps_growth": "PPS growth"}
    )
    rd_out = rd_out.rename(
        columns={
            "outcome": "Outcome",
            "window": "Window",
            "bandwidth": "BW",
            "coef": "Coef",
            "std_err": "SE",
            "p_value": "p",
            "n_obs": "N",
        }
    )
    rd_out = _fmt_numeric(rd_out, ["BW", "Coef", "SE", "p"], decimals=3)
    _write_tabular(rd_out, REPORT_TABLES / "rd_outcomes.tex")
    outputs["rd_outcomes"] = REPORT_TABLES / "rd_outcomes.tex"

    rd_placebo = _load_csv("rd_placebo_pretrend_v31.csv")[["window", "bandwidth", "coef", "std_err", "p_value", "n_obs"]]
    rd_placebo = rd_placebo.rename(
        columns={"window": "Window", "bandwidth": "BW", "coef": "Coef", "std_err": "SE", "p_value": "p", "n_obs": "N"}
    )
    rd_placebo = _fmt_numeric(rd_placebo, ["BW", "Coef", "SE", "p"], decimals=3)
    _write_tabular(rd_placebo, REPORT_TABLES / "rd_placebo.tex")
    outputs["rd_placebo"] = REPORT_TABLES / "rd_placebo.tex"

    iv_candidates = _load_csv("iv_first_stage_candidates_v31.csv")
    iv_candidates = iv_candidates[["candidate_id", "sample_type", "endogenous_var", "instrument_var", "f_stat", "partial_r2", "iv_first_stage_f_headline", "status"]]
    iv_candidates = iv_candidates.rename(
        columns={
            "candidate_id": "ID",
            "sample_type": "Sample",
            "endogenous_var": "Endogenous",
            "instrument_var": "Instrument",
            "f_stat": "F",
            "partial_r2": "Partial R2",
            "iv_first_stage_f_headline": "F (headline)",
            "status": "Status",
        }
    )
    iv_candidates = _fmt_numeric(iv_candidates, ["F", "Partial R2", "F (headline)"], decimals=2)
    _write_tabular(iv_candidates, REPORT_TABLES / "iv_candidates.tex")
    outputs["iv_candidates"] = REPORT_TABLES / "iv_candidates.tex"

    iv_cross = _load_csv("iv_cross_section_results_v31.csv")
    iv_cross = iv_cross[["outcome", "window", "instrument", "coef", "std_err", "p_value", "first_stage_f_stat", "n_obs"]]
    iv_cross["outcome"] = iv_cross["outcome"].replace(
        {"gdp_pc_real_growth": "Real growth", "gdp_pc_pps_growth": "PPS growth"}
    )
    iv_cross = iv_cross.rename(
        columns={
            "outcome": "Outcome",
            "window": "Window",
            "instrument": "Instrument",
            "coef": "Coef",
            "std_err": "SE",
            "p_value": "p",
            "first_stage_f_stat": "F",
            "n_obs": "N",
        }
    )
    iv_cross = _fmt_numeric(iv_cross, ["Coef", "SE", "p", "F"], decimals=3)
    _write_tabular(iv_cross, REPORT_TABLES / "iv_cross_section.tex")
    outputs["iv_cross_section"] = REPORT_TABLES / "iv_cross_section.tex"

    iv_panel = _load_csv("iv_2sls_results_v31.csv")[["outcome", "instrument", "coef", "std_err", "p_value", "first_stage_f_stat", "n_obs"]]
    iv_panel["outcome"] = iv_panel["outcome"].replace({"gdp_pc_real_growth": "Real growth"})
    iv_panel = iv_panel.rename(
        columns={
            "outcome": "Outcome",
            "instrument": "Instrument",
            "coef": "Coef",
            "std_err": "SE",
            "p_value": "p",
            "first_stage_f_stat": "F",
            "n_obs": "N",
        }
    )
    iv_panel = _fmt_numeric(iv_panel, ["Coef", "SE", "p", "F"], decimals=3)
    _write_tabular(iv_panel, REPORT_TABLES / "iv_panel.tex")
    outputs["iv_panel"] = REPORT_TABLES / "iv_panel.tex"

    comparison = _load_csv("model_comparison_summary_v31.csv")
    comparison = comparison[["estimator_family", "model", "window", "coef", "std_err", "p_value", "first_stage_f_stat", "is_headline_estimator"]]
    comparison = comparison.rename(
        columns={
            "estimator_family": "Family",
            "model": "Model",
            "window": "Window",
            "coef": "Coef",
            "std_err": "SE",
            "p_value": "p",
            "first_stage_f_stat": "F",
            "is_headline_estimator": "Headline",
        }
    )
    comparison = _fmt_numeric(comparison, ["Coef", "SE", "p", "F"], decimals=3)
    _write_tabular(comparison, REPORT_TABLES / "model_comparison.tex")
    outputs["model_comparison"] = REPORT_TABLES / "model_comparison.tex"

    beta = _load_csv("beta_convergence_results_v31.csv")
    beta = beta[
        (beta["model"] == "Model D")
        & (beta["term"].isin(["log_gdp_pc_real_l1", "log_gdp_pc_pps_l1"]))
    ][["outcome", "term", "coef", "std_err", "p_value"]]
    beta["outcome"] = beta["outcome"].replace(
        {"gdp_pc_real_growth": "Real growth", "gdp_pc_pps_growth": "PPS growth"}
    )
    beta["term"] = beta["term"].replace(
        {"log_gdp_pc_real_l1": "log real GDP pc lag", "log_gdp_pc_pps_l1": "log PPS GDP pc lag"}
    )
    beta = beta.rename(columns={"outcome": "Outcome", "term": "Term", "coef": "Coef", "std_err": "SE", "p_value": "p"})
    beta = _fmt_numeric(beta, ["Coef", "SE", "p"], decimals=3)
    _write_tabular(beta, REPORT_TABLES / "beta_convergence.tex")
    outputs["beta_convergence"] = REPORT_TABLES / "beta_convergence.tex"

    return outputs


def main() -> None:
    _ensure_dirs()
    _check_inputs_exist()
    _copy_inputs()
    snippets = _build_table_snippets()
    for name, path in snippets.items():
        print(f"[build_latex_tables] {name}: {path}")


if __name__ == "__main__":
    main()
