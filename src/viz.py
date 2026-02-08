from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_dynamic_lag_response(
    lag_effects: pd.DataFrame,
    output_path: Path,
    title: str = "Dynamic Lag Response",
) -> None:
    plt = _import_matplotlib()

    plot_df = lag_effects.sort_values("horizon").copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_df.empty:
        ax.text(0.5, 0.5, "No lag coefficients available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
        ax.errorbar(
            plot_df["horizon"],
            plot_df["coef"],
            yerr=1.96 * plot_df["std_err"],
            fmt="o-",
            color="#1f77b4",
            ecolor="#1f77b4",
            capsize=4,
        )
        ax.set_xticks(plot_df["horizon"].tolist())
        ax.set_xlabel("Lag horizon (years)")
        ax.set_ylabel("Coefficient on ERDF per capita")

    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_leads_lags_placebo(
    leads_lags_effects: pd.DataFrame,
    output_path: Path,
    title: str = "Placebo Leads and Lag Effects",
) -> None:
    plt = _import_matplotlib()

    plot_df = leads_lags_effects.sort_values("horizon").copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_df.empty:
        ax.text(0.5, 0.5, "No leads/lags coefficients available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
        ax.axvline(0.0, color="gray", linewidth=1, linestyle="--", alpha=0.7)

        ax.errorbar(
            plot_df["horizon"],
            plot_df["coef"],
            yerr=1.96 * plot_df["std_err"],
            fmt="o-",
            color="#ff7f0e",
            ecolor="#ff7f0e",
            capsize=4,
        )

        ax.set_xticks(plot_df["horizon"].tolist())
        ax.set_xlabel("Horizon (negative = lead, positive = lag)")
        ax.set_ylabel("Coefficient on ERDF per capita")

    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sigma_convergence_multi(
    sigma_df: pd.DataFrame,
    output_path: Path,
    period_start: int = 2014,
    period_end: int = 2020,
) -> None:
    plt = _import_matplotlib()

    plot_df = sigma_df.sort_values("year").copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_df.empty:
        ax.text(0.5, 0.5, "No sigma convergence data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        if "sigma_log_gdp_real" in plot_df.columns:
            ax.plot(
                plot_df["year"],
                plot_df["sigma_log_gdp_real"],
                marker="o",
                linewidth=2,
                label="Real GDP pc sigma",
            )
        if "sigma_log_gdp_pps" in plot_df.columns:
            ax.plot(
                plot_df["year"],
                plot_df["sigma_log_gdp_pps"],
                marker="o",
                linewidth=2,
                label="PPS GDP pc sigma",
            )
        if "sigma_log_gdp_nominal" in plot_df.columns:
            ax.plot(
                plot_df["year"],
                plot_df["sigma_log_gdp_nominal"],
                marker="o",
                linewidth=2,
                label="Nominal GDP pc sigma",
                alpha=0.6,
            )

        ax.axvspan(period_start, period_end, color="#2ca02c", alpha=0.15, label="2014-2020 period")
        ax.set_xlabel("Year")
        ax.set_ylabel("Sigma(log GDP per capita)")
        ax.legend()

    ax.set_title("Sigma Convergence Over Time")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_beta_marginal_effect(
    marginal_effect_df: pd.DataFrame,
    output_path: Path,
    title: str = "Marginal Effect of ERDF by Initial Income",
) -> None:
    plt = _import_matplotlib()

    plot_df = marginal_effect_df.sort_values("percentile").copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_df.empty:
        ax.text(0.5, 0.5, "No marginal effects available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.plot(plot_df["percentile"], plot_df["marginal_effect"], color="#9467bd", linewidth=2)
        ax.fill_between(
            plot_df["percentile"],
            plot_df["ci_lower"],
            plot_df["ci_upper"],
            color="#9467bd",
            alpha=0.2,
        )
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
        ax.set_xlabel("Percentile of lagged log GDP per capita")
        ax.set_ylabel("Marginal effect of ERDF on GDP growth")

    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rd_binned_scatter(
    region_df: pd.DataFrame,
    outcome_col: str,
    output_path: Path,
    cutoff: float = 75.0,
    bandwidth: float = 15.0,
    title: str = "RD around Eligibility Threshold",
) -> None:
    plt = _import_matplotlib()

    work = region_df[["r_value", outcome_col]].dropna().copy()
    work = work[(work["r_value"] >= cutoff - bandwidth) & (work["r_value"] <= cutoff + bandwidth)].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    if work.empty:
        ax.text(0.5, 0.5, "No RD data available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    work["side"] = (work["r_value"] >= cutoff).map({False: "left", True: "right"})
    work["bin"] = pd.cut(work["r_value"], bins=20, duplicates="drop")
    binned = (
        work.groupby(["bin", "side"], observed=False)
        .agg(r_value=("r_value", "mean"), y_mean=(outcome_col, "mean"))
        .dropna()
        .reset_index()
    )

    if not binned.empty:
        for side, color, label in [
            ("left", "#1f77b4", "Below cutoff"),
            ("right", "#d62728", "Above cutoff"),
        ]:
            side_df = binned[binned["side"] == side]
            if side_df.empty:
                continue
            ax.scatter(side_df["r_value"], side_df["y_mean"], color=color, alpha=0.8, label=label)

    for side, color in [("left", "#1f77b4"), ("right", "#d62728")]:
        side_df = work[work["side"] == side]
        if len(side_df) < 8:
            continue
        coeff = np.polyfit(side_df["r_value"], side_df[outcome_col], deg=1)
        x = np.linspace(side_df["r_value"].min(), side_df["r_value"].max(), 100)
        y = coeff[0] * x + coeff[1]
        ax.plot(x, y, color=color, linewidth=2)

    ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Running variable: GDP pc PPS relative to EU average (index)")
    ax.set_ylabel("Average growth in window")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rd_bandwidth_sensitivity(
    rd_sensitivity: pd.DataFrame,
    output_path: Path,
    title: str = "RD Bandwidth Sensitivity",
) -> None:
    plt = _import_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))
    if rd_sensitivity.empty:
        ax.text(0.5, 0.5, "No RD sensitivity estimates available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    plot_df = rd_sensitivity.sort_values(["outcome", "window", "bandwidth"]).copy()
    plot_df["label"] = plot_df["outcome"] + " | " + plot_df["window"]

    for label, group in plot_df.groupby("label", sort=False):
        ax.errorbar(
            group["bandwidth"],
            group["coef"],
            yerr=1.96 * group["std_err"],
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=label,
        )

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.set_xlabel("Bandwidth around cutoff")
    ax.set_ylabel("Estimated treatment effect")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
