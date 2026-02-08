from __future__ import annotations

from pathlib import Path
from typing import Optional

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
    title: str = "Dynamic Lag Response of GDP per Capita Growth",
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


def plot_sigma_convergence(
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
        ax.plot(plot_df["year"], plot_df["sigma_log_gdp"], marker="o", linewidth=2, color="#2ca02c")
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
