from __future__ import annotations

import re
from functools import reduce
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src import config
from src import ingest

KEY_COLUMNS = ["nuts2_id", "year"]
VALID_CATEGORIES = {"less_developed", "transition", "more_developed"}


def normalize_header(column_name: str) -> str:
    cleaned = column_name.replace("\ufeff", "").strip()
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[0].strip()
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned


def normalize_dataframe_headers(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    used: Dict[str, int] = {}

    for column in df.columns:
        normalized = normalize_header(str(column))
        if normalized in used:
            used[normalized] += 1
            normalized = f"{normalized}_{used[normalized]}"
        else:
            used[normalized] = 0
        rename_map[column] = normalized

    return df.rename(columns=rename_map)


def extract_code(value: object) -> object:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    string_value = str(value).strip()
    if not string_value or string_value in {":", "nan", "None"}:
        return np.nan

    if ":" in string_value:
        string_value = string_value.split(":", 1)[0].strip()

    return string_value


def normalize_code_column(series: pd.Series) -> pd.Series:
    return series.map(extract_code).astype("string").str.strip().str.upper()


def parse_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace({"": pd.NA, ":": pd.NA, "nan": pd.NA, "None": pd.NA})
    cleaned = cleaned.str.replace("\u00a0", "", regex=False)
    cleaned = cleaned.str.replace(" ", "", regex=False)
    cleaned = cleaned.str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_year_series(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(\d{4})", expand=False)
    years = pd.to_numeric(extracted, errors="coerce")
    years = years.where(years.between(1900, 2100), np.nan)
    return years.astype("Int64")


def ensure_required_raw_files_exist() -> None:
    missing_files = [
        file_name
        for file_name in config.REQUIRED_RAW_FILES
        if not (config.DATA_RAW_DIR / file_name).exists()
    ]

    if missing_files:
        formatted = "\n".join(f"  - {name}" for name in missing_files)
        raise FileNotFoundError(
            "Missing required raw files in data/raw/.\n"
            f"Please add these files and rerun:\n{formatted}"
        )


def ensure_output_directories() -> None:
    directories = [
        config.DATA_INTERIM_DIR,
        config.DATA_PROCESSED_DIR,
        config.OUTPUTS_TABLES_DIR,
        config.OUTPUTS_FIGURES_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def read_raw_csv(file_name: str) -> pd.DataFrame:
    path = config.DATA_RAW_DIR / file_name
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = normalize_dataframe_headers(df)
    return df


def apply_dimension_filters(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    filters = config.EUROSTAT_FILTERS.get(file_name, {})
    if not filters:
        return df

    filtered = df.copy()
    for column, expected_value in filters.items():
        if column not in filtered.columns:
            continue
        filtered[column] = normalize_code_column(filtered[column])
        filtered = filtered[filtered[column] == expected_value.upper()]

    return filtered


def _candidate_columns_by_keywords(df: pd.DataFrame, keywords: Sequence[str]) -> List[str]:
    return [column for column in df.columns if any(keyword in column for keyword in keywords)]


def infer_nuts2_column(df: pd.DataFrame) -> Optional[str]:
    candidates = _candidate_columns_by_keywords(df, ["nuts2", "geo", "region"])
    if not candidates:
        return None

    best_column = None
    best_score = 0.0

    for column in candidates:
        codes = normalize_code_column(df[column])
        valid = codes.str.fullmatch(config.NUTS2_REGEX, na=False) & ~codes.str.endswith(
            config.UNKNOWN_NUTS2_SUFFIXES
        )
        score = float(valid.mean()) if len(valid) else 0.0
        if score > best_score:
            best_score = score
            best_column = column

    return best_column if best_score > 0 else None


def infer_year_column(df: pd.DataFrame) -> Optional[str]:
    candidates = _candidate_columns_by_keywords(df, ["time_period", "year", "time"])
    if not candidates:
        return None

    best_column = None
    best_score = 0.0

    for column in candidates:
        years = parse_year_series(df[column])
        score = float(years.notna().mean()) if len(years) else 0.0
        if score > best_score:
            best_score = score
            best_column = column

    return best_column if best_score > 0 else None


def infer_value_column(df: pd.DataFrame, file_name: str) -> Optional[str]:
    fallback = config.COLUMN_FALLBACKS.get(file_name, {})
    for candidate in fallback.get("value_candidates", []):
        if candidate in df.columns:
            return candidate

    if "obs_value" in df.columns:
        return "obs_value"

    candidates = _candidate_columns_by_keywords(
        df,
        ["payment", "spending", "expenditure", "amount", "value", "obs", "index"],
    )

    best_column = None
    best_score = 0.0
    for column in candidates:
        parsed = parse_numeric_series(df[column])
        score = float(parsed.notna().mean()) if len(parsed) else 0.0
        if score > best_score:
            best_score = score
            best_column = column

    return best_column if best_score > 0 else None


def standardize_keys(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    fallback = config.COLUMN_FALLBACKS.get(file_name, {})

    nuts2_column = fallback.get("nuts2") if fallback.get("nuts2") in df.columns else None
    year_column = fallback.get("year") if fallback.get("year") in df.columns else None

    if nuts2_column is None:
        nuts2_column = infer_nuts2_column(df)
    if year_column is None:
        year_column = infer_year_column(df)

    if nuts2_column is None or year_column is None:
        raise ValueError(
            f"Could not infer nuts2/year columns for {file_name}. Available columns: {sorted(df.columns)}"
        )

    standardized = df.copy()
    standardized["nuts2_id"] = normalize_code_column(standardized[nuts2_column])
    standardized["year"] = parse_year_series(standardized[year_column])

    valid_nuts = standardized["nuts2_id"].str.fullmatch(config.NUTS2_REGEX, na=False)
    valid_nuts = valid_nuts & ~standardized["nuts2_id"].str.endswith(
        config.UNKNOWN_NUTS2_SUFFIXES
    )

    standardized = standardized[valid_nuts & standardized["year"].notna()].copy()
    standardized["year"] = standardized["year"].astype(int)

    return standardized


def assert_unique_keys(df: pd.DataFrame, label: str) -> None:
    duplicates = df.duplicated(KEY_COLUMNS).sum()
    if duplicates:
        raise ValueError(f"Dataset {label} has {duplicates} duplicated (nuts2_id, year) rows.")


def load_population() -> pd.DataFrame:
    file_name = config.RAW_FILES["population"]
    df = read_raw_csv(file_name)
    df = apply_dimension_filters(df, file_name)
    df = standardize_keys(df, file_name)

    value_column = infer_value_column(df, file_name)
    if value_column is None:
        raise ValueError(f"Could not infer population value column for {file_name}.")

    df["population"] = parse_numeric_series(df[value_column])
    df = df.dropna(subset=["population"])

    population = (
        df.groupby(KEY_COLUMNS, as_index=False, sort=True)["population"]
        .mean()
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    population = population[population["population"] > 0]
    assert_unique_keys(population, "population")
    return population


def load_nominal_gdp_series() -> pd.DataFrame:
    file_name = config.RAW_FILES["gdp"]
    df = read_raw_csv(file_name)
    df = apply_dimension_filters(df, file_name)
    df = standardize_keys(df, file_name)

    value_column = infer_value_column(df, file_name)
    if value_column is None:
        raise ValueError(f"Could not infer nominal GDP value column for {file_name}.")

    df["gdp_mio_eur"] = parse_numeric_series(df[value_column])
    df = df.dropna(subset=["gdp_mio_eur"])

    gdp = (
        df.groupby(KEY_COLUMNS, as_index=False, sort=True)["gdp_mio_eur"]
        .mean()
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    assert_unique_keys(gdp, "gdp_nominal")
    return gdp


def load_gdp_pc_pps_series() -> pd.DataFrame:
    file_name = config.RAW_FILES["gdp_pc_pps"]
    df = read_raw_csv(file_name)
    df = apply_dimension_filters(df, file_name)
    df = standardize_keys(df, file_name)

    value_column = infer_value_column(df, file_name)
    if value_column is None:
        raise ValueError(f"Could not infer GDP PPS value column for {file_name}.")

    df["gdp_pc_pps"] = parse_numeric_series(df[value_column])
    df = df.dropna(subset=["gdp_pc_pps"])

    gdp_pps = (
        df.groupby(KEY_COLUMNS, as_index=False, sort=True)["gdp_pc_pps"]
        .mean()
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    assert_unique_keys(gdp_pps, "gdp_pps")
    return gdp_pps


def load_gdp_real_index_series() -> pd.DataFrame:
    file_name = config.RAW_FILES["gdp_real"]
    df = read_raw_csv(file_name)
    df = apply_dimension_filters(df, file_name)
    df = standardize_keys(df, file_name)

    value_column = infer_value_column(df, file_name)
    if value_column is None:
        raise ValueError(f"Could not infer real GDP volume index column for {file_name}.")

    df["gdp_volume_index"] = parse_numeric_series(df[value_column])
    df = df.dropna(subset=["gdp_volume_index"])

    real_index = (
        df.groupby(KEY_COLUMNS, as_index=False, sort=True)["gdp_volume_index"]
        .mean()
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    assert_unique_keys(real_index, "gdp_real_index")
    return real_index


def load_indicator_series(file_key: str, value_name: str) -> pd.DataFrame:
    file_name = config.RAW_FILES[file_key]
    df = read_raw_csv(file_name)
    df = apply_dimension_filters(df, file_name)
    df = standardize_keys(df, file_name)

    value_column = infer_value_column(df, file_name)
    if value_column is None:
        raise ValueError(f"Could not infer value column for {file_name} ({value_name}).")

    df[value_name] = parse_numeric_series(df[value_column])
    df = df.dropna(subset=[value_name])

    result = (
        df.groupby(KEY_COLUMNS, as_index=False, sort=True)[value_name]
        .mean()
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    assert_unique_keys(result, value_name)
    return result


def load_treatment_erdf() -> pd.DataFrame:
    file_name = config.RAW_FILES["historic_payments"]
    df = read_raw_csv(file_name)

    fallback = config.COLUMN_FALLBACKS.get(file_name, {})
    fund_column = fallback.get("fund")
    if fund_column and fund_column in df.columns:
        funds = normalize_code_column(df[fund_column])
        df = df[funds == "ERDF"]

    df = standardize_keys(df, file_name)

    value_column = infer_value_column(df, file_name)
    if value_column is None:
        raise ValueError(f"Could not infer treatment value column for {file_name}.")

    df["erdf_eur"] = parse_numeric_series(df[value_column])
    df = df.dropna(subset=["erdf_eur"])

    treatment = (
        df.groupby(KEY_COLUMNS, as_index=False, sort=True)["erdf_eur"]
        .sum()
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    assert_unique_keys(treatment, "treatment_erdf")
    return treatment


def check_esif_nuts2_usability() -> str:
    file_name = config.RAW_FILES["esif_finance"]
    df = read_raw_csv(file_name)

    if "fund" in df.columns:
        funds = normalize_code_column(df["fund"])
        df = df[funds == "ERDF"]

    nuts2_column = infer_nuts2_column(df)
    if nuts2_column is None:
        return (
            "ESIF finance file does not contain a reliable NUTS2 key after header inspection; "
            "treatment is built from historic regionalized ERDF payments only."
        )

    codes = normalize_code_column(df[nuts2_column])
    valid_share = (
        codes.str.fullmatch(config.NUTS2_REGEX, na=False)
        & ~codes.str.endswith(config.UNKNOWN_NUTS2_SUFFIXES)
    ).mean()

    if float(valid_share) < 0.7:
        return (
            "ESIF finance file NUTS2 inference quality is low; treatment is built from historic "
            "regionalized ERDF payments only."
        )

    return "ESIF finance file appears to include a usable NUTS2 key."


def build_controls() -> pd.DataFrame:
    controls_frames = [
        load_indicator_series("unemployment", "unemp_rate"),
        load_indicator_series("employment", "emp_rate"),
        load_indicator_series("tertiary", "tertiary_share_25_64"),
        load_indicator_series("rd_gerd", "rd_gerd"),
        load_indicator_series("gva", "gva"),
    ]

    controls = reduce(
        lambda left, right: left.merge(
            right,
            on=KEY_COLUMNS,
            how="outer",
            sort=False,
            validate="one_to_one",
        ),
        controls_frames,
    )
    controls = controls.sort_values(KEY_COLUMNS).reset_index(drop=True)
    assert_unique_keys(controls, "controls")
    return controls


def load_eligibility_categories() -> pd.DataFrame:
    file_name = config.RAW_FILES["eligibility_categories"]
    df = read_raw_csv(file_name)

    fallback = config.COLUMN_FALLBACKS.get(file_name, {})
    nuts_col = fallback.get("nuts2") if fallback.get("nuts2") in df.columns else infer_nuts2_column(df)

    if nuts_col is None:
        raise ValueError(
            f"Could not infer nuts2_id column for eligibility mapping file {file_name}."
        )

    category_col = fallback.get("category")
    if category_col not in df.columns:
        category_candidates = [col for col in df.columns if "category" in col]
        category_col = category_candidates[0] if category_candidates else None

    if category_col is None:
        raise ValueError(
            f"Could not infer category_2014_2020 column in {file_name}."
        )

    eligibility = pd.DataFrame(
        {
            "nuts2_id": normalize_code_column(df[nuts_col]),
            "category_2014_2020": df[category_col].astype("string").str.strip().str.lower(),
        }
    )

    eligibility["category_2014_2020"] = (
        eligibility["category_2014_2020"]
        .str.replace(r"[^a-z]+", "_", regex=True)
        .str.strip("_")
        .replace(
            {
                "less_developed_region": "less_developed",
                "less_developed_regions": "less_developed",
                "transition_region": "transition",
                "transition_regions": "transition",
                "more_developed_region": "more_developed",
                "more_developed_regions": "more_developed",
            }
        )
    )

    valid_nuts = eligibility["nuts2_id"].str.fullmatch(config.NUTS2_REGEX, na=False)
    valid_nuts &= ~eligibility["nuts2_id"].str.endswith(config.UNKNOWN_NUTS2_SUFFIXES)

    eligibility = eligibility[valid_nuts].copy()
    eligibility = eligibility[eligibility["category_2014_2020"].isin(VALID_CATEGORIES)].copy()
    eligibility = eligibility.drop_duplicates(subset=["nuts2_id"], keep="first")
    eligibility = eligibility.sort_values("nuts2_id").reset_index(drop=True)

    if eligibility.empty:
        raise ValueError(
            "Eligibility categories mapping is empty after cleaning. "
            f"Check {config.DATA_RAW_DIR / file_name}."
        )

    return eligibility


def load_pps_relative_eu_series() -> pd.DataFrame:
    file_name = config.RAW_FILES["gdp_pc_pps_rel_eu"]
    df = read_raw_csv(file_name)
    df = apply_dimension_filters(df, file_name)
    df = standardize_keys(df, file_name)

    value_column = infer_value_column(df, file_name)
    if value_column is None:
        raise ValueError(f"Could not infer PPS relative-to-EU value column for {file_name}.")

    df["r_value_raw"] = parse_numeric_series(df[value_column])
    df = df.dropna(subset=["r_value_raw"])

    ratio = (
        df.groupby(KEY_COLUMNS, as_index=False, sort=True)["r_value_raw"]
        .mean()
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    assert_unique_keys(ratio, "pps_relative_eu")
    return ratio


def build_running_variable_eligibility(
    pps_relative: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for nuts2_id, region_df in pps_relative.groupby("nuts2_id", sort=True):
        region_sorted = region_df.sort_values("year")
        reference = region_sorted[region_sorted["year"].isin([2007, 2008, 2009])]

        if not reference.empty:
            r_value = float(reference["r_value_raw"].mean())
            ref_years_used = ",".join(str(year) for year in sorted(reference["year"].unique().tolist()))
        else:
            fallback = region_sorted[region_sorted["year"] <= 2013]
            if fallback.empty:
                continue
            r_value = float(fallback["r_value_raw"].mean())
            fallback_years = ",".join(str(year) for year in sorted(fallback["year"].unique().tolist()))
            ref_years_used = f"fallback_<=2013:{fallback_years}"

        records.append(
            {
                "nuts2_id": nuts2_id,
                "country": nuts2_id[:2],
                "r_value": r_value,
                "eligible_lt75": int(r_value < 75.0),
                "ref_years_used": ref_years_used,
            }
        )

    running = pd.DataFrame(records)
    if running.empty:
        raise ValueError(
            "Running variable construction produced an empty dataset. "
            "Check cached PPS relative-to-EU raw input."
        )

    running = running.merge(
        eligibility[["nuts2_id", "category_2014_2020"]],
        on="nuts2_id",
        how="left",
        validate="one_to_one",
    )

    # Fallback category assignment if any category is absent in mapping.
    missing_category = running["category_2014_2020"].isna()
    running.loc[missing_category & (running["r_value"] < 75), "category_2014_2020"] = "less_developed"
    running.loc[
        missing_category & running["r_value"].between(75, 90, inclusive="left"),
        "category_2014_2020",
    ] = "transition"
    running.loc[missing_category & (running["r_value"] >= 90), "category_2014_2020"] = "more_developed"

    running = running.sort_values("nuts2_id").reset_index(drop=True)
    return running[
        [
            "nuts2_id",
            "country",
            "r_value",
            "eligible_lt75",
            "category_2014_2020",
            "ref_years_used",
        ]
    ]


def build_erdf_cumulative_exposure(panel: pd.DataFrame) -> pd.DataFrame:
    base = panel.copy()
    base["erdf_eur_pc"] = pd.to_numeric(base["erdf_eur_pc"], errors="coerce").fillna(0.0)

    cum_2014_2020 = (
        base[base["year"].between(2014, 2020)]
        .groupby("nuts2_id", as_index=False)["erdf_eur_pc"]
        .sum()
        .rename(columns={"erdf_eur_pc": "erdf_eur_pc_cum_2014_2020"})
    )

    cum_2015_2020 = (
        base[base["year"].between(2015, 2020)]
        .groupby("nuts2_id", as_index=False)["erdf_eur_pc"]
        .sum()
        .rename(columns={"erdf_eur_pc": "erdf_eur_pc_cum_2015_2020"})
    )

    exposure = cum_2014_2020.merge(cum_2015_2020, on="nuts2_id", how="outer")
    exposure = exposure.merge(
        base[["nuts2_id", "country"]].drop_duplicates(subset=["nuts2_id"]),
        on="nuts2_id",
        how="left",
        validate="one_to_one",
    )
    exposure = exposure.sort_values("nuts2_id").reset_index(drop=True)
    return exposure[
        [
            "nuts2_id",
            "country",
            "erdf_eur_pc_cum_2014_2020",
            "erdf_eur_pc_cum_2015_2020",
        ]
    ]


def build_panel_skeleton(datasets: Iterable[pd.DataFrame]) -> pd.DataFrame:
    region_sets: List[set] = []
    min_year: Optional[int] = None
    max_year: Optional[int] = None

    for dataset in datasets:
        if dataset.empty:
            continue
        region_sets.append(set(dataset["nuts2_id"].dropna().unique().tolist()))
        dataset_min_year = int(dataset["year"].min())
        dataset_max_year = int(dataset["year"].max())

        min_year = dataset_min_year if min_year is None else min(min_year, dataset_min_year)
        max_year = dataset_max_year if max_year is None else max(max_year, dataset_max_year)

    if not region_sets or min_year is None or max_year is None:
        raise ValueError("Could not infer region/year skeleton from source datasets.")

    regions = sorted(set().union(*region_sets))
    years = list(range(min_year, max_year + 1))

    skeleton = (
        pd.MultiIndex.from_product([regions, years], names=KEY_COLUMNS)
        .to_frame(index=False)
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    skeleton["country"] = skeleton["nuts2_id"].str[:2]
    assert_unique_keys(skeleton, "panel_skeleton")
    return skeleton


def _construct_real_level_from_index(outcomes: pd.DataFrame) -> pd.DataFrame:
    result = outcomes.copy()
    result["gdp_pc_real"] = np.nan

    for region, region_df in result.groupby("nuts2_id", sort=False):
        idx = region_df.index
        nominal_values = pd.to_numeric(region_df["gdp_pc"], errors="coerce").astype(float)
        volume_index_values = pd.to_numeric(region_df["gdp_volume_index"], errors="coerce").astype(float)
        valid = (
            nominal_values.notna()
            & volume_index_values.notna()
            & volume_index_values.gt(0)
        )

        if not valid.any():
            continue

        reference_rows = region_df.loc[valid].copy().sort_values("year")
        preferred = reference_rows[reference_rows["year"] == 2015]
        reference = preferred.iloc[0] if not preferred.empty else reference_rows.iloc[0]

        reference_nominal_pc = float(pd.to_numeric(reference["gdp_pc"], errors="coerce"))
        reference_index = float(pd.to_numeric(reference["gdp_volume_index"], errors="coerce"))
        if reference_index <= 0:
            continue

        positive_index_mask = volume_index_values.gt(0).fillna(False).to_numpy()
        series = np.where(
            positive_index_mask,
            reference_nominal_pc * (volume_index_values.to_numpy() / reference_index),
            np.nan,
        )
        result.loc[idx, "gdp_pc_real"] = series

    return result


def build_outcomes(
    gdp_nominal: pd.DataFrame,
    population: pd.DataFrame,
    gdp_pps: pd.DataFrame,
    gdp_real_index: pd.DataFrame,
) -> pd.DataFrame:
    outcomes = gdp_nominal.merge(
        population,
        on=KEY_COLUMNS,
        how="left",
        sort=False,
        validate="one_to_one",
    )

    outcomes = outcomes.merge(
        gdp_pps,
        on=KEY_COLUMNS,
        how="outer",
        sort=False,
        validate="one_to_one",
    )

    outcomes = outcomes.merge(
        gdp_real_index,
        on=KEY_COLUMNS,
        how="outer",
        sort=False,
        validate="one_to_one",
    )

    outcomes = outcomes.sort_values(KEY_COLUMNS).reset_index(drop=True)

    outcomes["gdp_eur"] = outcomes["gdp_mio_eur"] * 1_000_000
    population_values = pd.to_numeric(outcomes["population"], errors="coerce").astype(float)
    valid_population = population_values.gt(0).fillna(False).to_numpy()

    outcomes["gdp_pc"] = np.where(
        valid_population,
        outcomes["gdp_eur"].astype(float).to_numpy() / population_values.to_numpy(),
        np.nan,
    )

    outcomes = _construct_real_level_from_index(outcomes)

    for level_column, log_column, growth_column in [
        ("gdp_pc", "log_gdp_pc", "gdp_pc_growth"),
        ("gdp_pc_pps", "log_gdp_pc_pps", "gdp_pc_pps_growth"),
        ("gdp_pc_real", "log_gdp_pc_real", "gdp_pc_real_growth"),
    ]:
        level_values = pd.to_numeric(outcomes[level_column], errors="coerce").astype(float)
        outcomes[level_column] = level_values
        positive_mask = level_values.gt(0).fillna(False).to_numpy()
        level_array = level_values.to_numpy()
        log_values = np.full(level_array.shape[0], np.nan, dtype=float)
        log_values[positive_mask] = np.log(level_array[positive_mask])
        outcomes[log_column] = log_values
        lag_column = f"{log_column}_l1"
        outcomes[lag_column] = outcomes.groupby("nuts2_id", sort=False)[log_column].shift(1)
        outcomes[growth_column] = 100.0 * (outcomes[log_column] - outcomes[lag_column])

    keep_columns = KEY_COLUMNS + [
        "gdp_mio_eur",
        "gdp_eur",
        "gdp_pc",
        "log_gdp_pc",
        "gdp_pc_growth",
        "gdp_pc_pps",
        "log_gdp_pc_pps",
        "gdp_pc_pps_growth",
        "gdp_volume_index",
        "gdp_pc_real",
        "log_gdp_pc_real",
        "gdp_pc_real_growth",
    ]

    outcomes = outcomes[keep_columns].sort_values(KEY_COLUMNS).reset_index(drop=True)
    assert_unique_keys(outcomes, "outcomes")
    return outcomes


def merge_left_preserve_rows(base: pd.DataFrame, other: pd.DataFrame, label: str) -> pd.DataFrame:
    if other.duplicated(KEY_COLUMNS).any():
        duplicate_count = int(other.duplicated(KEY_COLUMNS).sum())
        raise ValueError(f"Cannot merge {label}: {duplicate_count} duplicated key rows in source.")

    before_rows = len(base)
    merged = base.merge(other, on=KEY_COLUMNS, how="left", sort=False, validate="one_to_one")

    if len(merged) != before_rows:
        raise ValueError(
            f"Row count changed during merge of {label}: before={before_rows}, after={len(merged)}"
        )

    assert_unique_keys(merged, f"merged_{label}")
    return merged


def finalize_panel(
    skeleton: pd.DataFrame,
    population: pd.DataFrame,
    outcomes: pd.DataFrame,
    treatment: pd.DataFrame,
    controls: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    panel = skeleton.copy().sort_values(KEY_COLUMNS).reset_index(drop=True)

    panel = merge_left_preserve_rows(panel, population, "population")
    panel = merge_left_preserve_rows(panel, outcomes, "outcomes")
    panel = merge_left_preserve_rows(panel, treatment, "treatment")

    panel["erdf_eur"] = panel["erdf_eur"].fillna(0.0)
    population_values = pd.to_numeric(panel["population"], errors="coerce").astype(float)
    valid_population = population_values.gt(0).fillna(False).to_numpy()

    panel["erdf_eur_pc"] = np.where(
        valid_population,
        panel["erdf_eur"].astype(float).to_numpy() / population_values.to_numpy(),
        np.nan,
    )

    panel = panel.sort_values(KEY_COLUMNS).reset_index(drop=True)
    for lag in [1, 2, 3]:
        panel[f"erdf_eur_pc_l{lag}"] = panel.groupby("nuts2_id", sort=False)["erdf_eur_pc"].shift(lag)

    panel = merge_left_preserve_rows(panel, controls, "controls")

    panel = panel.merge(
        eligibility,
        on="nuts2_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )

    panel = panel.sort_values(KEY_COLUMNS).reset_index(drop=True)

    ordered_columns = [
        "nuts2_id",
        "country",
        "year",
        "population",
        "erdf_eur",
        "erdf_eur_pc",
        "erdf_eur_pc_l1",
        "erdf_eur_pc_l2",
        "erdf_eur_pc_l3",
        "gdp_mio_eur",
        "gdp_eur",
        "gdp_pc",
        "log_gdp_pc",
        "gdp_pc_growth",
        "gdp_pc_pps",
        "log_gdp_pc_pps",
        "gdp_pc_pps_growth",
        "gdp_volume_index",
        "gdp_pc_real",
        "log_gdp_pc_real",
        "gdp_pc_real_growth",
        "unemp_rate",
        "emp_rate",
        "tertiary_share_25_64",
        "rd_gerd",
        "gva",
        "category_2014_2020",
    ]

    panel = panel[ordered_columns]
    assert_unique_keys(panel, "panel_master")

    if panel["category_2014_2020"].notna().sum() == 0:
        raise ValueError(
            "category_2014_2020 is fully missing in panel_master. "
            f"Check {config.DATA_RAW_DIR / config.RAW_FILES['eligibility_categories']}."
        )

    return panel


def build_sigma_convergence(panel: pd.DataFrame) -> pd.DataFrame:
    sigma = (
        panel.groupby("year", as_index=False, sort=True)
        .agg(
            sigma_log_gdp_pps=("log_gdp_pc_pps", "std"),
            sigma_log_gdp_real=("log_gdp_pc_real", "std"),
            sigma_log_gdp_nominal=("log_gdp_pc", "std"),
            n_regions_pps=("log_gdp_pc_pps", "count"),
            n_regions_real=("log_gdp_pc_real", "count"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    return sigma


def build_data_quality_summary(panel: pd.DataFrame) -> pd.DataFrame:
    variables = [column for column in panel.columns if column not in ["nuts2_id", "country", "year"]]
    records = []

    for variable in variables:
        series = panel[variable]
        record = {
            "variable": variable,
            "dtype": str(series.dtype),
            "n_obs": len(series),
            "n_missing": int(series.isna().sum()),
            "missing_share": float(series.isna().mean()),
            "n_non_missing": int(series.notna().sum()),
            "min": np.nan,
            "median": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            record.update(
                {
                    "min": float(numeric.min()),
                    "median": float(numeric.median()),
                    "max": float(numeric.max()),
                    "mean": float(numeric.mean()),
                    "std": float(numeric.std()),
                }
            )

        records.append(record)

    summary = pd.DataFrame(records).sort_values("variable").reset_index(drop=True)
    return summary


def build_missingness_by_variable_year(panel: pd.DataFrame) -> pd.DataFrame:
    variables = [column for column in panel.columns if column not in ["nuts2_id", "country", "year"]]
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

    missingness = pd.concat(frames, ignore_index=True).sort_values(["variable", "year"])
    missingness = missingness.reset_index(drop=True)
    return missingness


def build_outliers_table(panel: pd.DataFrame) -> pd.DataFrame:
    sample = panel[["nuts2_id", "year", "erdf_eur", "erdf_eur_pc"]].copy()
    sample = sample[sample["erdf_eur_pc"].notna() & (sample["erdf_eur_pc"] > 0)].copy()

    if sample.empty:
        return pd.DataFrame(columns=["nuts2_id", "year", "erdf_eur", "erdf_eur_pc", "p99_erdf_eur_pc"])

    p99 = (
        sample.groupby("year", as_index=False, sort=True)["erdf_eur_pc"]
        .quantile(0.99)
        .rename(columns={"erdf_eur_pc": "p99_erdf_eur_pc"})
    )

    outliers = sample.merge(p99, on="year", how="left", sort=False)
    outliers = outliers[outliers["erdf_eur_pc"] >= outliers["p99_erdf_eur_pc"]]
    outliers = outliers.sort_values(["year", "erdf_eur_pc"], ascending=[True, False]).reset_index(drop=True)
    return outliers


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_missingness_heatmap(missingness_by_year: pd.DataFrame, output_path: Path) -> None:
    plt = _import_matplotlib()

    pivot = missingness_by_year.pivot(index="variable", columns="year", values="missing_share")
    pivot = pivot.sort_index().sort_index(axis=1)
    matrix = pivot.astype(float).to_numpy()

    fig_width = max(10, 0.45 * max(1, len(pivot.columns)))
    fig_height = max(6, 0.35 * max(1, len(pivot.index)))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=1)

    ax.set_title("Missingness by Variable and Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Variable")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), rotation=90)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Missing Share")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_distribution(series: pd.Series, title: str, xlabel: str, output_path: Path) -> None:
    plt = _import_matplotlib()
    values = pd.to_numeric(series, errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    if values.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(values, bins=50, color="#1f77b4", alpha=0.85, edgecolor="white")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_treated_vs_untreated_trends(panel: pd.DataFrame, output_path: Path) -> None:
    plt = _import_matplotlib()

    exposure = panel.groupby("nuts2_id", sort=True)["erdf_eur_pc"].mean()
    exposure = exposure.dropna()

    fig, ax = plt.subplots(figsize=(10, 6))

    if exposure.empty:
        ax.text(0.5, 0.5, "No treatment data available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    quantiles = exposure.quantile([0.25, 0.75])
    q25 = float(quantiles.loc[0.25])
    q75 = float(quantiles.loc[0.75])

    groups = pd.Series(index=exposure.index, data=pd.NA, dtype="string")
    groups.loc[exposure <= q25] = "Untreated (bottom quartile ERDF pc)"
    groups.loc[exposure >= q75] = "Treated (top quartile ERDF pc)"

    trend = panel.merge(groups.rename("group"), left_on="nuts2_id", right_index=True, how="left")
    trend = trend[trend["group"].notna()].copy()

    if trend.empty:
        ax.text(0.5, 0.5, "Insufficient treated/untreated data", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    trend_data = (
        trend.groupby(["year", "group"], as_index=False, sort=True)["gdp_pc"]
        .mean()
        .dropna(subset=["gdp_pc"])
    )

    for group_name, group_df in trend_data.groupby("group", sort=False):
        ax.plot(group_df["year"], group_df["gdp_pc"], marker="o", linewidth=2, label=group_name)

    ax.set_title("Mean Nominal GDP per Capita: Treated vs Untreated Quantiles")
    ax.set_xlabel("Year")
    ax.set_ylabel("Nominal GDP per capita (EUR)")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_growth_comparison(panel: pd.DataFrame, output_path: Path) -> None:
    plt = _import_matplotlib()

    columns = {
        "gdp_pc_growth": "Nominal growth",
        "gdp_pc_pps_growth": "PPS growth",
        "gdp_pc_real_growth": "Real growth",
    }

    available = [column for column in columns if column in panel.columns]
    trend = panel.groupby("year", sort=True)[available].mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    if trend.empty:
        ax.text(0.5, 0.5, "No growth data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        for column in available:
            ax.plot(trend.index, trend[column], marker="o", linewidth=2, label=columns[column])
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        ax.set_xlabel("Year")
        ax.set_ylabel("Average log-difference growth (x100)")
        ax.legend()
        ax.grid(alpha=0.2)

    ax.set_title("Nominal vs PPS vs Real GDP per capita growth trends")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.10g")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not write Parquet file. Install parquet support (e.g., pyarrow) and rerun."
        ) from exc


def build_dataset_pipeline(write_panel_csv: bool = True, fetch_missing_raw: bool = True) -> Dict[str, Path]:
    ensure_output_directories()
    ingest.ensure_v2_raw_inputs(fetch_missing=fetch_missing_raw)
    ensure_required_raw_files_exist()

    esif_message = check_esif_nuts2_usability()
    print(f"[build_dataset] {esif_message}")

    population = load_population()
    gdp_nominal = load_nominal_gdp_series()
    gdp_pps = load_gdp_pc_pps_series()
    gdp_real_index = load_gdp_real_index_series()
    pps_relative = load_pps_relative_eu_series()
    treatment = load_treatment_erdf()
    controls = build_controls()
    eligibility = load_eligibility_categories()
    running_variable = build_running_variable_eligibility(
        pps_relative=pps_relative,
        eligibility=eligibility,
    )

    skeleton = build_panel_skeleton([population, gdp_nominal, gdp_pps, gdp_real_index, treatment, controls])

    outcomes = build_outcomes(
        gdp_nominal=gdp_nominal,
        population=population,
        gdp_pps=gdp_pps,
        gdp_real_index=gdp_real_index,
    )

    panel = finalize_panel(
        skeleton=skeleton,
        population=population,
        outcomes=outcomes,
        treatment=treatment,
        controls=controls,
        eligibility=eligibility,
    )

    panel = panel.merge(
        running_variable[["nuts2_id", "r_value", "eligible_lt75", "ref_years_used"]],
        on="nuts2_id",
        how="left",
        validate="many_to_one",
    )

    erdf_cumulative = build_erdf_cumulative_exposure(panel)
    panel = panel.merge(
        erdf_cumulative,
        on=["nuts2_id", "country"],
        how="left",
        validate="many_to_one",
    )

    sigma = build_sigma_convergence(panel)

    # Interim outputs
    write_csv(skeleton[["nuts2_id", "year", "country"]], config.INTERIM_FILES["panel_skeleton"])

    treatment_interim = panel[
        [
            "nuts2_id",
            "year",
            "population",
            "erdf_eur",
            "erdf_eur_pc",
            "erdf_eur_pc_l1",
            "erdf_eur_pc_l2",
            "erdf_eur_pc_l3",
        ]
    ].copy()
    write_csv(treatment_interim, config.INTERIM_FILES["treatment_erdf"])

    write_csv(outcomes, config.INTERIM_FILES["outcomes_gdp"])
    write_csv(controls, config.INTERIM_FILES["controls"])
    write_csv(eligibility, config.INTERIM_FILES["eligibility_categories"])

    # Processed outputs
    write_parquet(panel, config.PANEL_MASTER_PARQUET)
    if write_panel_csv:
        write_csv(panel, config.PANEL_MASTER_CSV)
    write_csv(sigma, config.SIGMA_CONVERGENCE_CSV)
    write_csv(running_variable, config.RUNNING_VARIABLE_ELIGIBILITY_CSV)
    write_csv(erdf_cumulative, config.ERDF_CUMULATIVE_EXPOSURE_CSV)

    # QA tables
    quality_summary = build_data_quality_summary(panel)
    missingness_by_year = build_missingness_by_variable_year(panel)
    erdf_outliers = build_outliers_table(panel)

    write_csv(quality_summary, config.QA_TABLES["summary"])
    write_csv(missingness_by_year, config.QA_TABLES["missingness"])
    write_csv(erdf_outliers, config.QA_TABLES["outliers"])

    # QA figures
    plot_missingness_heatmap(missingness_by_year, config.QA_FIGURES["missingness_heatmap"])
    plot_distribution(
        panel["erdf_eur_pc"],
        "ERDF EUR per capita distribution",
        "ERDF EUR per capita",
        config.QA_FIGURES["erdf_distribution"],
    )
    plot_distribution(
        panel["gdp_pc"],
        "Nominal GDP per capita distribution",
        "Nominal GDP per capita (EUR)",
        config.QA_FIGURES["gdp_distribution"],
    )
    plot_distribution(
        panel["gdp_pc_pps"],
        "GDP per capita in PPS distribution",
        "GDP per capita (PPS)",
        config.QA_FIGURES["gdp_pps_distribution"],
    )
    plot_distribution(
        panel["gdp_pc_real"],
        "Real GDP per capita distribution (volume-index based)",
        "Real GDP per capita (2015-anchor units)",
        config.QA_FIGURES["gdp_real_distribution"],
    )
    plot_treated_vs_untreated_trends(panel, config.QA_FIGURES["treated_vs_untreated_trends"])
    plot_growth_comparison(panel, config.QA_FIGURES["growth_comparison_trend"])

    print(
        "[build_dataset] Built V3 panel_master, running-variable/cumulative exposure files, QA outputs, and sigma convergence."
    )

    return {
        "panel_master_parquet": config.PANEL_MASTER_PARQUET,
        "panel_master_csv": config.PANEL_MASTER_CSV,
        "sigma_convergence": config.SIGMA_CONVERGENCE_CSV,
        "running_variable_eligibility": config.RUNNING_VARIABLE_ELIGIBILITY_CSV,
        "erdf_cumulative_exposure": config.ERDF_CUMULATIVE_EXPOSURE_CSV,
        **{f"interim_{name}": path for name, path in config.INTERIM_FILES.items()},
        **{f"qa_table_{name}": path for name, path in config.QA_TABLES.items()},
        **{f"qa_figure_{name}": path for name, path in config.QA_FIGURES.items()},
    }
