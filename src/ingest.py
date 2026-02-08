from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src import config

EUROSTAT_BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


def _download_json(url: str, timeout_seconds: int = 180) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "EUcohesion-V2/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _decode_eurostat_json(dataset: dict) -> pd.DataFrame:
    ids = dataset.get("id", [])
    sizes = dataset.get("size", [])
    values = dataset.get("value", {})

    if not ids or not sizes:
        raise ValueError("Unexpected Eurostat response: missing dimension metadata.")

    category_codes_by_position: List[List[str]] = []
    for dim, size in zip(ids, sizes):
        category_index = dataset["dimension"][dim]["category"]["index"]
        codes = [None] * int(size)
        for code, position in category_index.items():
            codes[int(position)] = code
        category_codes_by_position.append(codes)

    rows: List[Dict[str, object]] = []
    for flat_index_string, obs_value in values.items():
        flat_index = int(flat_index_string)
        coordinates: List[int] = []

        for size in reversed(sizes):
            size_int = int(size)
            coordinates.append(flat_index % size_int)
            flat_index //= size_int
        coordinates.reverse()

        row = {
            dim: category_codes_by_position[idx][coord]
            for idx, (dim, coord) in enumerate(zip(ids, coordinates))
        }
        row["obs_value"] = obs_value
        rows.append(row)

    decoded = pd.DataFrame(rows)
    if decoded.empty:
        raise ValueError("Decoded Eurostat dataset is empty.")

    return decoded


def _fetch_eurostat_dataset(dataset_code: str, params: Dict[str, str]) -> pd.DataFrame:
    query = urllib.parse.urlencode(params)
    url = f"{EUROSTAT_BASE_URL}/{dataset_code}?{query}"

    try:
        response = _download_json(url)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to download Eurostat dataset from {url}") from exc

    return _decode_eurostat_json(response)


def _standardize_raw_eurostat_output(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in output.columns:
        if output[column].dtype == object:
            output[column] = output[column].astype("string").str.strip()
    output = output.sort_values(output.columns.tolist()).reset_index(drop=True)
    return output


def _is_nuts2(code_series: pd.Series) -> pd.Series:
    normalized = code_series.astype("string").str.strip().str.upper()
    mask = normalized.str.fullmatch(config.NUTS2_REGEX, na=False)
    mask &= ~normalized.str.endswith(config.UNKNOWN_NUTS2_SUFFIXES)
    return mask


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")


def download_gdp_pc_pps_nuts2(output_path: Path) -> None:
    dataset = _fetch_eurostat_dataset(
        dataset_code="nama_10r_2gdp",
        params={"freq": "A", "unit": "PPS_EU27_2020_HAB"},
    )
    dataset = _standardize_raw_eurostat_output(dataset)
    _write_csv(dataset, output_path)


def download_gdp_real_nuts2(output_path: Path) -> None:
    dataset = _fetch_eurostat_dataset(
        dataset_code="nama_10r_2gvagr",
        params={"freq": "A", "na_item": "B1GQ", "unit": "I15"},
    )
    dataset = _standardize_raw_eurostat_output(dataset)
    _write_csv(dataset, output_path)


def download_gdp_pc_pps_relative_eu_nuts2(output_path: Path) -> None:
    dataset = _fetch_eurostat_dataset(
        dataset_code="nama_10r_2gdp",
        params={"freq": "A", "unit": "PPS_HAB_EU27_2020"},
    )
    dataset = _standardize_raw_eurostat_output(dataset)
    _write_csv(dataset, output_path)


def build_eligibility_categories(output_path: Path) -> None:
    ratio_path = config.DATA_RAW_DIR / config.RAW_FILES["gdp_pc_pps_rel_eu"]
    if ratio_path.exists():
        ratio_dataset = pd.read_csv(ratio_path, dtype=str)
    else:
        ratio_dataset = _fetch_eurostat_dataset(
            dataset_code="nama_10r_2gdp",
            params={"freq": "A", "unit": "PPS_HAB_EU27_2020"},
        )

    ratio_dataset = ratio_dataset.rename(columns={"geo": "nuts2_id", "time": "year"})
    ratio_dataset["nuts2_id"] = ratio_dataset["nuts2_id"].astype("string").str.strip().str.upper()
    ratio_dataset = ratio_dataset[_is_nuts2(ratio_dataset["nuts2_id"])].copy()

    ratio_dataset["year"] = pd.to_numeric(ratio_dataset["year"], errors="coerce")
    ratio_dataset["pps_eu_rel"] = pd.to_numeric(ratio_dataset["obs_value"], errors="coerce")
    ratio_dataset = ratio_dataset.dropna(subset=["year", "pps_eu_rel"]).copy()
    ratio_dataset["year"] = ratio_dataset["year"].astype(int)

    reference_years = [2007, 2008, 2009]

    reference_average = (
        ratio_dataset[ratio_dataset["year"].isin(reference_years)]
        .groupby("nuts2_id", as_index=True)["pps_eu_rel"]
        .mean()
    )

    # Fallback for regions that do not have all reference years available.
    fallback_average = (
        ratio_dataset[ratio_dataset["year"] <= 2013]
        .groupby("nuts2_id", as_index=True)["pps_eu_rel"]
        .mean()
    )

    combined = reference_average.combine_first(fallback_average)

    categories = pd.Series(index=combined.index, dtype="string")
    categories.loc[combined < 75.0] = "less_developed"
    categories.loc[(combined >= 75.0) & (combined < 90.0)] = "transition"
    categories.loc[combined >= 90.0] = "more_developed"

    mapping = (
        pd.DataFrame(
            {
                "nuts2_id": combined.index,
                "category_2014_2020": categories.values,
            }
        )
        .dropna(subset=["category_2014_2020"])
        .sort_values("nuts2_id")
        .reset_index(drop=True)
    )

    if mapping.empty:
        raise ValueError(
            "Eligibility category reconstruction produced an empty mapping. "
            "Could not compute categories from Eurostat PPS relative-to-EU series."
        )

    _write_csv(mapping[["nuts2_id", "category_2014_2020"]], output_path)


def ensure_v2_raw_inputs(fetch_missing: bool = True) -> None:
    builders = {
        config.RAW_FILES["gdp_pc_pps"]: download_gdp_pc_pps_nuts2,
        config.RAW_FILES["gdp_real"]: download_gdp_real_nuts2,
        config.RAW_FILES["gdp_pc_pps_rel_eu"]: download_gdp_pc_pps_relative_eu_nuts2,
        config.RAW_FILES["eligibility_categories"]: build_eligibility_categories,
    }

    missing = [
        file_name
        for file_name in config.V2_DOWNLOAD_TARGETS
        if not (config.DATA_RAW_DIR / file_name).exists()
    ]

    if missing and not fetch_missing:
        formatted = "\n".join(f"  - {name}" for name in missing)
        raise FileNotFoundError(
            "Missing V2 cached raw inputs in data/raw/.\n"
            "Run with fetch_missing=True once to download official Eurostat inputs:\n"
            f"{formatted}"
        )

    for file_name in missing:
        output_path = config.DATA_RAW_DIR / file_name
        builder = builders[file_name]
        builder(output_path)

    still_missing = [
        file_name
        for file_name in config.V2_DOWNLOAD_TARGETS
        if not (config.DATA_RAW_DIR / file_name).exists()
    ]
    if still_missing:
        formatted = "\n".join(f"  - {name}" for name in still_missing)
        raise FileNotFoundError(
            "Failed to prepare required V2 raw files in data/raw/:\n"
            f"{formatted}"
        )
