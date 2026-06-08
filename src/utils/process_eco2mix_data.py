#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare regional eco2mix time series for the regional flexibility model.
"""

from __future__ import annotations

import logging
import os
import sys
import unicodedata
from datetime import datetime
from typing import Dict, Iterable

import matplotlib
import numpy as np
import pandas as pd
import yaml

# Add the project root to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_PALETTE = {
    "thermal": "#56B4E9",
    "nuclear": "#E69F00",
    "wind": "#009E73",
    "solar": "#F0E442",
    "hydro": "#0072B2",
    "biofuel": "#D55E00",
    "demand": "#000000",
}

BASE_OUTPUT_COLUMNS = [
    "demand",
    "thermal",
    "nuclear",
    "wind",
    "solar",
    "hydro",
    "biofuel",
]

OUTPUT_COLUMNS = BASE_OUTPUT_COLUMNS + ["ror", "hydro_flex", "residual_load", "hour", "month"]
ROR_LIKE_TECHS = {"fil de l eau", "eclusee"}

logger = logging.getLogger("process_eco2mix")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def format_title(label: str, region: str) -> str:
    """Return unified title with descriptor and region."""
    return f"{label} - {region}"


def load_palette(path: str | None) -> dict:
    """Load color palette from YAML file if provided."""
    palette = DEFAULT_PALETTE.copy()
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                user = yaml.safe_load(f) or {}
            user_palette = user.get("palette", user)
            palette.update({k: str(v) for k, v in user_palette.items()})
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to load palette %s: %s", path, exc)
    return palette


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_label(value: object) -> str:
    """Normalize labels across accents, punctuation, and spacing."""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    for old in ("'", "-", "_", "+", "/", "(", ")"):
        text = text.replace(old, " ")
    return " ".join(text.lower().split())


def _read_semicolon_csv(path: str) -> pd.DataFrame:
    """Read a semicolon-separated CSV with a few encoding fallbacks."""
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1", "ISO-8859-1"):
        try:
            return pd.read_csv(path, sep=";", encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to read {path}")


def _find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    """Find a column by normalized name among candidate spellings."""
    normalized_columns = {_normalize_label(col): col for col in df.columns}
    for candidate in candidates:
        match = normalized_columns.get(_normalize_label(candidate))
        if match is not None:
            return match
    if required:
        raise KeyError(f"Could not find any of columns: {list(candidates)}")
    return None


def _build_region_aliases(config: dict) -> dict[str, set[str]]:
    """Create normalized aliases from config region names and region_name_map."""
    def expand_names(raw_name: str) -> set[str]:
        names = {
            raw_name,
            raw_name.replace("_", " "),
            raw_name.replace("_", "-"),
            raw_name.replace("dAzur", "d Azur"),
            raw_name.replace("dAzur", "d'Azur"),
        }
        return {name for name in names if name}

    aliases: dict[str, set[str]] = {}
    region_name_map = config.get("region_name_map", {})

    for region in config.get("regions", []):
        names = set(expand_names(region))
        for source_name, target_name in region_name_map.items():
            if _normalize_label(target_name) == _normalize_label(region):
                names.update(expand_names(source_name))
        aliases[region] = {_normalize_label(name) for name in names}

    return aliases


def _build_time_index(year: int, resolution: str = "30min") -> pd.DatetimeIndex:
    """Create a canonical full-year half-hourly time index."""
    start = pd.Timestamp(f"{year}-01-01 00:00:00")
    end = pd.Timestamp(f"{year}-12-31 23:30:00")
    return pd.date_range(start=start, end=end, freq=resolution)


def _load_regional_raw(eco2mix_file: str, year: int) -> pd.DataFrame:
    """Load and standardize the regional eco2mix dataset."""
    df = _read_semicolon_csv(eco2mix_file)
    logger.info("Loaded %s rows from %s", len(df), eco2mix_file)

    column_map = {
        "region": _find_column(df, ["Région", "Region"]),
        "date": _find_column(df, ["Date"]),
        "time": _find_column(df, ["Heure"]),
        "demand": _find_column(df, ["Consommation (MW)", "Consommation"]),
        "thermal": _find_column(df, ["Thermique (MW)", "Thermique"]),
        "nuclear": _find_column(df, ["Nucléaire (MW)", "Nucleaire (MW)", "Nucléaire", "Nucleaire"]),
        "wind": _find_column(df, ["Eolien (MW)", "Éolien (MW)", "Eolien", "Éolien"]),
        "solar": _find_column(df, ["Solaire (MW)", "Solaire"]),
        "hydro": _find_column(df, ["Hydraulique (MW)", "Hydraulique"]),
        "biofuel": _find_column(df, ["Bioénergies (MW)", "Bioenergies (MW)", "Bioénergies", "Bioenergies"]),
    }

    timestamp = pd.to_datetime(
        df[column_map["date"]].astype(str).str.strip() + " " + df[column_map["time"]].astype(str).str.strip(),
        errors="coerce",
    )

    clean_df = pd.DataFrame(
        {
            "region": df[column_map["region"]].astype(str),
            "timestamp": timestamp,
            "demand": pd.to_numeric(df[column_map["demand"]], errors="coerce"),
            "thermal": pd.to_numeric(df[column_map["thermal"]], errors="coerce"),
            "nuclear": pd.to_numeric(df[column_map["nuclear"]], errors="coerce"),
            "wind": pd.to_numeric(df[column_map["wind"]], errors="coerce"),
            "solar": pd.to_numeric(df[column_map["solar"]], errors="coerce"),
            "hydro": pd.to_numeric(df[column_map["hydro"]], errors="coerce"),
            "biofuel": pd.to_numeric(df[column_map["biofuel"]], errors="coerce"),
        }
    )

    clean_df = clean_df.dropna(subset=["timestamp"]).copy()
    clean_df = clean_df[clean_df["timestamp"].dt.year == year].copy()
    clean_df["region_norm"] = clean_df["region"].map(_normalize_label)
    clean_df[BASE_OUTPUT_COLUMNS] = clean_df[BASE_OUTPUT_COLUMNS].fillna(0.0)

    return clean_df


def _load_national_ror_series(
    eco2mix_national_file: str | None,
    year: int,
    time_index: pd.DatetimeIndex,
    resolution: str,
) -> pd.Series:
    """Load the national 'fil de l'eau + éclusée' hydro time series."""
    zero_series = pd.Series(0.0, index=time_index, name="national_ror_like")
    if not eco2mix_national_file:
        return zero_series

    df = _read_semicolon_csv(eco2mix_national_file)
    ror_col = next(
        (
            col
            for col in df.columns
            if "hydraulique" in _normalize_label(col)
            and "fil de l eau" in _normalize_label(col)
            and "eclusee" in _normalize_label(col)
        ),
        None,
    )
    if ror_col is None:
        raise KeyError("Could not find national 'Hydraulique - Fil de l'eau + eclusee' column")

    date_col = _find_column(df, ["Date"])
    time_col = _find_column(df, ["Heure"])
    timestamp = pd.to_datetime(
        df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
        errors="coerce",
    )
    series = pd.Series(
        pd.to_numeric(df[ror_col], errors="coerce").to_numpy(),
        index=timestamp,
        name="national_ror_like",
    )
    series = series[series.index.notna()]
    series = series[series.index.year == year]
    series = series[series.notna()]
    series = series.groupby(level=0).mean().sort_index()
    series = series.resample(resolution).mean()
    series = series.reindex(time_index)
    series = series.interpolate(method="time", limit_direction="both")
    series = series.ffill().bfill()
    return series.fillna(0.0)


def _load_ror_capacities(
    odre_register_file: str | None,
    region_aliases: dict[str, set[str]],
) -> tuple[float, dict[str, float]]:
    """Load national and regional RoR-like capacities from the ODRÉ register."""
    if not odre_register_file:
        return 0.0, {region: 0.0 for region in region_aliases}

    df = _read_semicolon_csv(odre_register_file)
    region_col = _find_column(df, ["region", "Région"])
    tech_col = _find_column(df, ["technologie", "Technologie"])
    filiere_col = _find_column(df, ["filiere", "Filière"])
    power_col = _find_column(df, ["puisMaxInstallee"])

    frame = pd.DataFrame(
        {
            "region_norm": df[region_col].map(_normalize_label),
            "tech_norm": df[tech_col].map(_normalize_label),
            "filiere_norm": df[filiere_col].map(_normalize_label),
            "capacity_mw": pd.to_numeric(df[power_col], errors="coerce").fillna(0.0) / 1000.0,
        }
    )

    hydro_mask = frame["filiere_norm"] == "hydraulique"
    ror_mask = frame["tech_norm"].isin(ROR_LIKE_TECHS)
    national_capacity = float(frame.loc[hydro_mask & ror_mask, "capacity_mw"].sum())

    regional_capacities: dict[str, float] = {}
    for region, aliases in region_aliases.items():
        regional_mask = frame["region_norm"].isin(aliases)
        regional_capacities[region] = float(
            frame.loc[hydro_mask & ror_mask & regional_mask, "capacity_mw"].sum()
        )

    return national_capacity, regional_capacities


def _plot_region_statistics(region: str, data: pd.DataFrame, output_dir: str, palette: dict) -> None:
    """Save basic diagnostic plots for one region."""
    plot_df = data.reset_index().rename(columns={"index": "timestamp"})
    plot_df["hour"] = plot_df["timestamp"].dt.hour
    plot_df["month"] = plot_df["timestamp"].dt.month

    plt.figure(figsize=(15, 10))

    hourly_avg = plot_df.groupby("hour")[BASE_OUTPUT_COLUMNS].mean(numeric_only=True)
    plt.subplot(2, 2, 1)
    hourly_avg["demand"].plot(label="Demand", color=palette["demand"])
    plt.title(format_title("Average Daily Load Profile", region))
    plt.xlabel("Hour of Day")
    plt.ylabel("Power (MW)")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    tech_cols = ["thermal", "nuclear", "wind", "solar", "hydro", "biofuel"]
    hourly_avg[tech_cols].plot.area(
        stacked=True,
        color=[palette.get(t) for t in tech_cols],
    )
    plt.title(format_title("Technology Mix by Hour", region))
    plt.xlabel("Hour of Day")
    plt.ylabel("Power (MW)")
    plt.grid(True)

    monthly_avg = plot_df.groupby("month")[BASE_OUTPUT_COLUMNS].mean(numeric_only=True)
    plt.subplot(2, 2, 3)
    monthly_avg["demand"].plot(marker="o", color=palette["demand"])
    plt.title(format_title("Monthly Average Demand", region))
    plt.xlabel("Month")
    plt.ylabel("Power (MW)")
    plt.grid(True)
    plt.xticks(range(1, 13))

    plt.subplot(2, 2, 4)
    tech_total = plot_df[tech_cols].sum()
    tech_total.plot.pie(
        autopct="%1.1f%%",
        startangle=90,
        colors=[palette.get(t) for t in tech_cols],
    )
    plt.title(format_title("Technology Contribution", region))
    plt.axis("equal")

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_file = os.path.join(plot_dir, f"{region}_stats.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    logger.info("Saved statistics plot to %s", plot_file)


def process_eco2mix_data(
    eco2mix_file,
    output_dir,
    config,
    year=2022,
    palette_file=None,
    eco2mix_national_file: str | None = None,
    odre_register_file: str | None = None,
    output_suffix: str = "",
):
    """Process eco2mix data for the configured regions."""
    logger.info("Processing eco2mix data from %s for year %s", eco2mix_file, year)

    palette = load_palette(palette_file)
    os.makedirs(output_dir, exist_ok=True)

    regions = config.get("regions", [])
    if not regions:
        logger.error("No regions defined in config")
        return False

    resolution = ((config.get("time_settings") or {}).get("resolution")) or "30min"
    time_index = _build_time_index(year, resolution=resolution)
    region_aliases = _build_region_aliases(config)

    try:
        raw_regional = _load_regional_raw(eco2mix_file, year)
        national_ror = _load_national_ror_series(eco2mix_national_file, year, time_index, resolution)
        national_capacity, regional_ror_capacities = _load_ror_capacities(odre_register_file, region_aliases)

        logger.info("National RoR-like capacity: %.3f MW", national_capacity)
        for region in regions:
            logger.info("Regional RoR-like capacity for %s: %.3f MW", region, regional_ror_capacities.get(region, 0.0))

        regional_data: dict[str, pd.DataFrame] = {}
        if national_capacity > 0:
            national_cf = national_ror / national_capacity
        else:
            national_cf = pd.Series(0.0, index=time_index, name="national_ror_cf")

        for region in regions:
            region_mask = raw_regional["region_norm"].isin(region_aliases.get(region, {_normalize_label(region)}))
            region_df = raw_regional.loc[region_mask, ["timestamp"] + BASE_OUTPUT_COLUMNS].copy()
            if region_df.empty:
                logger.warning("No data found for region %s", region)
                continue

            region_df = region_df.groupby("timestamp", as_index=False).mean(numeric_only=True)
            region_df = region_df.set_index("timestamp").sort_index()
            region_df = region_df.resample(resolution).mean()
            region_df = region_df.reindex(time_index)
            region_df = region_df.interpolate(method="time", limit_direction="both")
            region_df = region_df.ffill().bfill()
            region_df[BASE_OUTPUT_COLUMNS] = region_df[BASE_OUTPUT_COLUMNS].fillna(0.0)

            raw_ror = national_cf * regional_ror_capacities.get(region, 0.0)
            region_df["ror"] = raw_ror.clip(lower=0.0)
            region_df["ror"] = pd.concat([region_df["ror"], region_df["hydro"]], axis=1).min(axis=1)
            region_df["hydro_flex"] = (region_df["hydro"] - region_df["ror"]).clip(lower=0.0)
            region_df["residual_load"] = (
                region_df["demand"] - region_df["solar"] - region_df["wind"] - region_df["ror"]
            )
            region_df["hour"] = region_df.index.hour.astype(int)
            region_df["month"] = region_df.index.month.astype(int)
            region_df = region_df[OUTPUT_COLUMNS]

            regional_data[region] = region_df
            logger.info("Prepared %s with %s periods", region, len(region_df))
            _plot_region_statistics(region, region_df, output_dir, palette)

        for region, data in regional_data.items():
            region_file = os.path.join(output_dir, f"{region}{output_suffix}.csv")
            data.to_csv(region_file)
            logger.info("Saved %s data to %s", region, region_file)

        time_index_name = f"time_index{output_suffix}.csv" if output_suffix else "time_index.csv"
        time_index_file = os.path.join(output_dir, time_index_name)
        pd.DataFrame({"timestamp": time_index}).to_csv(time_index_file, index=False)
        logger.info("Saved time index to %s", time_index_file)

        combined_data = {"time_index": time_index}
        combined_data.update(regional_data)
        return combined_data

    except Exception as e:  # pragma: no cover - surfaced to CLI
        logger.error("Error processing eco2mix data: %s", e)
        logger.exception("Detailed traceback")
        return False


def run_simulation(data, config_path, results_dir, time_period=None):
    """Run a small direct optimization using the current regional optimizer."""
    logger.info("Running regional optimizer simulation")
    os.makedirs(results_dir, exist_ok=True)

    from src.model.optimizer_regional_flex import RegionalFlexOptimizer

    try:
        optimizer = RegionalFlexOptimizer(config_path)
        regional_data = {k: v for k, v in data.items() if k != "time_index"}

        if time_period:
            start_time, end_time = time_period
            regional_data = {
                region: df.loc[start_time:end_time].copy()
                for region, df in regional_data.items()
            }

        optimizer.build_model(regional_data)
        status, _ = optimizer.solve()
        if status != 1:
            logger.error("Simulation failed with status %s", status)
            return False

        results = optimizer.get_results()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"regional_flex_results_{timestamp}.json")

        def convert_timestamps_in_dict(obj):
            if isinstance(obj, dict):
                return {
                    str(k) if isinstance(k, (pd.Timestamp, datetime)) else k: convert_timestamps_in_dict(v)
                    for k, v in obj.items()
                }
            if isinstance(obj, list):
                return [convert_timestamps_in_dict(item) for item in obj]
            if isinstance(obj, (pd.Timestamp, datetime)):
                return str(obj)
            if isinstance(obj, (np.int64, np.float64)):
                return float(obj)
            return obj

        import json

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(convert_timestamps_in_dict(results), f, indent=2)

        logger.info("Results saved to %s", results_file)
        return results
    except Exception as e:  # pragma: no cover - surfaced to CLI
        logger.error("Error in simulation: %s", e)
        logger.exception("Detailed traceback")
        return False


def main():
    """Main execution function."""
    logger.info("Starting eco2mix data processing and simulation")

    eco2mix_file = "Data/Raw/eco2mix-regional-cons-def_2023.csv"
    eco2mix_national_file = "Data/Raw/eco2mix-national-cons-def_2023.csv"
    odre_register_file = "Data/Raw/registre-national-installation-production-stockage-electricite-agrege-311223.csv"
    output_dir = "Data/processed"
    results_dir = "results/tech"
    config_path = "config/config_master.yaml"

    config = load_config(config_path)
    if not config:
        logger.error("Failed to load configuration")
        return False

    data = process_eco2mix_data(
        eco2mix_file,
        output_dir,
        config,
        year=2023,
        eco2mix_national_file=eco2mix_national_file,
        odre_register_file=odre_register_file,
        output_suffix="_2023",
    )
    if not data:
        logger.error("Failed to process eco2mix data")
        return False

    time_period = (
        data["time_index"][0],
        data["time_index"][min(len(data["time_index"]) - 1, 336)],
    )

    success = run_simulation(data, config_path, results_dir, time_period)
    if success:
        logger.info("Simulation completed successfully")
    else:
        logger.error("Simulation failed")

    return success


if __name__ == "__main__":
    main()
