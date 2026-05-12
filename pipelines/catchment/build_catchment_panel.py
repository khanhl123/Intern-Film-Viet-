"""Stage 4 — Build the cinema-week modelling panel.

Joins:
    sales_indian (daily) → weekly aggregate
    → cinema_sa4.csv  (geography + nearest competitor)
    → sa4_demographics.csv (catchment features)
    → indian_titles (film-side controls, via normalized title)

Output: data/catchment_panel.parquet
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from masalytics.catchment import (
    CATCHMENT_PANEL_PARQUET,
    CINEMA_SA4_CSV,
    SA4_DEMOGRAPHICS_CSV,
)
from masalytics.data_loader import get_all_columns, load_data, normalize_title

MIN_GROSS_WEEK = 100.0  # drop closed / promo weeks
MIN_WEEKS_PER_FILM = 3


def _load_indian_titles(conn: sqlite3.Connection) -> pd.DataFrame:
    cols = get_all_columns(conn, "indian_titles")
    keep = [c for c in ["title", "distributor", "release_date", "opening_screens"] if c in cols]
    df = pd.read_sql(f"SELECT {', '.join(keep)} FROM indian_titles", conn)
    df["title_norm"] = normalize_title(df["title"])
    return df.drop_duplicates(subset="title_norm")


def main() -> None:
    if not CINEMA_SA4_CSV.is_file():
        raise SystemExit(f"Missing {CINEMA_SA4_CSV}. Run build_cinema_sa4.py first.")
    if not SA4_DEMOGRAPHICS_CSV.is_file():
        raise SystemExit(f"Missing {SA4_DEMOGRAPHICS_CSV}. Run build_sa4_demographics.py first.")

    bundle = load_data(verbose=True)
    sales = bundle.sales_indian.dropna(subset=["actual_sales_date", "theatre_name", "gross_today"]).copy()
    sales["gross_today"] = pd.to_numeric(sales["gross_today"], errors="coerce").fillna(0.0)

    iso = sales["actual_sales_date"].dt.isocalendar()
    sales["iso_year"] = iso["year"].astype(int)
    sales["iso_week"] = iso["week"].astype(int)
    sales["year_week"] = sales["iso_year"] * 100 + sales["iso_week"]

    weekly = (
        sales.groupby(
            ["numero_film_id", "theatre_name", "city", "state", "iso_year", "iso_week", "year_week"],
            as_index=False,
        )["gross_today"]
        .sum()
        .rename(columns={"gross_today": "gross_week"})
    )
    weekly = weekly[weekly["gross_week"] >= MIN_GROSS_WEEK].copy()

    first_week = weekly.groupby("numero_film_id")["year_week"].transform("min")
    weekly["rel_week"] = (weekly["year_week"] - first_week)  # used categorically; absolute week-gap is fine

    keep_films = (
        weekly.groupby("numero_film_id")["year_week"].nunique().loc[lambda s: s >= MIN_WEEKS_PER_FILM].index
    )
    weekly = weekly[weekly["numero_film_id"].isin(keep_films)]

    sa4_map = pd.read_csv(CINEMA_SA4_CSV)
    sa4_map["sa4_code"] = sa4_map["sa4_code"].astype(str)
    weekly = weekly.merge(
        sa4_map[["theatre_name", "city", "state", "sa4_code", "sa4_name", "nearest_competitor_km", "lat", "lon"]],
        on=["theatre_name", "city", "state"],
        how="left",
    )

    demo = pd.read_csv(SA4_DEMOGRAPHICS_CSV)
    demo["sa4_code"] = demo["sa4_code"].astype(str)
    weekly = weekly.merge(demo, on="sa4_code", how="left")

    titles = _load_indian_titles(bundle.conn)
    fm = bundle.film_meta.copy()
    fm["title_norm"] = normalize_title(fm["title"])
    film_features = fm.merge(titles.drop(columns=["title"]), on="title_norm", how="left")
    weekly = weekly.merge(
        film_features[["numero_film_id", "distributor", "release_date", "opening_screens"]],
        on="numero_film_id",
        how="left",
    )

    weekly["log_gross_week"] = np.log(weekly["gross_week"])
    weekly["median_income_k"] = weekly["median_household_income_weekly"] / 1000.0
    with np.errstate(divide="ignore"):
        weekly["log_pop_density"] = np.log(weekly["pop_density_km2"].replace(0, np.nan))

    n_unmapped_sa4 = weekly["sa4_code"].isna().sum()
    if n_unmapped_sa4:
        print(f"WARN: {n_unmapped_sa4} cinema-weeks have no SA4 match — dropping them.")
        weekly = weekly.dropna(subset=["sa4_code"])

    n_missing_demo = weekly["pct_indian_ancestry"].isna().sum()
    if n_missing_demo:
        print(f"WARN: {n_missing_demo} rows missing demographic features — dropping.")
        weekly = weekly.dropna(subset=["pct_indian_ancestry", "median_income_k", "median_age", "log_pop_density"])

    CATCHMENT_PANEL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(CATCHMENT_PANEL_PARQUET, index=False)
    print(f"Wrote {CATCHMENT_PANEL_PARQUET} ({len(weekly)} rows, {weekly['numero_film_id'].nunique()} films, {weekly['theatre_name'].nunique()} cinemas)")
    print("\nGross-week summary by state:")
    print(weekly.groupby("state")["gross_week"].agg(["count", "median", "sum"]).round(0))


if __name__ == "__main__":
    main()
