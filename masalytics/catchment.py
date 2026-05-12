"""Shared helpers for the catchment / causal modelling pipeline."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DATA_DIR

CINEMA_GEOCODES_CSV = DATA_DIR / "cinema_geocodes.csv"
CINEMA_SA4_CSV = DATA_DIR / "cinema_sa4.csv"
SA4_DEMOGRAPHICS_CSV = DATA_DIR / "sa4_demographics.csv"
CATCHMENT_PANEL_PARQUET = DATA_DIR / "catchment_panel.parquet"
SA4_GEOJSON = DATA_DIR / "SA4_2021_AUST_GDA94_cleaned.geojson"
SA4_SHAPEFILE = DATA_DIR / "SA4_2021_AUST_GDA94.shp"
ABS_GCP_DIR = DATA_DIR / "abs_gcp_sa4"

SA4_CODE_COL_CANDIDATES = ["SA4_CODE_2021", "SA4_CODE21", "SA4_CODE"]
CINEMA_KEY = ["theatre_name", "city", "state"]


def sa4_geo_path():
    """Return the SA4 geometry source path, preferring the cleaned geojson if present."""
    if SA4_GEOJSON.is_file():
        return SA4_GEOJSON
    return SA4_SHAPEFILE


def load_geocode_cache() -> pd.DataFrame:
    """Return the existing geocode cache or an empty frame with the right schema."""
    cols = ["theatre_name", "city", "state", "lat", "lon", "confidence", "query_used"]
    if CINEMA_GEOCODES_CSV.is_file():
        df = pd.read_csv(CINEMA_GEOCODES_CSV)
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        return df[cols]
    return pd.DataFrame(columns=cols)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def unique_cinemas(sales_indian: pd.DataFrame) -> pd.DataFrame:
    """Distinct (theatre, city, state) triples present in Indian-film sales."""
    out = (
        sales_indian[CINEMA_KEY]
        .dropna(subset=["theatre_name"])
        .drop_duplicates()
        .sort_values(["state", "city", "theatre_name"])
        .reset_index(drop=True)
    )
    return out


def detect_sa4_columns(gdf) -> tuple[str, str]:
    """Return (code_col, name_col) for an SA4 GeoDataFrame, tolerant of ABS naming variants."""
    code_candidates = [c for c in gdf.columns if c.upper().startswith("SA4_CODE")]
    name_candidates = [c for c in gdf.columns if c.upper().startswith("SA4_NAME")]
    if not code_candidates or not name_candidates:
        raise RuntimeError(f"SA4 code/name columns not found. Have: {list(gdf.columns)}")
    return code_candidates[0], name_candidates[0]


def sa4_code_col(df: pd.DataFrame) -> str:
    """Return the first matching SA4 code column name in a plain DataFrame (ABS DataPacks)."""
    for c in SA4_CODE_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise RuntimeError(f"No SA4 code column in frame. Columns: {list(df.columns)[:10]}")


def load_sa4_geometry(target_crs: str = "EPSG:4326"):
    """Read the SA4 geometry, set source CRS to GDA94 if missing, and reproject to target_crs."""
    import geopandas as gpd  # local import: heavy optional dep

    sa4 = gpd.read_file(sa4_geo_path())
    if sa4.crs is None:
        sa4 = sa4.set_crs("EPSG:4283")
    return sa4.to_crs(target_crs)


def nearest_neighbor_km(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """For each point, return haversine distance in km to the nearest other point."""
    from sklearn.neighbors import BallTree

    if len(lats) < 2:
        return np.full(len(lats), np.nan)
    coords = np.radians(np.column_stack([lats, lons]))
    tree = BallTree(coords, metric="haversine")
    dists, _ = tree.query(coords, k=2)
    return dists[:, 1] * 6371.0088
