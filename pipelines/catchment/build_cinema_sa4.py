"""Stage 2 — Join cinemas to SA4 polygons and compute nearest-competitor distance."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import geopandas as gpd

from masalytics.catchment import (
    CINEMA_GEOCODES_CSV,
    CINEMA_SA4_CSV,
    detect_sa4_columns,
    load_sa4_geometry,
    nearest_neighbor_km,
)


def main() -> None:
    if not CINEMA_GEOCODES_CSV.is_file():
        raise SystemExit(f"Missing {CINEMA_GEOCODES_CSV}. Run build_cinema_geocodes.py first.")

    geo = pd.read_csv(CINEMA_GEOCODES_CSV)
    geo = geo.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    print(f"Loaded {len(geo)} geocoded cinemas")

    sa4 = load_sa4_geometry("EPSG:4326")
    code_col, name_col = detect_sa4_columns(sa4)

    cinemas_gdf = gpd.GeoDataFrame(
        geo,
        geometry=gpd.points_from_xy(geo["lon"], geo["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        cinemas_gdf,
        sa4[[code_col, name_col, "geometry"]],
        how="left",
        predicate="within",
    )
    unmatched = joined[joined[code_col].isna()]
    if len(unmatched) > 0:
        print(f"WARN: {len(unmatched)} cinemas fell outside any SA4 polygon")
        print(unmatched[["theatre_name", "city", "state", "lat", "lon"]].head(10))

    joined["nearest_competitor_km"] = nearest_neighbor_km(
        joined["lat"].to_numpy(), joined["lon"].to_numpy()
    )

    out = joined[
        [
            "theatre_name",
            "city",
            "state",
            "lat",
            "lon",
            code_col,
            name_col,
            "nearest_competitor_km",
        ]
    ].rename(columns={code_col: "sa4_code", name_col: "sa4_name"})

    out.to_csv(CINEMA_SA4_CSV, index=False)
    print(f"Wrote {CINEMA_SA4_CSV} ({len(out)} rows)")
    print("\nSA4 coverage (top 15):")
    print(out["sa4_name"].value_counts().head(15))


if __name__ == "__main__":
    main()
