"""Stage 3 — Extract per-SA4 demographic features from ABS 2021 GCP DataPacks.

Expected layout (after unzipping the DataPack):
    data/abs_gcp_sa4/
        2021Census_G01_AUST_SA4.csv   # selected person characteristics
        2021Census_G02_AUST_SA4.csv   # selected medians and averages
        2021Census_G08_AUST_SA4.csv   # ancestry by country of birth of parents (multi-part: A/B/C...)
        2021Census_G09A_AUST_SA4.csv  # country of birth of person (multi-part)
        ...

The script scans the directory and matches files by their Gxx table code, so
exact filename variations from the ABS website are tolerated.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from masalytics.catchment import (
    ABS_GCP_DIR,
    SA4_CODE_COL_CANDIDATES,
    SA4_DEMOGRAPHICS_CSV,
    load_sa4_geometry,
    sa4_code_col,
)


def _find_table_files(gcp_dir: Path, table_code: str) -> list[Path]:
    pattern = re.compile(rf"_{table_code}[A-Z]?_.*SA4.*\.csv$", re.IGNORECASE)
    return sorted(p for p in gcp_dir.glob("*.csv") if pattern.search(p.name))


def _load_concat(files: list[Path]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f) for f in files]
    merged = frames[0]
    for df in frames[1:]:
        join_col = next(
            (c for c in SA4_CODE_COL_CANDIDATES if c in merged.columns and c in df.columns),
            None,
        )
        if join_col is None:
            merged = pd.concat([merged, df], axis=1)
        else:
            merged = merged.merge(df, on=join_col, how="outer")
    return merged


def _pick_column(df: pd.DataFrame, patterns: list[str]) -> str | None:
    """Return the first column whose name matches any of the regex patterns (case-insensitive)."""
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in df.columns:
            if rx.search(c):
                return c
    return None


def _sa4_areas_km2() -> pd.DataFrame:
    """Compute SA4 polygon areas in km² from the local geojson."""
    sa4_eq = load_sa4_geometry("EPSG:3577")  # equal-area projection for Australia
    code_col = next((c for c in sa4_eq.columns if c.upper().startswith("SA4_CODE")), None)
    if code_col is None:
        raise RuntimeError("SA4 code column missing from geojson")
    return pd.DataFrame(
        {
            "sa4_code": sa4_eq[code_col].astype(str),
            "area_km2": sa4_eq.geometry.area.to_numpy() / 1e6,
        }
    )


def main() -> None:
    if not ABS_GCP_DIR.is_dir():
        raise SystemExit(
            f"Missing {ABS_GCP_DIR}.\n"
            "Download ABS 2021 Census DataPack → General Community Profile (SA4)\n"
            "from https://www.abs.gov.au/census/find-census-data/datapacks and unzip it there."
        )

    # G01 — total population (Tot_P_P)
    g01 = _load_concat(_find_table_files(ABS_GCP_DIR, "G01"))
    if g01.empty:
        raise SystemExit("G01 (selected person characteristics) not found in DataPack dir.")
    code = sa4_code_col(g01)
    pop_col = _pick_column(g01, [r"^Tot_P_P$", r"Tot.*P.*Persons", r"Total.*Persons"])
    if pop_col is None:
        print("WARN: total-persons column not auto-detected from G01; inspect columns:", file=sys.stderr)
        print(list(g01.columns)[:30], file=sys.stderr)
        raise SystemExit(2)
    pop = g01[[code, pop_col]].rename(columns={code: "sa4_code", pop_col: "total_pop"})

    # G02 — medians
    g02 = _load_concat(_find_table_files(ABS_GCP_DIR, "G02"))
    code2 = sa4_code_col(g02)
    med_age = _pick_column(g02, [r"Median_age_persons", r"Median.*age"])
    med_inc = _pick_column(g02, [r"Median.*tot.*hhd.*inc.*weekly", r"Median.*household.*income"])
    medians = g02[[code2, med_age, med_inc]].rename(
        columns={code2: "sa4_code", med_age: "median_age", med_inc: "median_household_income_weekly"}
    )

    def _extract_indian_count(table_code: str, out_col: str, patterns: list[str], missing_warn: str) -> pd.DataFrame:
        gxx = _load_concat(_find_table_files(ABS_GCP_DIR, table_code))
        if gxx.empty:
            print(f"WARN: {table_code} missing; {out_col} will be NaN", file=sys.stderr)
            return pd.DataFrame(columns=["sa4_code", out_col])
        code = sa4_code_col(gxx)
        col = _pick_column(gxx, patterns)
        if col is None:
            print(f"WARN: {missing_warn}", file=sys.stderr)
            return pd.DataFrame({"sa4_code": gxx[code].astype(str), out_col: np.nan})
        return gxx[[code, col]].rename(columns={code: "sa4_code", col: out_col})

    cob = _extract_indian_count(
        "G09", "india_born",
        [r"India_Total_Persons", r"Birthplace_India.*Persons", r"India.*Persons"],
        "India birthplace column not found; pct_india_born NaN",
    )
    anc = _extract_indian_count(
        "G08", "indian_ancestry",
        [r"Indian.*Persons", r"Ancestry.*Indian.*Persons", r"^Indian_"],
        "Indian ancestry column not found; pct_indian_ancestry NaN",
    )

    for df in (pop, medians, cob, anc):
        df["sa4_code"] = df["sa4_code"].astype(str)

    out = pop.merge(medians, on="sa4_code", how="left")
    out = out.merge(cob, on="sa4_code", how="left")
    out = out.merge(anc, on="sa4_code", how="left")

    out["pct_india_born"] = 100.0 * out["india_born"] / out["total_pop"]
    out["pct_indian_ancestry"] = 100.0 * out["indian_ancestry"] / out["total_pop"]

    areas = _sa4_areas_km2()
    out = out.merge(areas, on="sa4_code", how="left")
    out["pop_density_km2"] = out["total_pop"] / out["area_km2"]

    cols = [
        "sa4_code",
        "total_pop",
        "median_age",
        "median_household_income_weekly",
        "pct_india_born",
        "pct_indian_ancestry",
        "area_km2",
        "pop_density_km2",
    ]
    out = out[cols].sort_values("pct_indian_ancestry", ascending=False).reset_index(drop=True)
    out.to_csv(SA4_DEMOGRAPHICS_CSV, index=False)
    print(f"Wrote {SA4_DEMOGRAPHICS_CSV} ({len(out)} SA4s)")
    print("\nTop 10 SA4s by Indian ancestry %:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
