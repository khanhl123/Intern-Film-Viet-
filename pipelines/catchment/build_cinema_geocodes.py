"""Stage 1 — Geocode every unique cinema via Nominatim with a local cache.

Re-running is cheap: only cinemas missing from data/cinema_geocodes.csv hit the API.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

from masalytics.catchment import (
    CINEMA_GEOCODES_CSV,
    load_geocode_cache,
    unique_cinemas,
)
from masalytics.data_loader import load_data

USER_AGENT = "Masalytics/0.1 (khanhlieu38@gmail.com)"
RATE_LIMIT_SEC = 1.1

STATE_NORMALISE = {
    "New South Wales (inc ACT)": "New South Wales",
    "NSW": "New South Wales",
    "VIC": "Victoria",
    "QLD": "Queensland",
    "SA": "South Australia",
    "WA": "Western Australia",
    "TAS": "Tasmania",
    "NT": "Northern Territory",
    "ACT": "Australian Capital Territory",
}

# Cities that are actually in ACT but Numero groups under NSW
ACT_CITIES = {"Canberra"}


def _clean_theatre(theatre: str) -> str:
    """Strip trailing screen-count digits and stray punctuation."""
    s = re.sub(r"\s*[&,]\s*\d{1,3}\s*$", "", theatre)
    s = re.sub(r"\s+\d{1,3}\s*$", "", s)
    return s.strip()


def _clean_state(state: str | None, city: str | None) -> str | None:
    if pd.isna(state):
        return None
    if pd.notna(city) and city in ACT_CITIES:
        return "Australian Capital Territory"
    return STATE_NORMALISE.get(state, state)


def _query_variants(theatre: str, city: str | None, state: str | None) -> list[str]:
    cleaned = _clean_theatre(theatre)
    clean_state = _clean_state(state, city)
    variants = []
    if pd.notna(city) and clean_state:
        variants.append(f"{cleaned}, {city}, {clean_state}, Australia")
    if clean_state:
        variants.append(f"{cleaned}, {clean_state}, Australia")
    if pd.notna(city):
        variants.append(f"{cleaned}, {city}, Australia")
    variants.append(f"{cleaned}, Australia")
    return list(dict.fromkeys(variants))


def _geocode_one(geocoder: Nominatim, theatre: str, city, state) -> dict:
    for query in _query_variants(theatre, city, state):
        try:
            loc = geocoder.geocode(query, country_codes="au", timeout=15)
        except (GeocoderTimedOut, GeocoderServiceError) as exc:
            print(f"  warn: {exc} on {query!r}", file=sys.stderr)
            loc = None
        time.sleep(RATE_LIMIT_SEC)
        if loc is not None:
            confidence = float(loc.raw.get("importance", 0.0)) if hasattr(loc, "raw") else 0.0
            return {
                "lat": loc.latitude,
                "lon": loc.longitude,
                "confidence": confidence,
                "query_used": query,
            }
    return {"lat": None, "lon": None, "confidence": None, "query_used": None}


def main() -> None:
    bundle = load_data(verbose=True)
    cinemas = unique_cinemas(bundle.sales_indian)
    print(f"Unique cinemas to resolve: {len(cinemas)}")

    cache = load_geocode_cache()
    # Only treat *resolved* rows as cached; unresolved rows get retried with the cleanup logic.
    resolved_cache = cache[cache["lat"].notna()]
    cached_keys = set(
        zip(
            resolved_cache["theatre_name"].astype(str),
            resolved_cache["city"].astype(str),
            resolved_cache["state"].astype(str),
        )
    )
    # Drop unresolved rows from cache so they don't reappear as duplicates after merge.
    cache = resolved_cache

    key_tuples = list(
        zip(
            cinemas["theatre_name"].astype(str),
            cinemas["city"].astype(str),
            cinemas["state"].astype(str),
        )
    )
    todo = cinemas[[t not in cached_keys for t in key_tuples]].reset_index(drop=True)
    print(f"Cached: {len(cinemas) - len(todo)} | Need geocoding: {len(todo)}")

    if todo.empty:
        print("Nothing to do. Cache is complete.")
        return

    geocoder = Nominatim(user_agent=USER_AGENT, timeout=15)
    new_rows: list[dict] = []
    for i, row in todo.iterrows():
        result = _geocode_one(geocoder, row["theatre_name"], row["city"], row["state"])
        new_rows.append(
            {
                "theatre_name": row["theatre_name"],
                "city": row["city"],
                "state": row["state"],
                **result,
            }
        )
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(todo)} done")

    merged = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
    merged = merged.drop_duplicates(
        subset=["theatre_name", "city", "state"], keep="last"
    ).reset_index(drop=True)

    CINEMA_GEOCODES_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(CINEMA_GEOCODES_CSV, index=False)

    resolved = merged["lat"].notna().sum()
    print(f"Wrote {CINEMA_GEOCODES_CSV} | resolved {resolved}/{len(merged)}")


if __name__ == "__main__":
    main()
