# 🍿 Masalytics

**Competitive analysis of Indian film releases in Australia (2025), built for Film Viet Australia.**

Live report: https://khanhl123.github.io/Intern-Film-Viet-/

---

## What this is

Indian cinema is one of the biggest competitors for screens, marketing attention, and international audiences in Australia. This project tracks Indian film releases in 2025 and answers four questions for Film Viet Australia's release planning:

| Question | What it covers |
| --- | --- |
| **Q1 — Stable vs Volatile Locations** | Which cities/cinemas are safe bets vs unpredictable |
| **Q2 — Early Adopters vs Slow-burn** | Where audiences show up first vs where demand builds slowly |
| **Q3 — Seasonality** | Weekly highs and lows across the year by region |
| **Q4 — Catchment model (why)** | SA4-level model linking demand to demographics (Indian-ancestry share, income, density) |

Detailed write-ups live in [`docs/Q1_ANALYSIS_EXPLANATION.md`](docs/Q1_ANALYSIS_EXPLANATION.md), [`Q2`](docs/Q2_ANALYSIS_EXPLANATION.md), [`Q3`](docs/Q3_ANALYSIS_EXPLANATION.md).

---

## Data sources

| Source | What we use it for | Where it lives |
| --- | --- | --- |
| Numero (box-office) | Daily cinema-level gross for ~50 Indian films in 2025 | `data/numero_data.sqlite` (not in repo — see Setup) |
| ABS 2021 Census GCP (SA4) | Population, income, ancestry, country-of-birth per SA4 catchment | `data/abs_gcp_sa4/` |
| ABS SA4 shapefile (GDA94) | SA4 polygon geometries for spatial joins | `data/SA4_2021_AUST_GDA94.shp` |
| Nominatim (OpenStreetMap) | Geocoding cinema addresses → SA4 catchment | Cached to `data/cinema_geocodes.csv` |

---

## Setup

```bash
git clone https://github.com/khanhl123/Intern-Film-Viet-.git
cd Intern-Film-Viet-
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Place the Numero export at `data/numero_data.sqlite` (or set `MASALYTICS_DB=/path/to/db`).

Tested on Python 3.14.

---

## Repo layout

```
analysis/              Exploratory scripts (Q1–Q3 analyses, plotting)
pipelines/catchment/   5-stage causal pipeline (geocode → SA4 → demo → panel → OLS)
masalytics/            Shared package — paths, data loader, catchment helpers
notebooks/             Jupyter notebooks
data/                  Inputs (DB, ABS DataPack, SA4 shapefile, cached geocodes)
outputs/               Generated charts and reports
  ├── sales/           Q1 charts
  ├── titles/          Distributor analysis
  ├── location/        Q2 maps and HTML report
  └── catchment/       Q3 model outputs
docs/                  GitHub Pages site + analysis write-ups
```

---

## Running the analyses

All scripts run from the project root:

**Exploratory analyses** (Q1, Q2 location work, distributor charts):
```bash
python analysis/DataExplorationMain.py     # loads DB, builds shared dataframes
python analysis/SalesOverview.py           # → outputs/sales/
python analysis/TitlesDistributors.py      # → outputs/titles/
python analysis/LocationQuestions.py       # → outputs/location/
```

**Catchment model pipeline** (Q3, run in order):
```bash
python pipelines/catchment/build_cinema_geocodes.py    # 1. Geocode cinemas (cached)
python pipelines/catchment/build_cinema_sa4.py         # 2. Join to SA4 polygons
python pipelines/catchment/build_sa4_demographics.py   # 3. Extract ABS features
python pipelines/catchment/build_catchment_panel.py    # 4. Build modelling panel
python pipelines/catchment/fit_catchment_model.py      # 5. Fit OLS, write report
```

Each catchment stage writes a CSV/parquet under `data/` and `outputs/catchment/`. Later stages refuse to run if an earlier stage's output is missing.

---

## Headline finding (Q3 catchment model)

Holding the film, calendar week, and local competition constant, **a +5pp increase in the Indian-ancestry share of a cinema's SA4 catchment is associated with a measurable change in weekly gross** — the demand signal Q1's "safer vs volatile" labels were picking up indirectly. Full coefficients, effect curve, and residual analysis in [`outputs/catchment/catchment_section.html`](outputs/catchment/catchment_section.html).

---

## Roles

| Role | Responsibilities |
| --- | --- |
| **DS** | Sales benchmarks, timing models, catchment / clustering analysis |
| **DA/BA** | Data sourcing, cleaning, competitor insights, reporting |

---

## Project timeline

| Phase | Target |
| --- | --- |
| Week 1–2 | Collect & clean Indian-film AU cinema data |
| Week 3 | Identify distributor networks + regional strength |
| Week 4 | Analyse timing patterns + Film Viet overlap weeks |
| Week 5 | Build dashboards + structured insights |
| Week 6 | PDF report + screening strategy recommendations |
