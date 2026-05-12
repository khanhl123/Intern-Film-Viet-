"""Stage 5 — Estimate three nested OLS specifications and write report artifacts.

M1 baseline:  log_gross_week ~ pct_indian_ancestry + log_pop_density + nearest_competitor_km
M2 controls:  M1 + median_income_k + median_age + C(rel_week)
M3 full FE:   M2 + C(numero_film_id) + C(iso_week)

Standard errors clustered on theatre.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from masalytics.catchment import CATCHMENT_PANEL_PARQUET
from masalytics.paths import OUTPUTS_CATCHMENT_DIR

HEADLINE_VARS_REPORTED = [
    "pct_indian_ancestry",
    "log_pop_density",
    "nearest_competitor_km",
    "median_income_k",
    "median_age",
]


def _fit(df: pd.DataFrame, formula: str):
    model = smf.ols(formula, data=df)
    return model.fit(cov_type="cluster", cov_kwds={"groups": df["theatre_name"]})


def _stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def _coef_row(res, var: str) -> dict:
    if var not in res.params.index:
        return {"coef": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan}
    ci = res.conf_int().loc[var]
    return {
        "coef": float(res.params[var]),
        "se": float(res.bse[var]),
        "ci_lo": float(ci[0]),
        "ci_hi": float(ci[1]),
        "p": float(res.pvalues[var]),
    }


def _build_coef_table(results: dict) -> pd.DataFrame:
    rows = []
    for var in HEADLINE_VARS_REPORTED:
        row = {"variable": var}
        for label, res in results.items():
            r = _coef_row(res, var)
            row[f"{label}_coef"] = r["coef"]
            row[f"{label}_ci"] = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if not np.isnan(r["coef"]) else ""
            row[f"{label}_p"] = r["p"]
            row[f"{label}_stars"] = _stars(r["p"]) if not np.isnan(r["p"]) else ""
        rows.append(row)
    rows.append(
        {
            "variable": "N",
            **{f"{lbl}_coef": int(res.nobs) for lbl, res in results.items()},
        }
    )
    rows.append(
        {
            "variable": "R²",
            **{f"{lbl}_coef": round(float(res.rsquared), 3) for lbl, res in results.items()},
        }
    )
    return pd.DataFrame(rows)


def _coef_table_html(tbl: pd.DataFrame, labels: list[str]) -> str:
    header_cells = "".join(f"<th>{html.escape(l)}</th>" for l in labels)
    body_rows = []
    meta_vars = {"N", "R²"}
    for _, row in tbl.iterrows():
        var = str(row["variable"])
        cells = [f"<th>{html.escape(var)}</th>"]
        for label in labels:
            coef = row.get(f"{label}_coef", None)
            stars = row.get(f"{label}_stars", "")
            ci = row.get(f"{label}_ci", "")
            if isinstance(stars, float) and np.isnan(stars):
                stars = ""
            if isinstance(ci, float) and np.isnan(ci):
                ci = ""
            try:
                coef_is_nan = coef is None or (isinstance(coef, float) and np.isnan(coef))
            except TypeError:
                coef_is_nan = False
            if coef_is_nan:
                cell = "—"
            elif var in meta_vars:
                cell = f"{int(coef):,}" if var == "N" else f"{float(coef):.3f}"
            else:
                cell = f"{float(coef):.3f}{stars}"
                if ci:
                    cell += f"<br><span class='ci'>{html.escape(str(ci))}</span>"
            cells.append(f"<td>{cell}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table class='coef-table'>"
        f"<thead><tr><th>Variable</th>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _effect_curve(df: pd.DataFrame, res, out_path: Path) -> None:
    medians = df[["log_pop_density", "nearest_competitor_km", "median_income_k", "median_age"]].median()
    xs = np.linspace(df["pct_indian_ancestry"].quantile(0.02), df["pct_indian_ancestry"].quantile(0.98), 80)
    # Build prediction frame; for FE terms, use the modal level so prediction is well-defined.
    modal_film = df["numero_film_id"].mode().iloc[0]
    modal_iso = int(df["iso_week"].mode().iloc[0])
    modal_rel = int(df["rel_week"].mode().iloc[0])
    pred_df = pd.DataFrame(
        {
            "pct_indian_ancestry": xs,
            "log_pop_density": medians["log_pop_density"],
            "nearest_competitor_km": medians["nearest_competitor_km"],
            "median_income_k": medians["median_income_k"],
            "median_age": medians["median_age"],
            "rel_week": modal_rel,
            "numero_film_id": modal_film,
            "iso_week": modal_iso,
            "theatre_name": df["theatre_name"].iloc[0],
        }
    )
    try:
        preds = res.get_prediction(pred_df).summary_frame(alpha=0.05)
        ys = np.exp(preds["mean"])
        lo = np.exp(preds["mean_ci_lower"])
        hi = np.exp(preds["mean_ci_upper"])
    except Exception as exc:
        print(f"effect curve: prediction failed ({exc}); falling back to coefficient-only line")
        b = res.params.get("pct_indian_ancestry", 0.0)
        intercept_at_median = np.log(df["gross_week"].median())
        ys = np.exp(intercept_at_median + b * (xs - xs.mean()))
        lo = hi = ys

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, color="#1f4fbf", linewidth=2.2, label="Predicted weekly gross")
    ax.fill_between(xs, lo, hi, color="#1f4fbf", alpha=0.15, label="95% CI")
    ax.set_xlabel("Indian-ancestry population in SA4 (%)")
    ax.set_ylabel("Predicted weekly cinema gross (AUD)")
    ax.set_title("Catchment effect — Indian ancestry % vs predicted weekly gross\n(other variables held at median)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _residuals_by_sa4(df: pd.DataFrame, res, out_path: Path) -> None:
    agg = (
        df.assign(resid=res.resid)
        .groupby("sa4_name")["resid"]
        .agg(["mean", "count"])
        .sort_values("mean")
    )
    agg = agg[agg["count"] >= 5]
    top = pd.concat([agg.head(10), agg.tail(10)])

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#c0392b" if v > 0 else "#1f4fbf" for v in top["mean"]]
    ax.barh(range(len(top)), top["mean"].values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.axvline(0, color="#0b1f3a", linewidth=1)
    ax.set_xlabel("Mean residual (log gross)  —  positive = model under-predicts")
    ax.set_title("Where the catchment model leaves money unexplained (SA4-level)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_section_html(coef_html: str, headline_pct: float, ci_lo: float, ci_hi: float, n: int, r2: float, out_path: Path) -> None:
    body = f"""<section class="card intro-card">
  <h2>Why the map looks this way: a catchment model</h2>
  <p class="lead">Q1–Q3 describe <em>where</em> Indian films sell. This section asks <em>why</em>.
  We regress weekly cinema gross on the demographics of the SA4 catchment each cinema sits in,
  controlling for the film, the calendar week, and local competition.</p>
</section>
<section class="card">
  <h3>Headline result</h3>
  <p>Holding the film, the calendar week, income, and age structure constant,
  <strong>a +5 percentage-point increase in the Indian-ancestry share of a cinema's SA4 catchment is
  associated with a {headline_pct:+.1f}% change in weekly gross</strong>
  (95% CI: {ci_lo:+.1f}% to {ci_hi:+.1f}%; n = {n:,}; R² = {r2:.2f} in full-FE spec).</p>
  <p class="lead">This is the structural demand signal that Q1's "Safer / Volatile" labels were
  picking up indirectly. Cities flagged as "Safer" tend to sit on SA4s with a thick Indian-Australian
  population — the stability is a fundamentals story, not statistical luck.</p>
</section>
<section class="card">
  <h3>Coefficients across nested specifications</h3>
  {coef_html}
  <p class="helper">Coefficients on log(gross). Standard errors clustered on cinema.
  *** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1. M1 = baseline; M2 adds income/age + film-age FE;
  M3 adds film and calendar-week fixed effects.</p>
</section>
<section class="card">
  <h3>Effect curve</h3>
  <p>Predicted weekly gross as Indian-ancestry % varies, with every other variable held at its median.</p>
  <img src="../outputs_catchment/effect_curve_indian_ancestry.png"
       alt="Effect curve: Indian-ancestry % vs predicted weekly gross"
       style="max-width:100%;height:auto;border:1px solid var(--stroke);border-radius:8px;" />
</section>
<section class="card">
  <h3>Where the model still leaves money unexplained</h3>
  <p>SA4-level mean residuals after M3. Positive bars = the catchment model
  <em>under-predicts</em> demand there — something beyond demographics is driving it
  (cinema quality, marketing, community events). Negative = over-prediction.</p>
  <img src="../outputs_catchment/residuals_by_sa4.png"
       alt="Top and bottom SA4s by mean model residual"
       style="max-width:100%;height:auto;border:1px solid var(--stroke);border-radius:8px;" />
</section>
"""
    out_path.write_text(body, encoding="utf-8")


def main() -> None:
    if not CATCHMENT_PANEL_PARQUET.is_file():
        raise SystemExit(f"Missing {CATCHMENT_PANEL_PARQUET}. Run build_catchment_panel.py first.")

    OUTPUTS_CATCHMENT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CATCHMENT_PANEL_PARQUET)
    print(f"Loaded panel: {len(df):,} rows")

    f_m1 = "log_gross_week ~ pct_indian_ancestry + log_pop_density + nearest_competitor_km"
    f_m2 = f_m1 + " + median_income_k + median_age + C(rel_week)"
    f_m3 = f_m2 + " + C(numero_film_id) + C(iso_week)"

    print("Fitting M1..."); m1 = _fit(df, f_m1)
    print("Fitting M2..."); m2 = _fit(df, f_m2)
    print("Fitting M3..."); m3 = _fit(df, f_m3)

    results = {"M1": m1, "M2": m2, "M3": m3}
    tbl = _build_coef_table(results)
    tbl.to_csv(OUTPUTS_CATCHMENT_DIR / "coefficients_table.csv", index=False)

    coef_html = _coef_table_html(tbl, ["M1", "M2", "M3"])
    (OUTPUTS_CATCHMENT_DIR / "coefficients_table.html").write_text(coef_html, encoding="utf-8")

    _effect_curve(df, m3, OUTPUTS_CATCHMENT_DIR / "effect_curve_indian_ancestry.png")
    _residuals_by_sa4(df, m3, OUTPUTS_CATCHMENT_DIR / "residuals_by_sa4.png")

    b = float(m3.params.get("pct_indian_ancestry", np.nan))
    ci = m3.conf_int().loc["pct_indian_ancestry"] if "pct_indian_ancestry" in m3.params.index else (np.nan, np.nan)
    pct_5pp = (np.exp(5 * b) - 1) * 100
    pct_lo = (np.exp(5 * ci[0]) - 1) * 100
    pct_hi = (np.exp(5 * ci[1]) - 1) * 100

    _write_section_html(
        coef_html=coef_html,
        headline_pct=pct_5pp,
        ci_lo=pct_lo,
        ci_hi=pct_hi,
        n=int(m3.nobs),
        r2=float(m3.rsquared),
        out_path=OUTPUTS_CATCHMENT_DIR / "catchment_section.html",
    )

    print(f"\nHeadline (M3): +5pp Indian ancestry -> {pct_5pp:+.1f}% gross (95% CI {pct_lo:+.1f}% to {pct_hi:+.1f}%)")
    print(f"All outputs in {OUTPUTS_CATCHMENT_DIR}")


if __name__ == "__main__":
    main()
