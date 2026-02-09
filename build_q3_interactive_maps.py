from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import geopandas as gpd
except Exception:  # pragma: no cover - optional dependency
    gpd = None

if TYPE_CHECKING:
    import geopandas as gpd_type

    GeoDataFrame = gpd_type.GeoDataFrame
else:
    GeoDataFrame = Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

try:
    from shapely.geometry import shape
except Exception:  # pragma: no cover - optional dependency
    shape = None


OUTPUT_DIR = Path("outputs_locationquestions")

STATE_CSV = OUTPUT_DIR / "q3_state_seasonality.csv"
CITY_CSV = OUTPUT_DIR / "q3_city_seasonality.csv"

Q3_STATE_STORY_HTML = OUTPUT_DIR / "q3_story.html"

STATE_GEOJSON = OUTPUT_DIR / "q3_state_geometry_simplified.geojson"
CITY_GEOJSON = OUTPUT_DIR / "q3_city_sa4_geometry_simplified.geojson"

BASE_SA4_GEOJSON = Path("data/SA4_2021_AUST_GDA94_cleaned.geojson")

INCLUDE_PLOTLYJS = "cdn"  # Smaller HTML; requires internet to load plotly.js.
WRITE_DETAIL_OUTPUTS = False
STATE_TREND_TOP_N = 3
CITY_SPOTLIGHT_TOP_N = 5
CITY_HEATMAP_TOP_N = 12
CITY_SCATTER_TOP_N = 18
CITY_SCATTER_LABEL_TOP_N = 8
CINEMA_HEATMAP_TOP_N = 12
CINEMA_TREND_TOP_N = 12
CINEMA_SPOTLIGHT_TOP_N = 6
WEEK_SPOTLIGHT_TOP_N = 4
CITY_LABEL_MAX_COUNT = 12
CITY_LABEL_MIN_DISTANCE_DEG = 1.4
AUSTRALIA_BORDER_COLOR = "#5F5F5F"
AUSTRALIA_BORDER_WIDTH = 1.2
AUSTRALIA_SIMPLIFY_TOLERANCE = 0.05
CITY_PEAK_QUANTILE = 0.9
CITY_LOW_QUANTILE = 0.1
CINEMA_TOP_N = 20
WRITE_CITY_YEAR_SELECTOR = False

COLOR_SCALES = {
    "RdBu_r": px.colors.diverging.RdBu[::-1],
    "YlOrRd": px.colors.sequential.YlOrRd,
    "YlGn": px.colors.sequential.YlGn,
}

METRIC_LABELS = {
    "gross_z": "Seasonality (Z-Score)",
    "avg_titles": "Avg # Titles",
    "avg_cinemas": "Avg # Cinemas",
    "avg_cities": "Avg # Cities",
    "avg_gross": "Average Gross",
    "seasonality_idx": "Seasonality Index",
    "opportunity_score": "Opportunity Score",
    "peak_low_score": "Peaks/Lows",
    "weeks_to_50pct": "Weeks to 50% Gross",
}

METRIC_DESCRIPTIONS = {
    "gross_z": "Z-score of weekly average gross versus each location's baseline (red = above average, blue = below).",
    "avg_titles": "Average number of Indian titles active in that week (competition proxy).",
    "avg_cinemas": "Average number of cinemas screening Indian titles in that week (screen availability).",
    "avg_gross": "Average weekly gross by location across all years.",
    "seasonality_idx": "Weekly gross versus each location's median week (red = above median, blue = below).",
    "opportunity_score": "Demand minus competition: gross_z - 0.8 * titles_z.",
    "peak_low_score": "Top/bottom weeks per city (top 10% = peak, bottom 10% = low).",
    "weeks_to_50pct": "Week number when cumulative average gross reaches 50% of annual total (lower = faster demand).",
}


def load_geojson(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def gdf_to_geojson(gdf: GeoDataFrame) -> dict:
    return json.loads(gdf.to_json())


def normalize_key(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _flatten_coords(geom: dict) -> list[tuple[float, float]]:
    coords = geom.get("coordinates", [])
    gtype = geom.get("type")
    points: list[tuple[float, float]] = []
    if gtype == "Polygon":
        for ring in coords:
            points.extend(ring)
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                points.extend(ring)
    return points


def label_points_from_geojson(geojson: dict, property_name: str) -> dict[str, tuple[float, float]]:
    labels: dict[str, tuple[float, float]] = {}
    for feature in geojson.get("features", []):
        props = feature.get("properties", {}) or {}
        name = props.get(property_name)
        geom = feature.get("geometry")
        if not name or not geom:
            continue

        lon_lat = None
        if shape is not None:
            try:
                geom_obj = shape(geom)
                if not geom_obj.is_empty:
                    point = geom_obj.representative_point()
                    lon_lat = (float(point.x), float(point.y))
            except Exception:
                lon_lat = None

        if lon_lat is None:
            points = _flatten_coords(geom)
            if points:
                lons = [p[0] for p in points]
                lats = [p[1] for p in points]
                lon_lat = ((min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2)

        if lon_lat is not None:
            labels[str(name)] = lon_lat

    return labels


def add_week_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["iso_week_str"] = df["iso_week"].astype(int).astype(str).str.zfill(2)
    return df


def hover_format(metric: str) -> str:
    if metric in {"avg_titles", "avg_cinemas", "avg_cities"}:
        return ":.1f"
    if metric in {"gross_z", "seasonality_idx"}:
        return ":.2f"
    if metric == "peak_low_score":
        return ":.0f"
    if metric == "avg_gross":
        return ":,.0f"
    if metric == "weeks_to_50pct":
        return ":.0f"
    return ":.2f"


def hover_template(metric: str, label: str) -> str:
    fmt = hover_format(metric)
    return f"%{{location}}<br>ISO Week: %{{customdata}}<br>{label}: %{{z{fmt}}}<extra></extra>"


def build_peak_low_score(
    df: pd.DataFrame,
    metric: str,
    group_cols: list[str],
    high_quantile: float,
    low_quantile: float,
) -> pd.DataFrame:
    metric_df = df.copy()
    peak_threshold = metric_df.groupby(group_cols)[metric].transform(
        lambda x: x.quantile(high_quantile)
    )
    low_threshold = metric_df.groupby(group_cols)[metric].transform(
        lambda x: x.quantile(low_quantile)
    )
    metric_df["peak_low_score"] = 0.0
    metric_df.loc[metric_df[metric] >= peak_threshold, "peak_low_score"] = 1.0
    metric_df.loc[metric_df[metric] <= low_threshold, "peak_low_score"] = -1.0
    return metric_df


def build_event_highlight(
    df: pd.DataFrame,
    metric: str,
    label: str,
    top_n: int = 4,
    agg: str = "mean",
    scope_label: str = "cities",
) -> go.Figure:
    weekly = df.groupby("iso_week")[metric].agg(agg).dropna().sort_index()
    if weekly.empty:
        raise ValueError(f"No data for event highlight: {metric}")

    top_weeks = set(weekly.nlargest(top_n).index.tolist())
    bottom_weeks = set(weekly.nsmallest(top_n).index.tolist())
    colors = []
    for week in weekly.index.tolist():
        if week in top_weeks:
            colors.append("#d73027")
        elif week in bottom_weeks:
            colors.append("#4575b4")
        else:
            colors.append("#6b6b6b")

    fig = go.Figure(
        data=[
            go.Bar(
                x=weekly.index.tolist(),
                y=weekly.values.tolist(),
                marker_color=colors,
                marker_line_width=0,
                showlegend=False,
            )
        ]
    )
    fig.update_layout(
        title=f"{label} by ISO Week ({_agg_label(agg)} across {scope_label})",
        height=280,
        margin={"r": 20, "t": 40, "l": 50, "b": 40},
        template="plotly_white",
        xaxis_title="ISO Week",
        yaxis_title=label,
    )
    return fig


def build_location_highlight(
    df: pd.DataFrame,
    metric: str,
    label: str,
    location_col: str,
    top_n: int = 4,
    agg: str = "median",
    scope_label: str = "cinemas",
) -> go.Figure:
    location_stats = df.groupby(location_col)[metric].agg(agg).dropna()
    if location_stats.empty:
        raise ValueError(f"No data for location highlight: {metric}")

    top = location_stats.nlargest(top_n)
    bottom = location_stats.nsmallest(top_n)
    combined = pd.concat([bottom, top])
    colors = ["#4575b4"] * len(bottom) + ["#d73027"] * len(top)
    yaxis_title = scope_label[:-1].title() if scope_label.endswith("s") else scope_label.title()

    fig = go.Figure(
        data=[
            go.Bar(
                x=combined.values.tolist(),
                y=combined.index.tolist(),
                orientation="h",
                marker_color=colors,
                marker_line_width=0,
                showlegend=False,
            )
        ]
    )
    fig.update_layout(
        title=f"{label} by {scope_label.title()} ({_agg_label(agg)} across weeks)",
        height=320,
        margin={"r": 20, "t": 40, "l": 220, "b": 40},
        template="plotly_white",
        xaxis_title=label,
        yaxis_title=yaxis_title,
    )
    return fig


def build_city_stability_scatter(
    df: pd.DataFrame,
    top_n: int,
    label_top_n: int,
    height: int = 420,
) -> go.Figure:
    required = {"state", "city", "avg_gross", "seasonality_idx"}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise ValueError(f"Missing columns for city scatter: {sorted(missing)}")

    summary = (
        df.groupby(["state", "city"], as_index=False)
        .agg(
            avg_gross=("avg_gross", "mean"),
            seasonality_std=("seasonality_idx", "std"),
            seasonality_median=("seasonality_idx", "median"),
        )
        .dropna(subset=["avg_gross"])
    )
    if summary.empty:
        raise ValueError("No data for city scatter chart.")

    summary["seasonality_std"] = summary["seasonality_std"].fillna(0)
    summary = summary.sort_values("avg_gross", ascending=False)
    if top_n > 0:
        summary = summary.head(top_n)

    summary["label"] = ""
    if label_top_n > 0:
        summary.loc[summary.index[:label_top_n], "label"] = summary["city"]

    fig = px.scatter(
        summary,
        x="avg_gross",
        y="seasonality_std",
        color="state",
        size="avg_gross",
        size_max=28,
        text="label",
        height=height,
        labels={
            "avg_gross": "Avg weekly gross",
            "seasonality_std": "Seasonality volatility (std dev)",
            "state": "State",
        },
    )
    fig.update_traces(
        textposition="top center",
        marker={"size": 10, "opacity": 0.85, "line": {"width": 0.5, "color": "#ffffff"}},
        hovertemplate=(
            "City: %{customdata[0]}<br>State: %{customdata[1]}"
            "<br>Avg weekly gross: %{x:,.0f}"
            "<br>Volatility (std): %{y:.2f}<extra></extra>"
        ),
        customdata=summary[["city", "state"]],
    )
    fig.update_layout(
        title="City scale vs seasonality volatility",
        margin={"r": 20, "t": 50, "l": 60, "b": 40},
        template="plotly_white",
        legend={"title": "State"},
    )
    return fig


def _agg_label(agg: str) -> str:
    if agg == "mean":
        return "Avg"
    if agg == "median":
        return "Median"
    return str(agg).title()


def _fig_to_html(fig: go.Figure, include_plotlyjs: bool) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        full_html=False,
        config={"responsive": True},
    )


def _format_week_label(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    try:
        return f"W{int(value):02d}"
    except (TypeError, ValueError):
        return "n/a"


def select_top_states_by_gross(df: pd.DataFrame, top_n: int) -> list[str]:
    totals = df.groupby("state")["avg_gross"].sum().sort_values(ascending=False)
    return totals.head(top_n).index.tolist()


def build_state_violin_chart(
    df: pd.DataFrame,
    metric: str,
    title: str,
    height: int = 360,
) -> go.Figure:
    if metric not in df.columns:
        raise ValueError(f"Metric not found: {metric}")

    values = df[["state", metric]].dropna()
    if values.empty:
        raise ValueError(f"No data for violin chart: {metric}")

    order = (
        values.groupby("state")[metric]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )

    fig = go.Figure()
    for state in order:
        state_values = values.loc[values["state"] == state, metric]
        fig.add_trace(
            go.Violin(
                y=state_values,
                name=state,
                box_visible=True,
                meanline_visible=True,
                spanmode="hard",
                marker_color="#4C78A8",
                line_width=1,
                hovertemplate=f"State: {state}<br>Index: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        margin={"r": 20, "t": 50, "l": 60, "b": 40},
        template="plotly_white",
        xaxis_title="State",
        yaxis_title=METRIC_LABELS.get(metric, metric.replace("_", " ").title()),
    )
    return fig


def build_state_trend_chart(
    df: pd.DataFrame,
    metric: str,
    title: str,
    top_states: list[str],
    height: int = 360,
) -> go.Figure:
    if metric not in df.columns:
        raise ValueError(f"Metric not found: {metric}")

    df = df[df["state"].notna()].copy()
    df = add_week_str(df)
    weeks = sorted(df["iso_week_str"].unique(), key=lambda x: int(x))
    pivot = df.pivot_table(
        index="iso_week_str",
        columns="state",
        values=metric,
        aggfunc="mean",
    ).reindex(index=weeks)

    label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    fmt = hover_format(metric)
    fig = go.Figure()
    for state in top_states:
        if state not in pivot.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=pivot[state],
                mode="lines",
                name=state,
                line={"width": 2},
                hovertemplate=(
                    f"{state}<br>ISO Week: %{{x}}<br>{label}: %{{y{fmt}}}"
                    "<extra></extra>"
                ),
            )
        )

    median_series = pivot.median(axis=1, skipna=True)
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=median_series,
            mode="lines",
            name="Median (all states)",
            line={"width": 3, "color": "#111111"},
            hovertemplate=(
                f"Median (all states)<br>ISO Week: %{{x}}<br>{label}: %{{y{fmt}}}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        height=height,
        margin={"r": 20, "t": 50, "l": 60, "b": 40},
        template="plotly_white",
        xaxis_title="ISO Week",
        yaxis_title=label,
        xaxis={"categoryorder": "array", "categoryarray": weeks},
        legend={"title": "State"},
    )
    return fig


def build_city_spotlight_table(df: pd.DataFrame, top_n: int = 5) -> str:
    if df.empty:
        return ""

    summary = (
        df.groupby("city", as_index=False)["avg_gross"]
        .mean()
        .sort_values("avg_gross", ascending=False)
        .head(top_n)
    )
    if summary.empty:
        return ""

    rows = []
    for _, row in summary.iterrows():
        city = row["city"]
        city_df = df[df["city"] == city]
        if city_df.empty:
            continue
        peak_idx = city_df["seasonality_idx"].idxmax()
        low_idx = city_df["seasonality_idx"].idxmin()
        peak_week = _format_week_label(city_df.loc[peak_idx, "iso_week"])
        low_week = _format_week_label(city_df.loc[low_idx, "iso_week"])
        avg_gross = row["avg_gross"]
        avg_gross_label = "n/a"
        if pd.notna(avg_gross):
            avg_gross_label = f"{avg_gross:,.0f}"

        rows.append(
            "<tr>"
            f"<td>{city}</td>"
            f"<td>{avg_gross_label}</td>"
            f"<td>{peak_week}</td>"
            f"<td>{low_week}</td>"
            "</tr>"
        )

    if not rows:
        return ""

    return (
        "<table class=\"spotlight-table\">"
        "<thead><tr>"
        "<th>City</th>"
        "<th>Avg weekly gross</th>"
        "<th>Peak week</th>"
        "<th>Quiet week</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_cinema_spotlight_table(df: pd.DataFrame, top_n: int = 6) -> str:
    if df.empty:
        return ""

    summary = (
        df.groupby("cinema_label", as_index=False)["avg_gross"]
        .mean()
        .sort_values("avg_gross", ascending=False)
        .head(top_n)
    )
    if summary.empty:
        return ""

    rows = []
    for _, row in summary.iterrows():
        cinema = row["cinema_label"]
        cinema_df = df[df["cinema_label"] == cinema]
        if cinema_df.empty:
            continue
        peak_idx = cinema_df["seasonality_idx"].idxmax()
        low_idx = cinema_df["seasonality_idx"].idxmin()
        peak_week = _format_week_label(cinema_df.loc[peak_idx, "iso_week"])
        low_week = _format_week_label(cinema_df.loc[low_idx, "iso_week"])
        avg_gross = row["avg_gross"]
        avg_gross_label = "n/a"
        if pd.notna(avg_gross):
            avg_gross_label = f"{avg_gross:,.0f}"

        rows.append(
            "<tr>"
            f"<td>{cinema}</td>"
            f"<td>{avg_gross_label}</td>"
            f"<td>{peak_week}</td>"
            f"<td>{low_week}</td>"
            "</tr>"
        )

    if not rows:
        return ""

    return (
        "<table class=\"spotlight-table\">"
        "<thead><tr>"
        "<th>Cinema</th>"
        "<th>Avg weekly gross</th>"
        "<th>Peak week</th>"
        "<th>Quiet week</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_week_spotlight_table(
    df: pd.DataFrame,
    metric: str,
    top_n: int = 4,
    agg: str = "mean",
    value_label: str | None = None,
) -> str:
    if df.empty or metric not in df.columns:
        return ""

    weekly = df.groupby("iso_week")[metric].agg(agg).dropna().sort_index()
    if weekly.empty:
        return ""

    top_weeks = weekly.nlargest(top_n)
    bottom_weeks = weekly.nsmallest(top_n)

    rows = []
    for label, series, tag in (
        ("Peak", top_weeks, "Peak"),
        ("Quiet", bottom_weeks, "Quiet"),
    ):
        for week, value in series.items():
            rows.append(
                "<tr>"
                f"<td>{tag}</td>"
                f"<td>{_format_week_label(week)}</td>"
                f"<td>{value:.2f}</td>"
                "</tr>"
            )

    if not rows:
        return ""

    metric_label = value_label or METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    return (
        "<table class=\"spotlight-table\">"
        "<thead><tr>"
        "<th>Type</th>"
        "<th>Week</th>"
        f"<th>{metric_label}</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_top_n_line_chart(
    df: pd.DataFrame,
    metric: str,
    title: str,
    top_labels: list[str],
    median_label: str,
    height: int = 600,
) -> go.Figure:
    if metric not in df.columns:
        raise ValueError(f"Metric not found: {metric}")

    df = df[df["cinema_label"].notna()].copy()
    df = add_week_str(df)
    weeks = sorted(df["iso_week_str"].unique(), key=lambda x: int(x))
    pivot = df.pivot_table(
        index="iso_week_str",
        columns="cinema_label",
        values=metric,
        aggfunc="mean",
    ).reindex(index=weeks)
    if pivot.empty:
        raise ValueError(f"No data for metric: {metric}")

    label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    fmt = hover_format(metric)
    fig = go.Figure()
    for cinema in top_labels:
        if cinema not in pivot.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=pivot[cinema],
                mode="lines",
                name=cinema,
                line={"width": 1},
                opacity=0.6,
                hovertemplate=f"{cinema}<br>ISO Week: %{{x}}<br>{label}: %{{y{fmt}}}<extra></extra>",
            )
        )

    top_cols = [cinema for cinema in top_labels if cinema in pivot.columns]
    extrema_source = pivot[top_cols] if top_cols else pivot

    def _row_idxmax(row: pd.Series) -> str | None:
        return row.idxmax() if row.notna().any() else None

    def _row_idxmin(row: pd.Series) -> str | None:
        return row.idxmin() if row.notna().any() else None

    weekly_max = extrema_source.max(axis=1, skipna=True)
    weekly_min = extrema_source.min(axis=1, skipna=True)
    weekly_max_name = extrema_source.apply(_row_idxmax, axis=1)
    weekly_min_name = extrema_source.apply(_row_idxmin, axis=1)
    weekly_max_label = weekly_max_name.fillna("n/a").astype(str).tolist()
    weekly_min_label = weekly_min_name.fillna("n/a").astype(str).tolist()
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=weekly_max,
            mode="markers",
            name="Weekly max",
            marker={"color": "#d73027", "size": 6, "symbol": "triangle-up"},
            customdata=weekly_max_label,
            hovertemplate=(
                f"Weekly max<br>ISO Week: %{{x}}<br>Cinema: %{{customdata}}"
                f"<br>{label}: %{{y{fmt}}}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=weekly_min,
            mode="markers",
            name="Weekly min",
            marker={"color": "#4575b4", "size": 6, "symbol": "triangle-down"},
            customdata=weekly_min_label,
            hovertemplate=(
                f"Weekly min<br>ISO Week: %{{x}}<br>Cinema: %{{customdata}}"
                f"<br>{label}: %{{y{fmt}}}<extra></extra>"
            ),
        )
    )

    median_series = pivot.median(axis=1, skipna=True)
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=median_series,
            mode="lines",
            name=median_label,
            line={"width": 3, "color": "#111111"},
            hovertemplate=f"{median_label}<br>ISO Week: %{{x}}<br>{label}: %{{y{fmt}}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        height=height,
        margin={"r": 20, "t": 50, "l": 60, "b": 40},
        template="plotly_white",
        xaxis_title="ISO Week",
        yaxis_title=label,
        xaxis={"categoryorder": "array", "categoryarray": weeks},
        legend={"title": "Cinema"},
    )
    return fig


def compute_weeks_to_50pct(
    df: pd.DataFrame,
    group_col: str,
    week_col: str,
    value_col: str,
) -> pd.DataFrame:
    df = df[[group_col, week_col, value_col]].dropna().copy()
    if df.empty:
        return pd.DataFrame(columns=[group_col, "weeks_to_50pct"])

    df[week_col] = df[week_col].astype(int)
    weeks = sorted(df[week_col].unique())
    results = []
    for label, group in df.groupby(group_col):
        series = (
            group.set_index(week_col)[value_col]
            .reindex(weeks)
            .fillna(0)
        )
        total = series.sum()
        if total <= 0:
            weeks_to_50 = math.nan
        else:
            threshold = 0.5 * total
            cumulative = series.cumsum()
            week_hit = cumulative.index[cumulative >= threshold][0]
            weeks_to_50 = int(week_hit)
        results.append({group_col: label, "weeks_to_50pct": weeks_to_50})
    return pd.DataFrame(results)


def build_weeks_to_50pct_chart(
    df: pd.DataFrame,
    title: str,
    group_col: str,
    week_col: str,
    value_col: str,
    top_labels: list[str] | None = None,
    median_label: str | None = None,
    height: int = 600,
    yaxis_title: str | None = None,
) -> go.Figure:
    weeks_df = compute_weeks_to_50pct(
        df,
        group_col=group_col,
        week_col=week_col,
        value_col=value_col,
    )
    if weeks_df.empty:
        raise ValueError("No data for weeks_to_50pct chart.")

    if top_labels:
        top_df = weeks_df[weeks_df[group_col].isin(top_labels)].copy()
    else:
        top_df = weeks_df.copy()
    top_df = top_df.sort_values("weeks_to_50pct", ascending=True)

    fig = go.Figure(
        data=[
            go.Bar(
                x=top_df["weeks_to_50pct"],
                y=top_df[group_col],
                orientation="h",
                marker_color="#4C78A8",
                hovertemplate="Name: %{y}<br>Weeks to 50%: %{x:.0f}<extra></extra>",
            )
        ]
    )

    median_value = weeks_df["weeks_to_50pct"].median(skipna=True)
    if median_label and not math.isnan(median_value):
        fig.add_shape(
            type="line",
            x0=median_value,
            x1=median_value,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line={"color": "#111111", "width": 2},
        )
        fig.add_annotation(
            x=median_value,
            y=1,
            yref="paper",
            text=f"{median_label}: {median_value:.0f} w",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
        )

    fig.update_layout(
        title=title,
        height=height,
        margin={"r": 20, "t": 50, "l": 220, "b": 40},
        template="plotly_white",
        xaxis_title="Weeks to 50% of Annual Gross",
        yaxis_title=yaxis_title or group_col.replace("_", " ").title(),
        yaxis={"autorange": "reversed", "automargin": True},
    )
    return fig


def build_heatmap(
    df: pd.DataFrame,
    y_col: str,
    metric: str,
    title: str,
    color_scale: str,
    midpoint: float | None = None,
    range_percentiles: tuple[float, float] | None = None,
    y_order: list[str] | None = None,
    yaxis_title: str | None = None,
    height: int = 900,
) -> go.Figure:
    if metric not in df.columns:
        raise ValueError(f"Metric not found: {metric}")

    df = df[df[y_col].notna()].copy()
    df = add_week_str(df)
    weeks = sorted(df["iso_week_str"].unique(), key=lambda x: int(x))
    pivot = df.pivot_table(
        index=y_col,
        columns="iso_week_str",
        values=metric,
        aggfunc="mean",
    )
    if y_order:
        pivot = pivot.reindex(index=[name for name in y_order if name in pivot.index])
    else:
        pivot = pivot.sort_index()
    pivot = pivot.reindex(columns=weeks)

    label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    metric_series = df[metric].dropna()
    if metric_series.empty:
        raise ValueError(f"No data for metric: {metric}")

    if range_percentiles:
        low, high = range_percentiles
        zmin = float(metric_series.quantile(low / 100))
        zmax = float(metric_series.quantile(high / 100))
    else:
        zmin = float(metric_series.min())
        zmax = float(metric_series.max())

    if midpoint is not None:
        span = max(abs(zmin - midpoint), abs(zmax - midpoint))
        zmin = midpoint - span
        zmax = midpoint + span

    fmt = hover_format(metric)
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=weeks,
                y=pivot.index.tolist(),
                z=pivot.values,
                colorscale=COLOR_SCALES[color_scale],
                zmin=zmin,
                zmax=zmax,
                zmid=midpoint,
                colorbar={"title": label},
                hovertemplate=f"%{{y}}<br>ISO Week: %{{x}}<br>{label}: %{{z{fmt}}}<extra></extra>",
            )
        ]
    )
    y_title = yaxis_title or y_col.replace("_", " ").title()
    fig.update_layout(
        title=title,
        height=height,
        margin={"r": 20, "t": 50, "l": 160, "b": 40},
        template="plotly_white",
        xaxis_title="ISO Week",
        yaxis_title=y_title,
    )
    return fig


def pick_geojson(preferred: Path, fallback: Path) -> Path:
    if preferred.exists():
        return preferred
    return fallback


def select_spread_labels(
    candidates: list[str],
    label_points: dict[str, tuple[float, float]],
    max_count: int,
    min_distance: float,
) -> list[str]:
    selected: list[str] = []
    for name in candidates:
        if name not in label_points:
            continue
        lon, lat = label_points[name]
        if all(
            math.hypot(lon - label_points[other][0], lat - label_points[other][1])
            >= min_distance
            for other in selected
        ):
            selected.append(name)
        if len(selected) >= max_count:
            break

    if not selected:
        selected = [name for name in candidates if name in label_points][:max_count]

    return selected


def prepare_sales_for_seasonality() -> pd.DataFrame:
    from DataExplorationMain import sales_indian

    sales_for_seasonality = sales_indian.copy()
    sales_for_seasonality["actual_sales_date"] = pd.to_datetime(
        sales_for_seasonality["actual_sales_date"]
    )
    sales_for_seasonality["week_start"] = (
        sales_for_seasonality["actual_sales_date"]
        - pd.to_timedelta(
            sales_for_seasonality["actual_sales_date"].dt.dayofweek, unit="d"
        )
    )

    sales_for_seasonality["iso_year"] = (
        sales_for_seasonality["week_start"].dt.isocalendar().year
    )
    sales_for_seasonality["iso_week"] = (
        sales_for_seasonality["week_start"].dt.isocalendar().week
    )

    return sales_for_seasonality


def compute_seasonality_from_sales(
    sales_for_seasonality: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sales_for_seasonality is None:
        sales_for_seasonality = prepare_sales_for_seasonality()

    state_week = (
        sales_for_seasonality.groupby(["iso_year", "iso_week", "state"], as_index=False)
        .agg(
            {
                "gross_today": "sum",
                "numero_film_id": "nunique",
                "theatre_name": "nunique",
                "city": "nunique",
            }
        )
        .rename(
            columns={
                "gross_today": "total_gross",
                "numero_film_id": "n_titles",
                "theatre_name": "n_cinemas",
                "city": "n_cities",
            }
        )
    )

    state_seasonality = (
        state_week.groupby(["state", "iso_week"], as_index=False)
        .agg(
            {
                "total_gross": ["mean", "median"],
                "n_titles": "mean",
                "n_cinemas": "mean",
                "n_cities": "mean",
            }
        )
        .reset_index(drop=True)
    )

    state_seasonality.columns = [
        "state",
        "iso_week",
        "avg_gross",
        "med_gross",
        "avg_titles",
        "avg_cinemas",
        "avg_cities",
    ]

    state_seasonality["gross_z"] = state_seasonality.groupby("state")[
        "avg_gross"
    ].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

    state_seasonality["seasonality_idx"] = state_seasonality["avg_gross"] / (
        state_seasonality.groupby("state")["avg_gross"].transform("median")
    )

    city_week = (
        sales_for_seasonality.groupby(
            ["iso_year", "iso_week", "state", "city"], as_index=False
        )
        .agg(
            {
                "gross_today": "sum",
                "numero_film_id": "nunique",
                "theatre_name": "nunique",
            }
        )
        .rename(
            columns={
                "gross_today": "total_gross",
                "numero_film_id": "n_titles",
                "theatre_name": "n_cinemas",
            }
        )
    )

    city_seasonality = (
        city_week.groupby(["state", "city", "iso_week"], as_index=False)
        .agg(
            {
                "total_gross": ["mean", "median"],
                "n_titles": "mean",
                "n_cinemas": "mean",
            }
        )
        .reset_index(drop=True)
    )

    city_seasonality.columns = [
        "state",
        "city",
        "iso_week",
        "avg_gross",
        "med_gross",
        "avg_titles",
        "avg_cinemas",
    ]

    city_seasonality["seasonality_idx"] = city_seasonality["avg_gross"] / (
        city_seasonality.groupby(["state", "city"])["avg_gross"].transform("median")
    )

    city_seasonality["gross_z"] = city_seasonality.groupby(["state", "city"])[
        "avg_gross"
    ].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

    city_seasonality["titles_z"] = city_seasonality.groupby(["state", "city"])[
        "avg_titles"
    ].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

    city_seasonality["opportunity_score"] = (
        city_seasonality["gross_z"] - 0.8 * city_seasonality["titles_z"]
    )

    return state_seasonality, city_seasonality


def compute_city_week_by_year(
    sales_for_seasonality: pd.DataFrame,
) -> pd.DataFrame:
    city_week = (
        sales_for_seasonality.groupby(
            ["iso_year", "iso_week", "state", "city"], as_index=False
        )
        .agg(
            {
                "gross_today": "sum",
                "numero_film_id": "nunique",
                "theatre_name": "nunique",
            }
        )
        .rename(
            columns={
                "gross_today": "total_gross",
                "numero_film_id": "n_titles",
                "theatre_name": "n_cinemas",
            }
        )
    )

    city_week["seasonality_idx"] = city_week["total_gross"] / city_week.groupby(
        ["state", "city", "iso_year"]
    )["total_gross"].transform("median")

    city_week["gross_z"] = city_week.groupby(["state", "city", "iso_year"])[
        "total_gross"
    ].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

    city_week["titles_z"] = city_week.groupby(["state", "city", "iso_year"])[
        "n_titles"
    ].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

    city_week["opportunity_score"] = (
        city_week["gross_z"] - 0.8 * city_week["titles_z"]
    )

    return city_week


def compute_cinema_seasonality_from_sales(
    sales_for_seasonality: pd.DataFrame,
) -> pd.DataFrame:
    cinema_week = (
        sales_for_seasonality.groupby(
            ["iso_year", "iso_week", "state", "city", "theatre_name"], as_index=False
        )
        .agg(
            {
                "gross_today": "sum",
                "numero_film_id": "nunique",
            }
        )
        .rename(
            columns={
                "gross_today": "total_gross",
                "numero_film_id": "n_titles",
            }
        )
    )

    cinema_seasonality = (
        cinema_week.groupby(["state", "city", "theatre_name", "iso_week"], as_index=False)
        .agg(
            {
                "total_gross": ["mean", "median"],
                "n_titles": "mean",
            }
        )
        .reset_index(drop=True)
    )

    cinema_seasonality.columns = [
        "state",
        "city",
        "theatre_name",
        "iso_week",
        "avg_gross",
        "med_gross",
        "avg_titles",
    ]

    cinema_seasonality["seasonality_idx"] = cinema_seasonality["avg_gross"] / (
        cinema_seasonality.groupby(["state", "city", "theatre_name"])["avg_gross"].transform("median")
    )

    cinema_seasonality["gross_z"] = cinema_seasonality.groupby(
        ["state", "city", "theatre_name"]
    )["avg_gross"].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

    cinema_seasonality["cinema_label"] = (
        cinema_seasonality["theatre_name"].astype(str).str.strip()
        + " ("
        + cinema_seasonality["city"].astype(str).str.strip()
        + ", "
        + cinema_seasonality["state"].astype(str).str.strip()
        + ")"
    )

    return cinema_seasonality


def build_state_geometry(sa4: GeoDataFrame) -> GeoDataFrame:
    def map_state(name: str) -> str:
        if name in ["New South Wales", "Australian Capital Territory"]:
            return "New South Wales (inc ACT)"
        if name in ["Victoria", "Tasmania"]:
            return "Victoria (inc TAS)"
        return name

    state_geom = sa4.copy()
    state_geom["state"] = state_geom["STE_NAME21"].map(map_state)
    state_geom = state_geom.dissolve(by="state", as_index=False)
    state_geom = state_geom[["state", "geometry"]]
    state_geom["geometry"] = state_geom["geometry"].simplify(
        tolerance=0.05, preserve_topology=True
    )
    return state_geom


def build_city_geometry(
    sa4: GeoDataFrame, city_names: list[str]
) -> GeoDataFrame:
    sa4_names = set(sa4["SA4_NAME21"].dropna().astype(str).str.strip())
    sa4_name_map = {normalize_key(name): name for name in sa4_names}

    manual_map = {
        normalize_key("Canberra"): ["Australian Capital Territory"],
        normalize_key("Central and Far West"): ["Central West", "Far West and Orana"],
        normalize_key("City and Inner South"): ["Sydney - City and Inner South"],
        normalize_key("Eastern Suburbs"): ["Sydney - Eastern Suburbs"],
        normalize_key("Hills & Hawkesbury"): ["Sydney - Baulkham Hills and Hawkesbury"],
        normalize_key("Inner West"): ["Sydney - Inner West"],
        normalize_key("Murray and Riverina"): ["Murray", "Riverina"],
        normalize_key("North Sydney - Hornsby"): ["Sydney - North Sydney and Hornsby"],
        normalize_key("Northern Beaches"): ["Sydney - Northern Beaches"],
        normalize_key("Parramatta & Ryde"): ["Sydney - Parramatta", "Sydney - Ryde"],
        normalize_key("South West Sydney"): ["Sydney - South West"],
        normalize_key("Sutherland & St George"): ["Sydney - Sutherland"],
        normalize_key("West and Blue Mountains"): ["Sydney - Outer West and Blue Mountains"],
        normalize_key("N.T Outback"): ["Northern Territory - Outback"],
        normalize_key("Brisbane - Inner City"): ["Brisbane Inner City"],
        normalize_key("Cairns Region"): ["Cairns"],
        normalize_key("Ipswich Region"): ["Ipswich"],
        normalize_key("QLD - Outback"): ["Queensland - Outback"],
        normalize_key("Toowoomba - Darling Downs"): ["Toowoomba", "Darling Downs - Maranoa"],
        normalize_key("Townsville Region"): ["Townsville"],
        normalize_key("S.A  - Outback"): ["South Australia - Outback"],
        normalize_key("S.A - South East"): ["South Australia - South East"],
        normalize_key("Central Inner Melbourne"): ["Melbourne - Inner"],
        normalize_key("Inner East Melbourne"): ["Melbourne - Inner East"],
        normalize_key("Inner South Melbourne"): ["Melbourne - Inner South"],
        normalize_key("North East Melbourne"): ["Melbourne - North East"],
        normalize_key("North West Melbourne"): ["Melbourne - North West"],
        normalize_key("North West Victoria"): ["North West"],
        normalize_key("Outer East Melbourne"): ["Melbourne - Outer East"],
        normalize_key("South East Melbourne"): ["Melbourne - South East"],
        normalize_key("Tas - North East"): ["Launceston and North East"],
        normalize_key("Tas - North West"): ["West and North West"],
        normalize_key("West Melbourne"): ["Melbourne - West"],
        normalize_key("Mandurah - Bunbury"): ["Mandurah", "Bunbury"],
        normalize_key("W.A - Outback South"): ["Western Australia - Outback (South)"],
        normalize_key("W.A - Wheat Belt"): ["Western Australia - Wheat Belt"],
    }

    for targets in manual_map.values():
        for target in targets:
            if target not in sa4_names:
                raise SystemExit(f"Manual mapping target not in SA4 list: {target}")

    rows = []
    unmatched = []
    for city in city_names:
        key = normalize_key(city)
        if key in sa4_name_map:
            rows.append({"city": city, "sa4_name": sa4_name_map[key]})
        elif key in manual_map:
            for target in manual_map[key]:
                rows.append({"city": city, "sa4_name": target})
        else:
            unmatched.append(city)

    if unmatched:
        raise SystemExit(f"Unmatched city names: {', '.join(sorted(unmatched))}")

    crosswalk = pd.DataFrame(rows)
    merged = crosswalk.merge(
        sa4[["SA4_NAME21", "geometry"]],
        left_on="sa4_name",
        right_on="SA4_NAME21",
        how="left",
    )

    city_geom = gpd.GeoDataFrame(merged, geometry="geometry", crs=sa4.crs)
    city_geom = city_geom.dissolve(by="city", as_index=False)
    city_geom = city_geom[["city", "geometry"]]
    city_geom["geometry"] = city_geom["geometry"].simplify(
        tolerance=0.02, preserve_topology=True
    )
    return city_geom


def build_australia_outline(sa4: GeoDataFrame) -> dict:
    outline = sa4.dissolve()
    geom = outline.geometry.iloc[0]
    geom = geom.simplify(
        tolerance=AUSTRALIA_SIMPLIFY_TOLERANCE, preserve_topology=True
    )
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Australia"},
                "geometry": geom.__geo_interface__,
            }
        ],
    }


def build_animation(
    df: pd.DataFrame,
    geojson: dict,
    location_col: str,
    feature_key: str,
    metric: str,
    title: str,
    color_scale: str,
    output_path: Path | None,
    midpoint: float | None = None,
    label_points: dict[str, tuple[float, float]] | None = None,
    label_names: list[str] | set[str] | None = None,
    label_font_size: int = 12,
    label_textposition: str = "middle center",
    range_percentiles: tuple[float, float] | None = None,
    border_geojson: dict | None = None,
    border_color: str = AUSTRALIA_BORDER_COLOR,
    border_width: float = AUSTRALIA_BORDER_WIDTH,
    write_html: bool = True,
) -> go.Figure:
    if metric not in df.columns:
        raise ValueError(f"Metric not found: {metric}")

    df = df[df[location_col].notna()].copy()
    df = add_week_str(df)

    weeks = sorted(df["iso_week_str"].unique(), key=lambda x: int(x))
    pivot = df.pivot_table(
        index=location_col,
        columns="iso_week_str",
        values=metric,
        aggfunc="mean",
    )

    locations = sorted(pivot.index.astype(str))
    pivot = pivot.reindex(index=locations, columns=weeks)

    label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())

    metric_series = df[metric].dropna()
    if metric_series.empty:
        raise ValueError(f"No data for metric: {metric}")

    if range_percentiles:
        low, high = range_percentiles
        zmin = float(metric_series.quantile(low / 100))
        zmax = float(metric_series.quantile(high / 100))
    else:
        zmin = float(metric_series.min())
        zmax = float(metric_series.max())

    if midpoint is not None:
        span = max(abs(zmin - midpoint), abs(zmax - midpoint))
        zmin = midpoint - span
        zmax = midpoint + span

    base_week = weeks[0]
    base_z = pivot[base_week].tolist()

    data = [
        go.Choropleth(
            geojson=geojson,
            locations=locations,
            featureidkey=feature_key,
            z=base_z,
            colorscale=COLOR_SCALES[color_scale],
            zmin=zmin,
            zmax=zmax,
            zmid=midpoint,
            marker_line_width=0.5,
            marker_line_color="#FFFFFF",
            customdata=[base_week] * len(locations),
            hovertemplate=hover_template(metric, label),
        )
    ]

    if border_geojson:
        data.append(
            go.Choropleth(
                geojson=border_geojson,
                locations=["Australia"],
                featureidkey="properties.name",
                z=[1],
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                marker_line_color=border_color,
                marker_line_width=border_width,
                hoverinfo="skip",
            )
        )

    if label_points:
        if label_names:
            if isinstance(label_names, set):
                label_names = [loc for loc in locations if loc in label_names]
            else:
                label_names = [name for name in label_names if name in label_points]
        else:
            label_names = [loc for loc in locations if loc in label_points]

        label_lons = [label_points[name][0] for name in label_names]
        label_lats = [label_points[name][1] for name in label_names]
        data.append(
            go.Scattergeo(
                lon=label_lons,
                lat=label_lats,
                text=label_names,
                mode="text",
                textfont={"size": label_font_size, "color": "#1F1F1F"},
                textposition=label_textposition,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig = go.Figure(data=data)

    frames = []
    for week in weeks:
        frames.append(
            go.Frame(
                name=week,
                data=[
                    go.Choropleth(
                        z=pivot[week].tolist(),
                        locations=locations,
                        customdata=[week] * len(locations),
                    )
                ],
                traces=[0],
            )
        )

    fig.frames = frames

    steps = [
        {
            "label": week,
            "method": "animate",
            "args": [
                [week],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for week in weeks
    ]

    fig.update_layout(
        title=title,
        height=650,
        width=1000,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        template="plotly_white",
        geo={"fitbounds": "locations", "visible": False, "projection_type": "mercator"},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.01,
                "y": 0.05,
                "direction": "left",
                "pad": {"r": 10, "t": 0},
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 350, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "ISO Week: "},
                "pad": {"t": 40},
                "steps": steps,
            }
        ],
    )

    if write_html and output_path is not None:
        fig.write_html(output_path, include_plotlyjs=INCLUDE_PLOTLYJS)
    return fig


def write_selector_html(
    figures: dict[str, go.Figure],
    output_path: Path,
    options: list[str],
    label_map: dict[str, str],
    descriptions: dict[str, str],
    title: str,
    select_label: str = "Metric",
    highlight_sections: list[dict[str, object]] | None = None,
    highlight_sections_by_metric: dict[str, list[dict[str, object]]] | None = None,
) -> None:
    panels = []
    first = True
    for key in options:
        fig = figures.get(key)
        if fig is None:
            continue

        html = pio.to_html(
            fig,
            include_plotlyjs="cdn" if first else False,
            full_html=False,
            config={"responsive": True},
        )
        highlight_html = ""
        sections = []
        if highlight_sections_by_metric and key in highlight_sections_by_metric:
            sections = highlight_sections_by_metric[key]
        elif highlight_sections:
            sections = highlight_sections
        if sections:
            blocks = []
            for section in sections:
                heading_value = section.get("heading", "")
                heading = str(heading_value).strip() if heading_value is not None else ""
                note_value = section.get("note")
                note = str(note_value).strip() if note_value is not None else None
                if note == "":
                    note = None
                section_fig = section.get("fig")
                if not isinstance(section_fig, go.Figure):
                    continue
                section_html = pio.to_html(
                    section_fig,
                    include_plotlyjs=False,
                    full_html=False,
                    config={"responsive": True},
                )
                note_html = f'<div class="highlight-note">{note}</div>' if note else ""
                blocks.append(
                    f"""
        <div class="highlight-block">
          <h3>{heading}</h3>
          {note_html}
          {section_html}
        </div>
                    """.strip()
                )
            if blocks:
                highlight_html = (
                    '\n      <div class="highlight-wrap">\n        '
                    + "\n        ".join(blocks)
                    + "\n      </div>"
                )
        display = "block" if first else "none"
        panels.append(
            f'<div class="map-panel" id="panel-{key}" style="display:{display};">{html}{highlight_html}</div>'
        )
        first = False

    option_html = "\n".join(
        [
            f'<option value="{key}">{label_map.get(key, key)}</option>'
            for key in options
            if key in figures
        ]
    )
    description_map = {
        key: descriptions.get(key, "")
        for key in options
        if key in figures
    }

    wrapper = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      body {{ margin: 0; font-family: Arial, sans-serif; }}
      .toolbar {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        border-bottom: 1px solid #e5e5e5;
        background: #fafafa;
      }}
      .description {{
        padding: 0 16px 12px 16px;
        color: #333;
        font-size: 14px;
      }}
      .map-wrap {{ padding: 8px 0; }}
      .map-panel {{ width: 100%; }}
      .highlight-wrap {{ padding: 0 0 16px 0; }}
      .highlight-block {{ padding: 12px 16px 0 16px; }}
      .highlight-block h3 {{ margin: 12px 0 6px 0; font-size: 16px; }}
      .highlight-note {{ margin: 0 0 12px 0; color: #444; font-size: 13px; }}
    </style>
  </head>
  <body>
    <div class="toolbar">
      <label for="metricSelect"><strong>{select_label}</strong></label>
      <select id="metricSelect">{option_html}</select>
    </div>
    <div class="description" id="metricDescription"></div>
    <div class="map-wrap">
      {''.join(panels)}
    </div>
    <script>
      const descriptions = {json.dumps(description_map)};
      function showPanel(metric) {{
        document.querySelectorAll('.map-panel').forEach(panel => {{
          panel.style.display = 'none';
        }});
        const panel = document.getElementById('panel-' + metric);
        if (!panel) return;
        panel.style.display = 'block';
        panel.querySelectorAll('.plotly-graph-div').forEach((plotDiv) => {{
          if (plotDiv && window.Plotly) {{
            Plotly.Plots.resize(plotDiv);
          }}
        }});
        const desc = descriptions[metric] || '';
        const descEl = document.getElementById('metricDescription');
        if (descEl) descEl.textContent = desc;
      }}

      const select = document.getElementById('metricSelect');
      select.addEventListener('change', (event) => {{
        showPanel(event.target.value);
      }});

      // Default to the first option
      if (select.value) {{
        showPanel(select.value);
      }}
    </script>
  </body>
</html>
"""

    output_path.write_text(wrapper, encoding="utf-8")


def write_highlight_html(
    main_fig: go.Figure,
    highlight_fig: go.Figure,
    output_path: Path,
    highlight_heading: str = "HIGHLIGHT",
    highlight_note: str | None = None,
) -> None:
    main_html = pio.to_html(
        main_fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={"responsive": True},
    )
    highlight_html = pio.to_html(
        highlight_fig,
        include_plotlyjs=False,
        full_html=False,
        config={"responsive": True},
    )

    note_html = (
        f'<div class="highlight-note">{highlight_note}</div>' if highlight_note else ""
    )

    wrapper = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Q3 City Seasonality (Highlights)</title>
    <style>
      body {{ margin: 0; font-family: Arial, sans-serif; }}
      .section {{ padding: 12px 16px 0 16px; }}
      .section h2 {{ margin: 12px 0 6px 0; font-size: 18px; }}
      .highlight-note {{ margin: 0 0 12px 0; color: #444; font-size: 13px; }}
      .map-panel {{ width: 100%; }}
    </style>
  </head>
  <body>
    <div class="map-panel">{main_html}</div>
    <div class="section">
      <h2>{highlight_heading}</h2>
      {note_html}
    </div>
    <div class="map-panel">{highlight_html}</div>
  </body>
</html>
"""

    output_path.write_text(wrapper, encoding="utf-8")


def write_story_html(
    output_path: Path,
    map_fig: go.Figure,
    swing_fig: go.Figure,
    trend_fig: go.Figure,
    top_states: list[str],
    city_spotlight_html: str | None = None,
) -> None:
    map_html = pio.to_html(
        map_fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={"responsive": True},
    )
    swing_html = pio.to_html(
        swing_fig,
        include_plotlyjs=False,
        full_html=False,
        config={"responsive": True},
    )
    trend_html = pio.to_html(
        trend_fig,
        include_plotlyjs=False,
        full_html=False,
        config={"responsive": True},
    )

    top_states_label = ", ".join(top_states) if top_states else "top states"

    spotlight_section = ""
    if city_spotlight_html:
        spotlight_section = f"""
        <section class="card">
          <h2>City spotlights that drive the peaks</h2>
          <p class="lead">
            These are the highest-grossing cities and their peak and quiet weeks.
            Use this as a quick shortlist when you need extra detail without a full city map.
          </p>
          {city_spotlight_html}
          <p class="so-what">
            So what: add these cities in peak weeks to amplify demand, and scale them back
            during quiet weeks.
          </p>
        </section>
        """

    wrapper = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Q3 Seasonality Story</title>
    <style>
      body {{
        margin: 0;
        font-family: Arial, sans-serif;
        color: #1b1b1b;
        background: #f6f6f6;
      }}
      main {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 24px;
      }}
      h1 {{
        margin: 0 0 6px 0;
        font-size: 28px;
      }}
      .lead {{
        margin: 0 0 12px 0;
        color: #444;
        font-size: 15px;
      }}
      .card {{
        background: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
      }}
      .card h2 {{
        margin: 6px 0 12px 0;
        font-size: 20px;
      }}
      .so-what {{
        margin-top: 10px;
        font-size: 13px;
        color: #444;
      }}
      .plot {{
        width: 100%;
      }}
      .plot .plotly-graph-div {{
        width: 100% !important;
      }}
      .takeaways {{
        border-left: 4px solid #1f77b4;
        background: #f2f7fb;
      }}
      .takeaways ul {{
        margin: 8px 0 0 18px;
      }}
      .spotlight-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }}
      .spotlight-table th,
      .spotlight-table td {{
        padding: 8px 10px;
        border-bottom: 1px solid #e5e5e5;
        text-align: left;
      }}
      .spotlight-table th {{
        background: #f8f8f8;
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="card">
        <h1>Q3 Seasonality Story - When and Where Demand Peaks</h1>
        <p class="lead">
          This view keeps the story tight: one state map for timing, plus two supporting
          visuals for strength and trend. City and cinema maps are removed to avoid repetition.
        </p>
      </section>

      <section class="card">
        <h2>1. Weekly seasonality map by state</h2>
        <p class="lead">
          Seasonality index above 1 means above-median demand for that state.
          Use the week slider to scan peaks and troughs.
        </p>
        <div class="plot">{map_html}</div>
        <p class="so-what">
          So what: weeks that turn red across multiple states are the safest windows for
          tentpole releases; blue-heavy weeks are lower-pressure windows.
        </p>
      </section>

      <section class="card">
        <h2>2. Seasonality distribution by state</h2>
        <p class="lead">
          Each violin shows how a state's weekly seasonality index is distributed across the year.
        </p>
        <div class="plot">{swing_html}</div>
        <p class="so-what">
          So what: wider violins mean more variability, while tighter shapes signal steadier demand.
        </p>
      </section>

      <section class="card">
        <h2>3. Timing profile for {top_states_label}</h2>
        <p class="lead">
          These are the largest markets by total gross. The trend lines show how their
          seasonality index moves through the year.
        </p>
        <div class="plot">{trend_html}</div>
        <p class="so-what">
          So what: align release timing to weeks when the top states peak together, and avoid
          weeks where they all soften.
        </p>
      </section>

      {spotlight_section}

      <section class="card takeaways">
        <h2>Planning takeaways</h2>
        <ul>
          <li>Target weeks where multiple states peak on the map for national releases.</li>
          <li>Use high-swing states for seasonally driven films; lean on stable states for long runs.</li>
          <li>Plan calendars around {top_states_label} since they set the overall pattern.</li>
        </ul>
      </section>
    </main>
  </body>
</html>
"""

    output_path.write_text(wrapper, encoding="utf-8")


def write_story_levels_html(
    output_path: Path,
    levels: list[dict[str, object]],
    title: str,
    intro: str,
) -> None:
    level_blocks = []
    first_fig = True
    first_level = True

    for level in levels:
        key = str(level.get("key", "")).strip()
        label = str(level.get("label", key)).strip() or key
        intro_text = str(level.get("intro", "")).strip()

        chart_blocks = []
        for chart in level.get("charts", []):
            heading = str(chart.get("heading", "")).strip()
            lead = str(chart.get("lead", "")).strip()
            fig = chart.get("fig")
            if fig is None:
                continue
            chart_html = _fig_to_html(fig, include_plotlyjs=first_fig)
            first_fig = False
            chart_blocks.append(
                f"""
                <section class="card">
                  <h3>{heading}</h3>
                  <p class="lead">{lead}</p>
                  <div class="plot">{chart_html}</div>
                </section>
                """
            )

        highlight_block = ""
        highlight = level.get("highlight") or {}
        highlight_heading = str(highlight.get("heading", "")).strip()
        highlight_note = str(highlight.get("note", "")).strip()
        highlight_fig = highlight.get("fig")
        highlight_html = highlight.get("html", "")
        if highlight_fig is not None:
            highlight_html = _fig_to_html(highlight_fig, include_plotlyjs=first_fig)
            first_fig = False
        if highlight_heading and highlight_html:
            highlight_block = f"""
            <section class="card highlight-card">
              <h3>{highlight_heading}</h3>
              <p class="lead">{highlight_note}</p>
              <div class="plot">{highlight_html}</div>
            </section>
            """

        spotlight_block = ""
        spotlight = level.get("spotlight") or {}
        spotlight_heading = str(spotlight.get("heading", "")).strip()
        spotlight_note = str(spotlight.get("note", "")).strip()
        spotlight_html = spotlight.get("html", "")
        if spotlight_heading and spotlight_html:
            spotlight_block = f"""
            <section class="card spotlight-card">
              <h3>{spotlight_heading}</h3>
              <p class="lead">{spotlight_note}</p>
              {spotlight_html}
            </section>
            """

        display_style = "block" if first_level else "none"
        first_level = False
        level_blocks.append(
            f"""
            <div class="level-section" id="level-{key}" style="display: {display_style};">
              <section class="card">
                <h2>{label} level</h2>
                <p class="lead">{intro_text}</p>
              </section>
              {''.join(chart_blocks)}
              {highlight_block}
              {spotlight_block}
            </div>
            """
        )

    options_html = "\n".join(
        [f'<option value="{level.get("key")}">{level.get("label")}</option>' for level in levels]
    )

    wrapper = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      body {{
        margin: 0;
        font-family: Arial, sans-serif;
        color: #1b1b1b;
        background: #f6f6f6;
      }}
      main {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 24px;
      }}
      h1 {{
        margin: 0 0 6px 0;
        font-size: 28px;
      }}
      h2 {{
        margin: 6px 0 10px 0;
        font-size: 22px;
      }}
      h3 {{
        margin: 6px 0 10px 0;
        font-size: 18px;
      }}
      .lead {{
        margin: 0 0 12px 0;
        color: #444;
        font-size: 14px;
      }}
      .toolbar {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        background: #ffffff;
        margin-bottom: 18px;
      }}
      .card {{
        background: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
      }}
      .plot {{
        width: 100%;
      }}
      .plot .plotly-graph-div {{
        width: 100% !important;
      }}
      .spotlight-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }}
      .spotlight-table th,
      .spotlight-table td {{
        padding: 8px 10px;
        border-bottom: 1px solid #e5e5e5;
        text-align: left;
      }}
      .spotlight-table th {{
        background: #f8f8f8;
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="card">
        <h1>{title}</h1>
        <p class="lead">{intro}</p>
      </section>
      <div class="toolbar">
        <label for="levelSelect"><strong>Level</strong></label>
        <select id="levelSelect">{options_html}</select>
      </div>
      {''.join(level_blocks)}
    </main>
    <script>
      const select = document.getElementById('levelSelect');
      function showLevel(levelKey) {{
        document.querySelectorAll('.level-section').forEach(section => {{
          section.style.display = 'none';
        }});
        const active = document.getElementById('level-' + levelKey);
        if (!active) return;
        active.style.display = 'block';
        active.querySelectorAll('.plotly-graph-div').forEach(plotDiv => {{
          if (plotDiv && window.Plotly) {{
            Plotly.Plots.resize(plotDiv);
          }}
        }});
      }}

      select.addEventListener('change', event => {{
        showLevel(event.target.value);
      }});
      if (select.value) {{
        showLevel(select.value);
      }}
    </script>
  </body>
</html>
"""

    output_path.write_text(wrapper, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sales_for_seasonality = prepare_sales_for_seasonality()

    required_state_cols = {
        "state",
        "iso_week",
        "avg_gross",
        "seasonality_idx",
    }
    required_city_cols = {
        "city",
        "iso_week",
        "avg_gross",
        "seasonality_idx",
    }

    if STATE_CSV.exists() and CITY_CSV.exists():
        state_df = pd.read_csv(STATE_CSV)
        city_df = pd.read_csv(CITY_CSV)
        if not required_state_cols.issubset(set(state_df.columns)) or not required_city_cols.issubset(
            set(city_df.columns)
        ):
            state_df, city_df = compute_seasonality_from_sales(sales_for_seasonality)
    else:
        state_df, city_df = compute_seasonality_from_sales(sales_for_seasonality)

    if not BASE_SA4_GEOJSON.exists():
        raise SystemExit("Missing base SA4 GeoJSON file")

    sa4 = gpd.read_file(BASE_SA4_GEOJSON)

    state_geojson_path = pick_geojson(
        STATE_GEOJSON, OUTPUT_DIR / "q3_state_geometry.geojson"
    )
    if state_geojson_path.exists():
        state_geojson = load_geojson(state_geojson_path)
    else:
        state_geom = build_state_geometry(sa4)
        state_geojson = gdf_to_geojson(state_geom)

    state_labels = label_points_from_geojson(state_geojson, "state")

    state_map_fig = build_animation(
        state_df,
        state_geojson,
        location_col="state",
        feature_key="properties.state",
        metric="seasonality_idx",
        title="Q3 State Seasonality Index by ISO Week",
        color_scale="RdBu_r",
        output_path=None,
        midpoint=1.0,
        label_points=state_labels,
    )

    state_violin_fig = build_state_violin_chart(
        state_df,
        metric="seasonality_idx",
        title="Seasonality distribution by state (violin)",
        height=380,
    )

    top_states = select_top_states_by_gross(state_df, top_n=STATE_TREND_TOP_N)
    state_trend_fig = build_state_trend_chart(
        state_df,
        metric="seasonality_idx",
        title="Seasonality profile of top states",
        top_states=top_states,
        height=360,
    )

    state_event_fig = build_event_highlight(
        state_df,
        metric="seasonality_idx",
        label="Seasonality Index",
        agg="mean",
        scope_label="states",
    )
    state_week_spotlight = build_week_spotlight_table(
        state_df,
        metric="seasonality_idx",
        top_n=WEEK_SPOTLIGHT_TOP_N,
        agg="mean",
        value_label="Seasonality Index",
    )

    city_order = (
        city_df.groupby("city", as_index=False)["avg_gross"]
        .mean()
        .sort_values("avg_gross", ascending=False)["city"]
        .tolist()
    )
    top_city_list = city_order[:CITY_HEATMAP_TOP_N]
    city_heatmap_df = city_df
    if top_city_list:
        city_heatmap_df = city_df[city_df["city"].isin(top_city_list)]

    city_heatmap_fig = build_heatmap(
        city_heatmap_df,
        y_col="city",
        metric="seasonality_idx",
        title="City seasonality index by ISO week (top cities)",
        color_scale="RdBu_r",
        midpoint=1.0,
        range_percentiles=(5, 95),
        y_order=top_city_list or None,
        yaxis_title="City",
        height=520,
    )
    city_location_fig = build_city_stability_scatter(
        city_df,
        top_n=CITY_SCATTER_TOP_N,
        label_top_n=CITY_SCATTER_LABEL_TOP_N,
        height=420,
    )
    city_event_fig = build_event_highlight(
        city_df,
        metric="seasonality_idx",
        label="Seasonality Index",
        agg="mean",
        scope_label="cities",
    )
    city_spotlight_html = build_city_spotlight_table(city_df, top_n=CITY_SPOTLIGHT_TOP_N)

    cinema_df = compute_cinema_seasonality_from_sales(sales_for_seasonality)
    cinema_order = (
        cinema_df.groupby("cinema_label", as_index=False)["avg_gross"]
        .mean()
        .sort_values("avg_gross", ascending=False)["cinema_label"]
        .tolist()
    )
    top_cinema_heatmap = cinema_order[:CINEMA_HEATMAP_TOP_N]
    top_cinema_trend = cinema_order[:CINEMA_TREND_TOP_N]
    cinema_heatmap_df = cinema_df
    if top_cinema_heatmap:
        cinema_heatmap_df = cinema_df[cinema_df["cinema_label"].isin(top_cinema_heatmap)]

    cinema_heatmap_fig = build_heatmap(
        cinema_heatmap_df,
        y_col="cinema_label",
        metric="seasonality_idx",
        title="Cinema seasonality index by ISO week (top cinemas)",
        color_scale="RdBu_r",
        midpoint=1.0,
        range_percentiles=(5, 95),
        y_order=top_cinema_heatmap or None,
        yaxis_title="Cinema",
        height=520,
    )
    cinema_trend_fig = build_top_n_line_chart(
        cinema_df,
        metric="seasonality_idx",
        title="Seasonality index by week (top cinemas + median)",
        top_labels=top_cinema_trend,
        median_label="Median (top cinemas)",
        height=420,
    )
    cinema_weeks_fig = build_weeks_to_50pct_chart(
        cinema_df,
        title="Weeks to 50% of annual gross (top cinemas)",
        group_col="cinema_label",
        week_col="iso_week",
        value_col="avg_gross",
        top_labels=top_cinema_trend,
        median_label="All cinemas (median)",
        height=420,
        yaxis_title="Cinema",
    )
    cinema_event_fig = build_event_highlight(
        cinema_df,
        metric="seasonality_idx",
        label="Seasonality Index",
        agg="median",
        scope_label="cinemas",
    )
    cinema_spotlight_html = build_cinema_spotlight_table(
        cinema_df, top_n=CINEMA_SPOTLIGHT_TOP_N
    )

    levels = [
        {
            "key": "state",
            "label": "State",
            "intro": "National timing view. Use this to spot weeks that peak across multiple states.",
            "charts": [
                {
                    "heading": "Weekly seasonality map",
                    "lead": "Seasonality index above 1 means above-median demand for that state.",
                    "fig": state_map_fig,
                },
                {
                    "heading": "Seasonality distribution by state",
                    "lead": "Violin width shows how volatile each state is across the year.",
                    "fig": state_violin_fig,
                },
                {
                    "heading": "Top states timing profile",
                    "lead": "Largest markets by total gross with a median benchmark.",
                    "fig": state_trend_fig,
                },
            ],
            "highlight": {
                "heading": "National peaks and troughs by week",
                "note": "Average seasonality index across states by ISO week.",
                "fig": state_event_fig,
            },
            "spotlight": {
                "heading": "Peak and quiet weeks (national average)",
                "note": "Top and bottom weeks based on average seasonality index.",
                "html": state_week_spotlight,
            },
        },
        {
            "key": "city",
            "label": "City",
            "intro": "City timing shows which local markets drive the statewide peaks.",
            "charts": [
                {
                    "heading": "Top cities seasonality heatmap",
                    "lead": "Compare which weeks overperform for the biggest cities.",
                    "fig": city_heatmap_fig,
                },
                {
                    "heading": "City scale vs volatility",
                    "lead": "Bigger cities with low volatility are the most reliable.",
                    "fig": city_location_fig,
                },
            ],
            "highlight": {
                "heading": "City-wide peaks by week",
                "note": "Average seasonality index across cities by ISO week.",
                "fig": city_event_fig,
            },
            "spotlight": {
                "heading": "City spotlights",
                "note": "Highest-grossing cities with their peak and quiet weeks.",
                "html": city_spotlight_html,
            },
        },
        {
            "key": "cinema",
            "label": "Cinema",
            "intro": "Cinema timing reveals where seasonality is most pronounced at venue level.",
            "charts": [
                {
                    "heading": "Top cinemas seasonality heatmap",
                    "lead": "Weekly seasonality index for the largest cinemas.",
                    "fig": cinema_heatmap_fig,
                },
                {
                    "heading": "Top cinemas timing profile",
                    "lead": "Seasonality index by week with a median reference line.",
                    "fig": cinema_trend_fig,
                },
                {
                    "heading": "Weeks to reach 50% of annual gross",
                    "lead": "Faster ramp-up signals stronger early demand.",
                    "fig": cinema_weeks_fig,
                },
            ],
            "highlight": {
                "heading": "Cinema-wide peaks by week",
                "note": "Median seasonality index across cinemas by ISO week.",
                "fig": cinema_event_fig,
            },
            "spotlight": {
                "heading": "Cinema spotlights",
                "note": "Top cinemas with their peak and quiet weeks.",
                "html": cinema_spotlight_html,
            },
        },
    ]

    write_story_levels_html(
        output_path=Q3_STATE_STORY_HTML,
        levels=levels,
        title="Q3 Seasonality Story - State, City, and Cinema",
        intro=(
            "Use the level toggle to switch between state, city, and cinema views. "
            "Each level has its own charts and spotlight, so the highlight stays in sync."
        ),
    )

    if not WRITE_DETAIL_OUTPUTS:
        return

    city_geojson_path = pick_geojson(
        CITY_GEOJSON, OUTPUT_DIR / "q3_city_sa4_geometry.geojson"
    )

    state_weeks_df = compute_weeks_to_50pct(
        state_df,
        group_col="state",
        week_col="iso_week",
        value_col="avg_gross",
    )
    city_weeks_df = compute_weeks_to_50pct(
        city_df,
        group_col="city",
        week_col="iso_week",
        value_col="avg_gross",
    )

    australia_outline = build_australia_outline(sa4)

    if city_geojson_path.exists():
        city_geojson = load_geojson(city_geojson_path)
    else:
        city_names = sorted(city_df["city"].dropna().astype(str).unique().tolist())
        city_geom = build_city_geometry(sa4, city_names)
        city_geojson = gdf_to_geojson(city_geom)

    state_maps = [
        {
            "metric": "gross_z",
            "title": "Q3 State Seasonality (Z-score) by ISO Week",
            "scale": "RdBu_r",
            "midpoint": 0.0,
        },
        {
            "metric": "avg_titles",
            "title": "Q3 State Average Titles by ISO Week",
            "scale": "YlOrRd",
            "midpoint": None,
        },
        {
            "metric": "avg_cinemas",
            "title": "Q3 State Average Cinemas by ISO Week",
            "scale": "YlGn",
            "midpoint": None,
        },
        {
            "metric": "weeks_to_50pct",
            "title": "Q3 State Weeks to 50% Gross",
            "scale": "YlOrRd",
            "midpoint": None,
            "data": state_weeks_df,
        },
    ]

    state_figures: dict[str, go.Figure] = {}
    for spec in state_maps:
        metric = spec["metric"]
        if metric == "weeks_to_50pct":
            fig = build_weeks_to_50pct_chart(
                state_df,
                title=spec["title"],
                group_col="state",
                week_col="iso_week",
                value_col="avg_gross",
                top_labels=None,
                median_label="All states (median)",
                height=620,
                yaxis_title="State",
            )
        else:
            fig = build_animation(
                state_df,
                state_geojson,
                location_col="state",
                feature_key="properties.state",
                metric=metric,
                title=spec["title"],
                color_scale=spec["scale"],
                output_path=None,
                midpoint=spec["midpoint"],
                label_points=state_labels,
            )
        state_figures[spec["metric"]] = fig

    state_highlights_by_metric: dict[str, list[dict[str, object]]] = {}
    for spec in state_maps:
        metric = spec["metric"]
        if metric == "weeks_to_50pct":
            continue
        label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        event_fig = build_heatmap(
            state_df,
            y_col="state",
            metric=metric,
            title=f"Q3 State {label} Heatmap by ISO Week",
            color_scale=spec["scale"],
            midpoint=spec["midpoint"],
            range_percentiles=None,
            y_order=None,
            yaxis_title="State",
            height=360,
        )

        state_highlights_by_metric[metric] = [
            {
                "heading": "Event Highlight",
                "note": "Heatmap by state and ISO week.",
                "fig": event_fig,
            },
        ]

    write_selector_html(
        figures=state_figures,
        output_path=OUTPUT_DIR / "q3_state_seasonality_selector.html",
        options=[spec["metric"] for spec in state_maps],
        label_map=METRIC_LABELS,
        descriptions=METRIC_DESCRIPTIONS,
        title="Q3 State Seasonality Maps",
        highlight_sections_by_metric=state_highlights_by_metric,
    )

    city_labels = label_points_from_geojson(city_geojson, "city")
    city_candidates = (
        city_df.groupby("city", as_index=False)["avg_gross"]
        .mean()
        .sort_values("avg_gross", ascending=False)
        ["city"]
        .tolist()
    )
    top_cities = select_spread_labels(
        city_candidates,
        city_labels,
        CITY_LABEL_MAX_COUNT,
        CITY_LABEL_MIN_DISTANCE_DEG,
    )
    city_top_n = min(len(city_candidates), 20)
    top_city_list = city_candidates[:city_top_n]

    city_metric_specs = [
        {
            "metric": "gross_z",
            "title": "Q3 City Gross Z-Score by ISO Week",
            "scale": "RdBu_r",
            "midpoint": 0.0,
            "range_percentiles": None,
        },
        {
            "metric": "avg_titles",
            "title": "Q3 City Average Titles by ISO Week",
            "scale": "YlOrRd",
            "midpoint": None,
            "range_percentiles": (5, 95),
        },
        {
            "metric": "avg_cinemas",
            "title": "Q3 City Average Cinemas by ISO Week",
            "scale": "YlGn",
            "midpoint": None,
            "range_percentiles": (5, 95),
        },
        {
            "metric": "weeks_to_50pct",
            "title": "Q3 City Weeks to 50% Gross",
            "scale": "YlOrRd",
            "midpoint": None,
            "range_percentiles": None,
            "data": city_weeks_df,
        },
    ]

    city_figures: dict[str, go.Figure] = {}
    for spec in city_metric_specs:
        metric = spec["metric"]
        if metric == "weeks_to_50pct":
            fig = build_weeks_to_50pct_chart(
                city_df,
                title=spec["title"],
                group_col="city",
                week_col="iso_week",
                value_col="avg_gross",
                top_labels=top_city_list,
                median_label="All cities (median)",
                height=620,
                yaxis_title="City",
            )
        else:
            source_df = spec.get("data", city_df)
            fig = build_animation(
                source_df,
                city_geojson,
                location_col="city",
                feature_key="properties.city",
                metric=metric,
                title=spec["title"],
                color_scale=spec["scale"],
                output_path=None,
                midpoint=spec["midpoint"],
                label_points=city_labels,
                label_names=top_cities,
                label_font_size=9,
                label_textposition="top center",
                range_percentiles=spec["range_percentiles"],
                border_geojson=australia_outline,
                write_html=False,
            )
        city_figures[spec["metric"]] = fig

    city_highlights_by_metric: dict[str, list[dict[str, object]]] = {}
    for spec in city_metric_specs:
        metric = spec["metric"]
        if metric == "weeks_to_50pct":
            continue
        label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        peak_low_df = build_peak_low_score(
            city_df,
            metric=metric,
            group_cols=["state", "city"],
            high_quantile=CITY_PEAK_QUANTILE,
            low_quantile=CITY_LOW_QUANTILE,
        )
        peak_low_fig = build_animation(
            peak_low_df,
            city_geojson,
            location_col="city",
            feature_key="properties.city",
            metric="peak_low_score",
            title=f"Q3 City {label} Peak/Low Weeks by ISO Week",
            color_scale="RdBu_r",
            output_path=None,
            midpoint=0.0,
            label_points=city_labels,
            label_names=top_cities,
            label_font_size=9,
            label_textposition="top center",
            range_percentiles=None,
            border_geojson=australia_outline,
            write_html=False,
        )

        event_fig = build_event_highlight(city_df, metric=metric, label=label)
        location_fig = build_location_highlight(
            city_df,
            metric=metric,
            label=label,
            location_col="city",
            agg="median",
            scope_label="cities",
        )
        location_note = "Median across weeks; red = highest cities, blue = lowest cities."

        city_highlights_by_metric[metric] = [
            {
                "heading": "Peak/Low Highlight",
                "note": "Top 10% = peak, bottom 10% = low.",
                "fig": peak_low_fig,
            },
            {
                "heading": "Event Highlight",
                "note": "Avg across cities; red = highest weeks, blue = lowest weeks.",
                "fig": event_fig,
            },
            {
                "heading": "Event Highlight (Location)",
                "note": location_note,
                "fig": location_fig,
            },
        ]

    write_selector_html(
        figures=city_figures,
        output_path=OUTPUT_DIR / "q3_city_seasonality_selector.html",
        options=[spec["metric"] for spec in city_metric_specs],
        label_map=METRIC_LABELS,
        descriptions=METRIC_DESCRIPTIONS,
        title="Q3 City Seasonality Maps",
        highlight_sections_by_metric=city_highlights_by_metric,
    )

    cinema_df = compute_cinema_seasonality_from_sales(sales_for_seasonality)
    cinema_order = (
        cinema_df.groupby("cinema_label", as_index=False)["avg_gross"]
        .mean()
        .sort_values("avg_gross", ascending=False)
        ["cinema_label"]
        .tolist()
    )
    top_cinemas = cinema_order[:CINEMA_TOP_N]
    cinema_top_n = len(top_cinemas)
    median_label = "All cinemas (median)"

    cinema_metric_specs = [
        {
            "metric": "gross_z",
        },
        {
            "metric": "avg_titles",
        },
        {
            "metric": "avg_gross",
        },
        {
            "metric": "weeks_to_50pct",
        },
    ]

    cinema_figures: dict[str, go.Figure] = {}
    for spec in cinema_metric_specs:
        metric = spec["metric"]
        label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        title = (
            f"Q3 Cinema {label} by ISO Week (Top {cinema_top_n} cinemas + median)"
        )
        if metric == "weeks_to_50pct":
            title = f"Q3 Cinema {label} (Top {cinema_top_n} cinemas + median)"
            fig = build_weeks_to_50pct_chart(
                cinema_df,
                title=title,
                group_col="cinema_label",
                week_col="iso_week",
                value_col="avg_gross",
                top_labels=top_cinemas,
                median_label=median_label,
                height=620,
                yaxis_title="Cinema",
            )
        else:
            fig = build_top_n_line_chart(
                cinema_df,
                metric=metric,
                title=title,
                top_labels=top_cinemas,
                median_label=median_label,
                height=620,
            )
        cinema_figures[metric] = fig

    cinema_highlights_by_metric: dict[str, list[dict[str, object]]] = {}
    for spec in cinema_metric_specs:
        metric = spec["metric"]
        if metric == "weeks_to_50pct":
            continue
        label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        event_fig = build_event_highlight(
            cinema_df,
            metric=metric,
            label=label,
            agg="median",
            scope_label="cinemas",
        )
        location_fig = build_location_highlight(
            cinema_df,
            metric=metric,
            label=label,
            location_col="cinema_label",
            agg="median",
            scope_label="cinemas",
        )

        cinema_highlights_by_metric[metric] = [
            {
                "heading": "Event Highlight",
                "note": "Median across cinemas; red = highest weeks, blue = lowest weeks.",
                "fig": event_fig,
            },
            {
                "heading": "Event Highlight (Location)",
                "note": "Median across weeks; red = highest cinemas, blue = lowest cinemas.",
                "fig": location_fig,
            },
        ]

    write_selector_html(
        figures=cinema_figures,
        output_path=OUTPUT_DIR / "q3_cinema_seasonality_selector.html",
        options=[spec["metric"] for spec in cinema_metric_specs],
        label_map=METRIC_LABELS,
        descriptions=METRIC_DESCRIPTIONS,
        title="Q3 Cinema Seasonality Maps",
        highlight_sections_by_metric=cinema_highlights_by_metric,
    )

    if WRITE_CITY_YEAR_SELECTOR:
        city_year_week = compute_city_week_by_year(sales_for_seasonality)
        years = sorted(city_year_week["iso_year"].dropna().unique().tolist())

        year_figures: dict[str, go.Figure] = {}
        for year in years:
            year_df = city_year_week[city_year_week["iso_year"] == year].copy()
            fig = build_animation(
                year_df,
                city_geojson,
                location_col="city",
                feature_key="properties.city",
                metric="seasonality_idx",
                title=f"Q3 City Seasonality Index by ISO Week ({year})",
                color_scale="RdBu_r",
                output_path=None,
                midpoint=1.0,
                label_points=city_labels,
                label_names=top_cities,
                label_font_size=9,
                label_textposition="top center",
                range_percentiles=(5, 95),
                border_geojson=australia_outline,
                write_html=False,
            )
            year_figures[str(year)] = fig

        year_labels = {str(year): f"ISO Year {year}" for year in years}
        year_descriptions = {
            str(year): "Seasonality index within the selected year."
            for year in years
        }

        write_selector_html(
            figures=year_figures,
            output_path=OUTPUT_DIR / "q3_city_seasonality_year_selector.html",
            options=[str(year) for year in years],
            label_map=year_labels,
            descriptions=year_descriptions,
            title="Q3 City Seasonality by Year",
            select_label="Year",
        )


if __name__ == "__main__":
    main()
