"""
Shared Plotly chart theme aligned with Project Synthesis brand colors.
"""

from __future__ import annotations

import streamlit as st

BRAND_PRIMARY = "#667eea"
BRAND_SECONDARY = "#764ba2"
SCORE_GOOD = "#6bcb77"
SCORE_MID = "#ffd93d"
SCORE_LOW = "#ff6b6b"

BRAND_PALETTE = [BRAND_PRIMARY, BRAND_SECONDARY, SCORE_LOW, SCORE_MID, SCORE_GOOD]

DASHBOARD_SUBPLOT_TITLES = (
    "Data Overview",
    "Correlation Matrix",
    "Distribution Plot",
    "Quality Score",
    "Missing Values",
    "Data Types",
    "Feature Importance",
    "Outlier Analysis",
    "Privacy Score",
)


def plotly_template() -> str:
    """Pick Plotly template based on Streamlit theme."""
    try:
        base = st.get_option("theme.base")
        return "plotly_dark" if base == "dark" else "plotly_white"
    except Exception:
        return "plotly_dark"


def score_color(score_pct: float) -> str:
    """Return gauge/bar color for a 0–100 score."""
    if score_pct >= 70:
        return SCORE_GOOD
    if score_pct >= 40:
        return SCORE_MID
    return SCORE_LOW


def gauge_steps() -> list:
    return [
        {"range": [0, 40], "color": SCORE_LOW},
        {"range": [40, 70], "color": SCORE_MID},
        {"range": [70, 100], "color": SCORE_GOOD},
    ]


def gauge_config(score_pct: float, title: str, *, with_delta: bool = False) -> dict:
    """Build a Plotly Indicator gauge trace config."""
    gauge = {
        "axis": {"range": [0, 100], "tickwidth": 1},
        "bar": {"color": score_color(score_pct)},
        "steps": gauge_steps(),
    }
    cfg = {
        "mode": "gauge+number+delta" if with_delta else "gauge+number",
        "value": score_pct,
        "title": {"text": title, "font": {"size": 14}},
        "gauge": gauge,
    }
    if with_delta:
        cfg["delta"] = {
            "reference": 50,
            "increasing": {"color": SCORE_GOOD},
            "decreasing": {"color": SCORE_LOW},
        }
    return cfg


def apply_dashboard_layout(
    fig,
    *,
    title: str,
    height: int = 900,
    showlegend: bool = False,
) -> None:
    """Apply consistent dashboard layout to a Plotly figure."""
    template = plotly_template()
    title_color = "#fafafa" if template == "plotly_dark" else "#2c3e50"
    fig.update_layout(
        height=height,
        showlegend=showlegend,
        template=template,
        font=dict(size=12),
        title={
            "text": title,
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 22, "color": title_color},
        },
        margin=dict(l=50, r=50, t=80, b=50),
    )
