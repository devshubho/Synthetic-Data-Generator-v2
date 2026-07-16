"""
Shared Streamlit UI theme — CSS tokens, page chrome, and metric helpers.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

APP_NAME = "SynthSLM"
APP_TAGLINE = "Synthetic data generation for small language models"
APP_VERSION = "3.0"

CSS = """
<style>
:root {
    --brand-primary: #667eea;
    --brand-secondary: #764ba2;
    --score-good: #6bcb77;
    --score-mid: #ffd93d;
    --score-low: #ff6b6b;
}

.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.25rem;
    background: linear-gradient(135deg, var(--brand-primary) 0%, var(--brand-secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.page-header {
    font-size: 1.75rem;
    font-weight: 600;
    margin: 0 0 0.35rem 0;
    color: var(--text-color);
}

.page-subtitle {
    text-align: center;
    font-size: 1.05rem;
    color: rgba(128, 128, 128, 0.85);
    margin: 0 0 1.5rem 0;
}

.hero-subtitle {
    text-align: center;
    font-size: 1.05rem;
    color: rgba(128, 128, 128, 0.85);
    margin: 0 0 1.75rem 0;
}

.stats-card {
    background: linear-gradient(135deg, var(--brand-primary) 0%, var(--brand-secondary) 100%);
    color: white;
    padding: 1.25rem 1rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(102, 126, 234, 0.25);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    margin-bottom: 0.5rem;
}

.stats-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.35);
}

.stats-card h3 {
    margin: 0 0 0.25rem 0;
    font-size: 1.75rem;
    font-weight: 700;
    color: white;
}

.stats-card p {
    margin: 0;
    font-size: 0.875rem;
    opacity: 0.92;
    color: white;
}

.feature-card {
    background: var(--secondary-background-color);
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    border-left: 4px solid var(--brand-primary);
    margin: 0.5rem 0 1rem 0;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
}

.feature-card h3 {
    margin-top: 0;
    font-size: 1.15rem;
    color: var(--text-color);
}

.feature-card ul, .feature-card ol {
    margin-bottom: 0;
    padding-left: 1.25rem;
}

.upload-area {
    border: 2px dashed var(--brand-primary);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    background: var(--secondary-background-color);
    margin: 1rem 0 1.5rem 0;
    transition: border-color 0.15s ease, background 0.15s ease;
}

.upload-area:hover {
    border-color: var(--brand-secondary);
    background: rgba(102, 126, 234, 0.06);
}

.upload-area h3 {
    margin-top: 0;
    color: var(--text-color);
}

.upload-area p {
    color: rgba(128, 128, 128, 0.85);
    margin-bottom: 0;
}

.section-card {
    background: var(--secondary-background-color);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(128, 128, 128, 0.15);
}

.success-box {
    background: rgba(107, 203, 119, 0.12);
    border: 1px solid rgba(107, 203, 119, 0.35);
    border-radius: 8px;
    padding: 1rem;
    color: var(--score-good);
}

.empty-state {
    text-align: center;
    padding: 2.5rem 1.5rem;
    border-radius: 12px;
    background: var(--secondary-background-color);
    border: 1px dashed rgba(128, 128, 128, 0.3);
    margin: 1rem 0;
}

.empty-state h3 {
    margin: 0 0 0.5rem 0;
    color: var(--text-color);
}

.empty-state p {
    margin: 0;
    color: rgba(128, 128, 128, 0.85);
}

.sidebar-brand {
    font-size: 1.35rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}

.sidebar-tagline {
    font-size: 0.78rem;
    color: rgba(128, 128, 128, 0.85);
    margin-bottom: 0.75rem;
    line-height: 1.35;
}

.status-pill {
    display: inline-block;
    padding: 0.35rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    margin-bottom: 0.4rem;
    background: rgba(102, 126, 234, 0.15);
    color: var(--text-color);
    border: 1px solid rgba(102, 126, 234, 0.25);
}

.status-pill.ok {
    background: rgba(107, 203, 119, 0.12);
    border-color: rgba(107, 203, 119, 0.3);
}

.status-pill.muted {
    background: rgba(128, 128, 128, 0.1);
    border-color: rgba(128, 128, 128, 0.2);
}

.metric-band-good [data-testid="stMetricValue"] {
    color: var(--score-good) !important;
}

.metric-band-mid [data-testid="stMetricValue"] {
    color: var(--score-mid) !important;
}

.metric-band-low [data-testid="stMetricValue"] {
    color: var(--score-low) !important;
}

@media (max-width: 768px) {
    .main-header { font-size: 2rem; }
    .stats-card h3 { font-size: 1.4rem; }
}
</style>
"""


def inject_styles() -> None:
    """Inject global CSS into the Streamlit app."""
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: Optional[str] = None) -> None:
    """Consistent inner-page header."""
    st.markdown(f'<h2 class="page-header">{title}</h2>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="page-subtitle" style="text-align:left;">{subtitle}</p>', unsafe_allow_html=True)


def empty_state(title: str, message: str) -> None:
    """Styled placeholder when a page has no data yet."""
    st.markdown(
        f'<div class="empty-state"><h3>{title}</h3><p>{message}</p></div>',
        unsafe_allow_html=True,
    )


def format_score_pct(value: Optional[float], clamp: bool = True) -> str:
    """Format a 0–1 score as percentage; optionally clamp to [0, 1]."""
    if value is None:
        return "N/A"
    if clamp:
        value = max(0.0, min(1.0, float(value)))
    return f"{value:.1%}"


def _score_band(value: Optional[float]) -> str:
    if value is None:
        return ""
    v = max(0.0, min(1.0, float(value)))
    if v >= 0.7:
        return "metric-band-good"
    if v >= 0.4:
        return "metric-band-mid"
    return "metric-band-low"


def quality_metric(label: str, value: Optional[float], clamp: bool = True) -> None:
    """Render a color-banded quality metric."""
    band = _score_band(value) if value is not None else ""
    if band:
        st.markdown(f'<div class="{band}">', unsafe_allow_html=True)
    st.metric(label, format_score_pct(value, clamp=clamp))
    if band:
        st.markdown("</div>", unsafe_allow_html=True)


def styled_dataframe(df: pd.DataFrame, max_rows: int = 20) -> None:
    """Render a preview dataframe with sensible column formatting."""
    preview = df.head(max_rows).copy()
    column_config = {}
    for col in preview.columns:
        series = preview[col]
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                column_config[col] = st.column_config.NumberColumn(format="%d")
            else:
                column_config[col] = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(preview, use_container_width=True, column_config=column_config or None)


def sidebar_brand() -> None:
    """Render sidebar brand block."""
    st.sidebar.markdown(f'<div class="sidebar-brand">{APP_NAME}</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="sidebar-tagline">{APP_TAGLINE}</div>', unsafe_allow_html=True)


def status_pill(label: str, *, ok: bool = True) -> None:
    """Compact sidebar status indicator."""
    css_class = "status-pill ok" if ok else "status-pill muted"
    st.sidebar.markdown(f'<div class="{css_class}">{label}</div>', unsafe_allow_html=True)
