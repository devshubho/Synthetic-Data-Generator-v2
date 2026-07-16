"""
Quality Dashboard - Interactive Visualizations
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from logger import get_logger
from visualization.chart_theme import (
    BRAND_PRIMARY,
    BRAND_PALETTE,
    SCORE_LOW,
    DASHBOARD_SUBPLOT_TITLES,
    gauge_config,
    apply_dashboard_layout,
)

logger = get_logger()

class Dashboard:
    """Create interactive quality dashboard"""
    
    def create_quality_dashboard(self, data: pd.DataFrame, report: dict) -> go.Figure:
        """Create full quality dashboard"""
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=DASHBOARD_SUBPLOT_TITLES,
            specs=[
                [{'type': 'scatter'}, {'type': 'heatmap'}, {'type': 'histogram'}],
                [{'type': 'indicator'}, {'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'indicator'}]
            ]
        )
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=data[numeric_cols[0]][:500],
                    y=data[numeric_cols[1]][:500],
                    mode='markers',
                    marker=dict(size=5, opacity=0.5, color=BRAND_PRIMARY),
                    name='Data'
                ),
                row=1, col=1
            )
        
        if len(numeric_cols) > 1:
            corr = data[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale='RdBu',
                    zmid=0,
                    name='Correlation'
                ),
                row=1, col=2
            )
        
        if len(numeric_cols) > 0:
            fig.add_trace(
                go.Histogram(
                    x=data[numeric_cols[0]],
                    nbinsx=30,
                    name='Distribution',
                    marker_color=BRAND_PRIMARY
                ),
                row=1, col=3
            )
        
        score = report.get('overall_score', 0.5) * 100
        fig.add_trace(
            go.Indicator(**gauge_config(score, "Overall Quality")),
            row=2, col=1
        )
        
        nulls = data.isnull().sum()
        if nulls.sum() > 0:
            fig.add_trace(
                go.Bar(
                    x=nulls.index[:10],
                    y=nulls.values[:10],
                    name='Missing',
                    marker_color=SCORE_LOW
                ),
                row=2, col=2
            )
        
        types = data.dtypes.value_counts()
        if len(types) > 0:
            fig.add_trace(
                go.Pie(
                    labels=types.index.astype(str),
                    values=types.values,
                    name='Data Types',
                    marker=dict(colors=BRAND_PALETTE[:len(types)]),
                ),
                row=2, col=3
            )
        
        if len(numeric_cols) > 0:
            outlier_counts = {}
            for col in numeric_cols[:5]:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < Q1 - 1.5*IQR) | (data[col] > Q3 + 1.5*IQR)).sum()
                outlier_counts[col] = outliers
            
            if outlier_counts:
                fig.add_trace(
                    go.Bar(
                        x=list(outlier_counts.keys()),
                        y=list(outlier_counts.values()),
                        name='Outliers',
                        marker_color=SCORE_LOW
                    ),
                    row=3, col=1
                )
        
        if 'statistics' in report:
            stats = report['statistics']
            std_values = {}
            for col, info in stats.items():
                if 'std' in info:
                    std_values[col] = info['std']
            
            if std_values:
                sorted_vals = sorted(std_values.items(), key=lambda x: x[1], reverse=True)[:10]
                fig.add_trace(
                    go.Bar(
                        x=[v for _, v in sorted_vals],
                        y=[k for k, _ in sorted_vals],
                        orientation='h',
                        name='Std Dev',
                        marker_color=BRAND_PRIMARY
                    ),
                    row=3, col=2
                )
        
        privacy = report.get('privacy_score', 0.5) * 100
        fig.add_trace(
            go.Indicator(**gauge_config(privacy, "Privacy Score")),
            row=3, col=3
        )
        
        apply_dashboard_layout(fig, title="Synthetic Data Quality Dashboard", height=1000)
        return fig
