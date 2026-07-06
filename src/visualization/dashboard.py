"""
Quality Dashboard - Interactive Visualizations
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Dashboard:
    """Create interactive quality dashboard"""
    
    def create_quality_dashboard(self, data: pd.DataFrame, report: dict) -> go.Figure:
        """Create full quality dashboard"""
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Data Overview', 'Correlation', 'Distribution',
                'Quality Score', 'Missing Values', 'Data Types',
                'Outliers', 'Feature Stats', 'Privacy Score'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'heatmap'}, {'type': 'histogram'}],
                [{'type': 'indicator'}, {'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'indicator'}]
            ]
        )
        
        # Data Overview
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=data[numeric_cols[0]][:500],
                    y=data[numeric_cols[1]][:500],
                    mode='markers',
                    marker=dict(size=5, opacity=0.5),
                    name='Data'
                ),
                row=1, col=1
            )
        
        # Correlation
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
        
        # Distribution
        if len(numeric_cols) > 0:
            fig.add_trace(
                go.Histogram(
                    x=data[numeric_cols[0]],
                    nbinsx=30,
                    name='Distribution',
                    marker_color='#667eea'
                ),
                row=1, col=3
            )
        
        # Quality Score
        score = report.get('overall_score', 0.5) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Overall Quality"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#6bcb77' if score >= 70 else '#ffd93d' if score >= 40 else '#ff6b6b'},
                    'steps': [
                        {'range': [0, 40], 'color': "#ff6b6b"},
                        {'range': [40, 70], 'color': "#ffd93d"},
                        {'range': [70, 100], 'color': "#6bcb77"}
                    ]
                }
            ),
            row=2, col=1
        )
        
        # Missing Values
        nulls = data.isnull().sum()
        if nulls.sum() > 0:
            fig.add_trace(
                go.Bar(
                    x=nulls.index[:10],
                    y=nulls.values[:10],
                    name='Missing',
                    marker_color='#ff6b6b'
                ),
                row=2, col=2
            )
        
        # Data Types
        types = data.dtypes.value_counts()
        fig.add_trace(
            go.Pie(
                labels=types.index.astype(str),
                values=types.values,
                name='Data Types'
            ),
            row=2, col=3
        )
        
        # Outliers
        if len(numeric_cols) > 0:
            outlier_counts = {}
            for col in numeric_cols[:5]:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < Q1 - 1.5*IQR) | (data[col] > Q3 + 1.5*IQR)).sum()
                outlier_counts[col] = outliers
            
            fig.add_trace(
                go.Bar(
                    x=list(outlier_counts.keys()),
                    y=list(outlier_counts.values()),
                    name='Outliers',
                    marker_color='#ff6b6b'
                ),
                row=3, col=1
            )
        
        # Feature Statistics
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
                        marker_color='#667eea'
                    ),
                    row=3, col=2
                )
        
        # Privacy Score
        privacy = report.get('privacy_score', 0.5) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=privacy,
                title={'text': "Privacy Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#6bcb77' if privacy >= 70 else '#ffd93d' if privacy >= 40 else '#ff6b6b'},
                    'steps': [
                        {'range': [0, 40], 'color': "#ff6b6b"},
                        {'range': [40, 70], 'color': "#ffd93d"},
                        {'range': [70, 100], 'color': "#6bcb77"}
                    ]
                }
            ),
            row=3, col=3
        )
        
        fig.update_layout(height=1000, showlegend=False)
        return fig