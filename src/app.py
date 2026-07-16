"""
SynthSLM - Complete Application with Enhanced Visualization
B.Tech Final Year Project
Author: Team CSE_13
"""

import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
import random
import time
import io
import sys
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure src/ is on path when running via streamlit
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from engine.pipeline import GenerationPipeline
from analytics.quality_report import QualityReporter
from export.excel_export import make_excel_safe

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="SynthSLM", page_icon="🎲", layout="wide")

# ==================== SESSION STATE ====================
if 'init' not in st.session_state:
    st.session_state.init = True
    st.session_state.generated_data = None
    st.session_state.sample_data = None
    st.session_state.quality_report = None
    st.session_state.generation_history = []

# ==================== CSS ====================
st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: bold; text-align: center;
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stats-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  color: white; padding: 1.5rem; border-radius: 10px; text-align: center; }
    .feature-card { background: #f8f9fa; padding: 1.5rem; border-radius: 10px;
                    border-left: 4px solid #667eea; margin: 1rem 0; }
    .upload-area { border: 2px dashed #667eea; border-radius: 12px; padding: 2rem;
                   text-align: center; background: #f8f9fa; }
    .success-box { background: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; padding: 1rem; color: #155724; }
</style>
""", unsafe_allow_html=True)

# ==================== GENERATORS ====================

def generate_personal_data(n):
    fake = Faker()
    data = []
    for _ in range(n):
        first = fake.first_name()
        last = fake.last_name()
        data.append({
            'id': fake.uuid4(),
            'first_name': first,
            'last_name': last,
            'email': f"{first.lower()}.{last.lower()}@{fake.free_email_domain()}",
            'phone': fake.phone_number(),
            'address': fake.address().replace('\n', ', '),
            'city': fake.city(),
            'state': fake.state(),
            'zipcode': fake.zipcode(),
            'birth_date': fake.date_of_birth(minimum_age=18, maximum_age=80),
            'gender': random.choice(['Male', 'Female', 'Non-binary']),
            'occupation': fake.job(),
            'income': random.randint(30000, 200000),
            'education': random.choice(['High School', 'Bachelor', 'Master', 'PhD']),
            'active': random.choice([True, False])
        })
    return pd.DataFrame(data)

def generate_sales_data(n):
    fake = Faker()
    products = ['Laptop', 'Smartphone', 'Headphones', 'Monitor', 'Keyboard', 'Mouse', 'Desk', 'Chair']
    categories = ['Electronics', 'Accessories', 'Furniture']
    data = []
    for _ in range(n):
        product = random.choice(products)
        price = random.randint(100, 2000)
        qty = random.randint(1, 10)
        data.append({
            'transaction_id': fake.uuid4(),
            'customer_id': fake.uuid4(),
            'product': product,
            'category': random.choice(categories),
            'quantity': qty,
            'unit_price': price,
            'total': qty * price,
            'discount': random.choice([0, 5, 10, 15, 20]),
            'payment_method': random.choice(['Credit Card', 'Debit Card', 'PayPal', 'UPI']),
            'transaction_date': fake.date_time_between(start_date='-1y'),
            'region': random.choice(['North', 'South', 'East', 'West']),
            'rating': random.randint(1, 5)
        })
    return pd.DataFrame(data)

def generate_employee_data(n):
    fake = Faker()
    depts = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
    positions = ['Intern', 'Junior', 'Mid-Level', 'Senior', 'Lead', 'Manager', 'Director']
    data = []
    for _ in range(n):
        first = fake.first_name()
        last = fake.last_name()
        data.append({
            'employee_id': f"EMP{random.randint(10000, 99999)}",
            'first_name': first,
            'last_name': last,
            'email': f"{first.lower()}.{last.lower()}@company.com",
            'department': random.choice(depts),
            'position': random.choice(positions),
            'hire_date': fake.date_between(start_date='-10y'),
            'salary': random.randint(40000, 150000),
            'performance_rating': round(random.uniform(1, 5), 1),
            'experience': random.randint(0, 20),
            'education': random.choice(['Bachelor', 'Master', 'PhD', 'MBA']),
            'remote': random.choice([True, False])
        })
    return pd.DataFrame(data)

def generate_timeseries_data(n):
    start = datetime.now() - timedelta(days=n)
    dates = [start + timedelta(days=i) for i in range(n)]
    trend = np.linspace(0, 50, n)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n) / 30)
    noise = np.random.normal(0, 5, n)
    values = 100 + trend + seasonality + noise
    return pd.DataFrame({
        'date': dates,
        'value': values,
        'moving_avg_7d': pd.Series(values).rolling(7, min_periods=1).mean(),
        'anomaly': np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    })

def generate_logs_data(n):
    fake = Faker()
    levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    services = ['auth', 'payment', 'user', 'order', 'notification']
    data = []
    for _ in range(n):
        data.append({
            'timestamp': fake.date_time_between(start_date='-7d'),
            'log_level': random.choice(levels),
            'service': random.choice(services),
            'message': f"{random.choice(services)}: {fake.sentence()}",
            'source_ip': fake.ipv4() if random.random() < 0.5 else None,
            'user_id': fake.uuid4() if random.random() < 0.3 else None,
            'response_time_ms': random.randint(10, 5000),
            'status_code': random.choice([200, 201, 400, 401, 403, 404, 500])
        })
    return pd.DataFrame(data)

def generate_system_data(n):
    fake = Faker()
    hosts = [f'host-{i:02d}' for i in range(1, 11)]
    data = []
    for _ in range(n):
        data.append({
            'timestamp': fake.date_time_between(start_date='-7d'),
            'hostname': random.choice(hosts),
            'cpu_usage': round(random.uniform(10, 90), 2),
            'memory_usage': round(random.uniform(20, 85), 2),
            'disk_usage': round(random.uniform(10, 95), 2),
            'network_in_mbps': round(random.uniform(0.1, 100), 2),
            'network_out_mbps': round(random.uniform(0.1, 80), 2),
            'process_count': random.randint(50, 500)
        })
    return pd.DataFrame(data)

def generate_iot_data(n):
    fake = Faker()
    devices = [f'sensor-{i:04d}' for i in range(1, 51)]
    types = ['temperature', 'humidity', 'pressure', 'motion', 'light']
    data = []
    for _ in range(n):
        dtype = random.choice(types)
        value = {
            'temperature': 15 + random.random() * 20,
            'humidity': 20 + random.random() * 60,
            'pressure': 980 + random.random() * 50,
            'motion': random.randint(0, 100) if random.random() < 0.3 else 0,
            'light': random.random() * 1000
        }[dtype]
        data.append({
            'timestamp': fake.date_time_between(start_date='-7d'),
            'device_id': random.choice(devices),
            'device_type': dtype,
            'value': round(value, 2),
            'battery_level': random.randint(10, 100),
            'signal_strength': random.randint(1, 5)
        })
    return pd.DataFrame(data)

def generate_healthcare_data(n):
    fake = Faker()
    conditions = ['Healthy', 'Diabetes', 'Hypertension', 'Asthma', 'Heart Disease', 'Arthritis']
    data = []
    for _ in range(n):
        first = fake.first_name()
        last = fake.last_name()
        age = random.randint(18, 90)
        data.append({
            'patient_id': f"P{random.randint(10000, 99999)}",
            'first_name': first,
            'last_name': last,
            'age': age,
            'gender': random.choice(['Male', 'Female']),
            'weight_kg': round(random.uniform(50, 120), 1),
            'height_cm': round(random.uniform(150, 200), 1),
            'bmi': round(random.uniform(18, 35), 1),
            'blood_pressure_sys': random.randint(100, 180),
            'blood_pressure_dia': random.randint(60, 120),
            'heart_rate': random.randint(60, 100),
            'condition': random.choice(conditions),
            'medication': random.choice(['None', 'Metformin', 'Lisinopril', 'Albuterol']),
            'followup': random.choice([True, False])
        })
    return pd.DataFrame(data)

def generate_financial_data(n):
    fake = Faker()
    types = ['Deposit', 'Withdrawal', 'Transfer', 'Payment', 'Investment']
    currencies = ['USD', 'EUR', 'GBP', 'JPY']
    data = []
    for _ in range(n):
        data.append({
            'transaction_id': fake.uuid4(),
            'account_id': f"ACC{random.randint(10000, 99999)}",
            'type': random.choice(types),
            'amount': round(random.uniform(10, 10000), 2),
            'currency': random.choice(currencies),
            'timestamp': fake.date_time_between(start_date='-1y'),
            'status': random.choice(['Pending', 'Completed', 'Failed']),
            'category': random.choice(['Food', 'Transport', 'Entertainment', 'Bills', 'Shopping'])
        })
    return pd.DataFrame(data)

def generate_toll_data(n):
    fake = Faker()
    states = ['WB', 'JH', 'BR', 'UP', 'DL', 'MH', 'KA', 'TN']
    vehicle_types = ['Car', 'Truck', 'Bus', 'SUV', 'Motorcycle']
    toll_plazas = ['NH-16 Kolkata Toll Plaza', 'NH-8 Delhi Toll Plaza', 'NH-44 Chennai Toll Plaza']
    data = []
    for _ in range(n):
        state = random.choice(states)
        num = random.randint(10, 99)
        letter = random.choice(['AB','CD','EF','GH'])
        suffix = random.randint(1000, 9999)
        vehicle = f"{state}{num:02d}{letter}{suffix}"
        data.append({
            'transaction_id': f"TXN{datetime.now().strftime('%Y%m%d')}{str(len(data)+1).zfill(4)}",
            'toll_plaza': random.choice(toll_plazas),
            'vehicle_number': vehicle,
            'vehicle_type': random.choice(vehicle_types),
            'toll_amount': round(random.uniform(50, 500), 2),
            'payment_mode': random.choice(['FASTag', 'Cash', 'UPI', 'Card']),
            'transaction_date': fake.date_time_between(start_date='-7d'),
            'speed': random.randint(10, 80),
            'overloaded': random.choice(['No', 'No', 'No', 'Yes'])
        })
    return pd.DataFrame(data)

def generate_from_sample(sample, n):
    """Faker-driven custom generation via GenerationPipeline (not seed bootstrap)."""
    if sample is None or len(sample) == 0:
        return pd.DataFrame()
    pipeline = GenerationPipeline()
    return pipeline.generate_from_sample(
        sample,
        num_records=n,
        preserve_correlations=True,
        enable_privacy=True,
    )

# ==================== QUALITY ANALYZER ====================

def analyze_quality(data, sample=None, column_roles=None):
    if data is None or len(data) == 0:
        return {'overall_score': 0, 'completeness': 0, 'uniqueness': 0}
    reporter = QualityReporter()
    roles = column_roles or getattr(data, 'attrs', {}).get('column_roles')
    return reporter.generate_report(data, sample=sample, column_roles=roles)

# ==================== ENHANCED VISUALIZATION ====================

def create_enhanced_dashboard(data: pd.DataFrame, report: dict) -> go.Figure:
    """
    Create an enhanced interactive dashboard with multiple visualization types
    """
    
    # Create figure with 3x3 subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            '📊 Data Overview',
            '📈 Correlation Matrix',
            '📉 Distribution Plot',
            '🎯 Quality Score',
            '🔍 Missing Values',
            '📋 Data Types',
            '📊 Feature Importance',
            '📈 Outlier Analysis',
            '🔒 Privacy Score'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "heatmap"}, {"type": "histogram"}],
            [{"type": "indicator"}, {"type": "bar"}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "box"}, {"type": "indicator"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # ===== 1. DATA OVERVIEW (Scatter Plot) =====
    try:
        if len(numeric_cols) >= 2:
            sample_size = min(1000, len(data))
            fig.add_trace(
                go.Scatter(
                    x=data[numeric_cols[0]][:sample_size],
                    y=data[numeric_cols[1]][:sample_size],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=data[numeric_cols[0]][:sample_size],
                        colorscale='Viridis',
                        showscale=False,
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    name='Data Points',
                    hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add trend line
            if len(data) > 10:
                z = np.polyfit(data[numeric_cols[0]][:sample_size], data[numeric_cols[1]][:sample_size], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(data[numeric_cols[0]].min(), data[numeric_cols[0]].max(), 100)
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=p(x_trend),
                        mode='lines',
                        name='Trend Line',
                        line=dict(color='#ff6b6b', width=2, dash='dash'),
                        hovertemplate='Trend: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
    except Exception:
        pass
    
    # ===== 2. CORRELATION MATRIX (Heatmap) =====
    try:
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    name='Correlation',
                    text=corr_matrix.round(2).values,
                    texttemplate='%{text}',
                    textfont={"size": 8},
                    hovertemplate='<b>%{x}</b> ↔ <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
    except Exception:
        pass
    
    # ===== 3. DISTRIBUTION PLOT =====
    try:
        if numeric_cols:
            fig.add_trace(
                go.Histogram(
                    x=data[numeric_cols[0]].dropna(),
                    nbinsx=30,
                    name='Distribution',
                    marker_color='#667eea',
                    opacity=0.8,
                    hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=3
            )
            
            # Add mean line
            mean_val = data[numeric_cols[0]].mean()
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="#ff6b6b",
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="top",
                row=1, col=3
            )
    except Exception:
        pass
    
    # ===== 4. QUALITY SCORE (Gauge) =====
    try:
        score = report.get('overall_score', 0.5) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=score,
                title={
                    'text': "Overall Quality Score",
                    'font': {'size': 14}
                },
                delta={
                    'reference': 50,
                    'increasing': {'color': '#6bcb77'},
                    'decreasing': {'color': '#ff6b6b'}
                },
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': '#6bcb77' if score >= 70 else '#ffd93d' if score >= 40 else '#ff6b6b'},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': "#ff6b6b", 'name': 'Poor'},
                        {'range': [40, 70], 'color': "#ffd93d", 'name': 'Average'},
                        {'range': [70, 100], 'color': "#6bcb77", 'name': 'Good'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=1
        )
    except Exception:
        pass
    
    # ===== 5. MISSING VALUES =====
    try:
        nulls = data.isnull().sum()
        nulls = nulls[nulls > 0]
        if len(nulls) > 0:
            fig.add_trace(
                go.Bar(
                    x=nulls.index[:8],
                    y=nulls.values[:8],
                    name='Missing Values',
                    marker_color='#ff6b6b',
                    text=nulls.values[:8],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Missing: %{y}<extra></extra>'
                ),
                row=2, col=2
            )
        else:
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=0,
                    title={'text': "✅ No Missing Values"},
                    number={'font': {'color': '#6bcb77', 'size': 40}}
                ),
                row=2, col=2
            )
    except Exception:
        pass
    
    # ===== 6. DATA TYPES (Pie Chart) =====
    try:
        dtype_counts = data.dtypes.value_counts()
        colors = ['#667eea', '#764ba2', '#ff6b6b', '#ffd93d', '#6bcb77']
        fig.add_trace(
            go.Pie(
                labels=[str(dt) for dt in dtype_counts.index],
                values=dtype_counts.values,
                name='Data Types',
                hole=0.4,
                marker=dict(colors=colors[:len(dtype_counts)]),
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=2, col=3
        )
    except Exception:
        pass
    
    # ===== 7. FEATURE IMPORTANCE =====
    try:
        if len(numeric_cols) > 1:
            # Calculate feature importance using variance
            importance = data[numeric_cols].var().sort_values(ascending=False)
            top_features = importance.head(8)
            
            fig.add_trace(
                go.Bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation='h',
                    name='Feature Importance',
                    marker_color='#667eea',
                    text=top_features.values.round(2),
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>'
                ),
                row=3, col=1
            )
    except Exception:
        pass
    
    # ===== 8. OUTLIER ANALYSIS (Box Plot) =====
    try:
        if len(numeric_cols) > 0:
            # Select top numeric columns for box plot
            cols_to_show = numeric_cols[:4]
            fig.add_trace(
                go.Box(
                    y=[data[col].dropna() for col in cols_to_show],
                    name='Outliers',
                    boxmean='sd',
                    marker_color='#764ba2',
                    hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'
                ),
                row=3, col=2
            )
    except Exception:
        pass
    
    # ===== 9. PRIVACY SCORE (Gauge) =====
    try:
        privacy_score = report.get('privacy_score', 0.5) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=privacy_score,
                title={
                    'text': "🔒 Privacy Score",
                    'font': {'size': 14}
                },
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#6bcb77' if privacy_score >= 70 else '#ffd93d' if privacy_score >= 40 else '#ff6b6b'},
                    'steps': [
                        {'range': [0, 40], 'color': "#ff6b6b"},
                        {'range': [40, 70], 'color': "#ffd93d"},
                        {'range': [70, 100], 'color': "#6bcb77"}
                    ]
                }
            ),
            row=3, col=3
        )
    except Exception:
        pass
    
    # ===== Update Layout =====
    fig.update_layout(
        height=900,
        showlegend=False,
        template='plotly_white',
        font=dict(size=12),
        title={
            'text': '📊 Synthetic Data Quality Dashboard',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': '#2c3e50'}
        },
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_comparison_dashboard(original: pd.DataFrame, synthetic: pd.DataFrame, report: dict) -> go.Figure:
    """
    Create a comparison dashboard showing original vs synthetic data
    """
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Original vs Synthetic Distribution',
            'Statistical Comparison',
            'Correlation Comparison',
            'Quality Metrics Comparison'
        )
    )
    
    numeric_cols = original.select_dtypes(include=[np.number]).columns.tolist()
    
    # 1. Distribution Comparison
    if numeric_cols:
        col = numeric_cols[0]
        fig.add_trace(
            go.Histogram(
                x=original[col].dropna(),
                name='Original',
                opacity=0.6,
                marker_color='#667eea',
                nbinsx=20
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(
                x=synthetic[col].dropna(),
                name='Synthetic',
                opacity=0.6,
                marker_color='#ff6b6b',
                nbinsx=20
            ),
            row=1, col=1
        )
    
    # 2. Statistical Comparison (Bar Chart)
    try:
        stats_comparison = pd.DataFrame({
            'Mean': [original[numeric_cols].mean().mean(), synthetic[numeric_cols].mean().mean()],
            'Std': [original[numeric_cols].std().mean(), synthetic[numeric_cols].std().mean()],
            'Min': [original[numeric_cols].min().min(), synthetic[numeric_cols].min().min()],
            'Max': [original[numeric_cols].max().max(), synthetic[numeric_cols].max().max()]
        }, index=['Original', 'Synthetic'])
        
        fig.add_trace(
            go.Bar(
                x=stats_comparison.columns,
                y=stats_comparison.loc['Original'],
                name='Original',
                marker_color='#667eea'
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(
                x=stats_comparison.columns,
                y=stats_comparison.loc['Synthetic'],
                name='Synthetic',
                marker_color='#ff6b6b'
            ),
            row=1, col=2
        )
    except Exception:
        pass
    
    # 3. Correlation Comparison
    try:
        if len(numeric_cols) > 1:
            orig_corr = original[numeric_cols].corr().values.flatten()
            synth_corr = synthetic[numeric_cols].corr().values.flatten()
            fig.add_trace(
                go.Scatter(
                    x=orig_corr,
                    y=synth_corr,
                    mode='markers',
                    marker=dict(size=10, color='#667eea', opacity=0.6),
                    name='Correlation',
                    hovertemplate='Original: %{x:.2f}<br>Synthetic: %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
            # Add diagonal line
            fig.add_trace(
                go.Scatter(
                    x=[-1, 1],
                    y=[-1, 1],
                    mode='lines',
                    name='Perfect Match',
                    line=dict(color='#ff6b6b', dash='dash')
                ),
                row=2, col=1
            )
    except Exception:
        pass
    
    # 4. Quality Metrics Comparison
    try:
        metrics = ['completeness', 'uniqueness', 'privacy_score']
        orig_values = [report.get(m, 0) * 100 for m in metrics]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=orig_values,
                name='Synthetic',
                marker_color='#6bcb77',
                text=[f"{v:.1f}%" for v in orig_values],
                textposition='outside'
            ),
            row=2, col=2
        )
    except Exception:
        pass
    
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_white',
        font=dict(size=12),
        title={
            'text': '📊 Original vs Synthetic Data Comparison',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': '#2c3e50'}
        }
    )
    
    return fig

# ==================== PAGES ====================

def home_page():
    st.markdown('<h1 class="main-header">🎲 SynthSLM</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.2rem;'>Synthetic Data Generation for Small Language Models</p>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="stats-card"><h3>10+</h3><p>Data Types</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stats-card"><h3>100K</h3><p>Records</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stats-card"><h3>4</h3><p>Formats</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stats-card"><h3>100%</h3><p>Privacy</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.sample_data is None:
        st.markdown("""
        <div class="upload-area">
            <h3>📤 Start with Your Own Data</h3>
            <p style="color: #666;">Upload a sample dataset to generate synthetic data with the same patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🚀 Features
        - **10+ Data Types** - Personal, Sales, Employee, Time Series, Logs, System, IoT, Healthcare, Financial, Toll Plaza
        - **User-Defined Generation** - Upload your data, AI learns patterns
        - **Privacy Protection** - PII detection & anonymization
        - **Quality Analysis** - Statistical validation & metrics
        """)
    with col2:
        st.markdown("""
        ### 📊 Quick Start
        1. **Upload Sample** - Upload your dataset
        2. **Generate** - Configure and generate synthetic data
        3. **Analyze** - View quality metrics
        4. **Export** - Download in your preferred format
        """)

def upload_page():
    st.subheader("📤 Upload Sample Data")
    
    uploaded = st.file_uploader("Choose CSV, Excel, or JSON", type=['csv', 'xlsx', 'xls', 'json'])
    
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            elif uploaded.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_json(uploaded)
            
            st.session_state.sample_data = df
            st.success(f"✅ Loaded {len(df):,} records, {len(df.columns)} columns")
            st.dataframe(df.head(10))
            
            with st.expander("📋 Data Summary"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Unique': df.nunique(),
                    'Null %': (df.isnull().sum() / len(df) * 100).round(2)
                })
                st.dataframe(col_info)
        except Exception as e:
            st.error(f"Error: {e}")

def generate_page():
    st.subheader("⚙️ Generate Data")
    
    data_types = [
        "Personal/Customer Data",
        "Sales Transactions",
        "Employee Records",
        "Time Series Data",
        "Application Logs",
        "System Metrics",
        "IoT Sensor Data",
        "Healthcare Records",
        "Financial Transactions",
        "Toll Plaza Data",
        "User-Defined (Upload Sample)"
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        data_type = st.selectbox("Data Type", data_types)
    with col2:
        num_records = st.number_input("Records", 10, 100000, 1000, 100)
    
    if data_type == "User-Defined (Upload Sample)":
        if st.session_state.sample_data is not None:
            st.info(f"📊 Using: {len(st.session_state.sample_data):,} records")
        else:
            st.warning("⚠️ No sample loaded. Upload first.")
            return
    
    if st.button("🚀 Generate", use_container_width=True):
        with st.spinner(f"Generating {num_records} records..."):
            start = time.time()
            
            if data_type == "Personal/Customer Data":
                df = generate_personal_data(num_records)
            elif data_type == "Sales Transactions":
                df = generate_sales_data(num_records)
            elif data_type == "Employee Records":
                df = generate_employee_data(num_records)
            elif data_type == "Time Series Data":
                df = generate_timeseries_data(num_records)
            elif data_type == "Application Logs":
                df = generate_logs_data(num_records)
            elif data_type == "System Metrics":
                df = generate_system_data(num_records)
            elif data_type == "IoT Sensor Data":
                df = generate_iot_data(num_records)
            elif data_type == "Healthcare Records":
                df = generate_healthcare_data(num_records)
            elif data_type == "Financial Transactions":
                df = generate_financial_data(num_records)
            elif data_type == "Toll Plaza Data":
                df = generate_toll_data(num_records)
            else:
                df = generate_from_sample(st.session_state.sample_data, num_records)
            
            gen_time = time.time() - start
            
            st.session_state.generated_data = df
            sample_for_q = (
                st.session_state.sample_data
                if data_type == "User-Defined (Upload Sample)"
                else None
            )
            roles_for_q = getattr(df, 'attrs', {}).get('column_roles')
            report = analyze_quality(df, sample=sample_for_q, column_roles=roles_for_q)
            st.session_state.quality_report = report
            
            st.session_state.generation_history.append({
                'type': data_type,
                'records': len(df),
                'time': gen_time,
                'quality': report['overall_score']
            })
            
            st.success(f"✅ Generated {len(df):,} records in {gen_time:.2f}s")
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Records", f"{len(df):,}")
            with col2: st.metric("Columns", len(df.columns))
            with col3: st.metric("Quality", f"{report['overall_score']:.1%}")

            if data_type == "User-Defined (Upload Sample)":
                with st.expander("📊 Quality breakdown", expanded=True):
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        st.metric("Completeness", f"{report.get('completeness', 0):.1%}")
                        st.metric("ID Uniqueness", f"{report.get('id_uniqueness', 0):.1%}")
                    with b2:
                        st.metric("Date Diversity", f"{report.get('date_diversity', 0):.1%}")
                        st.metric("Open Diversity", f"{report.get('diversity', 0):.1%}")
                    with b3:
                        enum_f = report.get('enum_fidelity')
                        st.metric(
                            "Enum Fidelity",
                            f"{enum_f:.1%}" if enum_f is not None else "N/A",
                        )
                        coh = report.get('name_email_coherence')
                        st.metric(
                            "Name↔Email",
                            f"{coh:.1%}" if coh is not None else "N/A",
                        )
    
    if st.session_state.generated_data is not None:
        st.subheader("📊 Preview")
        st.dataframe(st.session_state.generated_data.head(20))

# ==================== DASHBOARD PAGE ====================

def dashboard_page():
    st.subheader("📈 Quality Dashboard")
    
    if st.session_state.generated_data is None:
        st.warning("No data to analyze")
        return
    
    df = st.session_state.generated_data
    report = st.session_state.quality_report or {}
    
    # Quick metrics cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Overall", f"{report.get('overall_score', 0):.1%}")
    with col2:
        st.metric("✅ Completeness", f"{report.get('completeness', 0):.1%}")
    with col3:
        st.metric("🔄 Uniqueness", f"{report.get('uniqueness', 0):.1%}")
    with col4:
        st.metric("🔒 Privacy", f"{report.get('privacy_score', 0):.1%}")
    with col5:
        st.metric("📈 Records", f"{len(df):,}")
    
    st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Quality Dashboard", "📈 Comparison View"])
    
    with tab1:
        try:
            fig = create_enhanced_dashboard(df, report)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            st.info("Showing data summary instead...")
            _show_data_summary(df, report)
    
    with tab2:
        if st.session_state.sample_data is not None:
            try:
                fig = create_comparison_dashboard(st.session_state.sample_data, df, report)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Comparison view error: {str(e)}")
        else:
            st.info("Upload a sample dataset to see comparison view")
    
    # Additional detailed metrics
    with st.expander("📊 Detailed Metrics & Statistics", expanded=False):
        _show_detailed_metrics(df, report)

def _show_data_summary(df: pd.DataFrame, report: dict):
    """Show data summary as fallback"""
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data Info**")
        st.write(f"- Records: {len(df):,}")
        st.write(f"- Columns: {len(df.columns)}")
        st.write(f"- Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    with col2:
        st.write("**Quality Metrics**")
        st.write(f"- Completeness: {report.get('completeness', 0):.1%}")
        st.write(f"- Uniqueness: {report.get('uniqueness', 0):.1%}")
        st.write(f"- Privacy: {report.get('privacy_score', 0):.1%}")
    
    # Show column info
    st.write("**Column Information**")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Unique': df.nunique(),
        'Null %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(col_info, use_container_width=True)

def _show_detailed_metrics(df: pd.DataFrame, report: dict):
    """Show detailed metrics"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Column Statistics")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    with col2:
        st.subheader("📋 Data Profile")
        profile = {
            'Total Records': len(df),
            'Total Columns': len(df.columns),
            'Numeric Columns': len(df.select_dtypes(include=[np.number]).columns),
            'Categorical Columns': len(df.select_dtypes(include=['object']).columns),
            'Missing Values': df.isnull().sum().sum(),
            'Duplicate Rows': df.duplicated().sum(),
            'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }
        st.json(profile)

# ==================== EXPORT PAGE (FIXED) ====================

def export_page():
    st.subheader("💾 Export Data")
    
    if st.session_state.generated_data is None:
        st.warning("No data to export")
        return
    
    df = st.session_state.generated_data
    
    col1, col2 = st.columns(2)
    with col1:
        # Map display names to actual format values
        format_options = {
            "CSV (.csv)": "CSV",
            "Excel (.xlsx)": "Excel",
            "JSON (.json)": "JSON",
            "Parquet (.parquet)": "Parquet"
        }
        display_format = st.selectbox(
            "Export Format",
            list(format_options.keys()),
            index=0,
            help="Choose the format to export your data"
        )
        export_format = format_options[display_format]
        
    with col2:
        filename = st.text_input(
            "File Name",
            f"data_{datetime.now().strftime('%Y%m%d')}",
            help="Enter a name for your exported file"
        )
    
    if st.button("📥 Export Data", use_container_width=True):
        try:
            if export_format == "CSV":
                data = df.to_csv(index=False).encode()
                mime = "text/csv"
                file_ext = "csv"
            elif export_format == "JSON":
                data = df.to_json(orient='records', indent=2).encode()
                mime = "application/json"
                file_ext = "json"
            elif export_format == "Parquet":
                output = io.BytesIO()
                df.to_parquet(output, index=False)
                data = output.getvalue()
                mime = "application/octet-stream"
                file_ext = "parquet"
            else:  # Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    make_excel_safe(df).to_excel(writer, index=False, sheet_name='Data')
                data = output.getvalue()
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                file_ext = "xlsx"
            
            file_size = len(data)
            if file_size > 1024 * 1024:
                size_str = f"{file_size / (1024*1024):.2f} MB"
            elif file_size > 1024:
                size_str = f"{file_size / 1024:.2f} KB"
            else:
                size_str = f"{file_size} B"
            
            st.success(f"✅ Data ready for download ({size_str})")

            st.download_button(
                label=f"📥 Download {display_format}",
                data=data,
                file_name=f"{filename}.{file_ext}",
                mime=mime,
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")

# ==================== MAIN ====================

def main():
    st.sidebar.title("🎲 SynthSLM")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Navigation", [
        "🏠 Home",
        "📤 Upload Sample",
        "⚙️ Generate",
        "📈 Dashboard",
        "💾 Export"
    ])
    
    st.sidebar.markdown("---")
    
    if st.session_state.sample_data is not None:
        st.sidebar.success(f"📊 Sample: {len(st.session_state.sample_data):,} rows")
    else:
        st.sidebar.info("📊 No sample loaded")
    
    if st.session_state.generated_data is not None:
        st.sidebar.info(f"📦 Generated: {len(st.session_state.generated_data):,} rows")
    else:
        st.sidebar.info("📦 No data generated")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("v3.0 | Team CSE_13 | B.Tech Final Year Project")
    
    if page == "🏠 Home":
        home_page()
    elif page == "📤 Upload Sample":
        upload_page()
    elif page == "⚙️ Generate":
        generate_page()
    elif page == "📈 Dashboard":
        dashboard_page()
    elif page == "💾 Export":
        export_page()

if __name__ == "__main__":
    main()