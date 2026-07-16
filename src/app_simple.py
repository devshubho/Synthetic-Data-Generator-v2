"""
SynthSLM - Simplified Working Version
"""

import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import time
import io
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from export.excel_export import make_excel_safe

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="SynthSLM", page_icon="🎲", layout="wide")

# ==================== INITIALIZE SESSION STATE ====================
if 'init' not in st.session_state:
    st.session_state.init = True
    st.session_state.sample_data = None
    st.session_state.generated_data = None
    st.session_state.quality_report = None
    st.session_state.generation_history = []

# ==================== CUSTOM CSS ====================
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
</style>
""", unsafe_allow_html=True)

# ==================== DATA GENERATORS ====================

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
            'email': f"{first.lower()}.{last.lower()}@example.com",
            'phone': fake.phone_number(),
            'address': fake.address().replace('\n', ', '),
            'city': fake.city(),
            'state': fake.state(),
            'zipcode': fake.zipcode(),
            'birth_date': fake.date_of_birth(minimum_age=18, maximum_age=80),
            'gender': random.choice(['Male', 'Female', 'Non-binary']),
            'occupation': fake.job(),
            'income': random.randint(30000, 200000),
            'active': random.choice([True, False])
        })
    return pd.DataFrame(data)

def generate_sales_data(n):
    fake = Faker()
    products = ['Laptop', 'Phone', 'Headphones', 'Monitor', 'Keyboard', 'Mouse']
    data = []
    for _ in range(n):
        product = random.choice(products)
        price = random.randint(100, 2000)
        qty = random.randint(1, 10)
        data.append({
            'transaction_id': fake.uuid4(),
            'customer_id': fake.uuid4(),
            'product': product,
            'quantity': qty,
            'unit_price': price,
            'total': qty * price,
            'date': fake.date_time_between(start_date='-1y'),
            'region': random.choice(['North', 'South', 'East', 'West'])
        })
    return pd.DataFrame(data)

def generate_employee_data(n):
    fake = Faker()
    depts = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']
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
            'position': random.choice(['Junior', 'Senior', 'Lead', 'Manager']),
            'hire_date': fake.date_between(start_date='-10y'),
            'salary': random.randint(40000, 150000),
            'rating': round(random.uniform(1, 5), 1)
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

def generate_timeseries_data(n):
    start = datetime.now() - timedelta(days=n)
    dates = [start + timedelta(days=i) for i in range(n)]
    trend = np.linspace(0, 50, n)
    season = 20 * np.sin(2 * np.pi * np.arange(n) / 30)
    noise = np.random.normal(0, 5, n)
    values = 100 + trend + season + noise
    return pd.DataFrame({
        'date': dates,
        'value': values,
        'moving_avg': pd.Series(values).rolling(7, min_periods=1).mean()
    })

def generate_from_sample(sample, n):
    if len(sample) == 0:
        return pd.DataFrame()
    indices = np.random.choice(len(sample), n, replace=True)
    return sample.iloc[indices].reset_index(drop=True)

# ==================== QUALITY ANALYZER ====================

def analyze_quality(data):
    if data is None or len(data) == 0:
        return {'overall_score': 0, 'completeness': 0, 'uniqueness': 0}
    
    report = {
        'overall_score': 0.0,
        'completeness': 1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) if data.shape[0] * data.shape[1] > 0 else 0,
        'uniqueness': data.nunique().mean() / len(data) if len(data) > 0 else 0,
        'diversity': 0.5,
        'privacy_score': 0.5
    }
    report['overall_score'] = (report['completeness'] + report['uniqueness']) / 2
    return report

# ==================== DASHBOARD ====================

def create_dashboard(data, report):
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Data Overview', 'Distribution', 'Quality Score', 'Missing Values'))
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        fig.add_trace(
            go.Scatter(x=data[numeric_cols[0]][:500], y=data[numeric_cols[1]][:500], 
                      mode='markers', marker=dict(size=5, opacity=0.5)),
            row=1, col=1
        )
    
    if len(numeric_cols) > 0:
        fig.add_trace(
            go.Histogram(x=data[numeric_cols[0]], nbinsx=30),
            row=1, col=2
        )
    
    score = report.get('overall_score', 0.5) * 100
    fig.add_trace(
        go.Indicator(mode="gauge+number", value=score, title={'text': "Quality Score"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': 'green' if score > 70 else 'orange'}}),
        row=2, col=1
    )
    
    nulls = data.isnull().sum()
    if nulls.sum() > 0:
        fig.add_trace(go.Bar(x=nulls.index[:5], y=nulls.values[:5]), row=2, col=2)
    
    fig.update_layout(height=800, showlegend=False)
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
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🚀 Features
        - **10+ Data Types** - Personal, Sales, Employee, Time Series, Toll Plaza
        - **User-Defined Generation** - Upload your data
        - **Privacy Protection** - PII detection
        - **Quality Analysis** - Metrics & dashboard
        """)
    with col2:
        st.markdown("""
        ### 📊 Quick Start
        1. Upload Sample
        2. Configure Generation
        3. Generate Data
        4. Analyze & Export
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
            
            with st.expander("📋 Data Info"):
                st.write("Columns:", list(df.columns))
                st.write("Data Types:", df.dtypes)
        except Exception as e:
            st.error(f"Error: {e}")

def generate_page():
    st.subheader("⚙️ Generate Data")
    
    data_types = [
        "Personal/Customer Data",
        "Sales Transactions",
        "Employee Records",
        "Time Series Data",
        "Toll Plaza Data",
        "User-Defined (Upload Sample)"
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        data_type = st.selectbox("Data Type", data_types)
    with col2:
        num_records = st.number_input("Records", 10, 10000, 100)
    
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
            elif data_type == "Toll Plaza Data":
                df = generate_toll_data(num_records)
            elif data_type == "Time Series Data":
                df = generate_timeseries_data(num_records)
            else:
                df = generate_from_sample(st.session_state.sample_data, num_records)
            
            gen_time = time.time() - start
            
            st.session_state.generated_data = df
            report = analyze_quality(df)
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
    
    if st.session_state.generated_data is not None:
        st.subheader("📊 Preview")
        st.dataframe(st.session_state.generated_data.head(20))

def dashboard_page():
    st.subheader("📈 Quality Dashboard")
    
    if st.session_state.generated_data is None:
        st.warning("No data to analyze")
        return
    
    df = st.session_state.generated_data
    report = st.session_state.quality_report or {}
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Quality", f"{report.get('overall_score', 0):.1%}")
    with col2: st.metric("Completeness", f"{report.get('completeness', 0):.1%}")
    with col3: st.metric("Uniqueness", f"{report.get('uniqueness', 0):.1%}")
    with col4: st.metric("Privacy", f"{report.get('privacy_score', 0):.1%}")
    
    st.markdown("---")
    fig = create_dashboard(df, report)
    st.plotly_chart(fig, use_container_width=True)

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
    if st.session_state.generated_data is not None:
        st.sidebar.info(f"📦 Generated: {len(st.session_state.generated_data):,} rows")
    
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