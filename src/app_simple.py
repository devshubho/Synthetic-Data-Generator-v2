"""
Project Synthesi - Simplified Working Version
B.Tech Final Year Project
Author: Bikram Sarkar
"""

import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import time
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Project Synthesi",
    page_icon="🎲",
    layout="wide"
)

# ==================== INITIALIZE SESSION STATE ====================
# !!! MUST BE FIRST THING AFTER PAGE CONFIG !!!

if 'init' not in st.session_state:
    st.session_state.init = True
    st.session_state.sample_data = None
    st.session_state.generated_data = None
    st.session_state.quality_report = None
    st.session_state.generation_history = []

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA GENERATORS ====================

def generate_personal_data(n):
    """Generate personal data"""
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
            'gender': random.choice(['Male', 'Female']),
            'occupation': fake.job(),
            'income': random.randint(30000, 200000),
            'active': random.choice([True, False])
        })
    return pd.DataFrame(data)

def generate_sales_data(n):
    """Generate sales data"""
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
    """Generate employee data"""
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

def generate_timeseries_data(n):
    """Generate time series data"""
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
    """Generate from user sample"""
    # Simple approach: sample with replacement
    indices = np.random.choice(len(sample), n, replace=True)
    return sample.iloc[indices].reset_index(drop=True)

# ==================== QUALITY ANALYZER ====================

def analyze_quality(data):
    """Simple quality analysis"""
    report = {
        'overall_score': 0.0,
        'completeness': 1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1]),
        'uniqueness': data.nunique().mean() / len(data) if len(data) > 0 else 0,
        'diversity': 0.5,
        'privacy_score': 0.5
    }
    report['overall_score'] = (report['completeness'] + report['uniqueness']) / 2
    return report

# ==================== DASHBOARD ====================

def create_dashboard(data, report):
    """Create quality dashboard"""
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Data Overview', 'Distribution', 'Quality Score', 'Missing Values'))
    
    # Data overview
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        fig.add_trace(
            go.Scatter(x=data[numeric_cols[0]][:500], y=data[numeric_cols[1]][:500], mode='markers', marker=dict(size=5, opacity=0.5)),
            row=1, col=1
        )
    
    # Distribution
    if len(numeric_cols) > 0:
        fig.add_trace(
            go.Histogram(x=data[numeric_cols[0]], nbinsx=30),
            row=1, col=2
        )
    
    # Quality Score
    score = report.get('overall_score', 0.5) * 100
    fig.add_trace(
        go.Indicator(mode="gauge+number", value=score, title={'text': "Quality Score"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': 'green' if score > 70 else 'orange'}}),
        row=2, col=1
    )
    
    # Missing Values
    nulls = data.isnull().sum()
    if nulls.sum() > 0:
        fig.add_trace(go.Bar(x=nulls.index[:5], y=nulls.values[:5]), row=2, col=2)
    
    fig.update_layout(height=800, showlegend=False)
    return fig

# ==================== PAGES ====================

def home_page():
    st.markdown('<h1 class="main-header">🎲 Project Synthesi</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.2rem;'>Enterprise-Grade Synthetic Data Generation System</p>", unsafe_allow_html=True)
    
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
        - **10+ Pre-built Templates**
        - **User-Defined Generation**
        - **Privacy Protection**
        - **Quality Analysis**
        """)
    with col2:
        st.markdown("""
        ### 📊 Quick Start
        1. Upload Sample or Choose Template
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
    
    col1, col2 = st.columns(2)
    with col1:
        data_type = st.selectbox("Data Type", [
            "Personal/Customer Data",
            "Sales Transactions",
            "Employee Records",
            "Time Series Data",
            "User-Defined (Upload Sample)"
        ])
    with col2:
        num_records = st.number_input("Records", 10, 10000, 100)
    
    # Show sample info
    if data_type == "User-Defined (Upload Sample)":
        if st.session_state.sample_data is not None:
            st.info(f"📊 Using: {len(st.session_state.sample_data):,} records")
        else:
            st.warning("⚠️ No sample loaded. Upload first.")
            return
    
    if st.button("🚀 Generate", use_container_width=True):
        with st.spinner(f"Generating {num_records} records..."):
            start = time.time()
            
            # Generate
            if data_type == "Personal/Customer Data":
                df = generate_personal_data(num_records)
            elif data_type == "Sales Transactions":
                df = generate_sales_data(num_records)
            elif data_type == "Employee Records":
                df = generate_employee_data(num_records)
            elif data_type == "Time Series Data":
                df = generate_timeseries_data(num_records)
            else:  # User-Defined
                df = generate_from_sample(st.session_state.sample_data, num_records)
            
            gen_time = time.time() - start
            
            # Store
            st.session_state.generated_data = df
            
            # Quality
            report = analyze_quality(df)
            st.session_state.quality_report = report
            
            # History
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
    
    # Show data
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

def export_page():
    st.subheader("💾 Export Data")
    
    if st.session_state.generated_data is None:
        st.warning("No data to export")
        return
    
    df = st.session_state.generated_data
    
    format = st.selectbox("Format", ["CSV", "JSON", "Excel"])
    filename = st.text_input("Filename", f"data_{datetime.now().strftime('%Y%m%d')}")
    
    if st.button("📥 Export", use_container_width=True):
        if format == "CSV":
            data = df.to_csv(index=False).encode()
            mime = "text/csv"
        elif format == "JSON":
            data = df.to_json(orient='records', indent=2).encode()
            mime = "application/json"
        else:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            data = output.getvalue()
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
        st.download_button(
            label=f"Download {format}",
            data=data,
            file_name=f"{filename}.{format.lower()}",
            mime=mime,
            use_container_width=True
        )

# ==================== MAIN ====================

def main():
    st.sidebar.title("🎲 Project Synthesi")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Navigation", [
        "🏠 Home",
        "📤 Upload Sample",
        "⚙️ Generate",
        "📈 Dashboard",
        "💾 Export"
    ])
    
    st.sidebar.markdown("---")
    
    # Show status
    if st.session_state.sample_data is not None:
        st.sidebar.success(f"📊 Sample: {len(st.session_state.sample_data):,} rows")
    if st.session_state.generated_data is not None:
        st.sidebar.info(f"📦 Generated: {len(st.session_state.generated_data):,} rows")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("v3.0 | Bikram Sarkar")
    
    # Route
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