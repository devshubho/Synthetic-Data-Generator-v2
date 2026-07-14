"""
SynthSLM - Main Application
B.Tech Final Year Project
Author: Subham Sarkar
"""

import streamlit as st
import pandas as pd
import time
import traceback
from datetime import datetime

# Import from config
from config import Config
from logger import setup_logger, get_logger, log_error
from engine.pipeline import GenerationPipeline
from analytics.quality_report import QualityReporter
from visualization.dashboard import Dashboard
from export.csv_export import CSVExporter
from export.json_export import JSONExporter
from export.parquet_export import ParquetExporter
from export.excel_export import ExcelExporter
from database.history import HistoryManager
from utils.helpers import format_size
from utils.exceptions import GenerationError, SampleError, ExportError

# Setup
logger = setup_logger()
st.set_page_config(page_title=Config.APP_NAME, page_icon="🎲", layout="wide")

# ==================== SESSION STATE ====================

def init_session():
    """Initialize session state"""
    try:
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.generated_data = None
            st.session_state.sample_data = None
            st.session_state.sample_data_type = None
            st.session_state.quality_report = None
            st.session_state.history = HistoryManager()
            st.session_state.pipeline = GenerationPipeline()
            st.session_state.error = None
            logger.info("Session state initialized")
    except Exception as e:
        logger.error(f"Session initialization failed: {str(e)}")
        st.error(f"Failed to initialize application: {str(e)}")

init_session()

# ==================== ERROR HANDLING ====================

def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
            with st.expander("🔍 Technical Details"):
                st.code(traceback.format_exc())
    return wrapper

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    .main-header { font-size: 3.5rem; font-weight: bold; text-align: center;
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stats-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  color: white; padding: 1.5rem; border-radius: 10px; text-align: center; }
    .feature-card { background: #f8f9fa; padding: 1.5rem; border-radius: 10px;
                    border-left: 4px solid #667eea; margin: 1rem 0; }
    .success-box { background: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; padding: 1rem; color: #155724; }
    .info-box { background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 8px; padding: 1rem; color: #0c5460; }
    .warning-box { background: #fff3cd; border: 1px solid #ffeeba; border-radius: 8px; padding: 1rem; color: #856404; }
</style>
""", unsafe_allow_html=True)

# ==================== PAGES ====================

@handle_errors
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

@handle_errors
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
            st.error(f"Error: {str(e)}")

@handle_errors
def generate_page():
    st.subheader("⚙️ Generate Synthetic Data")
    
    pipeline = st.session_state.pipeline
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_type = st.selectbox("Select Data Type", Config.DATA_TYPES)
    
    with col2:
        num_records = st.number_input("Number of Records", 10, 100000, 1000, 100)
    
    if data_type == "User-Defined (Upload Sample)":
        if st.session_state.sample_data is not None:
            st.info(f"📊 Using: {len(st.session_state.sample_data):,} records")
        else:
            st.warning("⚠️ No sample data loaded. Upload a sample first.")
            if st.button("📤 Go to Upload Sample"):
                st.session_state.page = "Upload Sample"
                st.rerun()
            return
    
    with st.expander("⚙️ Advanced Options"):
        preserve_correlations = st.checkbox("Preserve Correlations", value=True)
        enable_privacy = st.checkbox("Enable Privacy Protection", value=True)
    
    if st.button("🚀 Generate Data", use_container_width=True):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Initializing...")
            progress_bar.progress(10)
            
            with st.spinner(f"Generating {num_records} records..."):
                start_time = time.time()
                
                status_text.text(f"📊 Generating {num_records} records...")
                progress_bar.progress(30)
                
                if data_type == "User-Defined (Upload Sample)":
                    if st.session_state.sample_data is not None:
                        df = pipeline.generate_from_sample(
                            st.session_state.sample_data,
                            num_records,
                            preserve_correlations=preserve_correlations,
                            enable_privacy=enable_privacy
                        )
                    else:
                        st.error("No sample data available!")
                        return
                else:
                    df = pipeline.generate_template(data_type, num_records)
                
                gen_time = time.time() - start_time
                
                status_text.text("✅ Processing...")
                progress_bar.progress(70)
                
                st.session_state.generated_data = df
                
                status_text.text("📊 Analyzing quality...")
                progress_bar.progress(85)
                
                reporter = QualityReporter()
                report = reporter.generate_report(df)
                st.session_state.quality_report = report
                
                st.session_state.history.save({
                    'type': data_type,
                    'records': len(df),
                    'columns': len(df.columns),
                    'time': gen_time,
                    'quality': report.get('overall_score', 0)
                })
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                st.success(f"✅ Generated {len(df):,} records in {gen_time:.2f} seconds!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    quality = report.get('overall_score', 0)
                    st.metric("Quality Score", f"{quality:.1%}")
                
                progress_bar.empty()
                status_text.empty()
                
        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")
            logger.error(f"Generation error: {str(e)}\n{traceback.format_exc()}")
    
    if st.session_state.generated_data is not None:
        df = st.session_state.generated_data
        
        st.subheader("📊 Generated Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        with st.expander("📋 Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Unique': df.nunique(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)

@handle_errors
def dashboard_page():
    st.subheader("📈 Quality Dashboard")
    
    if st.session_state.generated_data is None:
        st.warning("No data generated yet.")
        return
    
    df = st.session_state.generated_data
    report = st.session_state.quality_report or {}
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Quality", f"{report.get('overall_score', 0):.1%}")
    with col2:
        st.metric("Completeness", f"{report.get('completeness', 0):.1%}")
    with col3:
        st.metric("Uniqueness", f"{report.get('uniqueness', 0):.1%}")
    with col4:
        st.metric("Privacy Score", f"{report.get('privacy_score', 0):.1%}")
    
    dashboard = Dashboard()
    fig = dashboard.create_quality_dashboard(df, report)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📊 Detailed Metrics"):
        st.json(report)

@handle_errors
def export_page():
    st.subheader("💾 Export Data")
    
    if st.session_state.generated_data is None:
        st.warning("No data to export.")
        return
    
    df = st.session_state.generated_data
    
    col1, col2 = st.columns(2)
    with col1:
        export_format = st.selectbox("Format", Config.EXPORT_FORMATS)
    with col2:
        filename = st.text_input("File Name", f"synthetic_data_{datetime.now().strftime('%Y%m%d')}")
    
    if st.button("📥 Export Data", use_container_width=True):
        try:
            exporters = {
                "CSV": CSVExporter,
                "JSON": JSONExporter,
                "Parquet": ParquetExporter,
                "Excel": ExcelExporter
            }
            
            if export_format in exporters:
                exporter = exporters[export_format]()
                data_bytes = exporter.export(df)
                file_size = format_size(len(data_bytes))
                
                st.success(f"✅ Ready for download ({file_size})")
                
                st.download_button(
                    label=f"📥 Download {export_format}",
                    data=data_bytes,
                    file_name=f"{filename}.{export_format.lower()}",
                    mime="application/octet-stream",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")

# ==================== MAIN ====================

def main():
    st.sidebar.title("🎲 SynthSLM")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Navigation", ["🏠 Home", "📤 Upload Sample", "⚙️ Generate", "📈 Dashboard", "💾 Export"])
    
    st.sidebar.markdown("---")
    
    if st.session_state.sample_data is not None:
        st.sidebar.success(f"📊 Sample: {len(st.session_state.sample_data):,} rows")
    else:
        st.sidebar.info("📊 No sample loaded")
    
    if st.session_state.generated_data is not None:
        st.sidebar.info(f"📦 Generated: {len(st.session_state.generated_data):,} rows")
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"v{Config.VERSION} | {Config.AUTHOR}")
    
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