"""
SynthSLM - Main Application with Error Handling
"""

import streamlit as st
import pandas as pd
import time
import traceback
from datetime import datetime

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
    """Initialize session state with error handling"""
    try:
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.generated_data = None
            st.session_state.sample_data = None
            st.session_state.quality_report = None
            st.session_state.history = HistoryManager()
            st.session_state.pipeline = GenerationPipeline()
            st.session_state.error = None
            logger.info("Session state initialized")
    except Exception as e:
        logger.error(f"Session initialization failed: {str(e)}")
        st.error(f"Failed to initialize application: {str(e)}")

init_session()

# ==================== ERROR HANDLING DECORATOR ====================

def handle_errors(func):
    """Decorator for error handling in Streamlit"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except GenerationError as e:
            st.error(f"❌ Generation Error: {str(e)}")
            logger.error(f"GenerationError: {str(e)}")
        except SampleError as e:
            st.error(f"❌ Sample Error: {str(e)}")
            logger.error(f"SampleError: {str(e)}")
        except ExportError as e:
            st.error(f"❌ Export Error: {str(e)}")
            logger.error(f"ExportError: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")
            logger.error(f"Unexpected Error: {str(e)}\n{traceback.format_exc()}")
            with st.expander("🔍 Technical Details"):
                st.code(traceback.format_exc())
    return wrapper

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
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
        color: #2d3748 !important;
    }
    .feature-card h3 {
        color: #1a2332 !important;
        margin-bottom: 1rem;
    }
    .feature-card ol {
        color: #2d3748 !important;
        padding-left: 1.25rem;
    }
    .feature-card li {
        color: #2d3748 !important;
        margin-bottom: 0.5rem;
    }
    .feature-card strong {
        color: #1a2332 !important;
    }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        color: #1a2332 !important;
    }
    .upload-area h3 {
        color: #1a2332 !important;
        margin-bottom: 0.75rem;
    }
    .upload-area p {
        color: #2d3748 !important;
    }
    .upload-area .upload-hint {
        color: #4a5568 !important;
        font-size: 0.9rem;
    }
    .upload-area:hover {
        background: #e8ecf1;
        border-color: #764ba2;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        color: #155724;
    }
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
        st.markdown('<div class="stats-card"><h3>100%</h3><p>Privacy Guaranteed</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upload Callout
    if st.session_state.sample_data is None:
        st.markdown("""
        <div class="upload-area">
            <h3>📤 Start with Your Own Data</h3>
            <p>Upload a sample dataset to generate synthetic data with the same patterns</p>
            <p class="upload-hint">Supported: CSV, Excel, JSON</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🚀 Key Features
        
        **📊 Pre-built Templates**
        - Personal/Customer Data
        - Sales Transactions
        - Employee Records
        - Time Series Data
        - Application Logs
        - System Metrics
        - IoT Sensor Data
        - Healthcare Records
        
        **📤 User-Defined Generation**
        - Upload any dataset
        - AI learns patterns automatically
        - Generate realistic synthetic data
        """)
    
    with col2:
        st.markdown("""
        ### ✨ Advanced Features
        
        **🔒 Privacy First**
        - No real data exposure
        - PII Detection & Removal
        - GDPR/CCPA Compliant
        
        **📈 Quality Analysis**
        - Statistical Similarity
        - Distribution Matching
        - Interactive Dashboard
        """)
    
    st.markdown("---")
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 Quick Start Guide</h3>
        <ol>
            <li><strong>Upload Sample</strong> - Upload your dataset or use pre-built templates</li>
            <li><strong>Generate</strong> - Configure parameters and generate synthetic data</li>
            <li><strong>Analyze</strong> - View quality metrics and visualizations</li>
            <li><strong>Export</strong> - Download in your preferred format</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

@handle_errors
def upload_page():
    st.subheader("📤 Upload Sample Data")
    st.markdown("Upload a sample dataset to generate synthetic data with the same patterns")
    
    uploaded = st.file_uploader(
        "Choose a CSV, Excel, or JSON file",
        type=['csv', 'xlsx', 'xls', 'json']
    )
    
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            elif uploaded.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_json(uploaded)
            
            st.session_state.sample_data = df
            
            st.markdown(f"""
            <div class="success-box">
                ✅ Successfully loaded <strong>{len(df):,}</strong> records with <strong>{len(df.columns)}</strong> columns
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(df.head(10))
            
            with st.expander("📋 Data Summary"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    missing = df.isnull().sum().sum()
                    st.metric("Missing Values", f"{missing:,}")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            logger.error(f"File upload error: {str(e)}\n{traceback.format_exc()}")

@handle_errors
def generate_page():
    st.subheader("⚙️ Generate Synthetic Data")
    
    pipeline = st.session_state.pipeline
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_type = st.selectbox("Select Data Type", Config.DATA_TYPES)
    
    with col2:
        num_records = st.number_input(
            "Number of Records",
            min_value=Config.MIN_RECORDS,
            max_value=Config.MAX_RECORDS,
            value=Config.DEFAULT_RECORDS,
            step=100
        )
    
    if data_type == "User-Defined (Upload Sample)":
        if st.session_state.sample_data is not None:
            st.info(f"📊 Using sample data: {len(st.session_state.sample_data):,} records")
        else:
            st.warning("⚠️ No sample data loaded. Please upload a sample first.")
            if st.button("📤 Go to Upload Sample"):
                st.session_state.page = "Upload Sample"
                st.rerun()
            return
    
    with st.expander("⚙️ Advanced Options"):
        preserve_correlations = st.checkbox("Preserve Correlations", value=True)
        enable_privacy = st.checkbox("Enable Privacy Protection", value=True)
    
    if st.button("🚀 Generate Data", use_container_width=True):
        try:
            with st.spinner(f"🔄 Generating {num_records} synthetic records..."):
                start_time = time.time()
                
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
                
                st.session_state.generated_data = df
                
                # Quality Report
                with st.spinner("📊 Analyzing quality..."):
                    import importlib
                    import analytics.quality_report as _qr_mod
                    importlib.reload(_qr_mod)
                    reporter = _qr_mod.QualityReporter()
                    sample_for_report = None
                    roles_for_report = None
                    if data_type == "User-Defined (Upload Sample)":
                        sample_for_report = st.session_state.sample_data
                        roles_for_report = getattr(df, 'attrs', {}).get('column_roles')
                    try:
                        report = reporter.generate_report(
                            df,
                            sample=sample_for_report,
                            column_roles=roles_for_report,
                        )
                    except TypeError:
                        # Stale module without sample/column_roles kwargs
                        report = reporter.generate_report(df)
                    st.session_state.quality_report = report
                
                # Save History
                st.session_state.history.save({
                    'type': data_type,
                    'records': len(df),
                    'columns': len(df.columns),
                    'time': gen_time,
                    'quality': report.get('overall_score', 0)
                })
                
                st.markdown(f"""
                <div class="success-box">
                    ✅ Generated <strong>{len(df):,}</strong> records in {gen_time:.2f} seconds!
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    quality = report.get('overall_score', 0)
                    st.metric("Quality Score", f"{quality:.1%}")

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
                    num_f = report.get('numeric_fidelity')
                    if num_f is not None:
                        st.caption(f"Numeric fidelity: {num_f:.1%}")
                
        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")
            logger.error(f"Generation error: {str(e)}\n{traceback.format_exc()}")
            with st.expander("🔍 Technical Details"):
                st.code(traceback.format_exc())
    
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
        st.warning("No data generated yet. Go to Generate page first.")
        return
    
    df = st.session_state.generated_data
    report = st.session_state.quality_report or {}
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Quality", f"{report.get('overall_score', 0):.1%}")
    with col2:
        st.metric("Completeness", f"{report.get('completeness', 0):.1%}")
    with col3:
        st.metric("Open Uniqueness", f"{report.get('uniqueness', 0) or 0:.1%}")
    with col4:
        st.metric("ID Uniqueness", f"{report.get('id_uniqueness', 0):.1%}")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Date Diversity", f"{report.get('date_diversity', 0):.1%}")
    with m2:
        enum_f = report.get('enum_fidelity')
        st.metric("Enum Fidelity", f"{enum_f:.1%}" if enum_f is not None else "N/A")
    with m3:
        coh = report.get('name_email_coherence')
        st.metric("Name↔Email", f"{coh:.1%}" if coh is not None else "N/A")
    with m4:
        num_f = report.get('numeric_fidelity')
        st.metric("Numeric Fidelity", f"{num_f:.1%}" if num_f is not None else "N/A")
    
    st.markdown("---")
    
    try:
        dashboard = Dashboard()
        fig = dashboard.create_quality_dashboard(df, report)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Dashboard creation failed: {str(e)}")
        logger.error(f"Dashboard error: {str(e)}")
    
    with st.expander("📊 Detailed Metrics Report"):
        st.json(report)

@handle_errors
def export_page():
    st.subheader("💾 Export Data")
    
    if st.session_state.generated_data is None:
        st.warning("No data to export. Generate data first.")
        return
    
    df = st.session_state.generated_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox("Export Format", Config.EXPORT_FORMATS)
    
    with col2:
        filename = st.text_input(
            "File Name",
            f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    if st.button("📥 Export Data", use_container_width=True):
        try:
            with st.spinner("🔄 Exporting data..."):
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
                    
                    st.success(f"✅ Data ready for download ({file_size})")
                    
                    extension = "xlsx" if export_format == "Excel" else export_format.lower()
                    st.download_button(
                        label=f"📥 Download {export_format} ({file_size})",
                        data=data_bytes,
                        file_name=f"{filename}.{extension}",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
            logger.error(f"Export error: {str(e)}\n{traceback.format_exc()}")

# ==================== MAIN ====================

def main():
    """Main Application"""
    
    st.sidebar.title("🎲 SynthSLM")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📤 Upload Sample", "⚙️ Generate", "📈 Dashboard", "💾 Export"]
    )
    
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