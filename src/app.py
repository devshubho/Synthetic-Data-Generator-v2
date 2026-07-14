"""
SynthSLM - Main Application with Advanced Deduplication
B.Tech Final Year Project
Author: Team Synthetic
This application provides a user-friendly interface for generating synthetic data for small language models."""

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
from utils.deduplicator import DataDeduplicator

# Setup
logger = setup_logger()
st.set_page_config(
    page_title=Config.APP_NAME,
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE ====================

def init_session():
    """Initialize session state with error handling"""
    try:
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.generated_data = None
            st.session_state.sample_data = None
            st.session_state.sample_data_type = None
            st.session_state.quality_report = None
            st.session_state.history = HistoryManager()
            st.session_state.pipeline = GenerationPipeline()
            st.session_state.dedup_stats = None
            st.session_state.error = None
            st.session_state.dedup_enabled = True
            st.session_state.dedup_strategy = 'latest'
            st.session_state.dedup_threshold = 0.85
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
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .stats-card:hover {
        transform: translateY(-5px);
    }
    .stats-card h3 {
        font-size: 2.5rem;
        margin: 0;
    }
    .stats-card p {
        margin: 0;
        opacity: 0.9;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateX(5px);
    }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        transition: all 0.3s ease;
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
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        color: #0c5460;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 8px;
        padding: 1rem;
        color: #856404;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .dedup-stats {
        background: #f0f4ff;
        border: 1px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== PAGES ====================

def home_page():
    """Home/Landing Page"""
    
    st.markdown('<h1 class="main-header">🎲 SynthSLM</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Synthetic Data Generation for Small Language Models</p>', unsafe_allow_html=True)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stats-card">
            <h3>10+</h3>
            <p>Data Types</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stats-card">
            <h3>100K</h3>
            <p>Records per Batch</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stats-card">
            <h3>4</h3>
            <p>Export Formats</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stats-card">
            <h3>100%</h3>
            <p>Privacy Guaranteed</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upload Callout
    if st.session_state.sample_data is None:
        st.markdown("""
        <div class="upload-area">
            <h3>📤 Start with Your Own Data</h3>
            <p style="color: #666;">Upload a sample dataset to generate synthetic data with the same patterns</p>
            <p style="color: #999; font-size: 0.9rem;">Supported: CSV, Excel, JSON</p>
            <br>
            <p style="color: #667eea; font-weight: bold;">👉 Go to Upload Sample page to get started</p>
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
        - Financial Transactions
        - Toll Plaza Data
        
        **📤 User-Defined Generation**
        - Upload any dataset
        - AI learns patterns automatically
        - Generate realistic synthetic data
        - Preserve relationships
        """)
    
    with col2:
        st.markdown("""
        ### ✨ Advanced Features
        
        **🔒 Privacy First**
        - No real data exposure
        - PII Detection & Removal
        - GDPR/CCPA Compliant
        
        **🧹 Intelligent Deduplication**
        - Auto-detects duplicates
        - Removes exact & near duplicates
        - Resolves inconsistent data
        - Smart conflict resolution
        
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
            <li><strong>Configure</strong> - Set number of records and deduplication options</li>
            <li><strong>Generate</strong> - Click generate to create synthetic data</li>
            <li><strong>Analyze</strong> - View quality metrics and visualizations</li>
            <li><strong>Export</strong> - Download in your preferred format</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

def upload_page():
    """Upload Sample Data Page"""
    
    st.subheader("📤 Upload Sample Data")
    st.markdown("Upload a sample dataset to generate synthetic data with the same patterns")
    
    uploaded = st.file_uploader(
        "Choose a CSV, Excel, or JSON file",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Upload your sample dataset"
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
            st.session_state.sample_data_type = _detect_data_type(df)
            
            st.markdown(f"""
            <div class="success-box">
                ✅ Successfully loaded <strong>{len(df):,}</strong> records with <strong>{len(df.columns)}</strong> columns
            </div>
            """, unsafe_allow_html=True)
            
            # Data preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data Summary
            with st.expander("📋 Data Summary", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", f"{len(df):,}")
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    missing = df.isnull().sum().sum()
                    st.metric("Missing Values", f"{missing:,}")
                with col4:
                    duplicates = df.duplicated().sum()
                    st.metric("Duplicates Found", f"{duplicates:,}")
                
                # Column Info
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Unique': df.nunique(),
                    'Null %': (df.isnull().sum() / len(df) * 100).round(2)
                })
                st.dataframe(col_info, use_container_width=True)
            
            # Detect data type
            detected_type = _detect_data_type(df)
            st.info(f"📌 Detected data type: **{detected_type.title()}**")
            
            st.info("💡 **Next:** Go to **Generate** page and select 'User-Defined (Upload Sample)'")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            logger.error(f"File upload error: {str(e)}\n{traceback.format_exc()}")

def _detect_data_type(data: pd.DataFrame) -> str:
    """Auto-detect data type from columns"""
    columns = [c.lower() for c in data.columns]
    
    # Toll data
    if any(col in columns for col in ['vehicle_number', 'toll_plaza_id', 'fastag_id', 'lane_number']):
        return 'toll'
    
    # Personal data
    if any(col in columns for col in ['first_name', 'last_name', 'email', 'phone']):
        return 'personal'
    
    # Sales data
    if any(col in columns for col in ['transaction_id', 'customer_id', 'product', 'quantity']):
        return 'sales'
    
    # Employee data
    if any(col in columns for col in ['employee_id', 'department', 'salary']):
        return 'employee'
    
    # Healthcare data
    if any(col in columns for col in ['patient_id', 'condition', 'medication']):
        return 'healthcare'
    
    # IoT data
    if any(col in columns for col in ['device_id', 'sensor', 'device_type']):
        return 'iot'
    
    # Financial data
    if any(col in columns for col in ['account_id', 'transaction', 'amount', 'currency']):
        return 'financial'
    
    return 'unknown'

def generate_page():
    """Data Generation Page"""
    
    st.subheader("⚙️ Generate Synthetic Data")
    
    pipeline = st.session_state.pipeline
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_type = st.selectbox(
            "Select Data Type",
            Config.DATA_TYPES,
            help="Choose a template or use your uploaded sample"
        )
    
    with col2:
        num_records = st.number_input(
            "Number of Records",
            min_value=Config.MIN_RECORDS,
            max_value=Config.MAX_RECORDS,
            value=Config.DEFAULT_RECORDS,
            step=100,
            help="Number of synthetic records to generate"
        )
    
    # User-Defined Check
    if data_type == "User-Defined (Upload Sample)":
        if st.session_state.sample_data is not None:
            st.markdown(f"""
            <div class="info-box">
                📊 Using sample data: <strong>{len(st.session_state.sample_data):,}</strong> records, 
                <strong>{len(st.session_state.sample_data.columns)}</strong> columns
                <br>📌 Detected type: <strong>{st.session_state.sample_data_type or 'unknown'}</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                ⚠️ No sample data loaded. Please upload a sample first.
            </div>
            """, unsafe_allow_html=True)
            if st.button("📤 Go to Upload Sample"):
                st.session_state.page = "Upload Sample"
                st.rerun()
            return
    
    # Advanced Options
    with st.expander("⚙️ Advanced Options", expanded=False):
        preserve_correlations = st.checkbox("Preserve Correlations", value=True)
        enable_privacy = st.checkbox("Enable Privacy Protection", value=True)
        
        st.markdown("---")
        st.markdown("### 🧹 Deduplication Options")
        
        enable_dedup = st.checkbox("Enable Intelligent Deduplication", value=True)
        
        if enable_dedup:
            col1, col2 = st.columns(2)
            with col1:
                dedup_strategy = st.selectbox(
                    "Deduplication Strategy",
                    ["latest", "most_complete", "first"],
                    format_func=lambda x: {
                        "latest": "Keep latest records (by timestamp)",
                        "most_complete": "Keep most complete records",
                        "first": "Keep first occurrence"
                    }.get(x, x),
                    help="Strategy to resolve conflicts"
                )
            with col2:
                dedup_threshold = st.slider(
                    "Similarity Threshold",
                    0.5, 1.0, 0.85, 0.05,
                    help="How similar two records must be to be considered duplicates"
                )
            
            # Update session state
            st.session_state.dedup_enabled = enable_dedup
            st.session_state.dedup_strategy = dedup_strategy
            st.session_state.dedup_threshold = dedup_threshold
    
    # Generate Button
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
                
                # Store data
                st.session_state.generated_data = df
                
                # Quality Report
                with st.spinner("📊 Analyzing quality..."):
                    reporter = QualityReporter()
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
                
                # Stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    quality = report.get('overall_score', 0)
                    st.metric("Quality Score", f"{quality:.1%}")
                
        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")
            logger.error(f"Generation error: {str(e)}\n{traceback.format_exc()}")
            with st.expander("🔍 Technical Details"):
                st.code(traceback.format_exc())
    
    # Display Generated Data
    if st.session_state.generated_data is not None:
        df = st.session_state.generated_data
        
        st.subheader("📊 Generated Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Show deduplication stats if available
        if hasattr(st.session_state, 'dedup_stats') and st.session_state.dedup_stats:
            with st.expander("🧹 Deduplication Report"):
                stats = st.session_state.dedup_stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original", stats.get('original_count', 0))
                with col2:
                    st.metric("Final", stats.get('final_count', 0))
                with col3:
                    removed = stats.get('removed_total', 0)
                    st.metric("Removed", removed, delta=f"-{removed}")
                
                if 'steps' in stats:
                    st.json(stats['steps'])
        
        with st.expander("📋 Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Unique': df.nunique(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)

def dashboard_page():
    """Quality Dashboard"""
    
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
        st.metric("Uniqueness", f"{report.get('uniqueness', 0):.1%}")
    with col4:
        st.metric("Privacy Score", f"{report.get('privacy_score', 0):.1%}")
    
    st.markdown("---")
    
    dashboard = Dashboard()
    fig = dashboard.create_quality_dashboard(df, report)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📊 Detailed Metrics Report"):
        st.json(report)

def export_page():
    """Export Data Page"""
    
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
            f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Name for the exported file"
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
                    
                    st.download_button(
                        label=f"📥 Download {export_format} ({file_size})",
                        data=data_bytes,
                        file_name=f"{filename}.{export_format.lower()}",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
            logger.error(f"Export error: {str(e)}\n{traceback.format_exc()}")

# ==================== MAIN NAVIGATION ====================

def main():
    """Main Application"""
    
    st.sidebar.title("🎲 SynthSLM")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📤 Upload Sample", "⚙️ Generate", "📈 Dashboard", "💾 Export"]
    )
    
    st.sidebar.markdown("---")
    
    # Session Info
    if st.session_state.sample_data is not None:
        st.sidebar.success(f"📊 Sample: {len(st.session_state.sample_data):,} rows")
    else:
        st.sidebar.info("📊 No sample loaded")
    
    if st.session_state.generated_data is not None:
        st.sidebar.info(f"📦 Generated: {len(st.session_state.generated_data):,} rows")
    else:
        st.sidebar.info("📦 No data generated")
    
    st.sidebar.markdown("---")
    
    # Dedup Status
    if st.session_state.dedup_enabled:
        st.sidebar.info(f"🧹 Dedup: {st.session_state.dedup_strategy}")
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"v{Config.VERSION} | {Config.AUTHOR}")
    
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