"""
Configuration Settings
"""

class Config:
    """Application Configuration"""
    
    APP_NAME = "Project Synthesi"
    VERSION = "3.0.0"
    AUTHOR = "Bikram Sarkar"
    
    # Generation Limits
    MAX_RECORDS = 100000
    MIN_RECORDS = 10
    DEFAULT_RECORDS = 1000
    
    # Data Types
    DATA_TYPES = [
        "Personal/Customer Data",
        "Sales Transactions",
        "Employee Records",
        "Time Series Data",
        "Application Logs",
        "System Metrics",
        "Correlated VM Data",
        "IoT Sensor Data",
        "Healthcare Records",
        "Financial Transactions",
        "User-Defined (Upload Sample)"
    ]
    
    # Export Formats
    EXPORT_FORMATS = ["CSV", "JSON", "Parquet", "Excel"]
    
    # Random Seed
    SEED = 42