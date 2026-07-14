"""
Logging Setup for SynthSLM
"""

import os
import logging
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Global logger instance
_logger = None

def setup_logger():
    """Configure logging with rotation and error tracking"""
    global _logger
    
    if _logger is not None:
        return _logger
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('SynthSLM')
    logger.setLevel(logging.DEBUG)
    
    # File handler with rotation (10MB per file, keep 5 backups)
    log_file = f"logs/synthslm_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)
    
    # Error file handler
    error_file = f"logs/error_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = RotatingFileHandler(
        error_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(exc_info)s')
    )
    logger.addHandler(error_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter('%(levelname)s - %(message)s')
    )
    logger.addHandler(console_handler)
    
    _logger = logger
    return logger

def get_logger():
    """Get the logger instance"""
    global _logger
    if _logger is None:
        return setup_logger()
    return _logger

def log_error(error: Exception, context: str = ""):
    """Log error with full traceback"""
    logger = get_logger()
    error_msg = f"ERROR in {context}: {str(error)}\n{traceback.format_exc()}"
    logger.error(error_msg)
    return error_msg