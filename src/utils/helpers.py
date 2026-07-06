"""
Helper Functions
"""

import os
from datetime import datetime

def format_size(size_bytes: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def generate_id() -> str:
    """Generate unique ID"""
    import uuid
    return str(uuid.uuid4())[:8]

def get_file_info(file_path: str) -> dict:
    """Get file information"""
    return {
        'name': os.path.basename(file_path),
        'size': os.path.getsize(file_path),
        'size_readable': format_size(os.path.getsize(file_path)),
        'created': datetime.fromtimestamp(os.path.getctime(file_path)),
        'modified': datetime.fromtimestamp(os.path.getmtime(file_path))
    }