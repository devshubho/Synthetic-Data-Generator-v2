"""
Generation History - Track all generations
"""

import pandas as pd
import sqlite3
import json
from datetime import datetime
import os

class HistoryManager:
    """Manage generation history"""
    
    def __init__(self):
        self.db_path = "datasets/history.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                data_type TEXT,
                records INTEGER,
                columns INTEGER,
                generation_time REAL,
                quality_score REAL,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save(self, data: dict):
        """Save generation record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO generations 
            (timestamp, data_type, records, columns, generation_time, quality_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            data.get('type', 'Unknown'),
            data.get('records', 0),
            data.get('columns', 0),
            data.get('time', 0),
            data.get('quality', 0),
            json.dumps(data.get('metadata', {}))
        ))
        
        conn.commit()
        conn.close()
    
    def get_history(self, limit: int = 20) -> pd.DataFrame:
        """Get generation history"""
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
            SELECT timestamp, data_type, records, columns, 
                   generation_time, quality_score
            FROM generations
            ORDER BY id DESC
            LIMIT {limit}
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df