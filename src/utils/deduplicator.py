"""
Universal Data Deduplication Engine
Handles duplicates across ALL data types with intelligent cleaning
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter
import re
from logger import get_logger

logger = get_logger()


class DataDeduplicator:
    """
    Universal deduplication engine for ANY data type
    Automatically detects and resolves duplicates intelligently
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.default_config = {
            'remove_exact_duplicates': True,
            'remove_near_duplicates': True,
            'consistency_check': True,
            'auto_resolve_conflicts': True,
            'keep_strategy': 'latest',  # 'latest', 'first', 'most_complete'
            'similarity_threshold': 0.85,
            'log_deduplication': True
        }
        self.config = {**self.default_config, **self.config}
        self.dedup_stats = {}
        self.patterns = {}
        logger.info("DataDeduplicator initialized")
    
    def deduplicate(self, df: pd.DataFrame, data_type: str = "unknown") -> pd.DataFrame:
        """
        Main deduplication pipeline - works for ANY data type
        """
        if df is None or len(df) == 0:
            logger.warning("Empty DataFrame provided")
            return df
        
        original_count = len(df)
        logger.info(f"Starting deduplication for {data_type} data: {original_count} rows")
        
        self.dedup_stats = {
            'original_count': original_count,
            'steps': {},
            'removed_counts': {},
            'final_count': 0
        }
        
        # Step 1: Detect data type and configure
        self._detect_data_type_patterns(df, data_type)
        
        # Step 2: Remove exact duplicates
        if self.config['remove_exact_duplicates']:
            df = self._remove_exact_duplicates(df)
        
        # Step 3: Remove near duplicates (based on similarity)
        if self.config['remove_near_duplicates']:
            df = self._remove_near_duplicates(df)
        
        # Step 4: Resolve inconsistent duplicates
        if self.config['consistency_check']:
            df = self._resolve_inconsistent_duplicates(df)
        
        # Step 5: Intelligent duplicate resolution
        if self.config['auto_resolve_conflicts']:
            df = self._auto_resolve_duplicates(df)
        
        self.dedup_stats['final_count'] = len(df)
        self.dedup_stats['removed_total'] = original_count - len(df)
        
        if self.config['log_deduplication']:
            self._log_dedup_stats()
        
        logger.info(f"Deduplication complete: {original_count} → {len(df)} rows (removed {original_count - len(df)})")
        
        return df
    
    def _detect_data_type_patterns(self, df: pd.DataFrame, data_type: str):
        """
        Auto-detect patterns and configure deduplication for specific data types
        """
        columns = df.columns.tolist()
        column_lower = [c.lower() for c in columns]
        
        # Store identified patterns
        self.patterns = {
            'id_columns': [],
            'unique_columns': [],
            'conflict_columns': [],
            'timestamp_columns': [],
            'strict_columns': []
        }
        
        # Detect ID columns
        id_keywords = ['id', 'uuid', 'key', 'number', 'code', 'reference']
        for col, col_lower in zip(columns, column_lower):
            if any(kw in col_lower for kw in id_keywords):
                self.patterns['id_columns'].append(col)
        
        # Detect timestamp columns
        timestamp_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'datetime']
        for col, col_lower in zip(columns, column_lower):
            if any(kw in col_lower for kw in timestamp_keywords):
                self.patterns['timestamp_columns'].append(col)
        
        # Data type specific patterns
        data_type_patterns = {
            'toll': {
                'unique': ['vehicle_number', 'transaction_id', 'fastag_id'],
                'conflict': ['vehicle_type', 'vehicle_class'],
                'strict': ['vehicle_number']
            },
            'personal': {
                'unique': ['email', 'phone', 'id'],
                'conflict': ['first_name', 'last_name', 'address'],
                'strict': ['email']
            },
            'sales': {
                'unique': ['transaction_id', 'customer_id'],
                'conflict': ['product', 'category', 'quantity'],
                'strict': ['transaction_id']
            },
            'employee': {
                'unique': ['employee_id', 'email'],
                'conflict': ['department', 'position', 'salary'],
                'strict': ['employee_id']
            },
            'healthcare': {
                'unique': ['patient_id', 'email'],
                'conflict': ['condition', 'medication'],
                'strict': ['patient_id']
            }
        }
        
        # Apply data type specific patterns
        for key, pattern in data_type_patterns.items():
            if key in data_type.lower() or any(key in col_lower for col_lower in column_lower):
                for col in pattern.get('unique', []):
                    matching_cols = [c for c in columns if col in c.lower()]
                    self.patterns['unique_columns'].extend(matching_cols)
                
                for col in pattern.get('conflict', []):
                    matching_cols = [c for c in columns if col in c.lower()]
                    self.patterns['conflict_columns'].extend(matching_cols)
                
                for col in pattern.get('strict', []):
                    matching_cols = [c for c in columns if col in c.lower()]
                    self.patterns['strict_columns'].extend(matching_cols)
                break
        
        # Remove duplicates from patterns
        for key in self.patterns:
            self.patterns[key] = list(set(self.patterns[key]))
        
        logger.debug(f"Detected patterns: {self.patterns}")
        return self.patterns
    
    def _remove_exact_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicate rows"""
        before = len(df)
        
        subset = self.patterns['strict_columns'] if self.patterns['strict_columns'] else None
        
        if subset:
            df = df.drop_duplicates(subset=subset, keep='last')
        else:
            df = df.drop_duplicates()
        
        removed = before - len(df)
        self.dedup_stats['steps']['exact_duplicates'] = {'before': before, 'after': len(df), 'removed': removed}
        logger.debug(f"Removed {removed} exact duplicates")
        
        return df
    
    def _remove_near_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove near-duplicate rows based on similarity"""
        before = len(df)
        
        if len(df) < 2:
            return df
        
        unique_cols = self.patterns['unique_columns'] if self.patterns['unique_columns'] else df.columns.tolist()
        
        to_drop = []
        seen_items = {}
        
        for idx, row in df.iterrows():
            fingerprint = self._create_fingerprint(row, unique_cols)
            
            duplicate_found = False
            for existing_fp, existing_idx in seen_items.items():
                similarity = self._calculate_similarity(fingerprint, existing_fp)
                if similarity >= self.config['similarity_threshold']:
                    if self._is_more_complete(row, df.loc[existing_idx]):
                        to_drop.append(existing_idx)
                        seen_items[existing_fp] = idx
                    else:
                        to_drop.append(idx)
                    duplicate_found = True
                    break
            
            if not duplicate_found:
                seen_items[fingerprint] = idx
        
        df = df.drop(index=to_drop)
        
        removed = before - len(df)
        self.dedup_stats['steps']['near_duplicates'] = {'before': before, 'after': len(df), 'removed': removed}
        logger.debug(f"Removed {removed} near duplicates")
        
        return df
    
    def _resolve_inconsistent_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resolve duplicate entries with inconsistent values"""
        before = len(df)
        
        if self.patterns['conflict_columns'] and self.patterns['strict_columns']:
            strict_col = self.patterns['strict_columns'][0]
            conflict_cols = self.patterns['conflict_columns']
            
            for strict_val in df[strict_col].unique():
                group = df[df[strict_col] == strict_val]
                if len(group) > 1:
                    for conflict_col in conflict_cols:
                        if conflict_col in df.columns:
                            unique_values = group[conflict_col].unique()
                            if len(unique_values) > 1:
                                most_common = group[conflict_col].mode()[0]
                                df.loc[df[strict_col] == strict_val, conflict_col] = most_common
                                logger.debug(f"Resolved conflict in {conflict_col} for {strict_col}={strict_val}")
        
        df = df.drop_duplicates()
        
        removed = before - len(df)
        self.dedup_stats['steps']['inconsistent_duplicates'] = {'before': before, 'after': len(df), 'removed': removed}
        
        return df
    
    def _auto_resolve_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-resolve duplicate entries using intelligent strategies"""
        before = len(df)
        
        if self.config['keep_strategy'] == 'most_complete':
            completeness_scores = df.notna().sum(axis=1) / len(df.columns)
            df['_completeness_score'] = completeness_scores
            
            df = df.sort_values('_completeness_score', ascending=False)
            
            subset = self.patterns['strict_columns'] if self.patterns['strict_columns'] else ['_completeness_score']
            df = df.drop_duplicates(subset=subset, keep='first')
            
            df = df.drop(columns=['_completeness_score'])
        
        elif self.config['keep_strategy'] == 'latest':
            if self.patterns['timestamp_columns']:
                ts_col = self.patterns['timestamp_columns'][0]
                if ts_col in df.columns:
                    if not pd.api.types.is_datetime64_dtype(df[ts_col]):
                        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
                    
                    df = df.sort_values(ts_col, ascending=False)
                    
                    subset = self.patterns['strict_columns'] if self.patterns['strict_columns'] else None
                    df = df.drop_duplicates(subset=subset, keep='first')
        
        removed = before - len(df)
        self.dedup_stats['steps']['auto_resolve'] = {'before': before, 'after': len(df), 'removed': removed}
        
        return df
    
    def _create_fingerprint(self, row: pd.Series, columns: List[str]) -> str:
        """Create a string fingerprint for similarity comparison"""
        if not columns:
            columns = row.index.tolist()
        
        parts = []
        for col in columns:
            if col in row.index:
                val = str(row[col]) if pd.notna(row[col]) else ''
                val = val.lower().strip()
                val = re.sub(r'\s+', ' ', val)
                parts.append(val)
        
        return '|'.join(parts)[:1000]
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1, str2).ratio()
    
    def _is_more_complete(self, row1: pd.Series, row2: pd.Series) -> bool:
        """Check if row1 is more complete than row2"""
        nulls1 = row1.isnull().sum()
        nulls2 = row2.isnull().sum()
        return nulls1 < nulls2
    
    def _log_dedup_stats(self):
        """Log deduplication statistics"""
        logger.info("=" * 60)
        logger.info("DEDUPLICATION STATISTICS")
        logger.info(f"Original records: {self.dedup_stats['original_count']}")
        logger.info(f"Final records: {self.dedup_stats['final_count']}")
        logger.info(f"Total removed: {self.dedup_stats['removed_total']}")
        logger.info("-" * 40)
        for step, stats in self.dedup_stats['steps'].items():
            if 'removed' in stats:
                logger.info(f"{step}: removed {stats['removed']} records")
        logger.info("=" * 60)
    
    def get_dedup_report(self) -> Dict[str, Any]:
        """Get detailed deduplication report"""
        return self.dedup_stats
    
    def get_duplicates_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze and return duplicate information"""
        duplicate_info = {
            'total_duplicates': int(df.duplicated().sum()),
            'duplicate_percentage': float((df.duplicated().sum() / len(df)) * 100) if len(df) > 0 else 0,
            'duplicate_columns': {},
            'duplicate_rows': []
        }
        
        for col in df.columns:
            duplicate_count = int(df[col].duplicated().sum())
            if duplicate_count > 0:
                duplicate_info['duplicate_columns'][col] = duplicate_count
        
        duplicate_rows = df[df.duplicated(keep=False)]
        if not duplicate_rows.empty:
            duplicate_info['duplicate_rows'] = duplicate_rows.to_dict('records')
        
        return duplicate_info