"""
Privacy Protection - Anonymization
"""

import pandas as pd
import re
from utils.constants import PII_PATTERNS
from logger import get_logger

logger = get_logger()

class Anonymizer:
    """Apply privacy protection to data"""
    
    def __init__(self):
        self.pii_patterns = PII_PATTERNS
        logger.info("Anonymizer initialized")
    
    def detect_pii(self, data: pd.DataFrame) -> dict:
        """Detect PII in dataframe"""
        detected = {}
        for col in data.columns:
            if data[col].dtype == 'object':
                for pattern_name, pattern in self.pii_patterns.items():
                    matches = data[col].astype(str).str.contains(pattern, regex=True).sum()
                    if matches > 0:
                        if col not in detected:
                            detected[col] = []
                        detected[col].append(pattern_name)
        return detected
    
    def apply_privacy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply privacy protection"""
        df = data.copy()
        pii_columns = self.detect_pii(df)
        
        for col, patterns in pii_columns.items():
            for pattern in patterns:
                if pattern == 'email':
                    df[col] = df[col].apply(lambda x: self._anonymize_email(str(x)))
                elif pattern == 'phone':
                    df[col] = df[col].apply(lambda x: self._anonymize_phone(str(x)))
                elif pattern == 'ssn':
                    df[col] = df[col].apply(lambda x: self._anonymize_ssn(str(x)))
                elif pattern == 'credit_card':
                    df[col] = df[col].apply(lambda x: self._anonymize_credit_card(str(x)))
        
        logger.info(f"Applied privacy protection to {len(pii_columns)} columns")
        return df
    
    def _anonymize_email(self, email: str) -> str:
        parts = email.split('@')
        if len(parts) == 2:
            return f"{parts[0][0]}***@{parts[1]}"
        return email
    
    def _anonymize_phone(self, phone: str) -> str:
        digits = re.sub(r'\D', '', phone)
        if len(digits) >= 10:
            return digits[:3] + '***' + digits[-4:]
        return '***' + phone[-4:] if len(phone) > 4 else '***'
    
    def _anonymize_ssn(self, ssn: str) -> str:
        if len(ssn) >= 9:
            return '***-**-' + ssn[-4:]
        return ssn
    
    def _anonymize_credit_card(self, cc: str) -> str:
        digits = re.sub(r'\D', '', cc)
        if len(digits) >= 16:
            return '****-****-****-' + digits[-4:]
        return '****' + cc[-4:] if len(cc) > 4 else cc