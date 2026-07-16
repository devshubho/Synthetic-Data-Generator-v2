"""
Quality Report - Role-aware analysis for synthetic custom data.
"""

import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from logger import get_logger

logger = get_logger()

ID_COL_RE = re.compile(
    r'(^id$|_id$|order_id|transaction_id|employee_id|account_id|patient_id|claim_id|'
    r'policy_id|kyc_id|ref_id|uuid|sku|imei|msisdn|loan_id|booking_id)',
    re.IGNORECASE,
)

OPEN_ROLES = {
    'person_name', 'first_name', 'last_name', 'email', 'city', 'address',
    'phone', 'company', 'job', 'state', 'country', 'text_open',
    'aadhaar', 'pan', 'ifsc', 'upi', 'pincode', 'gstin',
}
CLOSED_ROLES = {'closed_categorical', 'product'}
EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


class QualityReporter:
    """Generate comprehensive quality reports for synthetic data."""

    def generate_report(
        self,
        data: pd.DataFrame,
        sample: Optional[pd.DataFrame] = None,
        column_roles: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        roles = column_roles or self._infer_roles(data, sample)
        report = {
            'overall_score': self._overall_score(data, sample, roles),
            'completeness': self._completeness(data),
            'uniqueness': self._open_field_uniqueness(data, roles),
            'diversity': self._open_field_diversity(data, roles),
            'id_uniqueness': self._id_uniqueness(data),
            'date_diversity': self._date_diversity(data),
            'enum_fidelity': self._enum_fidelity(data, sample, roles),
            'name_email_coherence': self._name_email_coherence(data, roles),
            'numeric_fidelity': self._numeric_fidelity(data, sample, roles),
            'privacy_score': self._privacy_score(data),
            'statistics': self._statistics(data),
            'column_roles': roles,
        }
        return report

    def _overall_score(
        self,
        data: pd.DataFrame,
        sample: Optional[pd.DataFrame],
        roles: Dict[str, str],
    ) -> float:
        scores = [
            self._completeness(data),
            self._id_uniqueness(data),
            self._date_diversity(data),
            self._open_field_diversity(data, roles),
        ]
        for metric in (
            self._enum_fidelity(data, sample, roles),
            self._name_email_coherence(data, roles),
            self._numeric_fidelity(data, sample, roles),
            self._open_field_uniqueness(data, roles),
        ):
            if metric is not None:
                scores.append(metric)
        return round(float(np.mean(scores)), 3)

    def _infer_roles(
        self,
        data: pd.DataFrame,
        sample: Optional[pd.DataFrame] = None,
    ) -> Dict[str, str]:
        ref = sample if sample is not None else data
        roles: Dict[str, str] = {}
        for col in data.columns:
            name = str(col).strip().lower().replace(' ', '_')
            series = ref[col] if col in ref.columns else data[col]
            if ID_COL_RE.search(str(col)):
                roles[col] = 'id'
                continue
            if pd.api.types.is_numeric_dtype(series):
                roles[col] = 'numeric'
                continue
            parsed = pd.to_datetime(series, errors='coerce')
            if parsed.notna().mean() > 0.8:
                roles[col] = 'datetime'
                continue
            non_null = series.dropna().astype(str).str.strip()
            if 'email' in name or (
                len(non_null) and non_null.map(lambda x: bool(EMAIL_RE.match(x))).mean() > 0.6
            ):
                roles[col] = 'email'
            elif any(k in name for k in (
                'full_name', 'customer_name', 'employee_name', 'patient_name', 'doctor',
            )) or name in ('name',):
                roles[col] = 'person_name'
            elif 'city' in name:
                roles[col] = 'city'
            elif any(k in name for k in ('phone', 'mobile')):
                roles[col] = 'phone'
            else:
                unique_count = non_null.nunique() if len(non_null) else 0
                unique_ratio = unique_count / len(non_null) if len(non_null) else 0
                roles[col] = (
                    'closed_categorical'
                    if unique_count and unique_ratio < 0.55
                    else 'text_open'
                )
        return roles

    def _completeness(self, data: pd.DataFrame) -> float:
        total = data.shape[0] * data.shape[1]
        if total == 0:
            return 0.0
        return float(1 - (data.isnull().sum().sum() / total))

    def _open_field_uniqueness(
        self, data: pd.DataFrame, roles: Dict[str, str]
    ) -> Optional[float]:
        cols = [
            c for c in data.columns
            if roles.get(c) in OPEN_ROLES or roles.get(c) == 'id'
        ]
        if not cols:
            return None
        n = len(data)
        if n == 0:
            return 0.0
        return float(np.mean([data[c].nunique(dropna=True) / n for c in cols]))

    def _open_field_diversity(self, data: pd.DataFrame, roles: Dict[str, str]) -> float:
        scores = []
        for col in data.columns:
            role = roles.get(col, '')
            if role in CLOSED_ROLES or role in ('id', 'datetime', 'numeric'):
                continue
            if role not in OPEN_ROLES:
                continue
            probs = data[col].value_counts(normalize=True)
            if len(probs) == 0:
                scores.append(0.0)
                continue
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            max_entropy = np.log(len(probs) + 1e-10)
            scores.append(float(entropy / max_entropy) if max_entropy > 0 else 0.0)
        return float(np.mean(scores)) if scores else 1.0

    def _enum_fidelity(
        self,
        data: pd.DataFrame,
        sample: Optional[pd.DataFrame],
        roles: Dict[str, str],
    ) -> Optional[float]:
        if sample is None:
            return None
        closed_cols = [
            c for c in data.columns
            if roles.get(c) in CLOSED_ROLES and c in sample.columns
        ]
        if not closed_cols:
            return None
        scores = []
        for col in closed_cols:
            seed = sample[col].dropna().astype(str).str.strip()
            syn = data[col].dropna().astype(str).str.strip()
            if len(seed) == 0 or len(syn) == 0:
                scores.append(0.0)
                continue
            seed_set, syn_set = set(seed), set(syn)
            vocab_score = len(syn_set & seed_set) / len(syn_set) if syn_set else 0.0
            seed_p = seed.value_counts(normalize=True)
            syn_p = syn.value_counts(normalize=True)
            all_vals = sorted(seed_set | syn_set)
            p = np.array([seed_p.get(v, 0.0) for v in all_vals], dtype=float)
            q = np.array([syn_p.get(v, 0.0) for v in all_vals], dtype=float)
            tv = 0.5 * np.abs(p - q).sum()
            scores.append(0.5 * vocab_score + 0.5 * float(1.0 - tv))
        return float(np.mean(scores)) if scores else None

    def _name_email_coherence(
        self, data: pd.DataFrame, roles: Dict[str, str]
    ) -> Optional[float]:
        name_col = next(
            (c for c, r in roles.items() if r in ('person_name', 'first_name') and c in data.columns),
            None,
        )
        email_col = next(
            (c for c, r in roles.items() if r == 'email' and c in data.columns),
            None,
        )
        if not name_col or not email_col:
            for c in data.columns:
                cl = c.lower()
                if name_col is None and 'name' in cl and 'file' not in cl:
                    name_col = c
                if email_col is None and 'email' in cl:
                    email_col = c
        if not name_col or not email_col:
            return None
        hits = total = 0
        for name, email in zip(data[name_col].astype(str), data[email_col].astype(str)):
            if not email or '@' not in email or name in ('nan', 'None'):
                continue
            total += 1
            local = email.split('@', 1)[0].lower()
            parts = re.sub(r'[^a-z0-9\s]', '', name.lower()).split()
            if any(p in local for p in parts if len(p) >= 2):
                hits += 1
        return float(hits / total) if total else None

    def _numeric_fidelity(
        self,
        data: pd.DataFrame,
        sample: Optional[pd.DataFrame],
        roles: Dict[str, str],
    ) -> Optional[float]:
        if sample is None:
            return None
        numeric_cols = [
            c for c in data.columns
            if roles.get(c) == 'numeric' and c in sample.columns
        ]
        if not numeric_cols:
            return None
        scores = []
        for col in numeric_cols:
            s = pd.to_numeric(sample[col], errors='coerce').dropna()
            d = pd.to_numeric(data[col], errors='coerce').dropna()
            if len(s) < 2 or len(d) < 2:
                scores.append(0.5)
                continue
            s_mean, s_std = float(s.mean()), float(s.std()) or 1.0
            d_mean, d_std = float(d.mean()), float(d.std()) or 1.0
            mean_ok = abs(d_mean - s_mean) <= 2.0 * s_std
            std_ratio = min(d_std, s_std) / max(d_std, s_std)
            scores.append(1.0 if mean_ok else 0.5)
            scores.append(float(std_ratio))
        return float(np.mean(scores)) if scores else None

    def _privacy_score(self, data: pd.DataFrame) -> float:
        """Score privacy as inverse column uniqueness (lower uniqueness = better privacy)."""
        n = len(data)
        if n == 0:
            return 0.0
        scores = [
            1.0 - min(data[col].nunique(dropna=True) / n, 1.0)
            for col in data.columns
        ]
        return float(np.mean(scores)) if scores else 0.0

    def _id_uniqueness(self, data: pd.DataFrame) -> float:
        id_cols = [c for c in data.columns if ID_COL_RE.search(str(c))]
        if not id_cols:
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    continue
                s = data[col].dropna().astype(str)
                if len(s) == 0:
                    continue
                if s.nunique() / len(s) >= 0.95 and s.str.match(r'^[A-Za-z]+[-_]?\d+$').mean() > 0.5:
                    id_cols.append(col)
        if not id_cols:
            return 1.0
        n = len(data)
        if n == 0:
            return 0.0
        return float(np.mean([data[c].nunique(dropna=True) / n for c in id_cols]))

    def _date_diversity(self, data: pd.DataFrame) -> float:
        scores = []
        for col in data.columns:
            parsed = pd.to_datetime(data[col], errors='coerce')
            if parsed.notna().mean() < 0.8:
                continue
            vals = parsed.dropna()
            if len(vals) == 0:
                continue
            span = max((vals.max().normalize() - vals.min().normalize()).days, 0)
            denom = max(1, min(len(vals), span + 1 if span > 0 else vals.nunique()))
            scores.append(min(1.0, vals.nunique() / denom))
        return float(np.mean(scores)) if scores else 1.0

    def _statistics(self, data: pd.DataFrame) -> Dict:
        stats_dict = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                stats_dict[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                }
            else:
                vc = data[col].value_counts()
                stats_dict[col] = {
                    'unique': data[col].nunique(),
                    'most_common': vc.index[0] if len(vc) > 0 else None,
                }
        return stats_dict
