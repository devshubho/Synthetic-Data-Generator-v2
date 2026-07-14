"""
Quality Report - Role-aware analysis for synthetic custom data.
"""

import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


ID_COL_RE = re.compile(
    r'(^id$|_id$|order_id|transaction_id|employee_id|account_id|patient_id|claim_id|'
    r'policy_id|kyc_id|ref_id|uuid|sku|imei|msisdn)',
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
        """Generate full quality report.

        When sample/column_roles are provided (custom generation path),
        overall_score uses role-aware metrics that do not punish closed enums.
        """
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
            'null_analysis': self._null_analysis(data),
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
        enum = self._enum_fidelity(data, sample, roles)
        if enum is not None:
            scores.append(enum)
        coherence = self._name_email_coherence(data, roles)
        if coherence is not None:
            scores.append(coherence)
        numeric = self._numeric_fidelity(data, sample, roles)
        if numeric is not None:
            scores.append(numeric)
        # Prefer open-field uniqueness when open columns exist
        open_u = self._open_field_uniqueness(data, roles)
        if open_u is not None:
            scores.append(open_u)
        return round(float(np.mean(scores)), 3)

    def _infer_roles(
        self,
        data: pd.DataFrame,
        sample: Optional[pd.DataFrame] = None,
    ) -> Dict[str, str]:
        """Lightweight role inference when generator roles are unavailable."""
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
                continue
            if any(k in name for k in (
                'full_name', 'customer_name', 'employee_name', 'patient_name',
                'account_holder', 'holder_name', 'farmer_name',
            )) or name in ('name',):
                roles[col] = 'person_name'
                continue
            if 'city' in name or name in ('town', 'village', 'district'):
                roles[col] = 'city'
                continue
            if any(k in name for k in ('phone', 'mobile', 'tel')):
                roles[col] = 'phone'
                continue
            if 'aadhaar' in name:
                roles[col] = 'aadhaar'
                continue
            if name in ('pan', 'pan_number') or name.endswith('_pan'):
                roles[col] = 'pan'
                continue
            if 'ifsc' in name:
                roles[col] = 'ifsc'
                continue
            if 'upi' in name or name in ('vpa',):
                roles[col] = 'upi'
                continue
            if 'pincode' in name or 'pin_code' in name or name == 'zip':
                roles[col] = 'pincode'
                continue
            if 'gstin' in name or 'gst' == name:
                roles[col] = 'gstin'
                continue

            unique_count = non_null.nunique() if len(non_null) else 0
            unique_ratio = unique_count / len(non_null) if len(non_null) else 0
            enum_like = any(k in name for k in (
                'status', 'method', 'department', 'designation', 'type',
                'mode', 'channel', 'gender', 'payment', 'priority',
            ))
            if unique_count > 0 and unique_count <= max(25, int(0.5 * max(len(non_null), 1))) and (
                enum_like or unique_ratio < 0.55
            ):
                roles[col] = 'closed_categorical'
            elif unique_ratio >= 0.8:
                roles[col] = 'text_open'
            else:
                roles[col] = 'closed_categorical'
        return roles

    def _completeness(self, data: pd.DataFrame) -> float:
        total = data.shape[0] * data.shape[1]
        if total == 0:
            return 0.0
        nulls = data.isnull().sum().sum()
        return float(1 - (nulls / total))

    def _open_field_uniqueness(
        self,
        data: pd.DataFrame,
        roles: Dict[str, str],
    ) -> Optional[float]:
        """Uniqueness only for open/high-cardinality identity-like fields."""
        cols = [
            c for c in data.columns
            if roles.get(c) in OPEN_ROLES or roles.get(c) == 'id'
        ]
        if not cols:
            return None
        ratios = []
        for col in cols:
            n = len(data)
            if n == 0:
                ratios.append(0.0)
            else:
                ratios.append(data[col].nunique(dropna=True) / n)
        return float(np.mean(ratios))

    def _open_field_diversity(
        self,
        data: pd.DataFrame,
        roles: Dict[str, str],
    ) -> float:
        """Entropy diversity on open text fields; ignore closed enums."""
        scores = []
        for col in data.columns:
            role = roles.get(col, '')
            if role in CLOSED_ROLES or role in ('id', 'datetime', 'numeric'):
                continue
            if role not in OPEN_ROLES and not (
                not pd.api.types.is_numeric_dtype(data[col])
                and data[col].nunique(dropna=True) / max(len(data), 1) > 0.5
            ):
                continue

            if pd.api.types.is_numeric_dtype(data[col]):
                mean = data[col].mean()
                if mean != 0 and not pd.isna(mean):
                    cv = data[col].std() / abs(mean)
                    scores.append(min(float(cv), 1.0) if not pd.isna(cv) else 0.5)
                else:
                    scores.append(0.5)
            else:
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
        """How well closed enums preserve seed vocab and frequency."""
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

            seed_set = set(seed.tolist())
            syn_set = set(syn.tolist())
            # Penalize inventing new labels
            if not syn_set:
                scores.append(0.0)
                continue
            vocab_score = len(syn_set & seed_set) / len(syn_set)

            seed_p = seed.value_counts(normalize=True)
            syn_p = syn.value_counts(normalize=True)
            all_vals = sorted(seed_set | syn_set)
            p = np.array([seed_p.get(v, 0.0) for v in all_vals], dtype=float)
            q = np.array([syn_p.get(v, 0.0) for v in all_vals], dtype=float)
            # Total variation distance → similarity in [0, 1]
            tv = 0.5 * np.abs(p - q).sum()
            freq_score = float(1.0 - tv)
            scores.append(0.5 * vocab_score + 0.5 * freq_score)
        return float(np.mean(scores)) if scores else None

    def _name_email_coherence(
        self,
        data: pd.DataFrame,
        roles: Dict[str, str],
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
            # Heuristic column names
            for c in data.columns:
                cl = c.lower()
                if name_col is None and ('name' in cl and 'file' not in cl):
                    name_col = c
                if email_col is None and 'email' in cl:
                    email_col = c
        if not name_col or not email_col:
            return None

        hits = 0
        total = 0
        for name, email in zip(data[name_col].astype(str), data[email_col].astype(str)):
            if not email or '@' not in email or name in ('nan', 'None'):
                continue
            total += 1
            local = email.split('@', 1)[0].lower()
            parts = re.sub(r'[^a-z0-9\s]', '', name.lower()).split()
            if not parts:
                continue
            # Any name token (len>=2) appears in local-part
            if any(p in local for p in parts if len(p) >= 2):
                hits += 1
        if total == 0:
            return None
        return float(hits / total)

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
            # fallback by dtype
            numeric_cols = [
                c for c in data.columns
                if c in sample.columns
                and pd.api.types.is_numeric_dtype(data[c])
                and pd.api.types.is_numeric_dtype(sample[c])
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
            # Within 2 std of seed mean / comparable scale
            mean_ok = abs(d_mean - s_mean) <= 2.0 * s_std
            std_ratio = min(d_std, s_std) / max(d_std, s_std)
            scores.append(1.0 if mean_ok else 0.5)
            scores.append(float(std_ratio))
        return float(np.mean(scores)) if scores else None

    def _id_uniqueness(self, data: pd.DataFrame) -> float:
        """1.0 when all ID-like columns are collision-free."""
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
        scores = []
        for col in id_cols:
            n = len(data)
            if n == 0:
                scores.append(0.0)
            else:
                scores.append(data[col].nunique(dropna=True) / n)
        return float(np.mean(scores))

    def _date_diversity(self, data: pd.DataFrame) -> float:
        """Reward date columns that span many distinct values."""
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

    def _privacy_score(self, data: pd.DataFrame) -> float:
        if len(data) == 0:
            return 0.0
        return self._id_uniqueness(data)

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

    def _null_analysis(self, data: pd.DataFrame) -> Dict:
        return {
            'total_nulls': int(data.isnull().sum().sum()),
            'null_by_column': data.isnull().sum().to_dict(),
            'null_percentage': (data.isnull().sum() / max(len(data), 1) * 100).to_dict(),
        }
