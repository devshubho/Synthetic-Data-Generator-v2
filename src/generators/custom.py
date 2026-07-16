"""
Custom Generator - Faker-driven authentic synthetic data from user samples.

Open fields use Faker (en_IN). Closed enums resample seed frequencies.
Person/email stay coherent. Product groups jointly sample seed rows.
IDs are unique; dates span seed min/max. Domain roles cover ecom, banking,
UPI, KYC, EMR, telecom, agri, insurance.
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from faker import Faker
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import warnings

from logger import get_logger
from utils.exceptions import GenerationError

warnings.filterwarnings('ignore')
logger = get_logger()

PRODUCT_GROUP_SPECS = [
    ['product_name', 'category', 'unit_price'],
    ['product', 'category', 'unit_price'],
    ['medicine_name', 'category', 'unit_price'],
    ['crop_name', 'crop_type', 'yield_kg'],
]

PERSON_NAME_COLS = {
    'customer_name', 'account_holder', 'full_name', 'employee_name', 'name',
    'patient_name', 'holder_name', 'beneficiary_name', 'insured_name',
    'farmer_name', 'subscriber_name', 'user_name', 'username',
    'doctor', 'physician', 'surgeon', 'nurse', 'specialist', 'consultant',
    'practitioner', 'agent_name', 'driver', 'doctor_name', 'physician_name',
}

PERSON_ROLE_KEYS = (
    'doctor', 'physician', 'surgeon', 'nurse', 'specialist', 'consultant',
    'practitioner', 'agent_name', 'driver', 'doctor_name', 'physician_name',
    'attending', 'caregiver',
)

VENDOR_LABEL_KEYS = (
    'courier', 'carrier', 'logistics', 'vendor', 'provider', 'bank_name',
    'insurer', 'shipping_partner', 'delivery_partner', 'fleet',
)

ID_NAME_RE = re.compile(
    r'(^id$|_id$|_id_|order_id|orderid|transaction_id|employee_id|account_id|'
    r'patient_id|claim_id|policy_id|kyc_id|ref_id|uuid|sku|imei|msisdn|'
    r'upi_ref|rrn|loan_id|booking_id)',
    re.IGNORECASE,
)
ID_VALUE_RE = re.compile(r'^([A-Za-z]+[-_]?)(\d+)$')
EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')

INDIAN_CITY_HINTS = {
    'delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore', 'bengaluru', 'hyderabad',
    'pune', 'ahmedabad', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'indore',
    'bhopal', 'patna', 'chandigarh', 'surat', 'kochi', 'gurgaon', 'gurugram',
    'noida', 'ghaziabad', 'faridabad', 'varanasi', 'agra', 'mysuru', 'vadodara',
}

INDIAN_CITIES = [
    'Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Ahmedabad', 'Chennai', 'Kolkata',
    'Pune', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Thane', 'Bhopal',
    'Visakhapatnam', 'Patna', 'Vadodara', 'Ghaziabad', 'Ludhiana', 'Agra', 'Nashik',
    'Faridabad', 'Meerut', 'Rajkot', 'Varanasi', 'Srinagar', 'Amritsar', 'Gurugram',
    'Noida', 'Coimbatore', 'Chandigarh', 'Guwahati', 'Mysuru', 'Raipur', 'Kota',
]

UPI_HANDLES = ['oksbi', 'okhdfcbank', 'okicici', 'paytm', 'ybl', 'apl', 'axl']


class CustomGenerator:
    """Semantic Faker-backed generator: novel authentic values + seed structure."""

    def __init__(self, sample_data: pd.DataFrame, locale: str = 'en_IN'):
        if sample_data is None:
            raise GenerationError("Sample data is None")
        if len(sample_data) < 2:
            logger.warning("Sample has fewer than 2 rows; duplicating for modeling")
            sample_data = pd.concat([sample_data, sample_data], ignore_index=True)

        self.sample = sample_data.copy().reset_index(drop=True)
        self.locale = locale
        self.india_mode = self._detect_india_context(self.sample)
        try:
            self.fake = Faker(self.locale)
        except Exception:
            self.fake = Faker('en_IN')
            self.locale = 'en_IN'

        self.models: Dict[str, dict] = {}
        self.column_types: Dict[str, str] = {}
        self.column_roles: Dict[str, str] = {}
        self.correlations = None
        self.linked_groups: List[List[str]] = []
        self.grouped_columns: Set[str] = set()
        self.person_name_col: Optional[str] = None
        self.email_col: Optional[str] = None
        self.first_name_col: Optional[str] = None
        self.last_name_col: Optional[str] = None
        self.email_domains: List[str] = ['gmail.com', 'yahoo.com', 'outlook.com']
        self.india_phone = False
        self.last_validation: Optional[dict] = None
        self._person_cache: List[Tuple[str, str]] = []

        self._validate_sample()
        self._build_models()
        self._detect_person_email_cols()
        self._detect_linked_groups()
        self._extract_email_domains()
        self._detect_india_phone_style()
        self._calculate_correlations()
        logger.info(
            f"CustomGenerator locale={self.locale} india_mode={self.india_mode} "
            f"roles={self.column_roles}"
        )

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        if np.isnan(value) or np.isinf(value):
            return default
        return value

    @staticmethod
    def _safe_std(values, default=1.0):
        if values is None or len(values) < 2:
            return default
        std = float(np.std(values))
        if np.isnan(std) or std <= 0:
            return default
        return std

    @staticmethod
    def _detect_india_context(sample: pd.DataFrame) -> bool:
        for col in sample.columns:
            name = str(col).lower()
            if 'city' in name or name in ('town', 'village', 'district'):
                hits = 0
                vals = sample[col].dropna().astype(str).head(50)
                for v in vals:
                    token = v.strip().lower()
                    if token in INDIAN_CITY_HINTS or any(h in token for h in INDIAN_CITY_HINTS):
                        hits += 1
                if len(vals) and hits / len(vals) >= 0.3:
                    return True
        blob = ' '.join(str(c).lower() for c in sample.columns)
        if any(k in blob for k in (
            'aadhaar', 'aadhar', 'ifsc', 'upi', 'gstin', 'pincode', 'kyc',
            'fasTag'.lower(), 'toll',
        )):
            return True
        return False

    def _detect_india_phone_style(self):
        for col, role in self.column_roles.items():
            if role != 'phone':
                continue
            vals = self.sample[col].dropna().astype(str).head(40)
            if len(vals) == 0:
                continue
            digits = vals.str.replace(r'\D', '', regex=True)
            if digits.str.startswith('91').mean() > 0.25 or digits.str.len().eq(10).mean() > 0.4:
                self.india_phone = True
                return
        self.india_phone = self.india_mode

    def _validate_sample(self):
        if len(self.sample) == 0:
            raise GenerationError("Sample is empty")
        if len(self.sample.columns) == 0:
            raise GenerationError("Sample has no columns")

    def _build_models(self):
        for col in self.sample.columns:
            col_type = self._detect_column_type(col, self.sample[col])
            role = self._detect_semantic_role(col, self.sample[col], col_type)
            self.column_types[col] = col_type
            self.column_roles[col] = role

            if col_type == 'numeric':
                model = self._build_numeric_model(self.sample[col])
            elif col_type == 'datetime':
                model = self._build_datetime_model(self.sample[col])
            elif col_type == 'id':
                model = self._build_id_model(col, self.sample[col])
            else:
                model = self._build_categorical_model(self.sample[col])

            model['role'] = role
            model['column'] = col
            self.models[col] = model

    def _detect_column_type(self, col_name: str, series: pd.Series) -> str:
        name = str(col_name).strip().lower().replace(' ', '_')
        if any(k in name for k in (
            'aadhaar', 'aadhar', 'ifsc', 'gstin', 'upi', 'phone', 'mobile',
            'contact_no', 'tel', 'msisdn',
        )) or name in (
            'pan', 'pan_number', 'pan_card', 'vpa', 'upi_id', 'gst', 'gst_number',
        ):
            return 'categorical'
        if name in ('pincode', 'pin_code', 'zip', 'zipcode', 'postal_code') or name.endswith('_pincode'):
            return 'categorical'

        if self._is_id_column(col_name, series):
            return 'id'
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        if pd.api.types.is_datetime64_dtype(series):
            return 'datetime'
        if pd.api.types.is_bool_dtype(series):
            return 'categorical'

        non_null = series.dropna()
        if len(non_null) == 0:
            return 'categorical'
        if self._is_boolean_like(non_null):
            return 'categorical'

        try:
            converted_dates = pd.to_datetime(non_null, errors='coerce')
            if converted_dates.notna().mean() > 0.8:
                return 'datetime'
        except Exception:
            pass

        try:
            converted_nums = pd.to_numeric(non_null, errors='coerce')
            if converted_nums.notna().mean() == 1.0:
                return 'numeric'
        except Exception:
            pass

        return 'categorical'

    def _detect_semantic_role(self, col_name: str, series: pd.Series, col_type: str) -> str:
        if col_type == 'id':
            return 'id'
        if col_type == 'numeric':
            return 'numeric'
        if col_type == 'datetime':
            return 'datetime'

        name = str(col_name).strip().lower().replace(' ', '_')
        non_null = series.dropna().astype(str).str.strip()

        if 'email' in name or 'e_mail' in name:
            return 'email'
        if len(non_null) and non_null.map(lambda x: bool(EMAIL_RE.match(x))).mean() > 0.6:
            return 'email'

        if 'aadhaar' in name or 'aadhar' in name:
            return 'aadhaar'
        if name in ('pan', 'pan_number', 'pan_card') or name.endswith('_pan'):
            return 'pan'
        if 'ifsc' in name:
            return 'ifsc'
        if 'upi' in name or name in ('vpa', 'upi_id'):
            return 'upi'
        if 'pincode' in name or 'pin_code' in name or name in ('zip', 'zipcode', 'postal_code'):
            return 'pincode'
        if 'gstin' in name or name in ('gst', 'gst_number'):
            return 'gstin'

        if name in ('first_name', 'firstname', 'fname', 'given_name'):
            return 'first_name'
        if name in ('last_name', 'lastname', 'lname', 'surname', 'family_name'):
            return 'last_name'
        if (
            name in PERSON_ROLE_KEYS
            or any(k in name for k in (
                'full_name', 'customer_name', 'account_holder', 'employee_name',
                'patient_name', 'holder_name', 'beneficiary', 'insured_name',
                'farmer_name', 'subscriber_name', 'user_name',
            ))
            or name in ('name', 'account_name')
            or any(name == k or name.endswith('_' + k) for k in PERSON_ROLE_KEYS)
        ):
            return 'person_name'

        if any(k in name for k in VENDOR_LABEL_KEYS) or name in VENDOR_LABEL_KEYS:
            return 'closed_categorical'

        if any(k in name for k in (
            'city', 'shipping_city', 'billing_city', 'town', 'village', 'district',
        )):
            return 'city'
        if name in ('state', 'province') or name.endswith('_state'):
            return 'state'
        if name in ('country', 'nation', 'country_name') or name.endswith('_country'):
            return 'country'
        if any(k in name for k in ('address', 'street', 'location', 'geo')):
            return 'address'
        if any(k in name for k in ('phone', 'mobile', 'contact_no', 'tel', 'msisdn')):
            return 'phone'
        if any(k in name for k in ('company', 'employer', 'organization', 'org_name', 'hospital')):
            return 'company'

        unique_count = non_null.nunique() if len(non_null) else 0
        unique_ratio = unique_count / len(non_null) if len(non_null) else 0
        avg_len = float(non_null.map(len).mean()) if len(non_null) else 0.0
        avg_words = float(non_null.map(lambda x: len(str(x).split())).mean()) if len(non_null) else 0.0
        looks_like_label = avg_len <= 40 and avg_words <= 6

        enum_like_name = any(k in name for k in (
            'status', 'method', 'department', 'designation', 'position',
            'type', 'mode', 'channel', 'gender', 'education', 'priority',
            'stage', 'plan', 'tier', 'band', 'crop_type', 'claim_status',
            'payment', 'order_status', 'txn_type', 'kyc_status', 'churn',
            'risk', 'coverage', 'diagnosis', 'blood_group', 'network',
        ))
        if unique_count > 0 and unique_count <= max(25, int(0.5 * len(non_null))) and (
            enum_like_name or unique_ratio < 0.55
        ):
            return 'closed_categorical'

        if any(k in name for k in ('job', 'job_title', 'occupation')) or name in ('title', 'role'):
            return 'job'

        if any(k in name for k in (
            'product_name', 'product', 'medicine', 'drug', 'sku_name', 'item_name', 'crop_name',
        )):
            return 'product'

        label_cap = max(50, len(non_null))
        if unique_count > 0 and unique_count <= label_cap and looks_like_label:
            return 'closed_categorical'
        if unique_count > 0 and (avg_len > 40 or avg_words > 6):
            return 'text_open'
        return 'closed_categorical' if looks_like_label else 'text_open'

    def _is_id_column(self, col_name: str, series: pd.Series) -> bool:
        if ID_NAME_RE.search(str(col_name).strip()):
            return True
        non_null = series.dropna().astype(str).str.strip()
        if len(non_null) < 2:
            return False
        unique_ratio = non_null.nunique() / len(non_null)
        pattern_hits = non_null.str.match(r'^[A-Za-z]+[-_]?\d+$').mean()
        return unique_ratio >= 0.9 and pattern_hits >= 0.7

    @staticmethod
    def _is_boolean_like(series: pd.Series) -> bool:
        normalized = (
            series.astype(str).str.strip().str.lower()
            .replace({
                '1': 'true', '0': 'false', 'yes': 'true', 'no': 'false',
                'y': 'true', 'n': 'false', 't': 'true', 'f': 'false',
            })
        )
        return normalized.isin({'true', 'false'}).mean() >= 0.8

    def _detect_person_email_cols(self):
        for col, role in self.column_roles.items():
            if role == 'person_name' and self.person_name_col is None:
                self.person_name_col = col
            if role == 'first_name' and self.first_name_col is None:
                self.first_name_col = col
            if role == 'last_name' and self.last_name_col is None:
                self.last_name_col = col
            if role == 'email' and self.email_col is None:
                self.email_col = col
        if self.person_name_col is None:
            for col in self.sample.columns:
                if col.lower().replace(' ', '_') in PERSON_NAME_COLS:
                    self.person_name_col = col
                    self.column_roles[col] = 'person_name'
                    break

    def _detect_linked_groups(self):
        col_lookup = {c.lower(): c for c in self.sample.columns}
        groups = []
        for spec in PRODUCT_GROUP_SPECS:
            resolved = []
            for name in spec:
                if name.lower() in col_lookup:
                    resolved.append(col_lookup[name.lower()])
            resolved = [c for c in resolved if c not in self.grouped_columns]
            if len(resolved) >= 2:
                groups.append(resolved)
                self.grouped_columns.update(resolved)
        self.linked_groups = groups

    def _extract_email_domains(self):
        domains = []
        for col, role in self.column_roles.items():
            if role != 'email':
                continue
            for v in self.sample[col].dropna().astype(str):
                if '@' in v:
                    domains.append(v.split('@', 1)[1].strip().lower())
        if domains:
            self.email_domains = sorted(set(domains))

    def _build_numeric_model(self, series: pd.Series) -> dict:
        values = pd.to_numeric(series, errors='coerce').dropna().values
        if len(values) == 0:
            return {
                'type': 'numeric', 'method': 'statistical',
                'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0,
                'as_integer': True, 'six_digit': False,
            }
        mean = self._safe_float(np.mean(values), 0.0)
        std = self._safe_std(values, 1.0)
        vmin = self._safe_float(np.min(values), mean)
        vmax = self._safe_float(np.max(values), mean)
        as_integer = bool(
            pd.api.types.is_integer_dtype(series)
            or np.allclose(values, np.round(values), equal_nan=True)
        )
        six_digit = bool(
            as_integer and len(values) > 0
            and np.all((values >= 100000) & (values <= 999999))
        )
        model = {
            'type': 'numeric', 'method': 'statistical',
            'mean': mean, 'std': std, 'min': vmin, 'max': vmax,
            'as_integer': as_integer, 'six_digit': six_digit,
        }
        if len(values) >= 5:
            try:
                kde = KernelDensity(kernel='gaussian', bandwidth='scott')
                kde.fit(values.reshape(-1, 1))
                model['method'] = 'kde'
                model['kde'] = kde
            except Exception:
                pass
        return model

    def _build_datetime_model(self, series: pd.Series) -> dict:
        dates = pd.to_datetime(series, errors='coerce').dropna()
        if getattr(dates.dtype, 'tz', None) is not None:
            dates = dates.dt.tz_localize(None)
        if len(dates) == 0:
            now = pd.Timestamp.now().normalize()
            return {
                'type': 'datetime', 'min': now, 'max': now,
                'unique_dates': [now], 'date_only': True,
                'date_range_days': 0, 'span_seconds': 0, 'seed_nunique': 0,
            }
        date_only = bool((dates == dates.dt.normalize()).all())
        min_d = dates.min().normalize() if date_only else dates.min()
        max_d = dates.max().normalize() if date_only else dates.max()
        unique = (
            sorted(dates.dt.normalize().unique())
            if date_only
            else sorted(dates.unique())
        )
        span_seconds = max(int((max_d - min_d).total_seconds()), 0)
        return {
            'type': 'datetime',
            'min': min_d, 'max': max_d,
            'unique_dates': list(unique),
            'date_only': date_only,
            'date_range_days': max((max_d.normalize() - min_d.normalize()).days, 0),
            'span_seconds': span_seconds,
            'seed_nunique': len(unique),
        }

    def _build_id_model(self, col_name: str, series: pd.Series) -> dict:
        values = series.dropna().astype(str).str.strip()
        prefix, width, start, use_uuid = '', 5, 1, False
        if len(values) == 0:
            use_uuid = True
        else:
            matches = [ID_VALUE_RE.match(v) for v in values]
            if all(m is not None for m in matches):
                prefixes = [m.group(1) for m in matches]
                prefix = prefixes[0]
                if not all(p == prefix for p in prefixes):
                    prefix = self._common_prefix(list(values))
                    nums = []
                    for v in values:
                        m = re.search(r'(\d+)$', v)
                        nums.append(int(m.group(1)) if m else 0)
                    width = max((len(str(n)) for n in nums), default=5)
                    start = max(nums) + 1 if nums else 1
                else:
                    nums = [int(m.group(2)) for m in matches]
                    width = max(len(m.group(2)) for m in matches)
                    start = max(nums) + 1
            else:
                nums, prefixes = [], []
                for v in values:
                    m = re.match(r'^(.*?)(\d+)$', v)
                    if m:
                        prefixes.append(m.group(1))
                        nums.append(int(m.group(2)))
                if nums and len(nums) >= len(values) * 0.7:
                    prefix = max(set(prefixes), key=prefixes.count) if prefixes else ''
                    width = max(len(str(n)) for n in nums)
                    start = max(nums) + 1
                else:
                    use_uuid = True
        return {
            'type': 'id', 'column': col_name, 'prefix': prefix,
            'width': width, 'start': start, 'use_uuid': use_uuid,
            'seed_values': set(values.tolist()),
        }

    @staticmethod
    def _common_prefix(strings: List[str]) -> str:
        if not strings:
            return ''
        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix) and prefix:
                prefix = prefix[:-1]
        m = re.match(r'^(.*?\D)', prefix)
        return m.group(1) if m else prefix

    def _build_categorical_model(self, series: pd.Series) -> dict:
        cleaned = (
            series.dropna()
            .map(lambda x: str(x).strip() if not isinstance(x, bool) else ('TRUE' if x else 'FALSE'))
        )
        cleaned = cleaned.replace({
            'True': 'TRUE', 'False': 'FALSE', 'true': 'TRUE', 'false': 'FALSE',
            '1': 'TRUE', '0': 'FALSE', 'yes': 'TRUE', 'no': 'FALSE',
        })
        value_counts = cleaned.value_counts(normalize=True)
        if len(value_counts) == 0:
            return {
                'type': 'categorical', 'values': ['Unknown'], 'probabilities': [1.0],
                'seed_values': {'Unknown'}, 'min_seed_len': 1,
            }
        seed_values = set(cleaned.tolist())
        lengths = [len(str(v)) for v in seed_values]
        return {
            'type': 'categorical',
            'values': value_counts.index.tolist(),
            'probabilities': value_counts.values.tolist(),
            'seed_values': seed_values,
            'min_seed_len': min(lengths) if lengths else 1,
        }

    def _calculate_correlations(self):
        numeric_cols = [
            col for col, typ in self.column_types.items()
            if typ == 'numeric' and col not in self.grouped_columns
        ]
        if len(numeric_cols) > 1:
            numeric_df = self.sample[numeric_cols].apply(pd.to_numeric, errors='coerce')
            self.correlations = numeric_df.corr()
        else:
            self.correlations = None

    def generate(
        self,
        n: int = None,
        preserve_correlations: bool = True,
        num_records: int = None,
    ) -> pd.DataFrame:
        if num_records is not None:
            n = num_records
        if n is None:
            raise GenerationError("Number of records must be provided")
        if n < 1:
            raise GenerationError(f"Number of records must be at least 1: {n}")

        df = pd.DataFrame(index=range(n))
        self._person_cache = []

        for group in self.linked_groups:
            row_idx = np.random.randint(0, len(self.sample), size=n)
            for col in group:
                df[col] = self.sample.iloc[row_idx][col].to_numpy()

        self._fill_person_and_email(df, n)

        for col in self.sample.columns:
            if col in df.columns:
                continue
            df[col] = self._generate_by_role(col, n)

        df = df[list(self.sample.columns)]

        if preserve_correlations and self.correlations is not None:
            df = self._apply_correlations(df)

        df = self._apply_constraints(df)
        df = self._cast_integer_columns(df)

        self.last_validation = self.validate_generation(df, self.sample)
        self._log_validation(self.last_validation)
        if self.last_validation.get('errors'):
            raise GenerationError('; '.join(self.last_validation['errors']))

        try:
            df.attrs['column_roles'] = dict(self.column_roles)
            df.attrs['generator_locale'] = self.locale
        except Exception:
            pass

        logger.info(f"Generated {len(df)} records")
        return df.reset_index(drop=True)

    def _fill_person_and_email(self, df: pd.DataFrame, n: int):
        person_cols = [c for c, r in self.column_roles.items() if r == 'person_name']
        has_split = self.first_name_col or self.last_name_col
        has_full = bool(person_cols) or self.person_name_col is not None
        has_email = self.email_col is not None
        if not (has_split or has_full or has_email):
            return

        firsts, lasts, fulls, emails = [], [], [], []
        for _ in range(n):
            first = self.fake.first_name()
            last = self.fake.last_name()
            self._person_cache.append((first, last))
            firsts.append(first)
            lasts.append(last)
            fulls.append(f"{first} {last}")
            domain = np.random.choice(self.email_domains)
            local = re.sub(r'[^a-z0-9]', '', f"{first}.{last}".lower())
            emails.append(f"{local}@{domain}")

        if self.first_name_col and self.first_name_col not in df.columns:
            df[self.first_name_col] = firsts
        if self.last_name_col and self.last_name_col not in df.columns:
            df[self.last_name_col] = lasts
        primary = self.person_name_col
        if primary and primary not in df.columns:
            df[primary] = fulls
        for col in person_cols:
            if col in df.columns:
                continue
            df[col] = [f"{self.fake.first_name()} {self.fake.last_name()}" for _ in range(n)]
        if self.email_col and self.email_col not in df.columns:
            df[self.email_col] = emails

    def _generate_by_role(self, col: str, n: int) -> np.ndarray:
        model = self.models[col]
        role = model.get('role') or self.column_roles.get(col, 'closed_categorical')

        if role == 'id' or model.get('type') == 'id':
            return self._generate_ids(model, n)
        if role == 'datetime' or model.get('type') == 'datetime':
            return self._generate_datetime(model, n)
        if role == 'numeric' or model.get('type') == 'numeric':
            return self._generate_numeric(model, n)
        if role in ('closed_categorical', 'product'):
            return self._generate_categorical(model, n)
        if role == 'city':
            return self._generate_cities(model, n)
        if role == 'state':
            return np.array([self.fake.state() for _ in range(n)], dtype=object)
        if role == 'country':
            if self.india_mode:
                return np.array(['India'] * n, dtype=object)
            return np.array([self.fake.country() for _ in range(n)], dtype=object)
        if role == 'address':
            return np.array(
                [self.fake.address().replace('\n', ', ') for _ in range(n)],
                dtype=object,
            )
        if role == 'phone':
            return self._generate_phones(n)
        if role == 'aadhaar':
            return self._generate_aadhaar(n)
        if role == 'pan':
            return self._generate_pan(n)
        if role == 'ifsc':
            return self._generate_ifsc(n)
        if role == 'upi':
            return self._generate_upi(n)
        if role == 'pincode':
            return self._generate_pincode(model, n)
        if role == 'gstin':
            return self._generate_gstin(n)
        if role == 'company':
            return np.array([self.fake.company() for _ in range(n)], dtype=object)
        if role == 'job':
            return np.array([self.fake.job() for _ in range(n)], dtype=object)
        if role in ('person_name', 'first_name', 'last_name', 'email'):
            if role == 'email':
                return np.array([self.fake.email() for _ in range(n)], dtype=object)
            if role == 'first_name':
                return np.array([self.fake.first_name() for _ in range(n)], dtype=object)
            if role == 'last_name':
                return np.array([self.fake.last_name() for _ in range(n)], dtype=object)
            return np.array([self.fake.name() for _ in range(n)], dtype=object)
        if role == 'text_open':
            return self._generate_text_open(col, model, n)
        return self._generate_categorical(model, n)

    def _generate_text_open(self, col: str, model: dict, n: int) -> np.ndarray:
        if model.get('seed_values') or model.get('values'):
            return self._generate_categorical(model, n)
        name = str(col).strip().lower()
        if any(k in name for k in ('note', 'comment', 'description', 'remark', 'summary')):
            return np.array([self.fake.sentence(nb_words=6) for _ in range(n)], dtype=object)
        return np.array([self.fake.name() for _ in range(n)], dtype=object)

    def _generate_cities(self, model: dict, n: int) -> np.ndarray:
        seed_cities = list(model.get('seed_values') or [])
        out = []
        seed_prob = 0.3 if self.india_mode else 0.2
        for _ in range(n):
            if seed_cities and np.random.random() < seed_prob:
                out.append(np.random.choice(seed_cities))
            elif self.india_mode and np.random.random() < 0.75:
                out.append(np.random.choice(INDIAN_CITIES))
            else:
                out.append(self.fake.city())
        return np.array(out, dtype=object)

    def _generate_phones(self, n: int) -> np.ndarray:
        if self.india_phone or self.india_mode:
            out = []
            for _ in range(n):
                first = str(np.random.randint(6, 10))
                rest = ''.join(str(np.random.randint(0, 10)) for _ in range(9))
                out.append(f"+91{first}{rest}")
            return np.array(out, dtype=object)
        return np.array([self.fake.phone_number() for _ in range(n)], dtype=object)

    def _generate_aadhaar(self, n: int) -> np.ndarray:
        used: Set[str] = set()
        out = []
        while len(out) < n:
            value = str(np.random.randint(2, 10)) + ''.join(
                str(np.random.randint(0, 10)) for _ in range(11)
            )
            if value not in used:
                used.add(value)
                out.append(value)
        return np.array(out, dtype=object)

    def _generate_pan(self, n: int) -> np.ndarray:
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        used: Set[str] = set()
        out = []
        while len(out) < n:
            body = ''.join(np.random.choice(list(letters)) for _ in range(5))
            nums = ''.join(str(np.random.randint(0, 10)) for _ in range(4))
            tail = np.random.choice(list(letters))
            value = f"{body}{nums}{tail}"
            if value not in used:
                used.add(value)
                out.append(value)
        return np.array(out, dtype=object)

    def _generate_ifsc(self, n: int) -> np.ndarray:
        banks = ['SBIN', 'HDFC', 'ICIC', 'AXIS', 'PUNB', 'UBIN', 'CNRB', 'KKBK']
        used: Set[str] = set()
        out = []
        while len(out) < n:
            bank = np.random.choice(banks)
            branch = ''.join(str(np.random.randint(0, 10)) for _ in range(6))
            value = f"{bank}0{branch}"
            if value not in used:
                used.add(value)
                out.append(value)
        return np.array(out, dtype=object)

    def _generate_upi(self, n: int) -> np.ndarray:
        out = []
        for i in range(n):
            if i < len(self._person_cache):
                first, last = self._person_cache[i]
                local = re.sub(r'[^a-z0-9]', '', f"{first}{last}".lower())[:12]
            else:
                local = re.sub(r'[^a-z0-9]', '', self.fake.user_name().lower())[:12]
            if not local:
                local = f"user{np.random.randint(1000, 9999)}"
            out.append(f"{local}@{np.random.choice(UPI_HANDLES)}")
        return np.array(out, dtype=object)

    def _generate_pincode(self, model: dict, n: int) -> np.ndarray:
        seed_vals = list(model.get('seed_values') or [])
        numeric_seed = []
        for v in seed_vals:
            digits = re.sub(r'\D', '', str(v))
            if len(digits) == 6:
                numeric_seed.append(digits)
        out = []
        for _ in range(n):
            if numeric_seed and np.random.random() < 0.3:
                out.append(np.random.choice(numeric_seed))
            else:
                first = str(np.random.randint(1, 9))
                rest = ''.join(str(np.random.randint(0, 10)) for _ in range(5))
                out.append(first + rest)
        return np.array(out, dtype=object)

    def _generate_gstin(self, n: int) -> np.ndarray:
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        used: Set[str] = set()
        out = []
        while len(out) < n:
            state = f"{np.random.randint(1, 38):02d}"
            pan_body = ''.join(np.random.choice(list(letters)) for _ in range(5))
            pan_nums = ''.join(str(np.random.randint(0, 10)) for _ in range(4))
            pan_tail = np.random.choice(list(letters))
            entity = str(np.random.randint(1, 10))
            check = np.random.choice(list(letters + '0123456789'))
            value = f"{state}{pan_body}{pan_nums}{pan_tail}{entity}Z{check}"
            if value not in used:
                used.add(value)
                out.append(value)
        return np.array(out, dtype=object)

    def _generate_numeric(self, model: dict, n: int) -> np.ndarray:
        if model.get('method') == 'kde' and 'kde' in model:
            try:
                samples = model['kde'].sample(n).flatten()
                samples = np.clip(
                    samples,
                    model['min'] - 2 * model['std'],
                    model['max'] + 2 * model['std'],
                )
                return self._finalize_numeric_samples(samples, model)
            except Exception:
                pass
        samples = np.random.normal(model['mean'], model['std'], n)
        return self._finalize_numeric_samples(samples, model)

    def _generate_datetime(self, model: dict, n: int) -> np.ndarray:
        min_d = pd.Timestamp(model['min'])
        max_d = pd.Timestamp(model['max'])
        if min_d.tzinfo is not None:
            min_d = min_d.tz_localize(None)
        if max_d.tzinfo is not None:
            max_d = max_d.tz_localize(None)
        if min_d > max_d:
            min_d, max_d = max_d, min_d
        date_only = bool(model.get('date_only', True))
        span_days = max((max_d.normalize() - min_d.normalize()).days, 0)
        span_seconds = int(model.get('span_seconds') or max((max_d - min_d).total_seconds(), 0))
        unique_dates = model.get('unique_dates') or []

        # Timestamp fields (Entry_Time / Exit_Time): sample uniformly in seconds
        if not date_only and span_seconds > 0:
            offsets = np.random.randint(0, span_seconds + 1, size=n)
            result = pd.to_datetime(
                [min_d + pd.Timedelta(seconds=int(o)) for o in offsets]
            )
            return result.to_numpy()

        if span_days == 0 and unique_dates:
            result = pd.to_datetime(np.random.choice(unique_dates, size=n))
        elif span_days == 0:
            result = pd.to_datetime([min_d] * n)
        else:
            offsets = np.random.randint(0, span_days + 1, size=n)
            base = min_d.normalize() if date_only else min_d
            if date_only:
                result = pd.to_datetime(
                    [base + pd.Timedelta(days=int(o)) for o in offsets]
                )
            else:
                # keep time-of-day flavor from min when spanning days
                result = pd.to_datetime(
                    [min_d + pd.Timedelta(days=int(o)) for o in offsets]
                )
        return result.to_numpy() if hasattr(result, 'to_numpy') else np.asarray(result)

    def _generate_ids(self, model: dict, n: int) -> np.ndarray:
        used: Set[str] = set(model.get('seed_values') or set())
        out = []
        counter = int(model.get('start', 1))
        prefix = model.get('prefix', '')
        width = int(model.get('width', 5))
        use_uuid = bool(model.get('use_uuid', False))
        for _ in range(n):
            if use_uuid:
                value = str(uuid.uuid4())
                while value in used:
                    value = str(uuid.uuid4())
            else:
                value = f"{prefix}{str(counter).zfill(width)}"
                while value in used:
                    counter += 1
                    value = f"{prefix}{str(counter).zfill(width)}"
                counter += 1
            used.add(value)
            out.append(value)
        return np.array(out, dtype=object)

    def _generate_categorical(self, model: dict, n: int) -> np.ndarray:
        values = model['values']
        probs = model.get('probabilities')
        if len(values) == 1:
            return np.array([values[0]] * n, dtype=object)
        return np.random.choice(values, n, p=probs)

    def _finalize_numeric_samples(self, samples: np.ndarray, model: dict) -> np.ndarray:
        samples = np.nan_to_num(samples, nan=model.get('mean', 0.0))
        vmin = model.get('min', np.min(samples))
        vmax = model.get('max', np.max(samples))
        samples = np.clip(samples, vmin, vmax)
        if model.get('as_integer', False):
            samples = np.rint(samples).astype(np.int64)
            if model.get('six_digit', False):
                samples = np.clip(samples, 100000, 999999)
            else:
                samples = np.clip(samples, int(round(vmin)), int(round(vmax)))
        return samples

    def _cast_integer_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, model in self.models.items():
            if col not in df.columns or model.get('type') != 'numeric':
                continue
            if model.get('as_integer', False):
                values = pd.to_numeric(df[col], errors='coerce').fillna(model.get('mean', 0)).values
                df[col] = self._finalize_numeric_samples(values, model)
        return df

    def _apply_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.correlations is None:
            return df
        numeric_cols = [
            col for col in self.correlations.columns
            if col in df.columns and col not in self.grouped_columns
        ]
        if len(numeric_cols) <= 1:
            return df
        try:
            gen_data = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
            scaler = StandardScaler()
            gen_scaled = scaler.fit_transform(gen_data)
            corr_matrix = self.correlations.loc[numeric_cols, numeric_cols].values
            L = np.linalg.cholesky(corr_matrix + np.eye(len(numeric_cols)) * 0.001)
            correlated = scaler.inverse_transform(gen_scaled @ L.T)
            for i, col in enumerate(numeric_cols):
                df[col] = correlated[:, i]
        except Exception:
            pass
        return df

    def _apply_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            model = self.models.get(col)
            if not model:
                continue
            if model.get('type') == 'numeric' and 'min' in model and 'max' in model:
                df[col] = np.clip(
                    pd.to_numeric(df[col], errors='coerce'), model['min'], model['max']
                )
            elif model.get('type') == 'datetime' and 'min' in model and 'max' in model:
                vals = pd.to_datetime(df[col], errors='coerce')
                df[col] = vals.clip(lower=model['min'], upper=model['max'])
        return df

    def validate_generation(self, df: pd.DataFrame, sample: pd.DataFrame) -> dict:
        errors: List[str] = []
        warnings_list: List[str] = []
        nunique = {col: int(df[col].nunique(dropna=True)) for col in df.columns}

        for col, role in self.column_roles.items():
            if role != 'id' or col not in df.columns:
                continue
            if df[col].nunique(dropna=True) != len(df):
                errors.append(f"ID collisions in '{col}'")

        for col, role in self.column_roles.items():
            if role != 'datetime' or col not in df.columns:
                continue
            vals = pd.to_datetime(df[col], errors='coerce').dropna()
            if len(vals) == 0:
                continue
            model = self.models.get(col, {})
            seed_nunique = int(
                model.get('seed_nunique')
                or len(model.get('unique_dates') or [])
                or 1
            )
            span = int(model.get('date_range_days') or 0)
            span_sec = int(model.get('span_seconds') or 0)
            gen_nunique = int(vals.nunique())
            has_range = span > 0 or span_sec > 60
            # Hard-fail only on total collapse (single timestamp) when the seed
            # actually had range/diversity. Low-but-nonzero diversity (e.g. 3
            # distinct Entry_Time values on a small toll span) is a warning.
            if gen_nunique <= 1 and (has_range or seed_nunique > 1):
                errors.append(
                    f"Date collapse in '{col}' (nunique={gen_nunique}, "
                    f"seed had {seed_nunique})"
                )
            elif has_range and gen_nunique < min(3, seed_nunique) and seed_nunique >= 3:
                warnings_list.append(
                    f"Low date diversity in '{col}': nunique={gen_nunique} "
                    f"(seed={seed_nunique})"
                )

        for col, role in self.column_roles.items():
            if role not in ('closed_categorical', 'product') or col not in df.columns:
                continue
            model = self.models.get(col, {})
            seed_set = set(model.get('seed_values') or [])
            if not seed_set:
                continue
            syn_set = set(df[col].dropna().astype(str).str.strip())
            invented = syn_set - {str(s) for s in seed_set}
            if invented:
                errors.append(f"Invented closed labels in '{col}': {list(invented)[:3]}")
            min_len = int(model.get('min_seed_len') or 1)
            too_short = df[col].dropna().astype(str).str.len() < max(1, min_len - 2)
            if too_short.mean() > 0.1:
                errors.append(f"Truncation smell in closed column '{col}'")

        if self.person_name_col and self.email_col:
            if self.person_name_col in df.columns and self.email_col in df.columns:
                hits = 0
                total = 0
                for name, email in zip(
                    df[self.person_name_col].astype(str),
                    df[self.email_col].astype(str),
                ):
                    if '@' not in email:
                        continue
                    total += 1
                    local = email.split('@', 1)[0].lower()
                    parts = re.sub(r'[^a-z0-9\s]', '', name.lower()).split()
                    if any(p in local for p in parts if len(p) >= 2):
                        hits += 1
                if total and hits / total < 0.85:
                    errors.append(
                        f"Name↔email coherence too low: {hits}/{total} "
                        f"({hits / total:.1%})"
                    )

        return {'errors': errors, 'warnings': warnings_list, 'nunique': nunique}

    def _log_validation(self, result: dict):
        nunique = result.get('nunique') or {}
        logger.info(
            "Generation nunique by field: "
            + ', '.join(f"{k}={v}" for k, v in nunique.items())
        )
        for w in result.get('warnings') or []:
            logger.warning(w)
        if result.get('errors'):
            for e in result['errors']:
                logger.error(e)
        else:
            logger.info("Generation validation passed")
