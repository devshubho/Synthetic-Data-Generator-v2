"""
Custom Generator - Learn from User Uploaded Data
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

warnings.filterwarnings("ignore")


class CustomGenerator:
    """Generate synthetic data from user-provided sample"""

    def __init__(self, sample_data: pd.DataFrame):
        self.sample = sample_data
        self.models = {}
        self._build_models()

    def _build_models(self):
        """Build generation models from sample"""

        for col in self.sample.columns:

            if pd.api.types.is_numeric_dtype(self.sample[col]):

                valid = self.sample[col].dropna().values.reshape(-1, 1)

                if len(valid) > 3:
                    try:
                        kde = KernelDensity(
                            kernel="gaussian",
                            bandwidth="scott"
                        )
                        kde.fit(valid)

                        self.models[col] = {
                            "type": "numeric",
                            "kde": kde,
                            "min": self.sample[col].min(),
                            "max": self.sample[col].max(),
                        }

                    except Exception:
                        self.models[col] = {
                            "type": "numeric",
                            "min": self.sample[col].min(),
                            "max": self.sample[col].max(),
                            "mean": self.sample[col].mean(),
                            "std": self.sample[col].std() or 1,
                        }

                else:
                    self.models[col] = {
                        "type": "numeric",
                        "min": self.sample[col].min(),
                        "max": self.sample[col].max(),
                        "mean": self.sample[col].mean(),
                        "std": self.sample[col].std() or 1,
                    }

            elif pd.api.types.is_datetime64_dtype(self.sample[col]):

                self.models[col] = {
                    "type": "datetime",
                    "min": self.sample[col].min(),
                    "max": self.sample[col].max(),
                }

            else:

                counts = self.sample[col].value_counts(normalize=True)

                self.models[col] = {
                    "type": "categorical",
                    "values": list(counts.index),
                    "probs": list(counts.values),
                }

    def generate(
        self,
        num_records: int,
        preserve_correlations: bool = True,
    ) -> pd.DataFrame:
        """Generate synthetic data"""

        data = {}

        for col, model in self.models.items():

            if model["type"] == "numeric":

                if "kde" in model:
                    samples = model["kde"].sample(num_records).flatten()
                else:
                    samples = np.random.normal(
                        model["mean"],
                        model["std"],
                        num_records,
                    )

                data[col] = np.clip(
                    samples,
                    model["min"],
                    model["max"],
                )

            elif model["type"] == "datetime":

                min_ts = model["min"].timestamp()
                max_ts = model["max"].timestamp()

                timestamps = np.random.uniform(
                    min_ts,
                    max_ts,
                    num_records,
                )

                data[col] = pd.to_datetime(
                    timestamps,
                    unit="s",
                )

            else:

                data[col] = np.random.choice(
                    model["values"],
                    num_records,
                    p=model["probs"],
                )

        df = pd.DataFrame(data)

        if preserve_correlations:
            df = self._apply_correlations(df)

        return df

    def _apply_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply correlations from original data"""

        numeric_cols = self.sample.select_dtypes(
            include=[np.number]
        ).columns

        numeric_cols = [
            c for c in numeric_cols if c in df.columns
        ]

        if len(numeric_cols) > 1:
            try:
                corr_matrix = self.sample[numeric_cols].corr().values

                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j:
                            corr = corr_matrix[i, j]

                            if abs(corr) > 0.3:
                                df[col2] = (
                                    df[col2]
                                    + corr * (df[col1] - df[col1].mean())
                                )

            except Exception:
                pass

        return df