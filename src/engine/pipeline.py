"""
Generation Pipeline - Complete Workflow
"""

import pandas as pd
from generators.factory import GeneratorFactory
from generators.custom import CustomGenerator
from privacy.anonymizer import Anonymizer
from engine.data_processor import DataProcessor
from engine.validator import DataValidator
from logger import setup_logger

logger = setup_logger()


class GenerationPipeline:
    """Complete generation workflow"""

    def __init__(self):
        self.processor = DataProcessor()
        self.validator = DataValidator()
        self.factory = GeneratorFactory()
        self.anonymizer = Anonymizer()

    def generate_template(
        self,
        data_type: str,
        num_records: int,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """Generate data from a pre-built template"""

        logger.info(f"Generating {num_records} records of {data_type}")

        generator = self.factory.get_generator(data_type)

        # ✅ FIX: Pass BOTH data_type and num_records as positional arguments
        df = generator.generate(data_type, num_records, random_seed)

        self.validator.validate(df)
        df = self.processor.clean(df)

        logger.info(f"Generated {len(df)} records")
        return df

    def generate_from_sample(
        self,
        sample: pd.DataFrame,
        num_records: int,
        preserve_correlations: bool = True,
        enable_privacy: bool = True,
    ) -> pd.DataFrame:
        """Generate synthetic data from uploaded sample"""

        logger.info(
            f"Generating {num_records} records from sample ({len(sample)} rows)"
        )

        # Validate sample has enough data
        if len(sample) < 2:
            raise ValueError("Sample must have at least 2 rows. Please upload more data.")

        self.validator.validate_sample(sample)
        sample = self.processor.process_sample(sample)

        generator = CustomGenerator(sample)

        df = generator.generate(
            n=num_records,
            preserve_correlations=preserve_correlations,
        )
        # Full structural validation already ran inside generate()

        # Synthetic Faker output is not real PII — skip anonymization so
        # name↔email coherence and authentic values survive to the UI.
        if enable_privacy:
            logger.info(
                "Privacy anonymization skipped for custom synthetic data "
                "(generated values are synthetic, not uploaded PII)"
            )

        # Attach generator metadata for quality scoring (attrs survive most exports)
        try:
            df.attrs['column_roles'] = dict(generator.column_roles)
            df.attrs['sample_columns'] = list(sample.columns)
            df.attrs['generator_locale'] = getattr(generator, 'locale', 'en_IN')
        except Exception:
            pass

        self.validator.validate(df)
        df = self.processor.clean(df)
        df = self.processor.preserve_numeric_style(df, sample)

        # Re-attach attrs after processor (some ops may drop them)
        try:
            df.attrs['column_roles'] = dict(generator.column_roles)
            df.attrs['generator_locale'] = getattr(generator, 'locale', 'en_IN')
        except Exception:
            pass

        nunique = {col: int(df[col].nunique(dropna=True)) for col in df.columns}
        logger.info(
            "Final nunique by field: "
            + ', '.join(f"{k}={v}" for k, v in nunique.items())
        )
        for col, typ in generator.column_types.items():
            if typ == 'id' and col in df.columns:
                if df[col].nunique(dropna=True) != len(df):
                    raise ValueError(f"Duplicate IDs in '{col}' after post-processing")
            if typ == 'datetime' and col in df.columns:
                vals = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(vals):
                    logger.info(
                        f"Datetime '{col}': nunique={vals.nunique()}, "
                        f"min={vals.min()}, max={vals.max()}"
                    )

        logger.info(f"Generated {len(df)} records from sample")
        return df