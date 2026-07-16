"""
Generation Pipeline - Complete Workflow
"""

import pandas as pd
from generators.factory import GeneratorFactory
from generators.custom import CustomGenerator
from privacy.anonymizer import Anonymizer
from engine.data_processor import DataProcessor
from engine.validator import DataValidator
from logger import get_logger, log_error
from utils.exceptions import GenerationError, SampleError

logger = get_logger()


class GenerationPipeline:
    """Complete generation workflow with error handling"""

    def __init__(self):
        try:
            self.processor = DataProcessor()
            self.validator = DataValidator()
            self.factory = GeneratorFactory()
            self.anonymizer = Anonymizer()
            logger.info("GenerationPipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GenerationPipeline: {str(e)}")
            raise

    def _detect_data_type(self, sample: pd.DataFrame) -> str:
        cols = ' '.join(str(c).lower() for c in sample.columns)
        if any(k in cols for k in ('patient', 'diagnosis', 'doctor', 'admission')):
            return 'healthcare'
        if any(k in cols for k in ('order_id', 'product', 'courier', 'shipping')):
            return 'ecommerce'
        if any(k in cols for k in ('employee', 'salary', 'department', 'designation')):
            return 'hr'
        if any(k in cols for k in ('upi', 'aadhaar', 'ifsc', 'kyc', 'account')):
            return 'banking'
        if any(k in cols for k in ('claim', 'policy', 'premium', 'insured')):
            return 'insurance'
        if any(k in cols for k in ('msisdn', 'churn', 'telecom', 'subscriber')):
            return 'telecom'
        if any(k in cols for k in ('crop', 'yield', 'farmer', 'acre')):
            return 'agriculture'
        return 'unknown'

    def generate_template(
        self,
        data_type: str,
        num_records: int,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        try:
            logger.info(f"Generating {num_records} records of {data_type}")
            if not data_type:
                raise GenerationError("Data type is required")
            if num_records < 1:
                raise GenerationError(f"Number of records must be at least 1: {num_records}")
            if num_records > 100000:
                raise GenerationError(f"Number of records exceeds limit: {num_records}")

            try:
                generator = self.factory.get_generator(data_type)
            except ValueError:
                raise GenerationError(f"Unknown data type: {data_type}")

            try:
                df = generator.generate(data_type, num_records, random_seed)
            except Exception as e:
                raise GenerationError(f"Generation failed: {str(e)}")

            try:
                self.validator.validate(df)
            except Exception as e:
                raise GenerationError(f"Validation failed: {str(e)}")

            try:
                df = self.processor.clean(df, data_type)
            except Exception as e:
                raise GenerationError(f"Data cleaning failed: {str(e)}")

            logger.info(f"Successfully generated {len(df)} records")
            return df
        except Exception as e:
            log_error(e, "generate_template")
            raise GenerationError(f"Template generation failed: {str(e)}")

    def generate_from_sample(
        self,
        sample: pd.DataFrame,
        num_records: int,
        preserve_correlations: bool = True,
        enable_privacy: bool = True,
    ) -> pd.DataFrame:
        """Generate synthetic data from uploaded sample via Faker CustomGenerator."""
        try:
            logger.info(
                f"Generating {num_records} records from sample ({len(sample)} rows)"
            )
            if sample is None:
                raise SampleError("Sample data is None")
            if len(sample) < 2:
                raise SampleError(
                    f"Sample must have at least 2 rows. Current: {len(sample)}. "
                    f"Please upload more data."
                )
            if num_records < 1:
                raise GenerationError(f"Number of records must be at least 1: {num_records}")

            try:
                self.validator.validate_sample(sample)
            except Exception as e:
                raise SampleError(f"Sample validation failed: {str(e)}")

            detected_type = self._detect_data_type(sample)
            try:
                sample = self.processor.process_sample(sample, detected_type)
            except Exception as e:
                raise SampleError(f"Sample processing failed: {str(e)}")

            try:
                generator = CustomGenerator(sample)
            except Exception as e:
                raise GenerationError(f"Failed to create custom generator: {str(e)}")

            try:
                df = generator.generate(
                    n=num_records,
                    preserve_correlations=preserve_correlations,
                )
            except Exception as e:
                raise GenerationError(f"Generation failed: {str(e)}")

            # Synthetic Faker output is not real PII — skip anonymization so
            # name↔email coherence survives to the UI.
            if enable_privacy:
                logger.info(
                    "Privacy anonymization skipped for custom synthetic data "
                    "(generated values are synthetic, not uploaded PII)"
                )

            try:
                df.attrs['column_roles'] = dict(generator.column_roles)
                df.attrs['generator_locale'] = getattr(generator, 'locale', 'en_IN')
            except Exception:
                pass

            try:
                self.validator.validate(df)
            except Exception as e:
                raise GenerationError(f"Validation failed: {str(e)}")

            try:
                # Avoid aggressive dedup on freshly minted unique IDs/names
                df = df.copy()
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].astype(str).str.strip()
            except Exception as e:
                logger.warning(f"Post-process warning: {e}")

            try:
                df.attrs['column_roles'] = dict(generator.column_roles)
                df.attrs['generator_locale'] = getattr(generator, 'locale', 'en_IN')
            except Exception:
                pass

            for col, typ in generator.column_types.items():
                if typ == 'id' and col in df.columns:
                    if df[col].nunique(dropna=True) != len(df):
                        raise GenerationError(
                            f"Duplicate IDs in '{col}' after post-processing"
                        )

            logger.info(f"Successfully generated {len(df)} records from sample")
            return df

        except (GenerationError, SampleError):
            raise
        except Exception as e:
            log_error(e, "generate_from_sample")
            raise GenerationError(f"Sample-based generation failed: {str(e)}")
