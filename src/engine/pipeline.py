"""
Generation Pipeline with Intelligent Deduplication
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
<<<<<<< HEAD
    """Complete generation workflow with intelligent deduplication"""
    
=======
    """Complete generation workflow with error handling"""

>>>>>>> 13afe48a2cd9ea54a2a46ac23ae25eac85a4de45
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
<<<<<<< HEAD
    
    def _detect_data_type(self, data: pd.DataFrame) -> str:
        """Auto-detect data type from columns"""
        columns = [c.lower() for c in data.columns]
        
        # Toll data
        if any(col in columns for col in ['vehicle_number', 'toll_plaza_id', 'fastag_id', 'lane_number']):
            return 'toll'
        
        # Personal data
        if any(col in columns for col in ['first_name', 'last_name', 'email', 'phone']):
            return 'personal'
        
        # Sales data
        if any(col in columns for col in ['transaction_id', 'customer_id', 'product', 'quantity']):
            return 'sales'
        
        # Employee data
        if any(col in columns for col in ['employee_id', 'department', 'salary']):
            return 'employee'
        
        # Healthcare data
        if any(col in columns for col in ['patient_id', 'condition', 'medication']):
            return 'healthcare'
        
        # IOT data
        if any(col in columns for col in ['device_id', 'sensor', 'device_type']):
            return 'iot'
        
        # Financial data
        if any(col in columns for col in ['account_id', 'transaction', 'amount', 'currency']):
            return 'financial'
        
        return 'unknown'
    
=======

>>>>>>> 13afe48a2cd9ea54a2a46ac23ae25eac85a4de45
    def generate_template(
        self,
        data_type: str,
        num_records: int,
        random_seed: int = 42,
    ) -> pd.DataFrame:
<<<<<<< HEAD
        """Generate data from a pre-built template"""
        
        try:
            logger.info(f"Generating {num_records} records of {data_type}")
            
=======
        """Generate data from a pre-built template with error handling"""

        try:
            logger.info(f"Generating {num_records} records of {data_type}")

>>>>>>> 13afe48a2cd9ea54a2a46ac23ae25eac85a4de45
            if not data_type:
                raise GenerationError("Data type is required")

            if num_records < 1:
                raise GenerationError(f"Number of records must be at least 1: {num_records}")

            if num_records > 100000:
                raise GenerationError(f"Number of records exceeds limit: {num_records}")
<<<<<<< HEAD
            
            generator = self.factory.get_generator(data_type)
            df = generator.generate(data_type, num_records, random_seed)
            
            self.validator.validate(df)
            
            # Clean with auto-detected data type
            detected_type = self._detect_data_type(df)
            df = self.processor.clean(df, detected_type)
            
=======

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
                df = self.processor.clean(df)
            except Exception as e:
                raise GenerationError(f"Data cleaning failed: {str(e)}")

>>>>>>> 13afe48a2cd9ea54a2a46ac23ae25eac85a4de45
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
<<<<<<< HEAD
        """Generate synthetic data from uploaded sample"""
        
        try:
            logger.info(f"Generating {num_records} records from sample ({len(sample)} rows)")
            
=======
        """Generate synthetic data from uploaded sample with error handling."""

        try:
            logger.info(
                f"Generating {num_records} records from sample ({len(sample)} rows)"
            )

>>>>>>> 13afe48a2cd9ea54a2a46ac23ae25eac85a4de45
            if sample is None:
                raise SampleError("Sample data is None")

            if len(sample) < 2:
                raise SampleError(
                    f"Sample must have at least 2 rows. Current: {len(sample)}. "
                    f"Please upload more data."
                )

            if num_records < 1:
                raise GenerationError(f"Number of records must be at least 1: {num_records}")
<<<<<<< HEAD
            
            self.validator.validate_sample(sample)
            
            # Process sample with auto-detected data type
            detected_type = self._detect_data_type(sample)
            sample = self.processor.process_sample(sample, detected_type)
            
            generator = CustomGenerator(sample)
            df = generator.generate(
                num_records=num_records,
                preserve_correlations=preserve_correlations,
            )
            
            if enable_privacy:
                try:
                    df = self.anonymizer.apply_privacy(df)
                except Exception as e:
                    logger.warning(f"Privacy application failed: {str(e)}")
            
            self.validator.validate(df)
            
            # Clean with auto-detected data type
            df = self.processor.clean(df, detected_type)
            
=======

            try:
                self.validator.validate_sample(sample)
            except Exception as e:
                raise SampleError(f"Sample validation failed: {str(e)}")

            try:
                sample = self.processor.process_sample(sample)
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
            # name↔email coherence and authentic values survive to the UI.
            if enable_privacy:
                logger.info(
                    "Privacy anonymization skipped for custom synthetic data "
                    "(generated values are synthetic, not uploaded PII)"
                )

            try:
                df.attrs['column_roles'] = dict(generator.column_roles)
                df.attrs['sample_columns'] = list(sample.columns)
                df.attrs['generator_locale'] = getattr(generator, 'locale', 'en_IN')
            except Exception:
                pass

            try:
                self.validator.validate(df)
            except Exception as e:
                raise GenerationError(f"Validation failed: {str(e)}")

            try:
                df = self.processor.clean(df)
                df = self.processor.preserve_numeric_style(df, sample)
            except Exception as e:
                raise GenerationError(f"Data cleaning failed: {str(e)}")

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
                        raise GenerationError(
                            f"Duplicate IDs in '{col}' after post-processing"
                        )
                if typ == 'datetime' and col in df.columns:
                    vals = pd.to_datetime(df[col], errors='coerce').dropna()
                    if len(vals):
                        logger.info(
                            f"Datetime '{col}': nunique={vals.nunique()}, "
                            f"min={vals.min()}, max={vals.max()}"
                        )

>>>>>>> 13afe48a2cd9ea54a2a46ac23ae25eac85a4de45
            logger.info(f"Successfully generated {len(df)} records from sample")
            return df

        except (GenerationError, SampleError):
            raise
        except Exception as e:
            log_error(e, "generate_from_sample")
            raise GenerationError(f"Sample-based generation failed: {str(e)}")
