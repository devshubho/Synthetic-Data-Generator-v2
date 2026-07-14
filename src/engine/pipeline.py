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
    """Complete generation workflow with intelligent deduplication"""
    
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
    
    def generate_template(
        self,
        data_type: str,
        num_records: int,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """Generate data from a pre-built template"""
        
        try:
            logger.info(f"Generating {num_records} records of {data_type}")
            
            if not data_type:
                raise GenerationError("Data type is required")
            
            if num_records < 1:
                raise GenerationError(f"Number of records must be at least 1: {num_records}")
            
            if num_records > 100000:
                raise GenerationError(f"Number of records exceeds limit: {num_records}")
            
            generator = self.factory.get_generator(data_type)
            df = generator.generate(data_type, num_records, random_seed)
            
            self.validator.validate(df)
            
            # Clean with auto-detected data type
            detected_type = self._detect_data_type(df)
            df = self.processor.clean(df, detected_type)
            
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
        """Generate synthetic data from uploaded sample"""
        
        try:
            logger.info(f"Generating {num_records} records from sample ({len(sample)} rows)")
            
            if sample is None:
                raise SampleError("Sample data is None")
            
            if len(sample) < 2:
                raise SampleError(
                    f"Sample must have at least 2 rows. Current: {len(sample)}. "
                    f"Please upload more data."
                )
            
            if num_records < 1:
                raise GenerationError(f"Number of records must be at least 1: {num_records}")
            
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
            
            logger.info(f"Successfully generated {len(df)} records from sample")
            return df
            
        except Exception as e:
            log_error(e, "generate_from_sample")
            raise GenerationError(f"Sample-based generation failed: {str(e)}")