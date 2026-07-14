"""
Generation Pipeline with Error Handling
"""

import pandas as pd
from src.generators.factory import GeneratorFactory
from src.generators.custom import CustomGenerator
from src.privacy.anonymizer import Anonymizer
from src.engine.data_processor import DataProcessor
from src.engine.validator import DataValidator
from src.logger import get_logger, log_error
from src.utils.exceptions import GenerationError, SampleError

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
    
    def generate_template(
        self,
        data_type: str,
        num_records: int,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """Generate data from a pre-built template with error handling"""
        
        try:
            logger.info(f"Generating {num_records} records of {data_type}")
            
            # Validate inputs
            if not data_type:
                raise GenerationError("Data type is required")
            
            if num_records < 1:
                raise GenerationError(f"Number of records must be at least 1: {num_records}")
            
            if num_records > 100000:
                raise GenerationError(f"Number of records exceeds limit: {num_records}")
            
            # Get generator
            try:
                generator = self.factory.get_generator(data_type)
            except ValueError as e:
                raise GenerationError(f"Unknown data type: {data_type}")
            
            # Generate data
            try:
                df = generator.generate(data_type, num_records, random_seed)
            except Exception as e:
                raise GenerationError(f"Generation failed: {str(e)}")
            
            # Validate
            try:
                self.validator.validate(df)
            except Exception as e:
                raise GenerationError(f"Validation failed: {str(e)}")
            
            # Clean
            try:
                df = self.processor.clean(df)
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
        """Generate synthetic data from uploaded sample with error handling"""
        
        try:
            logger.info(f"Generating {num_records} records from sample ({len(sample)} rows)")
            
            # Validate inputs
            if sample is None:
                raise SampleError("Sample data is None")
            
            if len(sample) < 2:
                raise SampleError(
                    f"Sample must have at least 2 rows. Current: {len(sample)}. "
                    f"Please upload more data."
                )
            
            if num_records < 1:
                raise GenerationError(f"Number of records must be at least 1: {num_records}")
            
            # Validate sample
            try:
                self.validator.validate_sample(sample)
            except Exception as e:
                raise SampleError(f"Sample validation failed: {str(e)}")
            
            # Process sample
            try:
                sample = self.processor.process_sample(sample)
            except Exception as e:
                raise SampleError(f"Sample processing failed: {str(e)}")
            
            # Create custom generator
            try:
                generator = CustomGenerator(sample)
            except Exception as e:
                raise GenerationError(f"Failed to create custom generator: {str(e)}")
            
            # Generate
            try:
                df = generator.generate(
                    num_records=num_records,
                    preserve_correlations=preserve_correlations,
                )
            except Exception as e:
                raise GenerationError(f"Generation failed: {str(e)}")
            
            # Apply privacy
            if enable_privacy:
                try:
                    df = self.anonymizer.apply_privacy(df)
                except Exception as e:
                    logger.warning(f"Privacy application failed: {str(e)}")
                    # Continue without privacy
            
            # Validate
            try:
                self.validator.validate(df)
            except Exception as e:
                raise GenerationError(f"Validation failed: {str(e)}")
            
            # Clean
            try:
                df = self.processor.clean(df)
            except Exception as e:
                raise GenerationError(f"Data cleaning failed: {str(e)}")
            
            logger.info(f"Successfully generated {len(df)} records from sample")
            return df
            
        except Exception as e:
            log_error(e, "generate_from_sample")
            raise GenerationError(f"Sample-based generation failed: {str(e)}")