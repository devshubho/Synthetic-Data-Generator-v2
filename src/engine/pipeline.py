"""
Generation Pipeline - Complete Workflow
"""

import pandas as pd
from src.generators.factory import GeneratorFactory
from src.generators.custom import CustomGenerator
from src.privacy.anonymizer import Anonymizer
from src.engine.data_processor import DataProcessor
from src.engine.validator import DataValidator
from src.logger import setup_logger

logger = setup_logger()

class GenerationPipeline:
    """Complete generation workflow"""
    
    def __init__(self):
        self.processor = DataProcessor()
        self.validator = DataValidator()
        self.factory = GeneratorFactory()
        self.anonymizer = Anonymizer()
    
    def generate_template(self, data_type: str, num_records: int, 
                         random_seed: int = 42) -> pd.DataFrame:
        """Generate from template"""
        
        logger.info(f"Generating {num_records} records of {data_type}")
        
        generator = self.factory.get_generator(data_type)
        df = generator.generate(num_records, random_seed=random_seed)
        
        self.validator.validate(df)
        df = self.processor.clean(df)
        
        logger.info(f"Generated {len(df)} records")
        return df
    
    def generate_from_sample(self, sample: pd.DataFrame, num_records: int,
                            preserve_correlations: bool = True,
                            enable_privacy: bool = True) -> pd.DataFrame:
        """Generate from user sample"""
        
        logger.info(f"Generating {num_records} records from sample ({len(sample)} rows)")
        
        self.validator.validate_sample(sample)
        sample = self.processor.process_sample(sample)
        
        generator = CustomGenerator(sample)
        df = generator.generate(num_records, preserve_correlations=preserve_correlations)
        
        if enable_privacy:
            df = self.anonymizer.apply_privacy(df)
        
        self.validator.validate(df)
        df = self.processor.clean(df)
        
        logger.info(f"Generated {len(df)} records from sample")
        return df