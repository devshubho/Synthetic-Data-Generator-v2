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
        
        # Process sample
        detected_type = self._detect_data_type(sample)
        sample = self.processor.process_sample(sample, detected_type)
        
        # Create custom generator
        generator = CustomGenerator(sample)
        
        # Generate - FIX: Use positional argument 'n' not keyword 'num_records'
        df = generator.generate(
            n=num_records,  # <-- FIX: Use 'n' not 'num_records'
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