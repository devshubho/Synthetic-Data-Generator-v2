"""
Custom Exceptions for SynthSLM
"""

class SynthSLMError(Exception):
    """Base exception for SynthSLM"""
    pass

class DataValidationError(SynthSLMError):
    """Raised when data validation fails"""
    pass

class GenerationError(SynthSLMError):
    """Raised when data generation fails"""
    pass

class PrivacyError(SynthSLMError):
    """Raised when privacy protection fails"""
    pass

class ExportError(SynthSLMError):
    """Raised when export fails"""
    pass

class SampleError(SynthSLMError):
    """Raised when sample data is invalid"""
    pass

class ConfigError(SynthSLMError):
    """Raised when configuration is invalid"""
    pass

class DatabaseError(SynthSLMError):
    """Raised when database operation fails"""
    pass