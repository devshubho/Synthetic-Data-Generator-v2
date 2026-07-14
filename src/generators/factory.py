"""
Generator Factory - Select appropriate generator
"""

from generators.template import TemplateGenerator

class GeneratorFactory:
    """Factory to get appropriate generator"""
    
    def __init__(self):
        self.template_generator = TemplateGenerator()
    
    def get_generator(self, data_type: str):
        """Get generator for data type"""
        
        template_types = [
            "Personal/Customer Data",
            "Sales Transactions",
            "Employee Records",
            "Time Series Data",
            "Application Logs",
            "System Metrics",
            "Correlated VM Data",
            "IoT Sensor Data",
            "Healthcare Records",
            "Financial Transactions",
            "Toll Plaza Data"
        ]
        
        if data_type in template_types:
            return self.template_generator
        else:
            raise ValueError(f"Unknown data type: {data_type}")