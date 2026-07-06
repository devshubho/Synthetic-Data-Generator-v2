"""
Pre-built Data Templates
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

class TemplateGenerator:
    """Generate data from pre-built templates"""
    
    def __init__(self):
        self.fake = Faker()
    
    def generate(self, data_type: str, num_records: int, random_seed: int = 42) -> pd.DataFrame:
        """Generate based on template type"""
        
        Faker.seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        generators = {
            "Personal/Customer Data": self._personal,
            "Sales Transactions": self._sales,
            "Employee Records": self._employee,
            "Time Series Data": self._timeseries,
            "Application Logs": self._logs,
            "System Metrics": self._system,
            "Correlated VM Data": self._vm,
            "IoT Sensor Data": self._iot,
            "Healthcare Records": self._healthcare,
            "Financial Transactions": self._financial
        }
        
        if data_type in generators:
            return generators[data_type](num_records)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _personal(self, n: int) -> pd.DataFrame:
        """Personal/Customer Data"""
        data = []
        for _ in range(n):
            first = self.fake.first_name()
            last = self.fake.last_name()
            data.append({
                'id': self.fake.uuid4(),
                'first_name': first,
                'last_name': last,
                'email': f"{first.lower()}.{last.lower()}@{self.fake.free_email_domain()}",
                'phone': self.fake.phone_number(),
                'address': self.fake.address().replace('\n', ', '),
                'city': self.fake.city(),
                'state': self.fake.state(),
                'zipcode': self.fake.zipcode(),
                'birth_date': self.fake.date_of_birth(minimum_age=18, maximum_age=80),
                'gender': random.choice(['Male', 'Female', 'Non-binary']),
                'occupation': self.fake.job(),
                'income': random.randint(30000, 200000),
                'education': random.choice(['High School', 'Bachelor', 'Master', 'PhD']),
                'active': random.choice([True, False]),
                'created_at': self.fake.date_time_between(start_date='-2y', end_date='now')
            })
        return pd.DataFrame(data)
    
    def _sales(self, n: int) -> pd.DataFrame:
        """Sales Transactions"""
        products = {
            'Laptop': (500, 2000), 'Smartphone': (300, 1000),
            'Headphones': (50, 300), 'Monitor': (200, 800),
            'Keyboard': (30, 150), 'Mouse': (20, 100),
            'Desk': (150, 500), 'Chair': (200, 600)
        }
        
        data = []
        for _ in range(n):
            product = random.choice(list(products.keys()))
            price = random.randint(products[product][0], products[product][1])
            qty = random.randint(1, 10)
            
            data.append({
                'transaction_id': self.fake.uuid4(),
                'customer_id': self.fake.uuid4(),
                'product': product,
                'category': random.choice(['Electronics', 'Accessories', 'Furniture']),
                'quantity': qty,
                'unit_price': price,
                'total': qty * price,
                'discount': random.choice([0, 5, 10, 15, 20]),
                'payment_method': random.choice(['Credit Card', 'Debit Card', 'PayPal']),
                'transaction_date': self.fake.date_time_between(start_date='-1y'),
                'shipping_address': self.fake.address().replace('\n', ', '),
                'region': random.choice(['North', 'South', 'East', 'West']),
                'rating': random.randint(1, 5),
                'returned': random.choice([True, False]) if random.random() < 0.05 else False
            })
        return pd.DataFrame(data)
    
    def _employee(self, n: int) -> pd.DataFrame:
        """Employee Records"""
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
        positions = ['Intern', 'Junior', 'Mid-Level', 'Senior', 'Lead', 'Manager', 'Director']
        
        data = []
        for _ in range(n):
            first = self.fake.first_name()
            last = self.fake.last_name()
            dept = random.choice(departments)
            
            data.append({
                'employee_id': f"EMP{random.randint(10000, 99999)}",
                'first_name': first,
                'last_name': last,
                'email': f"{first.lower()}.{last.lower()}@company.com",
                'department': dept,
                'position': random.choice(positions),
                'hire_date': self.fake.date_between(start_date='-10y'),
                'salary': random.randint(40000, 150000),
                'manager_id': f"EMP{random.randint(10000, 99999)}",
                'performance_rating': round(random.uniform(1, 5), 1),
                'experience': random.randint(0, 20),
                'education': random.choice(['Bachelor', 'Master', 'PhD', 'MBA']),
                'remote': random.choice([True, False]),
                'bonus_eligible': random.choice([True, False])
            })
        return pd.DataFrame(data)
    
    def _timeseries(self, n: int) -> pd.DataFrame:
        """Time Series Data"""
        start = datetime.now() - timedelta(days=n)
        dates = [start + timedelta(days=i) for i in range(n)]
        
        # Multiple patterns
        trend = np.linspace(0, 50, n)
        seasonality = 20 * np.sin(2 * np.pi * np.arange(n) / 30)
        weekly = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
        noise = np.random.normal(0, 5, n)
        
        values = 100 + trend + seasonality + weekly + noise
        
        return pd.DataFrame({
            'date': dates,
            'value': values,
            'moving_avg_7d': pd.Series(values).rolling(7, min_periods=1).mean(),
            'moving_avg_30d': pd.Series(values).rolling(30, min_periods=1).mean(),
            'volatility': np.abs(np.random.normal(0, 2, n)),
            'anomaly': np.random.choice([0, 1], size=n, p=[0.95, 0.05])
        })
    
    def _logs(self, n: int) -> pd.DataFrame:
        """Application Logs"""
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        services = ['auth', 'payment', 'user', 'order', 'notification']
        
        data = []
        for _ in range(n):
            data.append({
                'timestamp': self.fake.date_time_between(start_date='-7d'),
                'log_level': random.choice(levels),
                'service': random.choice(services),
                'message': f"{random.choice(services)}: {self.fake.sentence()}",
                'source_ip': self.fake.ipv4() if random.random() < 0.5 else None,
                'user_id': self.fake.uuid4() if random.random() < 0.3 else None,
                'session_id': self.fake.uuid4()[:8] if random.random() < 0.3 else None,
                'response_time_ms': random.randint(10, 5000),
                'status_code': random.choice([200, 201, 400, 401, 403, 404, 500]),
                'endpoint': f"/api/v1/{random.choice(['users', 'orders', 'products'])}"
            })
        return pd.DataFrame(data)
    
    def _system(self, n: int) -> pd.DataFrame:
        """System Metrics"""
        hosts = [f'host-{i:02d}' for i in range(1, 11)]
        
        data = []
        for _ in range(n):
            data.append({
                'timestamp': self.fake.date_time_between(start_date='-7d'),
                'hostname': random.choice(hosts),
                'cpu_usage': round(random.uniform(10, 90), 2),
                'memory_usage': round(random.uniform(20, 85), 2),
                'disk_usage': round(random.uniform(10, 95), 2),
                'network_in_mbps': round(random.uniform(0.1, 100), 2),
                'network_out_mbps': round(random.uniform(0.1, 80), 2),
                'load_avg_1min': round(random.uniform(0, 4), 2),
                'process_count': random.randint(50, 500),
                'uptime_hours': random.randint(1, 8760)
            })
        return pd.DataFrame(data)
    
    def _vm(self, n: int) -> pd.DataFrame:
        """Correlated VM Data"""
        hosts = ['web-01', 'web-02', 'api-01', 'api-02', 'db-01', 'db-02']
        
        data = []
        for i in range(n):
            ts = datetime.now() - timedelta(minutes=i*5)
            host = random.choice(hosts)
            
            # Correlated metrics
            has_issue = random.random() < 0.2
            cpu = random.uniform(10, 60) + (random.uniform(30, 60) if has_issue else 0)
            memory = random.uniform(30, 70) + (random.uniform(20, 40) if has_issue else 0)
            
            data.append({
                'timestamp': ts,
                'hostname': host,
                'cpu_usage': round(min(100, cpu), 2),
                'memory_usage': round(min(100, memory), 2),
                'disk_usage': round(random.uniform(20, 85), 2),
                'network_in_mbps': round(random.uniform(1, 50), 2),
                'network_out_mbps': round(random.uniform(1, 40), 2),
                'has_issue': has_issue
            })
        
        return pd.DataFrame(data)
    
    def _iot(self, n: int) -> pd.DataFrame:
        """IoT Sensor Data"""
        devices = [f'sensor-{i:04d}' for i in range(1, 51)]
        types = ['temperature', 'humidity', 'pressure', 'motion', 'light']
        
        data = []
        for _ in range(n):
            dtype = random.choice(types)
            value = {
                'temperature': 15 + random.random() * 20,
                'humidity': 20 + random.random() * 60,
                'pressure': 980 + random.random() * 50,
                'motion': random.randint(0, 100) if random.random() < 0.3 else 0,
                'light': random.random() * 1000
            }[dtype]
            
            data.append({
                'timestamp': self.fake.date_time_between(start_date='-7d'),
                'device_id': random.choice(devices),
                'device_type': dtype,
                'value': round(value, 2),
                'battery_level': random.randint(10, 100),
                'signal_strength': random.randint(1, 5),
                'status': random.choice(['active', 'idle', 'maintenance'])
            })
        return pd.DataFrame(data)
    
    def _healthcare(self, n: int) -> pd.DataFrame:
        """Healthcare Records"""
        conditions = ['Healthy', 'Diabetes', 'Hypertension', 'Asthma', 'Heart Disease', 'Arthritis']
        
        data = []
        for _ in range(n):
            first = self.fake.first_name()
            last = self.fake.last_name()
            age = random.randint(18, 90)
            
            data.append({
                'patient_id': f"P{random.randint(10000, 99999)}",
                'first_name': first,
                'last_name': last,
                'age': age,
                'gender': random.choice(['Male', 'Female']),
                'weight_kg': round(random.uniform(50, 120), 1),
                'height_cm': round(random.uniform(150, 200), 1),
                'bmi': round(random.uniform(18, 35), 1),
                'blood_pressure_sys': random.randint(100, 180),
                'blood_pressure_dia': random.randint(60, 120),
                'heart_rate': random.randint(60, 100),
                'temperature_c': round(random.uniform(36.0, 38.0), 1),
                'blood_sugar': random.randint(70, 200),
                'condition': random.choice(conditions),
                'diagnosis_date': self.fake.date_between(start_date='-5y'),
                'medication': random.choice(['None', 'Metformin', 'Lisinopril', 'Albuterol']),
                'followup': random.choice([True, False])
            })
        return pd.DataFrame(data)
    
    def _financial(self, n: int) -> pd.DataFrame:
        """Financial Transactions"""
        types = ['Deposit', 'Withdrawal', 'Transfer', 'Payment', 'Investment']
        currencies = ['USD', 'EUR', 'GBP', 'JPY']
        
        data = []
        for _ in range(n):
            data.append({
                'transaction_id': self.fake.uuid4(),
                'account_id': f"ACC{random.randint(10000, 99999)}",
                'type': random.choice(types),
                'amount': round(random.uniform(10, 10000), 2),
                'currency': random.choice(currencies),
                'timestamp': self.fake.date_time_between(start_date='-1y'),
                'status': random.choice(['Pending', 'Completed', 'Failed']),
                'description': self.fake.sentence(),
                'category': random.choice(['Food', 'Transport', 'Entertainment', 'Bills']),
                'location': self.fake.city()
            })
        return pd.DataFrame(data)