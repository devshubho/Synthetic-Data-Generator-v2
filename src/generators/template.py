"""
Pre-built Data Templates - COMPLETE OPTIMIZED VERSION
Includes ALL data types with fast vectorized generation
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

class TemplateGenerator:
    """Generate data from pre-built templates - Optimized for speed"""
    
    def __init__(self):
        self.fake = Faker()
        # Pre-generate common values for speed
        self._cache_common_values()
    
    def _cache_common_values(self):
        """Cache commonly used random choices"""
        self.cached_products = ['Laptop', 'Smartphone', 'Headphones', 'Monitor', 'Keyboard', 'Mouse', 'Desk', 'Chair']
        self.cached_categories = ['Electronics', 'Accessories', 'Furniture']
        self.cached_payment = ['Credit Card', 'Debit Card', 'PayPal', 'UPI', 'Cash']
        self.cached_regions = ['North', 'South', 'East', 'West', 'Central']
        self.cached_departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations', 'R&D']
        self.cached_positions = ['Intern', 'Junior', 'Mid-Level', 'Senior', 'Lead', 'Manager', 'Director', 'VP']
        self.cached_education = ['High School', 'Bachelor', 'Master', 'PhD', 'MBA']
        self.cached_genders = ['Male', 'Female', 'Non-binary', 'Prefer not to say']
        self.cached_conditions = ['Healthy', 'Diabetes', 'Hypertension', 'Asthma', 'Heart Disease', 'Arthritis', 'Cancer']
        self.cached_medications = ['None', 'Metformin', 'Lisinopril', 'Albuterol', 'Aspirin', 'Insulin']
        self.cached_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'INR']
        self.cached_payment_status = ['Success', 'Success', 'Success', 'Failed', 'Pending']
        self.cached_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        self.cached_services = ['auth', 'payment', 'user', 'order', 'notification', 'api-gateway', 'database']
        self.cached_status_codes = [200, 201, 202, 400, 401, 403, 404, 500, 502, 503]
        self.cached_hosts = [f'host-{i:02d}' for i in range(1, 21)]
        self.cached_vehicle_types = ['Car', 'Truck', 'Bus', 'SUV', 'Motorcycle', 'Auto']
        self.cached_states = ['WB', 'JH', 'BR', 'UP', 'DL', 'MH', 'KA', 'TN', 'AP', 'KL', 'GJ', 'RJ']
        self.cached_toll_plazas = ['NH-16 Kolkata Toll Plaza', 'NH-8 Delhi Toll Plaza', 'NH-44 Chennai Toll Plaza', 
                                   'NH-48 Mumbai Toll Plaza', 'NH-7 Bangalore Toll Plaza']
        self.cached_device_types = ['temperature', 'humidity', 'pressure', 'motion', 'light', 'vibration', 'gas']
        self.cached_sensor_status = ['active', 'active', 'active', 'idle', 'maintenance']
    
    def generate(self, data_type: str, num_records: int, random_seed: int = 42) -> pd.DataFrame:
        """Generate based on template type - FAST"""
        
        Faker.seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        generators = {
            "Personal/Customer Data": self._personal_fast,
            "Sales Transactions": self._sales_fast,
            "Employee Records": self._employee_fast,
            "Time Series Data": self._timeseries_fast,
            "Application Logs": self._logs_fast,
            "System Metrics": self._system_fast,
            "Correlated VM Data": self._vm_fast,
            "IoT Sensor Data": self._iot_fast,
            "Healthcare Records": self._healthcare_fast,
            "Financial Transactions": self._financial_fast,
            "Toll Plaza Data": self._toll_fast
        }
        
        if data_type in generators:
            return generators[data_type](num_records)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _toll_fast(self, n: int) -> pd.DataFrame:
        """Toll Plaza Data - FAST"""
        # Generate unique vehicle numbers using numpy
        state_codes = np.random.choice(self.cached_states, n)
        numbers = np.random.randint(10, 100, n)
        letters = np.random.choice(['AB','CD','EF','GH','JK','LM','NP','QR','ST','UV'], n)
        suffixes = np.random.randint(1000, 9999, n)
        
        vehicle_numbers = [f"{s}{num:02d}{l}{suf}" for s, num, l, suf in zip(state_codes, numbers, letters, suffixes)]
        
        # Assign consistent vehicle types
        vehicle_type_map = {v: np.random.choice(self.cached_vehicle_types) for v in set(vehicle_numbers)}
        vehicle_types_list = [vehicle_type_map[v] for v in vehicle_numbers]
        
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        return pd.DataFrame({
            'Transaction_ID': [f"TXN{datetime.now().strftime('%Y%m%d')}{str(i+1).zfill(4)}" for i in range(n)],
            'Toll_Plaza_ID': [f"TP{np.random.randint(1,50):03d}" for _ in range(n)],
            'Toll_Plaza_Name': np.random.choice(self.cached_toll_plazas, n),
            'Lane_Number': np.random.randint(1, 11, n),
            'Transaction_Date': date_str,
            'Transaction_Time': [self.fake.time() for _ in range(n)],
            'Vehicle_Number': vehicle_numbers,
            'Vehicle_Type': vehicle_types_list,
            'Vehicle_Class': np.random.choice(['Private', 'Commercial', 'Government'], n),
            'Vehicle_Brand': np.random.choice(['Hyundai', 'Tata', 'Maruti', 'Toyota', 'Honda', 'Ford', 'Mahindra'], n),
            'Vehicle_Model': np.random.choice(['Creta', 'Swift', 'Innova', 'XUV700', 'Scorpio', 'Fortuner', 'Aura'], n),
            'Vehicle_Color': np.random.choice(['White', 'Black', 'Silver', 'Red', 'Blue', 'Grey', 'Orange'], n),
            'Registration_State': np.random.choice(self.cached_states, n),
            'Owner_Name': [self.fake.name() for _ in range(n)],
            'Driver_Name': [self.fake.name() for _ in range(n)],
            'FASTag_ID': [f"FT{np.random.randint(100000000000, 999999999999)}" for _ in range(n)],
            'FASTag_Status': np.random.choice(['Active', 'Active', 'Active', 'Inactive', 'Suspended'], n),
            'Toll_Amount': np.round(np.random.uniform(20, 800, n), 2),
            'Payment_Mode': np.random.choice(['FASTag', 'UPI', 'Cash', 'Card', 'QR Code'], n),
            'Payment_Status': np.random.choice(['Success', 'Success', 'Success', 'Success', 'Failed'], n, p=[0.7, 0.1, 0.1, 0.05, 0.05]),
            'Entry_Direction': np.random.choice(['Northbound', 'Southbound', 'Eastbound', 'Westbound'], n),
            'Exit_Direction': np.random.choice(['Northbound', 'Southbound', 'Eastbound', 'Westbound'], n),
            'Speed_kmph': np.random.randint(5, 80, n),
            'Overloaded': np.random.choice(['No', 'No', 'No', 'Yes', 'Yes'], n, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
            'CCTV_Camera_ID': [f"CCTV-{np.random.randint(1,50):02d}" for _ in range(n)],
            'Booth_Operator_ID': [f"OPR{np.random.randint(100, 999)}" for _ in range(n)],
            'Remarks': np.random.choice(['Normal Transaction', 'Normal Transaction', 'Normal Transaction', 
                                        'Overload Alert', 'Speed Violation', 'FASTag Error', 'Manual Entry'], n,
                                        p=[0.5, 0.2, 0.1, 0.07, 0.05, 0.04, 0.04])
        })
    
    def _personal_fast(self, n: int) -> pd.DataFrame:
        """Personal/Customer Data - FAST"""
        first_names = [self.fake.first_name() for _ in range(n)]
        last_names = [self.fake.last_name() for _ in range(n)]
        
        return pd.DataFrame({
            'id': [self.fake.uuid4() for _ in range(n)],
            'first_name': first_names,
            'last_name': last_names,
            'email': [f"{f.lower()}.{l.lower()}@{self.fake.free_email_domain()}" for f, l in zip(first_names, last_names)],
            'phone': [self.fake.phone_number() for _ in range(n)],
            'address': [self.fake.address().replace('\n', ', ') for _ in range(n)],
            'city': [self.fake.city() for _ in range(n)],
            'state': [self.fake.state() for _ in range(n)],
            'zipcode': [self.fake.zipcode() for _ in range(n)],
            'birth_date': [self.fake.date_of_birth(minimum_age=18, maximum_age=80) for _ in range(n)],
            'age': np.random.randint(18, 80, n),
            'gender': np.random.choice(self.cached_genders, n),
            'occupation': [self.fake.job() for _ in range(n)],
            'income': np.random.randint(20000, 250000, n),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n),
            'dependents': np.random.randint(0, 5, n),
            'education': np.random.choice(self.cached_education, n),
            'active': np.random.choice([True, False], n, p=[0.85, 0.15]),
            'created_at': [self.fake.date_time_between(start_date='-2y', end_date='now') for _ in range(n)]
        })
    
    def _sales_fast(self, n: int) -> pd.DataFrame:
        """Sales Transactions - FAST"""
        products = np.random.choice(self.cached_products, n)
        prices = []
        for p in products:
            if p in ['Laptop', 'Smartphone']:
                prices.append(np.random.randint(500, 2000))
            elif p in ['Monitor', 'Desk', 'Chair']:
                prices.append(np.random.randint(200, 800))
            else:
                prices.append(np.random.randint(20, 300))
        
        quantities = np.random.randint(1, 11, n)
        
        return pd.DataFrame({
            'transaction_id': [self.fake.uuid4() for _ in range(n)],
            'customer_id': [self.fake.uuid4() for _ in range(n)],
            'product': products,
            'category': np.random.choice(self.cached_categories, n),
            'quantity': quantities,
            'unit_price': prices,
            'total': np.array(prices) * quantities,
            'discount': np.random.choice([0, 5, 10, 15, 20, 25, 30], n),
            'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'UPI', 'Net Banking'], n),
            'transaction_date': [self.fake.date_time_between(start_date='-1y') for _ in range(n)],
            'shipping_address': [self.fake.address().replace('\n', ', ') for _ in range(n)],
            'shipping_method': np.random.choice(['Standard', 'Express', 'Overnight', 'Same Day'], n),
            'region': np.random.choice(self.cached_regions, n),
            'rating': np.random.randint(1, 6, n),
            'customer_rating': np.round(np.random.uniform(1, 5, n), 1),
            'returned': np.random.choice([True, False], n, p=[0.05, 0.95]),
            'refund_status': np.random.choice(['None', 'None', 'None', 'Pending', 'Completed'], n, p=[0.7, 0.15, 0.07, 0.04, 0.04])
        })
    
    def _employee_fast(self, n: int) -> pd.DataFrame:
        """Employee Records - FAST"""
        first_names = [self.fake.first_name() for _ in range(n)]
        last_names = [self.fake.last_name() for _ in range(n)]
        departments = np.random.choice(self.cached_departments, n)
        
        return pd.DataFrame({
            'employee_id': [f"EMP{np.random.randint(10000, 99999)}" for _ in range(n)],
            'first_name': first_names,
            'last_name': last_names,
            'email': [f"{f.lower()}.{l.lower()}@company.com" for f, l in zip(first_names, last_names)],
            'department': departments,
            'position': [f"{np.random.choice(self.cached_positions)} {d}" for d in departments],
            'hire_date': [self.fake.date_between(start_date='-10y') for _ in range(n)],
            'salary': np.random.randint(30000, 200000, n),
            'manager_id': [f"EMP{np.random.randint(10000, 99999)}" for _ in range(n)],
            'performance_rating': np.round(np.random.uniform(1, 5, n), 1),
            'experience_years': np.random.randint(0, 25, n),
            'education': np.random.choice(self.cached_education, n),
            'remote_work': np.random.choice([True, False], n, p=[0.4, 0.6]),
            'bonus_eligible': np.random.choice([True, False], n, p=[0.6, 0.4]),
            'travel_percentage': np.random.randint(0, 100, n),
            'projects_completed': np.random.randint(0, 30, n),
            'certifications': np.random.randint(0, 8, n),
            'termination_date': [self.fake.date_between(start_date='-2y', end_date='now') if np.random.random() < 0.1 else None for _ in range(n)]
        })
    
    def _timeseries_fast(self, n: int) -> pd.DataFrame:
        """Time Series Data - FAST"""
        start = datetime.now() - timedelta(days=n)
        dates = [start + timedelta(days=i) for i in range(n)]
        
        # Vectorized operations with multiple patterns
        trend = np.linspace(0, 50, n)
        seasonality_monthly = 20 * np.sin(2 * np.pi * np.arange(n) / 30)
        seasonality_weekly = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
        seasonality_yearly = 15 * np.sin(2 * np.pi * np.arange(n) / 365)
        noise = np.random.normal(0, 5, n)
        
        values = 100 + trend + seasonality_monthly + seasonality_weekly + seasonality_yearly + noise
        
        return pd.DataFrame({
            'date': dates,
            'value': values,
            'moving_avg_7d': pd.Series(values).rolling(7, min_periods=1).mean(),
            'moving_avg_30d': pd.Series(values).rolling(30, min_periods=1).mean(),
            'volatility': np.abs(np.random.normal(0, 2, n)),
            'anomaly': np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
            'series_2': 50 + 0.5 * values + np.random.normal(0, 10, n),
            'series_3': np.random.normal(0, 1, n).cumsum()  # Random walk
        })
    
    def _logs_fast(self, n: int) -> pd.DataFrame:
        """Application Logs - FAST"""
        return pd.DataFrame({
            'timestamp': [self.fake.date_time_between(start_date='-7d') for _ in range(n)],
            'log_level': np.random.choice(self.cached_log_levels, n),
            'service': np.random.choice(self.cached_services, n),
            'message': [f"{np.random.choice(self.cached_services)}: {self.fake.sentence()}" for _ in range(n)],
            'source_ip': [self.fake.ipv4() if np.random.random() < 0.5 else None for _ in range(n)],
            'user_id': [self.fake.uuid4() if np.random.random() < 0.3 else None for _ in range(n)],
            'session_id': [self.fake.uuid4()[:8] if np.random.random() < 0.3 else None for _ in range(n)],
            'response_time_ms': np.random.randint(10, 5000, n),
            'status_code': np.random.choice(self.cached_status_codes, n),
            'endpoint': [f"/api/v1/{np.random.choice(['users', 'orders', 'products', 'payments', 'auth'])}" for _ in range(n)],
            'method': np.random.choice(['GET', 'POST', 'PUT', 'DELETE', 'PATCH'], n),
            'bytes_sent': np.random.randint(100, 1000000, n),
            'error_message': [self.fake.sentence() if np.random.random() < 0.1 else None for _ in range(n)]
        })
    
    def _system_fast(self, n: int) -> pd.DataFrame:
        """System Metrics - FAST"""
        return pd.DataFrame({
            'timestamp': [self.fake.date_time_between(start_date='-7d') for _ in range(n)],
            'hostname': np.random.choice(self.cached_hosts, n),
            'cpu_usage': np.round(np.random.uniform(5, 95, n), 2),
            'memory_usage': np.round(np.random.uniform(15, 90, n), 2),
            'disk_usage': np.round(np.random.uniform(10, 98, n), 2),
            'network_in_mbps': np.round(np.random.uniform(0.1, 200, n), 2),
            'network_out_mbps': np.round(np.random.uniform(0.1, 150, n), 2),
            'load_avg_1min': np.round(np.random.uniform(0, 5, n), 2),
            'load_avg_5min': np.round(np.random.uniform(0, 4, n), 2),
            'load_avg_15min': np.round(np.random.uniform(0, 3, n), 2),
            'process_count': np.random.randint(30, 800, n),
            'uptime_hours': np.random.randint(1, 8760, n),
            'temperature_celsius': np.round(np.random.uniform(30, 75, n), 1),
            'power_consumption_watts': np.random.randint(50, 1000, n),
            'open_file_descriptors': np.random.randint(100, 65536, n)
        })
    
    def _vm_fast(self, n: int) -> pd.DataFrame:
        """Correlated VM Data - FAST"""
        hosts = ['web-01', 'web-02', 'web-03', 'api-01', 'api-02', 'db-01', 'db-02', 'cache-01', 'worker-01', 'worker-02']
        
        # Vectorized operations with correlations
        has_issue = np.random.random(n) < 0.15
        cpu_base = np.random.uniform(10, 60, n)
        cpu_spike = np.where(has_issue, np.random.uniform(30, 70, n), 0)
        memory_base = np.random.uniform(30, 70, n)
        memory_spike = np.where(has_issue, np.random.uniform(20, 50, n), 0)
        
        timestamps = [datetime.now() - timedelta(minutes=i*5) for i in range(n)]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'hostname': np.random.choice(hosts, n),
            'cpu_usage': np.round(np.minimum(100, cpu_base + cpu_spike), 2),
            'memory_usage': np.round(np.minimum(100, memory_base + memory_spike), 2),
            'disk_usage': np.round(np.random.uniform(20, 90, n), 2),
            'network_in_mbps': np.round(np.random.uniform(1, 100, n), 2),
            'network_out_mbps': np.round(np.random.uniform(1, 80, n), 2),
            'disk_read_ops': np.random.randint(0, 2000, n),
            'disk_write_ops': np.random.randint(0, 1000, n),
            'active_connections': np.random.randint(10, 500, n),
            'running_processes': np.random.randint(50, 400, n),
            'has_issue': has_issue,
            'issue_type': np.where(has_issue, np.random.choice(['High CPU', 'Memory Leak', 'Disk I/O', 'Network Spike'], n), 'None')
        })
    
    def _iot_fast(self, n: int) -> pd.DataFrame:
        """IoT Sensor Data - FAST"""
        devices = [f'sensor-{i:04d}' for i in range(1, 101)]
        types = self.cached_device_types
        
        dtypes = np.random.choice(types, n)
        
        # Vectorized value generation
        values = np.zeros(n)
        for i, dt in enumerate(dtypes):
            if dt == 'temperature':
                values[i] = 10 + np.random.random() * 30
            elif dt == 'humidity':
                values[i] = 20 + np.random.random() * 70
            elif dt == 'pressure':
                values[i] = 950 + np.random.random() * 80
            elif dt == 'motion':
                values[i] = np.random.randint(0, 100) if np.random.random() < 0.3 else 0
            elif dt == 'light':
                values[i] = np.random.random() * 1000
            elif dt == 'vibration':
                values[i] = np.random.random() * 2
            else:  # gas
                values[i] = np.random.random() * 100
        
        return pd.DataFrame({
            'timestamp': [self.fake.date_time_between(start_date='-30d') for _ in range(n)],
            'device_id': np.random.choice(devices, n),
            'device_type': dtypes,
            'value': np.round(values, 2),
            'unit': np.where(dtypes == 'temperature', '°C', 
                           np.where(dtypes == 'humidity', '%',
                           np.where(dtypes == 'pressure', 'hPa',
                           np.where(dtypes == 'light', 'lux', 'mV')))).astype(str),
            'battery_level': np.random.randint(5, 100, n),
            'signal_strength': np.random.randint(1, 6, n),
            'location_lat': np.round(np.random.uniform(-90, 90, n), 4),
            'location_lon': np.round(np.random.uniform(-180, 180, n), 4),
            'status': np.random.choice(self.cached_sensor_status, n),
            'firmware_version': [f"v{np.random.randint(1,3)}.{np.random.randint(0,9)}.{np.random.randint(0,20)}" for _ in range(n)]
        })
    
    def _healthcare_fast(self, n: int) -> pd.DataFrame:
        """Healthcare Records - FAST"""
        first_names = [self.fake.first_name() for _ in range(n)]
        last_names = [self.fake.last_name() for _ in range(n)]
        
        return pd.DataFrame({
            'patient_id': [f"P{np.random.randint(10000, 99999)}" for _ in range(n)],
            'first_name': first_names,
            'last_name': last_names,
            'age': np.random.randint(18, 90, n),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n),
            'weight_kg': np.round(np.random.uniform(40, 150, n), 1),
            'height_cm': np.round(np.random.uniform(140, 210, n), 1),
            'bmi': np.round(np.random.uniform(16, 40, n), 1),
            'blood_pressure_sys': np.random.randint(90, 200, n),
            'blood_pressure_dia': np.random.randint(50, 130, n),
            'heart_rate': np.random.randint(50, 120, n),
            'temperature_c': np.round(np.random.uniform(35.5, 39.0, n), 1),
            'blood_sugar': np.random.randint(60, 250, n),
            'cholesterol': np.random.randint(120, 350, n),
            'condition': np.random.choice(self.cached_conditions, n),
            'diagnosis_date': [self.fake.date_between(start_date='-5y') for _ in range(n)],
            'medication': np.random.choice(self.cached_medications, n),
            'allergies': np.random.choice(['None', 'Pollen', 'Peanuts', 'Dairy', 'Latex', 'None', 'None'], n),
            'emergency_contact': [self.fake.phone_number() for _ in range(n)],
            'insurance_provider': np.random.choice(['Aetna', 'BlueCross', 'Cigna', 'Medicare', 'United', 'None'], n),
            'followup_required': np.random.choice([True, False], n, p=[0.3, 0.7])
        })
    
    def _financial_fast(self, n: int) -> pd.DataFrame:
        """Financial Transactions - FAST"""
        types = ['Deposit', 'Withdrawal', 'Transfer', 'Payment', 'Investment', 'Loan']
        categories = ['Food', 'Transport', 'Entertainment', 'Bills', 'Shopping', 'Healthcare', 'Education', 'Investment']
        
        return pd.DataFrame({
            'transaction_id': [self.fake.uuid4() for _ in range(n)],
            'account_id': [f"ACC{np.random.randint(10000, 99999)}" for _ in range(n)],
            'type': np.random.choice(types, n),
            'amount': np.round(np.random.uniform(10, 100000, n), 2),
            'currency': np.random.choice(self.cached_currencies, n),
            'exchange_rate': np.round(np.random.uniform(0.5, 100, n), 2),
            'timestamp': [self.fake.date_time_between(start_date='-1y') for _ in range(n)],
            'status': np.random.choice(['Pending', 'Completed', 'Completed', 'Completed', 'Failed'], n, p=[0.05, 0.5, 0.25, 0.1, 0.1]),
            'description': [self.fake.sentence() for _ in range(n)],
            'category': np.random.choice(categories, n),
            'location': [self.fake.city() for _ in range(n)],
            'payment_channel': np.random.choice(['Online', 'Mobile', 'ATM', 'Branch', 'UPI'], n),
            'reference_number': [f"REF{np.random.randint(100000, 999999)}" for _ in range(n)]
        })