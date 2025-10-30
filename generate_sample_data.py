"""Generate sample ship engine data for testing."""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 11934  # Original dataset size mentioned in research

# Generate normal operating data
engine_rpm = np.random.normal(500, 50, n_samples)
lub_oil_pressure = np.random.normal(5, 0.5, n_samples)
fuel_pressure = np.random.normal(7, 0.7, n_samples)
coolant_pressure = np.random.normal(3, 0.3, n_samples)
lub_oil_temp = np.random.normal(50, 5, n_samples)
coolant_temp = np.random.normal(35, 3, n_samples)

# Add some anomalies (3-5% of data)
n_anomalies = int(n_samples * 0.03)
anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)

# Create anomalies by adding extreme values
engine_rpm[anomaly_indices] += np.random.uniform(150, 300, n_anomalies)
lub_oil_pressure[anomaly_indices] += np.random.uniform(-2, -1, n_anomalies)
fuel_pressure[anomaly_indices] += np.random.uniform(3, 5, n_anomalies)
coolant_pressure[anomaly_indices] += np.random.uniform(-1, -0.5, n_anomalies)
lub_oil_temp[anomaly_indices] += np.random.uniform(20, 40, n_anomalies)
coolant_temp[anomaly_indices] += np.random.uniform(15, 25, n_anomalies)

# Create DataFrame with exact column names from original dataset
df = pd.DataFrame({
    'Engine RPM': engine_rpm,
    'Lubrication oil pressure': lub_oil_pressure,
    'Fuel pressure': fuel_pressure,
    'Coolant pressure': coolant_pressure,
    'Lubrication oil temperature': lub_oil_temp,
    'Coolant temperature': coolant_temp
})

# Save to CSV
output_file = 'data/raw/engine.csv'
df.to_csv(output_file, index=False)

print(f"Generated {n_samples} samples with ~{n_anomalies} anomalies")
print(f"Saved to: {output_file}")
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nStatistics:")
print(df.describe())
