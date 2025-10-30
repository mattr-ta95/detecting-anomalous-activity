# Dataset Information

## Original Dataset

The original dataset is from:
- **Author:** Devabrat Mohakul (2022)
- **Title:** "Predictive Maintenance on Ship's Main Engine using AI"
- **Source:** IEEE DataPort
- **DOI:** 10.21227/g3za-v415
- **URL:** https://ieee-dataport.org/open-access/predictive-maintenance-ships-main-engine-using-ai
- **Access:** Open Access (requires IEEE DataPort login)

## Dataset Details

The dataset contains 6 parameters for monitoring ship engine condition:

1. **Engine RPM** - Revolutions per minute
2. **Lubrication oil pressure** - Bar
3. **Fuel pressure** - Bar
4. **Coolant pressure** - Bar
5. **Lubrication oil temperature** - Celsius
6. **Coolant temperature** - Celsius

**Original Size:** ~11,934 samples

## Getting the Dataset

### Option 1: Generate Sample Data (Recommended for Testing)

Run the provided data generator:

```bash
python generate_sample_data.py
```

This will create a synthetic dataset with the same structure and characteristics as the original, including:
- Normal operating conditions
- ~3% anomalies
- Realistic value ranges
- Same column names

### Option 2: Download from IEEE DataPort

1. Visit: https://ieee-dataport.org/open-access/predictive-maintenance-ships-main-engine-using-ai
2. Create an account or login to IEEE DataPort
3. Download the dataset
4. Place it at: `data/raw/engine.csv`

### Option 3: Use Your Own Data

If you have ship engine sensor data, format it as a CSV with these columns:
- Engine RPM
- Lubrication oil pressure
- Fuel pressure
- Coolant pressure
- Lubrication oil temperature
- Coolant temperature

Place your CSV file at: `data/raw/engine.csv`

## Data Format

The CSV file should have:
- Header row with column names (exactly as shown above)
- Numeric values only
- No missing values (or they will be handled by preprocessing)

Example:
```csv
Engine RPM,Lubrication oil pressure,Fuel pressure,Coolant pressure,Lubrication oil temperature,Coolant temperature
500.5,5.2,7.1,3.0,50.3,35.2
495.3,5.1,7.0,2.9,49.8,34.8
...
```

## Updating the Dataset URL

If you find an alternative public URL for the dataset, update `config.yaml`:

```yaml
data:
  url: "YOUR_NEW_URL_HERE"
  checksum: null  # Optional: Add SHA256 checksum
```
