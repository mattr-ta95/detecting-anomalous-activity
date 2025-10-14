# Ship Engine Anomaly Detection System

A comprehensive machine learning system for detecting anomalous activity in ship engine operations using statistical methods and unsupervised learning algorithms.

## Overview

This project implements an anomaly detection system that analyzes ship engine sensor data to identify potential maintenance issues before they become critical problems. The system uses multiple approaches including statistical methods (IQR) and machine learning algorithms (One-Class SVM and Isolation Forest) to detect anomalies in engine performance.

## Features

- **Statistical Analysis**: Interquartile Range (IQR) method for outlier detection
- **Machine Learning Models**: One-Class SVM and Isolation Forest algorithms
- **Dimensionality Reduction**: Principal Component Analysis (PCA) for visualization
- **Comprehensive Visualization**: Box plots, histograms, correlation matrices, and 2D scatter plots
- **Automated Reporting**: Detailed analysis reports with recommendations

## Dataset

The system analyzes six critical engine parameters:

- **Engine RPM**: Revolutions per minute - indicates engine speed and potential stress
- **Lubrication Oil Pressure**: Critical for engine lubrication and preventing wear
- **Fuel Pressure**: Affects combustion efficiency and engine performance
- **Coolant Pressure**: Essential for temperature regulation and preventing overheating
- **Lubrication Oil Temperature**: Indicates oil condition and engine heat levels
- **Coolant Temperature**: Key indicator of engine thermal state

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ship-engine-anomaly-detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis script:

```bash
python ship_engine_anomaly_detection.py
```

The script will:
1. Load the engine dataset from the remote source
2. Perform exploratory data analysis
3. Apply statistical anomaly detection methods
4. Train and evaluate machine learning models
5. Generate comprehensive visualizations
6. Provide detailed analysis reports

## Methodology

### 1. Statistical Approach (IQR)
- Calculates interquartile ranges for each feature
- Identifies outliers beyond 1.5 × IQR from quartiles
- Flags samples with multiple outlier features

### 2. One-Class SVM
- Uses RBF kernel with optimized parameters
- Targets 1-5% anomaly rate (typically 3%)
- Provides decision boundaries for anomaly classification

### 3. Isolation Forest
- Ensemble method for anomaly detection
- Works well with high-dimensional data
- Provides anomaly scores for ranking

### 4. Dimensionality Reduction
- PCA reduces 6D feature space to 2D for visualization
- Maintains ~37% of original variance
- Enables 2D plotting of anomaly detection results

## Results

The system typically identifies:
- **IQR Method**: ~2-3% of samples as anomalies
- **One-Class SVM**: ~3% of samples as anomalies  
- **Isolation Forest**: ~3% of samples as anomalies

All methods consistently identify the expected 1-5% anomaly rate, with key findings including:
- Engine RPM and lubrication oil temperature are primary indicators
- Multiple outlier features often co-occur
- Anomalies are well-separated in PCA space

## Key Insights

1. **Critical Monitoring Parameters**: Engine RPM and lubrication oil temperature require close monitoring
2. **Early Warning System**: Anomalies can be detected before critical failures
3. **Maintenance Optimization**: Proactive maintenance based on anomaly patterns
4. **Cost Reduction**: Prevents expensive breakdowns and downtime

## Technical Details

- **Language**: Python 3.7+
- **Key Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Data Source**: Remote CSV file from GitHub
- **Processing**: StandardScaler for feature normalization
- **Visualization**: Matplotlib and Seaborn for comprehensive plotting

## File Structure

```
ship-engine-anomaly-detection/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── ship_engine_anomaly_detection.py   # Main analysis script
└── engine.csv                         # Dataset (downloaded automatically)
```

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Author

**Matthew Russell**  
Data Science & Machine Learning Engineer

## Acknowledgments

- Dataset sourced from Devabrat, M. (2022) - Predictive Maintenance on Ship's Main Engine using AI
- Built using scikit-learn, pandas, and other open-source libraries
- Inspired by real-world industrial maintenance challenges
# detecting-anomalous-activity
