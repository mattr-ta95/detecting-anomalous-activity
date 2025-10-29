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
git clone https://github.com/mattr-ta95/detecting-anomalous-activity.git
cd detecting-anomalous-activity
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install the package:
```bash
pip install -e .
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

- **Language**: Python 3.8-3.11
- **Key Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, PyYAML
- **Data Source**: Remote CSV file from GitHub (with checksum verification support)
- **Processing**: StandardScaler for feature normalization
- **Visualization**: Matplotlib and Seaborn for comprehensive plotting
- **Configuration**: YAML-based configuration with extensive customization options
- **Security**: Input validation, checksum verification, and secure data handling

## File Structure

```
detecting-anomalous-activity/
├── .github/
│   └── workflows/              # CI/CD pipelines
├── data/
│   ├── raw/                   # Original dataset
│   └── processed/             # Processed data
├── src/                       # Source code
│   ├── __init__.py
│   ├── anomaly_detection.py   # Anomaly detection algorithms
│   ├── preprocessing.py       # Data preprocessing
│   ├── visualization.py       # Plotting functions
│   └── utils.py              # Utility functions
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_anomaly_detection.py
│   ├── test_preprocessing.py
│   └── test_utils.py
├── notebooks/                 # Jupyter notebooks
│   └── anomaly_detection_example.ipynb
├── results/                   # Output directory
│   ├── plots/                # Generated plots
│   └── reports/              # Analysis reports
├── .gitignore                # Git ignore file
├── LICENSE                   # MIT License
├── README.md                 # Project documentation
├── config.yaml               # Configuration file
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
└── ship_engine_anomaly_detection.py  # Main script
```

## Dependencies

### Core Dependencies
- numpy == 1.24.3
- pandas == 2.0.3
- scikit-learn == 1.3.0
- matplotlib == 3.7.2
- seaborn == 0.12.2
- PyYAML == 6.0.1
- requests == 2.31.0

### Development Dependencies
- pytest == 7.4.0
- pytest-cov == 4.1.0
- black == 23.7.0
- flake8 == 6.1.0
- mypy == 1.4.1

See `requirements.txt` for the complete list.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_anomaly_detection.py
```

### Code Quality

```bash
# Format code with black
black src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Format code (`black .`)
7. Submit a pull request

Please ensure your code follows PEP 8 guidelines and includes appropriate tests.

## License

This project is open source and available under the MIT License.

## Author

**Matthew Russell**  
Data Science & Machine Learning Engineer

## Troubleshooting

### Common Issues

1. **Dataset download fails**
   - Check internet connection
   - Verify URL in `config.yaml` is accessible
   - Try downloading manually and placing in `data/raw/`

2. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Verify Python version is 3.8 or higher

3. **Visualization errors**
   - Ensure matplotlib backend is properly configured
   - For headless environments, set `MPLBACKEND=Agg`

## Acknowledgments

- Dataset sourced from: Devabrat, M. (2022). "Predictive Maintenance on Ship's Main Engine using AI"
  - Available at: https://github.com/Devabrat-glitch/Predictive-Maintenance-of-Ships-Main-Engine-using-Deep-Learning
- Built using scikit-learn, pandas, and other open-source libraries
- Inspired by real-world industrial maintenance challenges
