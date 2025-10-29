"""Tests for anomaly detection functions."""

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import AnomalyDetector, create_anomaly_summary


@pytest.fixture
def sample_dataframe():
    """Fixture for sample DataFrame."""
    np.random.seed(42)
    n_samples = 100

    # Create normal data
    data = {
        "feature1": np.random.normal(0, 1, n_samples),
        "feature2": np.random.normal(0, 1, n_samples),
        "feature3": np.random.normal(0, 1, n_samples),
    }

    # Add some outliers
    data["feature1"][0] = 10  # Clear outlier
    data["feature2"][0] = 10  # Clear outlier
    data["feature1"][1] = -10  # Clear outlier

    return pd.DataFrame(data)


@pytest.fixture
def sample_config():
    """Fixture for sample configuration."""
    return {
        "models": {
            "iqr": {
                "enabled": True,
                "multiplier": 1.5,
                "min_outlier_features": 2,
            },
            "one_class_svm": {
                "enabled": True,
                "kernel": "rbf",
                "gamma": "auto",
                "nu": 0.1,
            },
            "isolation_forest": {
                "enabled": True,
                "n_estimators": 100,
                "contamination": 0.1,
                "max_samples": "auto",
                "random_state": 42,
            },
        },
        "pca": {
            "enabled": True,
            "n_components": 2,
            "random_state": 42,
        },
    }


class TestAnomalyDetector:
    """Test AnomalyDetector class."""

    def test_init(self, sample_config):
        """Test detector initialization."""
        detector = AnomalyDetector(sample_config)
        assert detector.config == sample_config
        assert detector.models == {}
        assert detector.results == {}

    def test_detect_iqr_anomalies(self, sample_dataframe, sample_config):
        """Test IQR anomaly detection."""
        detector = AnomalyDetector(sample_config)
        labels, results = detector.detect_iqr_anomalies(sample_dataframe)

        assert len(labels) == len(sample_dataframe)
        assert results["n_anomalies"] > 0
        assert 0 <= results["percentage"] <= 100
        assert "top_features" in results
        assert all(label in [1, -1] for label in labels)

    def test_detect_one_class_svm_anomalies(self, sample_dataframe, sample_config):
        """Test One-Class SVM anomaly detection."""
        detector = AnomalyDetector(sample_config)
        labels, results = detector.detect_one_class_svm_anomalies(sample_dataframe)

        assert len(labels) == len(sample_dataframe)
        assert results["n_anomalies"] > 0
        assert 0 <= results["percentage"] <= 100
        assert "scores" in results
        assert "model" in results
        assert all(label in [1, -1] for label in labels)

    def test_detect_isolation_forest_anomalies(self, sample_dataframe, sample_config):
        """Test Isolation Forest anomaly detection."""
        detector = AnomalyDetector(sample_config)
        labels, results = detector.detect_isolation_forest_anomalies(sample_dataframe)

        assert len(labels) == len(sample_dataframe)
        assert results["n_anomalies"] > 0
        assert 0 <= results["percentage"] <= 100
        assert "scores" in results
        assert "model" in results
        assert all(label in [1, -1] for label in labels)

    def test_apply_pca(self, sample_dataframe, sample_config):
        """Test PCA dimensionality reduction."""
        detector = AnomalyDetector(sample_config)
        pca_data, pca_model = detector.apply_pca(sample_dataframe, n_components=2)

        assert pca_data.shape == (len(sample_dataframe), 2)
        assert pca_model is not None
        assert hasattr(pca_model, "explained_variance_ratio_")

    def test_detect_all_anomalies(self, sample_dataframe, sample_config):
        """Test running all detection methods."""
        detector = AnomalyDetector(sample_config)
        results, pca_data = detector.detect_all_anomalies(sample_dataframe)

        assert "total_samples" in results
        assert results["total_samples"] == len(sample_dataframe)
        assert "iqr" in results
        assert "one_class_svm" in results
        assert "isolation_forest" in results
        assert "recommendations" in results
        assert pca_data is not None
        assert pca_data.shape[1] == 2


class TestAnomalySummary:
    """Test anomaly summary creation."""

    def test_create_anomaly_summary(self):
        """Test creating summary DataFrame."""
        results = {
            "iqr": {"n_anomalies": 5, "percentage": 5.0},
            "one_class_svm": {"n_anomalies": 3, "percentage": 3.0},
            "isolation_forest": {"n_anomalies": 4, "percentage": 4.0},
        }

        summary_df = create_anomaly_summary(results)

        assert len(summary_df) == 3
        assert "Method" in summary_df.columns
        assert "Anomalies Detected" in summary_df.columns
        assert "Percentage" in summary_df.columns

    def test_create_anomaly_summary_empty(self):
        """Test creating summary with no results."""
        results = {}
        summary_df = create_anomaly_summary(results)

        assert len(summary_df) == 0
