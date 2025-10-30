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

    def test_create_anomaly_summary_partial(self):
        """Test creating summary with partial results."""
        results = {
            "iqr": {"n_anomalies": 5, "percentage": 5.0},
        }

        summary_df = create_anomaly_summary(results)

        assert len(summary_df) == 1
        assert summary_df.iloc[0]["Method"] == "Iqr"


class TestGenerateRecommendations:
    """Test recommendation generation."""

    def test_generate_recommendations_high_anomaly_rate(self, sample_config):
        """Test recommendations with high anomaly rate."""
        detector = AnomalyDetector(sample_config)

        results = {
            "total_samples": 100,
            "iqr": {
                "n_anomalies": 10,
                "percentage": 10.0,
                "top_features": ["Engine RPM", "Temperature"],
            },
        }

        recommendations = detector._generate_recommendations(results)

        # Should recommend investigating sensor calibration
        assert any("calibration" in rec.lower() for rec in recommendations)
        # Should mention top features
        assert any("Engine RPM" in rec for rec in recommendations)

    def test_generate_recommendations_consensus_anomalies(self, sample_config):
        """Test recommendations with consensus anomalies."""
        detector = AnomalyDetector(sample_config)

        results = {
            "total_samples": 100,
            "iqr": {
                "labels": np.array([1, -1, -1, 1, 1]),
                "percentage": 2.0,
                "top_features": ["feature1"],
            },
            "one_class_svm": {
                "labels": np.array([1, -1, -1, 1, 1]),
                "percentage": 2.0,
            },
            "isolation_forest": {
                "labels": np.array([1, -1, -1, 1, 1]),
                "percentage": 2.0,
            },
        }

        recommendations = detector._generate_recommendations(results)

        # Should mention consensus anomalies
        assert any("flagged by all three methods" in rec for rec in recommendations)


class TestAnomalyDetectorEdgeCases:
    """Test edge cases in anomaly detection."""

    def test_detect_all_anomalies_disabled_methods(self, sample_dataframe):
        """Test with some methods disabled."""
        config = {
            "models": {
                "iqr": {"enabled": False},
                "one_class_svm": {"enabled": True, "kernel": "rbf", "gamma": "auto", "nu": 0.1},
                "isolation_forest": {
                    "enabled": True,
                    "n_estimators": 100,
                    "contamination": 0.1,
                    "random_state": 42,
                },
            },
            "pca": {"enabled": True, "n_components": 2, "random_state": 42},
        }

        detector = AnomalyDetector(config)
        results, pca_data = detector.detect_all_anomalies(sample_dataframe)

        # IQR should not be in results
        assert "iqr" not in results
        # Other methods should be present
        assert "one_class_svm" in results
        assert "isolation_forest" in results

    def test_detect_all_anomalies_no_pca(self, sample_dataframe):
        """Test with PCA disabled."""
        config = {
            "models": {
                "iqr": {"enabled": True, "multiplier": 1.5, "min_outlier_features": 2},
                "one_class_svm": {"enabled": False},
                "isolation_forest": {"enabled": False},
            },
            "pca": {"enabled": False},
        }

        detector = AnomalyDetector(config)
        results, pca_data = detector.detect_all_anomalies(sample_dataframe)

        # PCA data should be None
        assert pca_data is None
        # IQR results should be present
        assert "iqr" in results
