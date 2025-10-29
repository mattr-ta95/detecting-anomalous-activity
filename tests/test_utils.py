"""Tests for utility functions."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils import (
    calculate_checksum,
    format_anomaly_report,
    load_config,
    verify_checksum,
)


class TestConfigLoading:
    """Test configuration loading."""

    def test_load_config_valid(self, tmp_path):
        """Test loading valid config file."""
        config_data = {"test": "value", "nested": {"key": "value"}}
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))
        assert config["test"] == "value"
        assert config["nested"]["key"] == "value"

    def test_load_config_missing_file(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestChecksumFunctions:
    """Test checksum calculation and verification."""

    def test_calculate_checksum(self, tmp_path):
        """Test checksum calculation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        checksum = calculate_checksum(str(test_file))
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 produces 64 hex characters

    def test_calculate_checksum_missing_file(self):
        """Test checksum calculation with missing file."""
        with pytest.raises(FileNotFoundError):
            calculate_checksum("nonexistent.txt")

    def test_verify_checksum_valid(self, tmp_path):
        """Test checksum verification with correct checksum."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        checksum = calculate_checksum(str(test_file))
        assert verify_checksum(str(test_file), checksum) is True

    def test_verify_checksum_invalid(self, tmp_path):
        """Test checksum verification with incorrect checksum."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        assert verify_checksum(str(test_file), "wrong_checksum") is False


class TestAnomalyReport:
    """Test anomaly report formatting."""

    def test_format_anomaly_report_basic(self):
        """Test basic report formatting."""
        results = {
            "total_samples": 100,
            "iqr": {
                "n_anomalies": 5,
                "percentage": 5.0,
                "top_features": ["feature1", "feature2"],
            },
            "recommendations": ["Recommendation 1", "Recommendation 2"],
        }

        report = format_anomaly_report(results)
        assert "SHIP ENGINE ANOMALY DETECTION REPORT" in report
        assert "Total samples analyzed: 100" in report
        assert "Anomalies detected: 5" in report

    def test_format_anomaly_report_save_file(self, tmp_path):
        """Test saving report to file."""
        results = {
            "total_samples": 100,
            "recommendations": ["Test recommendation"],
        }

        output_file = tmp_path / "report.txt"
        report = format_anomaly_report(results, str(output_file))

        assert output_file.exists()
        assert output_file.read_text() == report


@pytest.fixture
def sample_config():
    """Fixture for sample configuration."""
    return {
        "data": {
            "url": "https://example.com/data.csv",
            "raw_path": "data/raw/data.csv",
            "processed_path": "data/processed/data.csv",
        },
        "output": {
            "plots_dir": "results/plots",
            "reports_dir": "results/reports",
        },
        "logging": {"level": "INFO", "console": True},
        "network": {
            "timeout": 30,
            "retries": 3,
            "backoff_factor": 2,
            "verify_ssl": True,
        },
    }
