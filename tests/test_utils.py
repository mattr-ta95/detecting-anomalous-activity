"""Tests for utility functions."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
import yaml

from src.utils import (
    calculate_checksum,
    create_session_with_retries,
    download_file,
    ensure_directories,
    format_anomaly_report,
    load_config,
    setup_logging,
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


    def test_format_anomaly_report_all_methods(self):
        """Test report formatting with all detection methods."""
        results = {
            "total_samples": 1000,
            "iqr": {
                "n_anomalies": 30,
                "percentage": 3.0,
                "top_features": ["Engine RPM", "Lubrication oil temperature"],
            },
            "one_class_svm": {
                "n_anomalies": 25,
                "percentage": 2.5,
            },
            "isolation_forest": {
                "n_anomalies": 28,
                "percentage": 2.8,
            },
            "recommendations": [
                "Monitor Engine RPM closely",
                "Check lubrication system",
            ],
        }

        report = format_anomaly_report(results)
        assert "IQR Method" in report
        assert "One-Class SVM" in report
        assert "Isolation Forest" in report
        assert "30" in report
        assert "25" in report
        assert "28" in report
        assert "Monitor Engine RPM closely" in report


class TestLogging:
    """Test logging setup."""

    def test_setup_logging_console_only(self, sample_config):
        """Test logging setup with console handler."""
        config = sample_config.copy()
        config["logging"]["file"] = None

        logger = setup_logging(config)

        assert logger.name == "anomaly_detection"
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 1

    def test_setup_logging_with_file(self, sample_config, tmp_path):
        """Test logging setup with file handler."""
        config = sample_config.copy()
        log_file = tmp_path / "test.log"
        config["logging"]["file"] = str(log_file)

        logger = setup_logging(config)

        assert logger.name == "anomaly_detection"
        # Log a message to verify file handler works
        logger.info("Test message")

        # Verify log file was created
        assert log_file.exists()

    def test_setup_logging_different_levels(self, sample_config, tmp_path):
        """Test logging with different log levels."""
        config = sample_config.copy()
        log_file = tmp_path / "test.log"
        config["logging"]["file"] = str(log_file)

        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            config["logging"]["level"] = level
            logger = setup_logging(config)
            assert logger.level == getattr(logging, level)


class TestSessionCreation:
    """Test session creation with retries."""

    def test_create_session_with_retries_default(self):
        """Test creating session with default parameters."""
        session = create_session_with_retries()

        assert isinstance(session, requests.Session)
        assert session.adapters["http://"]
        assert session.adapters["https://"]

    def test_create_session_with_retries_custom(self):
        """Test creating session with custom parameters."""
        session = create_session_with_retries(
            retries=5, backoff_factor=3, timeout=60
        )

        assert isinstance(session, requests.Session)


class TestDownloadFile:
    """Test file downloading."""

    @patch("src.utils.requests.Session")
    def test_download_file_success(
        self, mock_session_class, sample_config, tmp_path
    ):
        """Test successful file download."""
        # Setup mock
        mock_session = Mock()
        mock_response = Mock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content = Mock(
            return_value=[b"test", b"data", b"content"]
        )
        mock_session.get = Mock(return_value=mock_response)
        mock_session_class.return_value = mock_session

        output_file = tmp_path / "downloaded.csv"
        url = "https://example.com/data.csv"

        with patch("src.utils.create_session_with_retries", return_value=mock_session):
            result = download_file(url, str(output_file), sample_config)

        assert result is True
        assert output_file.exists()

    @patch("src.utils.requests.Session")
    def test_download_file_with_checksum(
        self, mock_session_class, sample_config, tmp_path
    ):
        """Test file download with checksum verification."""
        # Setup mock
        mock_session = Mock()
        mock_response = Mock()
        mock_response.headers = {"content-length": "12"}
        content = b"test content"
        mock_response.iter_content = Mock(return_value=[content])
        mock_session.get = Mock(return_value=mock_response)

        output_file = tmp_path / "downloaded.csv"
        url = "https://example.com/data.csv"

        # Calculate expected checksum
        import hashlib

        expected_checksum = hashlib.sha256(content).hexdigest()
        sample_config["data"]["checksum"] = expected_checksum

        with patch("src.utils.create_session_with_retries", return_value=mock_session):
            result = download_file(url, str(output_file), sample_config)

        assert result is True

    def test_download_file_invalid_url(self, sample_config, tmp_path):
        """Test download with invalid URL."""
        output_file = tmp_path / "downloaded.csv"
        url = "not-a-valid-url"

        with pytest.raises(ValueError):
            download_file(url, str(output_file), sample_config)

    @patch("src.utils.requests.Session")
    def test_download_file_network_error(
        self, mock_session_class, sample_config, tmp_path
    ):
        """Test download with network error."""
        # Setup mock to raise exception
        mock_session = Mock()
        mock_session.get = Mock(side_effect=requests.RequestException("Network error"))

        output_file = tmp_path / "downloaded.csv"
        url = "https://example.com/data.csv"

        with patch("src.utils.create_session_with_retries", return_value=mock_session):
            with pytest.raises(requests.RequestException):
                download_file(url, str(output_file), sample_config)


class TestEnsureDirectories:
    """Test directory creation."""

    def test_ensure_directories(self, sample_config, tmp_path):
        """Test ensuring all directories exist."""
        # Update config with temp paths
        config = sample_config.copy()
        config["data"]["raw_path"] = str(tmp_path / "data" / "raw" / "file.csv")
        config["data"]["processed_path"] = str(
            tmp_path / "data" / "processed" / "file.csv"
        )
        config["output"]["plots_dir"] = str(tmp_path / "results" / "plots")
        config["output"]["reports_dir"] = str(tmp_path / "results" / "reports")

        ensure_directories(config)

        # Verify all directories created
        assert (tmp_path / "data" / "raw").exists()
        assert (tmp_path / "data" / "processed").exists()
        assert (tmp_path / "results" / "plots").exists()
        assert (tmp_path / "results" / "reports").exists()

    def test_ensure_directories_already_exist(self, sample_config, tmp_path):
        """Test with directories that already exist."""
        # Create directories first
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        config = sample_config.copy()
        config["data"]["raw_path"] = str(tmp_path / "data.csv")
        config["data"]["processed_path"] = str(tmp_path / "data.csv")
        config["output"]["plots_dir"] = str(plots_dir)
        config["output"]["reports_dir"] = str(plots_dir)

        # Should not raise error
        ensure_directories(config)


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
