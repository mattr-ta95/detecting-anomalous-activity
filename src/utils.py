"""Utility functions for anomaly detection system."""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured logger instance
    """
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create logger
    logger = logging.getLogger("anomaly_detection")
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Console handler
    if log_config.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)

    # File handler
    log_file = log_config.get("file")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    return logger


def calculate_checksum(file_path: str, algorithm: str = "sha256") -> str:
    """
    Calculate checksum of a file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hexadecimal checksum string

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def verify_checksum(
    file_path: str, expected_checksum: str, algorithm: str = "sha256"
) -> bool:
    """
    Verify file checksum.

    Args:
        file_path: Path to file
        expected_checksum: Expected checksum value
        algorithm: Hash algorithm (default: sha256)

    Returns:
        True if checksum matches, False otherwise
    """
    actual_checksum = calculate_checksum(file_path, algorithm)
    return actual_checksum.lower() == expected_checksum.lower()


def create_session_with_retries(
    retries: int = 3, backoff_factor: float = 2, timeout: int = 30
) -> requests.Session:
    """
    Create requests session with retry logic.

    Args:
        retries: Number of retry attempts
        backoff_factor: Backoff multiplier for retries
        timeout: Request timeout in seconds

    Returns:
        Configured requests session
    """
    session = requests.Session()

    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def download_file(
    url: str,
    output_path: str,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Download file from URL with retry logic and checksum verification.

    Args:
        url: URL to download from
        output_path: Local path to save file
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        True if download successful, False otherwise

    Raises:
        ValueError: If URL is invalid
        requests.RequestException: If download fails after retries
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Validate URL
    parsed_url = urlparse(url)
    if not all([parsed_url.scheme, parsed_url.netloc]):
        raise ValueError(f"Invalid URL: {url}")

    # Upgrade HTTP to HTTPS if configured
    if parsed_url.scheme == "http":
        logger.warning("HTTP URL detected, upgrading to HTTPS")
        url = url.replace("http://", "https://")

    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Get network configuration
    network_config = config.get("network", {})
    timeout = network_config.get("timeout", 30)
    retries = network_config.get("retries", 3)
    backoff_factor = network_config.get("backoff_factor", 2)
    verify_ssl = network_config.get("verify_ssl", True)

    # Create session with retries
    session = create_session_with_retries(retries, backoff_factor, timeout)

    try:
        logger.info(f"Downloading file from {url}")
        response = session.get(url, timeout=timeout, verify=verify_ssl, stream=True)
        response.raise_for_status()

        # Download with progress
        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.debug(f"Download progress: {progress:.1f}%")

        logger.info(f"File downloaded successfully to {output_path}")

        # Verify checksum if provided
        expected_checksum = config.get("data", {}).get("checksum")
        if expected_checksum:
            logger.info("Verifying file checksum...")
            if verify_checksum(output_path, expected_checksum):
                logger.info("Checksum verification passed")
            else:
                logger.error("Checksum verification failed!")
                return False

        return True

    except requests.RequestException as e:
        logger.error(f"Failed to download file: {e}")
        raise
    finally:
        session.close()


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure all required directories exist.

    Args:
        config: Configuration dictionary
    """
    directories = [
        Path(config["data"]["raw_path"]).parent,
        Path(config["data"]["processed_path"]).parent,
        Path(config["output"]["plots_dir"]),
        Path(config["output"]["reports_dir"]),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def format_anomaly_report(
    results: Dict[str, Any], output_path: Optional[str] = None
) -> str:
    """
    Format anomaly detection results as a report.

    Args:
        results: Dictionary containing detection results
        output_path: Optional path to save report

    Returns:
        Formatted report string
    """
    report_lines = [
        "=" * 80,
        "SHIP ENGINE ANOMALY DETECTION REPORT",
        "=" * 80,
        "",
        f"Total samples analyzed: {results.get('total_samples', 0)}",
        "",
        "ANOMALY DETECTION RESULTS:",
        "-" * 80,
    ]

    # IQR results
    if "iqr" in results:
        iqr_results = results["iqr"]
        report_lines.extend(
            [
                "",
                "1. Interquartile Range (IQR) Method:",
                f"   Anomalies detected: {iqr_results.get('n_anomalies', 0)} "
                f"({iqr_results.get('percentage', 0):.2f}%)",
                f"   Most anomalous features: {', '.join(iqr_results.get('top_features', []))}",
            ]
        )

    # One-Class SVM results
    if "one_class_svm" in results:
        svm_results = results["one_class_svm"]
        report_lines.extend(
            [
                "",
                "2. One-Class SVM:",
                f"   Anomalies detected: {svm_results.get('n_anomalies', 0)} "
                f"({svm_results.get('percentage', 0):.2f}%)",
            ]
        )

    # Isolation Forest results
    if "isolation_forest" in results:
        if_results = results["isolation_forest"]
        report_lines.extend(
            [
                "",
                "3. Isolation Forest:",
                f"   Anomalies detected: {if_results.get('n_anomalies', 0)} "
                f"({if_results.get('percentage', 0):.2f}%)",
            ]
        )

    # Recommendations
    report_lines.extend(
        [
            "",
            "=" * 80,
            "RECOMMENDATIONS:",
            "-" * 80,
        ]
    )

    recommendations = results.get("recommendations", [])
    for i, rec in enumerate(recommendations, 1):
        report_lines.append(f"{i}. {rec}")

    report_lines.extend(["", "=" * 80, ""])

    report = "\n".join(report_lines)

    # Save to file if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report
