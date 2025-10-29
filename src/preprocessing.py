"""Data preprocessing functions for ship engine data."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(
    file_path: str,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Load ship engine data from CSV file.

    Args:
        file_path: Path to CSV file
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)

    # Validate columns
    expected_columns = config.get("data", {}).get("columns", [])
    if expected_columns:
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")

    return df


def validate_data(
    df: pd.DataFrame, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Validate data quality and return summary statistics.

    Args:
        df: Input DataFrame
        logger: Logger instance

    Returns:
        Dictionary with validation results
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    validation_results = {
        "n_samples": len(df),
        "n_features": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
        "data_types": df.dtypes.to_dict(),
    }

    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_values = {}
    for col in numeric_cols:
        n_inf = np.isinf(df[col]).sum()
        if n_inf > 0:
            inf_values[col] = n_inf
    validation_results["infinite_values"] = inf_values

    # Log validation results
    logger.info(f"Data validation: {validation_results['n_samples']} samples, "
                f"{validation_results['n_features']} features")

    if validation_results["total_missing"] > 0:
        logger.warning(f"Found {validation_results['total_missing']} missing values")
        for col, count in validation_results["missing_values"].items():
            if count > 0:
                logger.warning(f"  {col}: {count} missing ({count/len(df)*100:.1f}%)")

    if validation_results["duplicate_rows"] > 0:
        logger.warning(f"Found {validation_results['duplicate_rows']} duplicate rows")

    if inf_values:
        logger.warning(f"Found infinite values in: {list(inf_values.keys())}")

    return validation_results


def handle_missing_values(
    df: pd.DataFrame,
    method: str = "drop",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.

    Args:
        df: Input DataFrame
        method: Method to handle missing values (drop, fill_mean, fill_median)
        logger: Logger instance

    Returns:
        DataFrame with missing values handled

    Raises:
        ValueError: If method is not supported
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if df.isnull().sum().sum() == 0:
        logger.info("No missing values to handle")
        return df

    initial_rows = len(df)

    if method == "drop":
        df = df.dropna()
        logger.info(f"Dropped {initial_rows - len(df)} rows with missing values")

    elif method == "fill_mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        logger.info("Filled missing values with column means")

    elif method == "fill_median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        logger.info("Filled missing values with column medians")

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'drop', 'fill_mean', or 'fill_median'"
        )

    return df


def remove_duplicates(
    df: pd.DataFrame, logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.

    Args:
        df: Input DataFrame
        logger: Logger instance

    Returns:
        DataFrame without duplicates
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)

    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows")
    else:
        logger.info("No duplicate rows found")

    return df


def remove_infinite_values(
    df: pd.DataFrame, logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Remove rows with infinite values.

    Args:
        df: Input DataFrame
        logger: Logger instance

    Returns:
        DataFrame without infinite values
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    initial_rows = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Replace inf with NaN and drop
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    removed = initial_rows - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} rows with infinite values")

    return df


def scale_features(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler.

    Args:
        df: Input DataFrame
        scaler: Pre-fitted scaler (optional, for transform only)
        logger: Logger instance

    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if scaler is None:
        logger.info("Fitting StandardScaler on data")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_cols])
    else:
        logger.info("Transforming data with existing scaler")
        scaled_data = scaler.transform(df[numeric_cols])

    # Create scaled DataFrame
    df_scaled = pd.DataFrame(
        scaled_data, columns=numeric_cols, index=df.index
    )

    # Add back non-numeric columns if any
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        df_scaled[non_numeric_cols] = df[non_numeric_cols]

    logger.info(f"Scaled {len(numeric_cols)} features")

    return df_scaled, scaler


def preprocess_data(
    df: pd.DataFrame,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Complete preprocessing pipeline.

    Args:
        df: Input DataFrame
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Tuple of (preprocessed DataFrame, scaler if scaling was applied)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Starting data preprocessing pipeline")

    # Validate data
    validation_results = validate_data(df, logger)

    # Handle missing values
    missing_method = config.get("preprocessing", {}).get("handle_missing", "drop")
    df = handle_missing_values(df, method=missing_method, logger=logger)

    # Remove duplicates
    df = remove_duplicates(df, logger=logger)

    # Remove infinite values
    df = remove_infinite_values(df, logger=logger)

    # Scale features if configured
    scaler = None
    if config.get("preprocessing", {}).get("scaling", True):
        df, scaler = scale_features(df, logger=logger)

    logger.info(f"Preprocessing complete: {len(df)} samples ready for analysis")

    return df, scaler


def get_feature_statistics(
    df: pd.DataFrame, logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Calculate comprehensive feature statistics.

    Args:
        df: Input DataFrame
        logger: Logger instance

    Returns:
        DataFrame with feature statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    stats = df[numeric_cols].describe().T
    stats["missing"] = df[numeric_cols].isnull().sum()
    stats["missing_pct"] = (stats["missing"] / len(df) * 100).round(2)
    stats["skewness"] = df[numeric_cols].skew()
    stats["kurtosis"] = df[numeric_cols].kurtosis()

    logger.info("Calculated feature statistics")

    return stats
