"""Tests for preprocessing functions."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    handle_missing_values,
    remove_duplicates,
    remove_infinite_values,
    scale_features,
    validate_data,
)


@pytest.fixture
def sample_dataframe():
    """Fixture for sample DataFrame."""
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
            "feature3": [100, 200, 300, 400, 500],
        }
    )


class TestDataValidation:
    """Test data validation functions."""

    def test_validate_data_clean(self, sample_dataframe):
        """Test validation with clean data."""
        results = validate_data(sample_dataframe)

        assert results["n_samples"] == 5
        assert results["n_features"] == 3
        assert results["total_missing"] == 0
        assert results["duplicate_rows"] == 0

    def test_validate_data_with_missing(self):
        """Test validation with missing values."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, None, 4, 5],
                "feature2": [10, None, 30, 40, 50],
            }
        )

        results = validate_data(df)
        assert results["total_missing"] == 2
        assert results["missing_values"]["feature1"] == 1
        assert results["missing_values"]["feature2"] == 1

    def test_validate_data_with_duplicates(self):
        """Test validation with duplicate rows."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 2, 4],
                "feature2": [10, 20, 20, 40],
            }
        )

        results = validate_data(df)
        assert results["duplicate_rows"] == 1

    def test_validate_data_with_infinite(self):
        """Test validation with infinite values."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, np.inf, 4, 5],
                "feature2": [10, 20, 30, -np.inf, 50],
            }
        )

        results = validate_data(df)
        assert len(results["infinite_values"]) == 2


class TestMissingValueHandling:
    """Test missing value handling."""

    def test_handle_missing_drop(self):
        """Test dropping rows with missing values."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, None, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
            }
        )

        result = handle_missing_values(df, method="drop")
        assert len(result) == 4
        assert result.isnull().sum().sum() == 0

    def test_handle_missing_fill_mean(self):
        """Test filling missing values with mean."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, None, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        result = handle_missing_values(df, method="fill_mean")
        assert result.isnull().sum().sum() == 0
        # Mean of [1, 2, 4, 5] is 3.0
        assert result["feature1"].iloc[2] == 3.0

    def test_handle_missing_fill_median(self):
        """Test filling missing values with median."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, None, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        result = handle_missing_values(df, method="fill_median")
        assert result.isnull().sum().sum() == 0
        # Median of [1, 2, 4, 5] is 3.0
        assert result["feature1"].iloc[2] == 3.0

    def test_handle_missing_invalid_method(self):
        """Test with invalid method."""
        df = pd.DataFrame({"feature1": [1, 2, None]})

        with pytest.raises(ValueError):
            handle_missing_values(df, method="invalid_method")


class TestDuplicateRemoval:
    """Test duplicate row removal."""

    def test_remove_duplicates(self):
        """Test removing duplicate rows."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 2, 3],
                "feature2": [10, 20, 20, 30],
            }
        )

        result = remove_duplicates(df)
        assert len(result) == 3

    def test_remove_duplicates_none(self, sample_dataframe):
        """Test with no duplicates."""
        result = remove_duplicates(sample_dataframe)
        assert len(result) == len(sample_dataframe)


class TestInfiniteValueRemoval:
    """Test infinite value removal."""

    def test_remove_infinite_values(self):
        """Test removing rows with infinite values."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, np.inf, 4, 5],
                "feature2": [10, 20, 30, -np.inf, 50],
            }
        )

        result = remove_infinite_values(df)
        assert len(result) == 3  # Removed 2 rows

    def test_remove_infinite_values_none(self, sample_dataframe):
        """Test with no infinite values."""
        result = remove_infinite_values(sample_dataframe)
        assert len(result) == len(sample_dataframe)


class TestFeatureScaling:
    """Test feature scaling."""

    def test_scale_features_fit(self, sample_dataframe):
        """Test fitting scaler and transforming."""
        scaled_df, scaler = scale_features(sample_dataframe)

        assert scaled_df.shape == sample_dataframe.shape
        assert scaler is not None

        # Check that mean is approximately 0 and std is approximately 1
        for col in scaled_df.select_dtypes(include=[np.number]).columns:
            assert abs(scaled_df[col].mean()) < 1e-10
            # Use ddof=0 to match StandardScaler's population std behavior
            assert abs(scaled_df[col].std(ddof=0) - 1.0) < 1e-10

    def test_scale_features_transform(self, sample_dataframe):
        """Test transforming with existing scaler."""
        # First fit
        _, scaler = scale_features(sample_dataframe)

        # Then transform
        scaled_df, returned_scaler = scale_features(sample_dataframe, scaler=scaler)

        assert scaled_df.shape == sample_dataframe.shape
        assert returned_scaler is scaler
