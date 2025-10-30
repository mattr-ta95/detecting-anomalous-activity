"""Tests for preprocessing functions."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    get_feature_statistics,
    handle_missing_values,
    load_data,
    preprocess_data,
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


class TestLoadData:
    """Test data loading functions."""

    def test_load_data_success(self, tmp_path):
        """Test successful data loading."""
        # Create test CSV
        test_file = tmp_path / "test.csv"
        df = pd.DataFrame(
            {
                "Engine RPM": [1000, 2000, 3000],
                "Lubrication oil pressure": [10, 20, 30],
            }
        )
        df.to_csv(test_file, index=False)

        config = {
            "data": {
                "columns": ["Engine RPM", "Lubrication oil pressure"]
            }
        }

        loaded_df = load_data(str(test_file), config)

        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["Engine RPM", "Lubrication oil pressure"]

    def test_load_data_missing_file(self):
        """Test loading non-existent file."""
        config = {"data": {"columns": []}}

        with pytest.raises(FileNotFoundError):
            load_data("nonexistent.csv", config)

    def test_load_data_missing_columns(self, tmp_path):
        """Test loading data with missing required columns."""
        # Create test CSV
        test_file = tmp_path / "test.csv"
        df = pd.DataFrame({"col1": [1, 2, 3]})
        df.to_csv(test_file, index=False)

        config = {
            "data": {
                "columns": ["col1", "col2", "col3"]  # col2 and col3 missing
            }
        }

        with pytest.raises(ValueError, match="Missing required columns"):
            load_data(str(test_file), config)


class TestFeatureStatistics:
    """Test feature statistics calculation."""

    def test_get_feature_statistics(self):
        """Test calculating feature statistics."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
            }
        )

        stats = get_feature_statistics(df)

        assert "mean" in stats.columns
        assert "std" in stats.columns
        assert "min" in stats.columns
        assert "max" in stats.columns
        assert "missing" in stats.columns
        assert "skewness" in stats.columns
        assert "kurtosis" in stats.columns
        assert len(stats) == 2  # Two features

    def test_get_feature_statistics_with_missing(self):
        """Test statistics with missing values."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, None, 4, 5],
                "feature2": [10, None, None, 40, 50],
            }
        )

        stats = get_feature_statistics(df)

        assert stats.loc["feature1", "missing"] == 1
        assert stats.loc["feature2", "missing"] == 2


class TestPreprocessData:
    """Test complete preprocessing pipeline."""

    def test_preprocess_data_complete_pipeline(self):
        """Test full preprocessing pipeline."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 5],  # Duplicate row
                "feature2": [10, 20, 30, 40, 50, 50],  # Duplicate row
            }
        )

        config = {
            "preprocessing": {
                "handle_missing": "drop",
                "scaling": True,
            }
        }

        processed_df, scaler = preprocess_data(df, config)

        # Should have removed duplicate
        assert len(processed_df) == 5
        # Should be scaled
        assert scaler is not None
        # Mean should be ~0
        assert abs(processed_df["feature1"].mean()) < 1e-10

    def test_preprocess_data_no_scaling(self):
        """Test preprocessing without scaling."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
            }
        )

        config = {
            "preprocessing": {
                "handle_missing": "drop",
                "scaling": False,
            }
        }

        processed_df, scaler = preprocess_data(df, config)

        assert scaler is None
        # Data should not be scaled
        assert processed_df["feature1"].mean() == 3.0

    def test_preprocess_data_fill_mean(self):
        """Test preprocessing with fill_mean."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, None, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        config = {
            "preprocessing": {
                "handle_missing": "fill_mean",
                "scaling": False,
            }
        }

        processed_df, scaler = preprocess_data(df, config)

        # Should have no missing values
        assert processed_df.isnull().sum().sum() == 0
        # Missing value should be filled with mean
        assert processed_df["feature1"].iloc[2] == 3.0  # Mean of [1, 2, 4, 5]
