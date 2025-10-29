"""Tests for visualization functions."""

from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from src.visualization import (
    create_all_visualizations,
    plot_anomaly_scores,
    plot_box_plots,
    plot_correlation_matrix,
    plot_histograms,
    plot_pca_scatter,
    setup_plot_style,
)


@pytest.fixture
def sample_config():
    """Fixture for sample configuration."""
    return {
        "visualization": {
            "style": "whitegrid",
            "palette": "Set2",
            "figure_dpi": 300,
            "save_format": "png",
            "plots": {
                "box_plot": True,
                "histogram": True,
                "correlation_matrix": True,
                "pca_scatter": True,
            },
        },
        "output": {
            "plots_dir": "results/plots",
            "reports_dir": "results/reports",
        },
    }


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


@pytest.fixture
def sample_pca_data():
    """Fixture for sample PCA data."""
    return np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])


@pytest.fixture
def sample_anomaly_labels():
    """Fixture for sample anomaly labels."""
    return {
        "IQR": np.array([1, 1, -1, 1, 1]),
        "One-Class SVM": np.array([1, -1, 1, 1, 1]),
    }


class TestSetupPlotStyle:
    """Test plot style setup."""

    @patch("src.visualization.sns.set_style")
    @patch("src.visualization.sns.set_palette")
    @patch("src.visualization.plt.rcParams", {})
    def test_setup_plot_style(self, mock_set_palette, mock_set_style, sample_config):
        """Test plot style setup."""
        setup_plot_style(sample_config)

        mock_set_style.assert_called_once_with("whitegrid")
        mock_set_palette.assert_called_once_with("Set2")


class TestBoxPlots:
    """Test box plot creation."""

    @patch("src.visualization.plt.savefig")
    @patch("src.visualization.plt.close")
    @patch("src.visualization.plt.tight_layout")
    @patch("src.visualization.plt.subplots")
    @patch("src.visualization.sns.boxplot")
    def test_plot_box_plots(
        self,
        mock_boxplot,
        mock_subplots,
        mock_tight_layout,
        mock_close,
        mock_savefig,
        sample_dataframe,
        sample_config,
        tmp_path,
    ):
        """Test box plot creation."""
        # Setup mocks
        fig_mock = Mock()
        axes_mock = [Mock(), Mock(), Mock()]
        mock_subplots.return_value = (fig_mock, np.array(axes_mock))

        output_dir = str(tmp_path)
        plot_box_plots(sample_dataframe, output_dir, sample_config)

        # Verify subplots created
        assert mock_subplots.called
        # Verify boxplot called for each column
        assert mock_boxplot.call_count == 3
        # Verify saving
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("src.visualization.plt.close")
    def test_plot_box_plots_no_numeric_columns(
        self, mock_close, sample_config, tmp_path
    ):
        """Test box plots with no numeric columns."""
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        output_dir = str(tmp_path)

        plot_box_plots(df, output_dir, sample_config)

        # Should return early without creating plots
        mock_close.assert_not_called()


class TestHistograms:
    """Test histogram creation."""

    @patch("src.visualization.plt.savefig")
    @patch("src.visualization.plt.close")
    @patch("src.visualization.plt.tight_layout")
    @patch("src.visualization.plt.subplots")
    def test_plot_histograms(
        self,
        mock_subplots,
        mock_tight_layout,
        mock_close,
        mock_savefig,
        sample_dataframe,
        sample_config,
        tmp_path,
    ):
        """Test histogram creation."""
        # Setup mocks
        fig_mock = Mock()
        axes_mock = [Mock(), Mock(), Mock()]
        for ax in axes_mock:
            ax.hist = Mock()
        mock_subplots.return_value = (fig_mock, np.array(axes_mock))

        output_dir = str(tmp_path)
        plot_histograms(sample_dataframe, output_dir, sample_config)

        # Verify subplots created
        assert mock_subplots.called
        # Verify hist called for each column
        assert sum(ax.hist.called for ax in axes_mock) == 3
        # Verify saving
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("src.visualization.plt.close")
    def test_plot_histograms_no_numeric_columns(
        self, mock_close, sample_config, tmp_path
    ):
        """Test histograms with no numeric columns."""
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        output_dir = str(tmp_path)

        plot_histograms(df, output_dir, sample_config)

        # Should return early without creating plots
        mock_close.assert_not_called()


class TestCorrelationMatrix:
    """Test correlation matrix creation."""

    @patch("src.visualization.plt.savefig")
    @patch("src.visualization.plt.close")
    @patch("src.visualization.plt.tight_layout")
    @patch("src.visualization.plt.figure")
    @patch("src.visualization.sns.heatmap")
    def test_plot_correlation_matrix(
        self,
        mock_heatmap,
        mock_figure,
        mock_tight_layout,
        mock_close,
        mock_savefig,
        sample_dataframe,
        sample_config,
        tmp_path,
    ):
        """Test correlation matrix creation."""
        output_dir = str(tmp_path)
        plot_correlation_matrix(sample_dataframe, output_dir, sample_config)

        # Verify heatmap created
        mock_heatmap.assert_called_once()
        # Verify saving
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("src.visualization.plt.close")
    def test_plot_correlation_matrix_insufficient_columns(
        self, mock_close, sample_config, tmp_path
    ):
        """Test correlation matrix with < 2 numeric columns."""
        df = pd.DataFrame({"feature1": [1, 2, 3]})
        output_dir = str(tmp_path)

        plot_correlation_matrix(df, output_dir, sample_config)

        # Should return early without creating plots
        mock_close.assert_not_called()


class TestPCAScatter:
    """Test PCA scatter plot creation."""

    @patch("src.visualization.plt.savefig")
    @patch("src.visualization.plt.close")
    @patch("src.visualization.plt.tight_layout")
    @patch("src.visualization.plt.subplots")
    def test_plot_pca_scatter(
        self,
        mock_subplots,
        mock_tight_layout,
        mock_close,
        mock_savefig,
        sample_pca_data,
        sample_anomaly_labels,
        sample_config,
        tmp_path,
    ):
        """Test PCA scatter plot creation."""
        # Setup mocks
        fig_mock = Mock()
        axes_mock = [Mock(), Mock()]
        for ax in axes_mock:
            ax.scatter = Mock()
        mock_subplots.return_value = (fig_mock, axes_mock)

        output_dir = str(tmp_path)
        plot_pca_scatter(
            sample_pca_data, sample_anomaly_labels, output_dir, sample_config
        )

        # Verify subplots created
        assert mock_subplots.called
        # Verify scatter called (once for normal, once for anomaly per method)
        total_scatter_calls = sum(ax.scatter.call_count for ax in axes_mock)
        assert total_scatter_calls == 4  # 2 methods × 2 scatter calls each
        # Verify saving
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("src.visualization.plt.close")
    def test_plot_pca_scatter_no_labels(self, mock_close, sample_pca_data, sample_config, tmp_path):
        """Test PCA scatter with no anomaly labels."""
        output_dir = str(tmp_path)
        plot_pca_scatter(sample_pca_data, {}, output_dir, sample_config)

        # Should return early without creating plots
        mock_close.assert_not_called()

    @patch("src.visualization.plt.savefig")
    @patch("src.visualization.plt.close")
    @patch("src.visualization.plt.subplots")
    def test_plot_pca_scatter_single_method(
        self,
        mock_subplots,
        mock_close,
        mock_savefig,
        sample_pca_data,
        sample_config,
        tmp_path,
    ):
        """Test PCA scatter with single method."""
        # Setup mocks
        fig_mock = Mock()
        ax_mock = Mock()
        ax_mock.scatter = Mock()
        mock_subplots.return_value = (fig_mock, ax_mock)

        output_dir = str(tmp_path)
        single_label = {"IQR": np.array([1, 1, -1, 1, 1])}
        plot_pca_scatter(sample_pca_data, single_label, output_dir, sample_config)

        # Verify scatter called
        assert ax_mock.scatter.call_count == 2  # normal + anomaly
        mock_savefig.assert_called_once()


class TestAnomalyScores:
    """Test anomaly score plot creation."""

    @patch("src.visualization.plt.savefig")
    @patch("src.visualization.plt.close")
    @patch("src.visualization.plt.tight_layout")
    @patch("src.visualization.plt.subplots")
    def test_plot_anomaly_scores(
        self,
        mock_subplots,
        mock_tight_layout,
        mock_close,
        mock_savefig,
        sample_config,
        tmp_path,
    ):
        """Test anomaly score plot creation."""
        # Setup mocks
        fig_mock = Mock()
        ax_mock = Mock()
        ax_mock.hist = Mock()
        mock_subplots.return_value = (fig_mock, ax_mock)

        scores = {"Method1": np.array([1, 2, 3, 4, 5])}
        output_dir = str(tmp_path)

        plot_anomaly_scores(scores, output_dir, sample_config)

        # Verify hist called
        ax_mock.hist.assert_called_once()
        # Verify saving
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("src.visualization.plt.close")
    def test_plot_anomaly_scores_no_scores(self, mock_close, sample_config, tmp_path):
        """Test anomaly scores with no scores."""
        output_dir = str(tmp_path)
        plot_anomaly_scores({}, output_dir, sample_config)

        # Should return early without creating plots
        mock_close.assert_not_called()


class TestCreateAllVisualizations:
    """Test creating all visualizations."""

    @patch("src.visualization.plot_anomaly_scores")
    @patch("src.visualization.plot_pca_scatter")
    @patch("src.visualization.plot_correlation_matrix")
    @patch("src.visualization.plot_histograms")
    @patch("src.visualization.plot_box_plots")
    @patch("src.visualization.setup_plot_style")
    def test_create_all_visualizations(
        self,
        mock_setup_style,
        mock_box_plots,
        mock_histograms,
        mock_correlation,
        mock_pca_scatter,
        mock_anomaly_scores,
        sample_dataframe,
        sample_config,
    ):
        """Test creating all visualizations."""
        pca_data = np.array([[1, 2], [3, 4]])
        anomaly_results = {
            "iqr": {"labels": np.array([1, -1])},
            "one_class_svm": {"labels": np.array([1, 1])},
            "isolation_forest": {
                "labels": np.array([1, -1]),
                "scores": np.array([0.5, -0.5]),
            },
        }

        create_all_visualizations(
            sample_dataframe, pca_data, anomaly_results, sample_config
        )

        # Verify all visualization functions called
        mock_setup_style.assert_called_once()
        mock_box_plots.assert_called_once()
        mock_histograms.assert_called_once()
        mock_correlation.assert_called_once()
        mock_pca_scatter.assert_called_once()
        mock_anomaly_scores.assert_called_once()

    @patch("src.visualization.plot_box_plots")
    @patch("src.visualization.setup_plot_style")
    def test_create_all_visualizations_disabled_plots(
        self,
        mock_setup_style,
        mock_box_plots,
        sample_dataframe,
        sample_config,
    ):
        """Test creating visualizations with some disabled."""
        # Disable all plots
        sample_config["visualization"]["plots"] = {
            "box_plot": False,
            "histogram": False,
            "correlation_matrix": False,
            "pca_scatter": False,
        }

        create_all_visualizations(
            sample_dataframe, None, {}, sample_config
        )

        # Verify setup called but plots not created
        mock_setup_style.assert_called_once()
        mock_box_plots.assert_not_called()

    @patch("src.visualization.plot_pca_scatter")
    @patch("src.visualization.setup_plot_style")
    def test_create_all_visualizations_no_pca_data(
        self,
        mock_setup_style,
        mock_pca_scatter,
        sample_dataframe,
        sample_config,
    ):
        """Test creating visualizations without PCA data."""
        anomaly_results = {"iqr": {"labels": np.array([1, -1])}}

        create_all_visualizations(
            sample_dataframe, None, anomaly_results, sample_config
        )

        # PCA scatter should not be called without data
        mock_pca_scatter.assert_not_called()
