"""Visualization functions for anomaly detection results."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_plot_style(config: Dict[str, Any]) -> None:
    """
    Setup matplotlib and seaborn style.

    Args:
        config: Configuration dictionary
    """
    viz_config = config.get("visualization", {})
    style = viz_config.get("style", "whitegrid")
    palette = viz_config.get("palette", "Set2")

    sns.set_style(style)
    sns.set_palette(palette)
    plt.rcParams["figure.dpi"] = viz_config.get("figure_dpi", 300)


def plot_box_plots(
    df: pd.DataFrame,
    output_dir: str,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Create box plots for all numeric features.

    Args:
        df: Input DataFrame
        output_dir: Directory to save plots
        config: Configuration dictionary
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)

    if n_cols == 0:
        logger.warning("No numeric columns to plot")
        return

    # Calculate grid dimensions
    n_plot_cols = 3
    n_plot_rows = (n_cols + n_plot_cols - 1) // n_plot_cols

    fig, axes = plt.subplots(
        n_plot_rows, n_plot_cols, figsize=(15, 5 * n_plot_rows)
    )

    if n_plot_rows == 1 and n_plot_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_cols > 1 else [axes]

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        sns.boxplot(data=df, y=col, ax=ax, color="skyblue")
        ax.set_title(f"Box Plot: {col}")
        ax.set_ylabel(col)

    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir) / "box_plots.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_format = config.get("visualization", {}).get("save_format", "png")
    plt.savefig(output_path, format=save_format, bbox_inches="tight")
    plt.close()

    logger.info(f"Box plots saved to {output_path}")


def plot_histograms(
    df: pd.DataFrame,
    output_dir: str,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Create histograms for all numeric features.

    Args:
        df: Input DataFrame
        output_dir: Directory to save plots
        config: Configuration dictionary
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)

    if n_cols == 0:
        logger.warning("No numeric columns to plot")
        return

    # Calculate grid dimensions
    n_plot_cols = 3
    n_plot_rows = (n_cols + n_plot_cols - 1) // n_plot_cols

    fig, axes = plt.subplots(
        n_plot_rows, n_plot_cols, figsize=(15, 5 * n_plot_rows)
    )

    if n_plot_rows == 1 and n_plot_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_cols > 1 else [axes]

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        ax.hist(df[col].dropna(), bins=50, color="steelblue", edgecolor="black")
        ax.set_title(f"Histogram: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir) / "histograms.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_format = config.get("visualization", {}).get("save_format", "png")
    plt.savefig(output_path, format=save_format, bbox_inches="tight")
    plt.close()

    logger.info(f"Histograms saved to {output_path}")


def plot_correlation_matrix(
    df: pd.DataFrame,
    output_dir: str,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Create correlation matrix heatmap.

    Args:
        df: Input DataFrame
        output_dir: Directory to save plots
        config: Configuration dictionary
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 2:
        logger.warning("Need at least 2 numeric columns for correlation matrix")
        return

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir) / "correlation_matrix.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_format = config.get("visualization", {}).get("save_format", "png")
    plt.savefig(output_path, format=save_format, bbox_inches="tight")
    plt.close()

    logger.info(f"Correlation matrix saved to {output_path}")


def plot_pca_scatter(
    pca_data: np.ndarray,
    anomaly_labels: Dict[str, np.ndarray],
    output_dir: str,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Create PCA scatter plots with anomaly labels.

    Args:
        pca_data: PCA-transformed data (n_samples, 2)
        anomaly_labels: Dictionary mapping method names to anomaly labels
        output_dir: Directory to save plots
        config: Configuration dictionary
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    n_methods = len(anomaly_labels)
    if n_methods == 0:
        logger.warning("No anomaly labels provided")
        return

    # Create subplots
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))

    if n_methods == 1:
        axes = [axes]

    for idx, (method, labels) in enumerate(anomaly_labels.items()):
        ax = axes[idx]

        # Separate normal and anomalies
        normal_mask = labels == 1
        anomaly_mask = labels == -1

        # Plot normal points
        ax.scatter(
            pca_data[normal_mask, 0],
            pca_data[normal_mask, 1],
            c="blue",
            alpha=0.5,
            s=20,
            label="Normal",
        )

        # Plot anomalies
        ax.scatter(
            pca_data[anomaly_mask, 0],
            pca_data[anomaly_mask, 1],
            c="red",
            alpha=0.7,
            s=50,
            label="Anomaly",
            marker="x",
        )

        ax.set_xlabel("First Principal Component")
        ax.set_ylabel("Second Principal Component")
        ax.set_title(f"Anomaly Detection: {method}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir) / "pca_anomaly_scatter.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_format = config.get("visualization", {}).get("save_format", "png")
    plt.savefig(output_path, format=save_format, bbox_inches="tight")
    plt.close()

    logger.info(f"PCA scatter plots saved to {output_path}")


def plot_anomaly_scores(
    scores: Dict[str, np.ndarray],
    output_dir: str,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Plot anomaly score distributions.

    Args:
        scores: Dictionary mapping method names to anomaly scores
        output_dir: Directory to save plots
        config: Configuration dictionary
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    n_methods = len(scores)
    if n_methods == 0:
        logger.warning("No anomaly scores provided")
        return

    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))

    if n_methods == 1:
        axes = [axes]

    for idx, (method, score_values) in enumerate(scores.items()):
        ax = axes[idx]
        ax.hist(score_values, bins=50, color="steelblue", edgecolor="black")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Anomaly Score Distribution: {method}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir) / "anomaly_scores.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_format = config.get("visualization", {}).get("save_format", "png")
    plt.savefig(output_path, format=save_format, bbox_inches="tight")
    plt.close()

    logger.info(f"Anomaly score distributions saved to {output_path}")


def create_all_visualizations(
    df: pd.DataFrame,
    pca_data: Optional[np.ndarray],
    anomaly_results: Dict[str, Any],
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Create all configured visualizations.

    Args:
        df: Input DataFrame
        pca_data: PCA-transformed data (optional)
        anomaly_results: Dictionary with anomaly detection results
        config: Configuration dictionary
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Creating visualizations")

    # Setup plot style
    setup_plot_style(config)

    output_dir = config.get("output", {}).get("plots_dir", "results/plots")
    viz_config = config.get("visualization", {}).get("plots", {})

    # Create box plots
    if viz_config.get("box_plot", True):
        plot_box_plots(df, output_dir, config, logger)

    # Create histograms
    if viz_config.get("histogram", True):
        plot_histograms(df, output_dir, config, logger)

    # Create correlation matrix
    if viz_config.get("correlation_matrix", True):
        plot_correlation_matrix(df, output_dir, config, logger)

    # Create PCA scatter plots
    if viz_config.get("pca_scatter", True) and pca_data is not None:
        anomaly_labels = {}
        if "iqr" in anomaly_results:
            anomaly_labels["IQR"] = anomaly_results["iqr"]["labels"]
        if "one_class_svm" in anomaly_results:
            anomaly_labels["One-Class SVM"] = anomaly_results["one_class_svm"][
                "labels"
            ]
        if "isolation_forest" in anomaly_results:
            anomaly_labels["Isolation Forest"] = anomaly_results["isolation_forest"][
                "labels"
            ]

        if anomaly_labels:
            plot_pca_scatter(pca_data, anomaly_labels, output_dir, config, logger)

    # Create anomaly score plots
    anomaly_scores = {}
    if "isolation_forest" in anomaly_results:
        anomaly_scores["Isolation Forest"] = anomaly_results["isolation_forest"][
            "scores"
        ]

    if anomaly_scores:
        plot_anomaly_scores(anomaly_scores, output_dir, config, logger)

    logger.info("All visualizations created successfully")
