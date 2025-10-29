"""Anomaly detection methods for ship engine data."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class AnomalyDetector:
    """Anomaly detection using multiple methods."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize anomaly detector.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.models = {}
        self.results = {}

    def detect_iqr_anomalies(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect anomalies using Interquartile Range (IQR) method.

        Args:
            df: Input DataFrame with numeric features

        Returns:
            Tuple of (anomaly labels, detection results)
        """
        self.logger.info("Detecting anomalies using IQR method")

        iqr_config = self.config.get("models", {}).get("iqr", {})
        multiplier = iqr_config.get("multiplier", 1.5)
        min_outlier_features = iqr_config.get("min_outlier_features", 2)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_samples = len(df)

        # Calculate IQR for each feature
        outlier_counts = np.zeros(n_samples)
        outlier_features = {i: [] for i in range(n_samples)}

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            # Find outliers for this feature
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_indices = np.where(outliers)[0]

            for idx in outlier_indices:
                outlier_counts[idx] += 1
                outlier_features[idx].append(col)

        # Label samples as anomalies if they exceed threshold
        labels = np.ones(n_samples)
        labels[outlier_counts >= min_outlier_features] = -1

        n_anomalies = np.sum(labels == -1)
        percentage = (n_anomalies / n_samples) * 100

        # Find most anomalous features
        feature_outlier_counts = {}
        for idx, features in outlier_features.items():
            if labels[idx] == -1:
                for feature in features:
                    feature_outlier_counts[feature] = (
                        feature_outlier_counts.get(feature, 0) + 1
                    )

        top_features = sorted(
            feature_outlier_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]
        top_feature_names = [f[0] for f in top_features]

        results = {
            "n_anomalies": int(n_anomalies),
            "percentage": float(percentage),
            "labels": labels,
            "outlier_counts": outlier_counts,
            "top_features": top_feature_names,
            "feature_outlier_counts": feature_outlier_counts,
        }

        self.logger.info(
            f"IQR method: Found {n_anomalies} anomalies ({percentage:.2f}%)"
        )
        self.logger.info(f"Most anomalous features: {', '.join(top_feature_names)}")

        return labels, results

    def detect_one_class_svm_anomalies(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect anomalies using One-Class SVM.

        Args:
            df: Input DataFrame with numeric features

        Returns:
            Tuple of (anomaly labels, detection results)
        """
        self.logger.info("Detecting anomalies using One-Class SVM")

        svm_config = self.config.get("models", {}).get("one_class_svm", {})
        kernel = svm_config.get("kernel", "rbf")
        gamma = svm_config.get("gamma", "auto")
        nu = svm_config.get("nu", 0.03)
        random_state = svm_config.get("random_state", 42)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].values

        # Train One-Class SVM
        model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu, random_state=random_state)
        labels = model.fit_predict(X)

        n_samples = len(df)
        n_anomalies = np.sum(labels == -1)
        percentage = (n_anomalies / n_samples) * 100

        # Get decision function scores
        scores = model.decision_function(X)

        results = {
            "n_anomalies": int(n_anomalies),
            "percentage": float(percentage),
            "labels": labels,
            "scores": scores,
            "model": model,
        }

        self.models["one_class_svm"] = model

        self.logger.info(
            f"One-Class SVM: Found {n_anomalies} anomalies ({percentage:.2f}%)"
        )

        return labels, results

    def detect_isolation_forest_anomalies(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect anomalies using Isolation Forest.

        Args:
            df: Input DataFrame with numeric features

        Returns:
            Tuple of (anomaly labels, detection results)
        """
        self.logger.info("Detecting anomalies using Isolation Forest")

        if_config = self.config.get("models", {}).get("isolation_forest", {})
        n_estimators = if_config.get("n_estimators", 100)
        contamination = if_config.get("contamination", 0.03)
        max_samples = if_config.get("max_samples", "auto")
        random_state = if_config.get("random_state", 42)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].values

        # Train Isolation Forest
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
        )
        labels = model.fit_predict(X)

        n_samples = len(df)
        n_anomalies = np.sum(labels == -1)
        percentage = (n_anomalies / n_samples) * 100

        # Get anomaly scores (negative scores indicate anomalies)
        scores = model.score_samples(X)

        results = {
            "n_anomalies": int(n_anomalies),
            "percentage": float(percentage),
            "labels": labels,
            "scores": scores,
            "model": model,
        }

        self.models["isolation_forest"] = model

        self.logger.info(
            f"Isolation Forest: Found {n_anomalies} anomalies ({percentage:.2f}%)"
        )

        return labels, results

    def apply_pca(
        self, df: pd.DataFrame, n_components: int = 2
    ) -> Tuple[np.ndarray, PCA]:
        """
        Apply PCA for dimensionality reduction.

        Args:
            df: Input DataFrame with numeric features
            n_components: Number of principal components

        Returns:
            Tuple of (transformed data, PCA model)
        """
        self.logger.info(f"Applying PCA with {n_components} components")

        pca_config = self.config.get("pca", {})
        random_state = pca_config.get("random_state", 42)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].values

        pca = PCA(n_components=n_components, random_state=random_state)
        X_pca = pca.fit_transform(X)

        explained_variance = pca.explained_variance_ratio_.sum()
        self.logger.info(
            f"PCA: Explained variance = {explained_variance:.2%}"
        )

        return X_pca, pca

    def detect_all_anomalies(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """
        Run all configured anomaly detection methods.

        Args:
            df: Input DataFrame with numeric features

        Returns:
            Tuple of (detection results dictionary, PCA-transformed data)
        """
        self.logger.info("Running all anomaly detection methods")

        results = {"total_samples": len(df)}

        # IQR method
        if self.config.get("models", {}).get("iqr", {}).get("enabled", True):
            _, iqr_results = self.detect_iqr_anomalies(df)
            results["iqr"] = iqr_results

        # One-Class SVM
        if (
            self.config.get("models", {})
            .get("one_class_svm", {})
            .get("enabled", True)
        ):
            _, svm_results = self.detect_one_class_svm_anomalies(df)
            results["one_class_svm"] = svm_results

        # Isolation Forest
        if (
            self.config.get("models", {})
            .get("isolation_forest", {})
            .get("enabled", True)
        ):
            _, if_results = self.detect_isolation_forest_anomalies(df)
            results["isolation_forest"] = if_results

        # Apply PCA if configured
        pca_data = None
        if self.config.get("pca", {}).get("enabled", True):
            n_components = self.config.get("pca", {}).get("n_components", 2)
            pca_data, pca_model = self.apply_pca(df, n_components)
            results["pca"] = {"transformed_data": pca_data, "model": pca_model}

        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        results["recommendations"] = recommendations

        self.results = results

        self.logger.info("Anomaly detection complete")

        return results, pca_data

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on detection results.

        Args:
            results: Detection results dictionary

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check anomaly rates
        total_samples = results.get("total_samples", 0)

        if "iqr" in results:
            iqr_pct = results["iqr"]["percentage"]
            if iqr_pct > 5:
                recommendations.append(
                    f"IQR method detected {iqr_pct:.1f}% anomalies (above expected 1-5%). "
                    "Investigate sensor calibration or data quality issues."
                )

            top_features = results["iqr"].get("top_features", [])
            if top_features:
                recommendations.append(
                    f"Focus maintenance on: {', '.join(top_features)}. "
                    "These parameters show the most frequent anomalies."
                )

        # Cross-validate anomalies
        if "iqr" in results and "one_class_svm" in results and "isolation_forest" in results:
            iqr_labels = results["iqr"]["labels"]
            svm_labels = results["one_class_svm"]["labels"]
            if_labels = results["isolation_forest"]["labels"]

            # Find samples flagged by multiple methods
            consensus_anomalies = (
                (iqr_labels == -1) & (svm_labels == -1) & (if_labels == -1)
            )
            n_consensus = np.sum(consensus_anomalies)

            if n_consensus > 0:
                recommendations.append(
                    f"{n_consensus} samples flagged by all three methods. "
                    "These require immediate investigation as high-priority anomalies."
                )

        # General recommendations
        recommendations.extend(
            [
                "Implement real-time monitoring for the identified anomalous parameters.",
                "Schedule preventive maintenance for engines showing consistent anomalies.",
                "Review and validate sensor accuracy for parameters with high anomaly rates.",
                "Consider setting up automated alerts for critical anomaly thresholds.",
            ]
        )

        return recommendations


def create_anomaly_summary(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create summary DataFrame of anomaly detection results.

    Args:
        results: Detection results dictionary

    Returns:
        Summary DataFrame
    """
    summary_data = []

    for method in ["iqr", "one_class_svm", "isolation_forest"]:
        if method in results:
            method_results = results[method]
            summary_data.append(
                {
                    "Method": method.replace("_", " ").title(),
                    "Anomalies Detected": method_results["n_anomalies"],
                    "Percentage": f"{method_results['percentage']:.2f}%",
                }
            )

    summary_df = pd.DataFrame(summary_data)
    return summary_df
