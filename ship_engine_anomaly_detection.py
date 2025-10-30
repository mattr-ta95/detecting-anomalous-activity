#!/usr/bin/env python3
"""
Ship Engine Anomaly Detection System

Main script for running anomaly detection on ship engine sensor data.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from anomaly_detection import AnomalyDetector, create_anomaly_summary
from preprocessing import load_data, preprocess_data, get_feature_statistics
from utils import (
    download_file,
    ensure_directories,
    format_anomaly_report,
    load_config,
    setup_logging,
)
from visualization import create_all_visualizations


def main():
    """Main execution function."""
    # Load configuration
    print("Loading configuration...")
    config = load_config("config.yaml")

    # Setup logging
    logger = setup_logging(config)
    logger.info("=" * 80)
    logger.info("SHIP ENGINE ANOMALY DETECTION SYSTEM")
    logger.info("=" * 80)

    # Ensure directories exist
    ensure_directories(config)

    # Download data if needed
    data_url = config["data"]["url"]
    raw_data_path = config["data"]["raw_path"]

    if not Path(raw_data_path).exists():
        logger.info("Dataset not found locally. Downloading...")
        try:
            download_file(data_url, raw_data_path, config, logger)
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            logger.info("Please download the dataset manually and place it at:")
            logger.info(f"  {raw_data_path}")
            return 1
    else:
        logger.info(f"Using existing dataset: {raw_data_path}")

    # Load data
    try:
        df = load_data(raw_data_path, config, logger)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    # Display feature statistics
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE STATISTICS (Before Preprocessing)")
    logger.info("=" * 80)
    stats = get_feature_statistics(df, logger)
    if config.get("output", {}).get("verbose", True):
        print("\n" + stats.to_string())

    # Preprocess data
    logger.info("\n" + "=" * 80)
    logger.info("DATA PREPROCESSING")
    logger.info("=" * 80)
    try:
        df_processed, scaler = preprocess_data(df, config, logger)
    except Exception as e:
        logger.error(f"Failed to preprocess data: {e}")
        return 1

    # Save processed data
    processed_path = config["data"]["processed_path"]
    df_processed.to_csv(processed_path, index=False)
    logger.info(f"Saved processed data to {processed_path}")

    # Initialize anomaly detector
    logger.info("\n" + "=" * 80)
    logger.info("ANOMALY DETECTION")
    logger.info("=" * 80)
    detector = AnomalyDetector(config, logger)

    # Run anomaly detection
    try:
        results, pca_data = detector.detect_all_anomalies(df_processed)
    except Exception as e:
        logger.error(f"Failed to detect anomalies: {e}")
        return 1

    # Display summary
    logger.info("\n" + "=" * 80)
    logger.info("DETECTION SUMMARY")
    logger.info("=" * 80)
    summary_df = create_anomaly_summary(results)
    if config.get("output", {}).get("verbose", True):
        print("\n" + summary_df.to_string(index=False))

    # Generate report
    if config.get("output", {}).get("save_report", True):
        report_path = Path(config["output"]["reports_dir"]) / "anomaly_report.txt"
        report = format_anomaly_report(results, report_path)
        logger.info(f"\nReport saved to {report_path}")

        if config.get("output", {}).get("verbose", True):
            print("\n" + report)

    # Create visualizations
    if config.get("output", {}).get("save_plots", True):
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 80)
        try:
            create_all_visualizations(df_processed, pca_data, results, config, logger)
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
            logger.warning("Continuing without visualizations...")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total samples analyzed: {len(df_processed)}")
    logger.info(f"Results saved to: {config['output']['reports_dir']}")
    logger.info(f"Plots saved to: {config['output']['plots_dir']}")
    logger.info("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
