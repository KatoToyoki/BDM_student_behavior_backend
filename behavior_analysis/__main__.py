"""
Main application entry point for Behavior Analysis.

This module orchestrates the complete workflow:
1. Convert SPSS files to Parquet format
2. Load data using Spark
3. Perform statistical analysis
"""

import logging
import sys
import warnings

from pyspark.sql import DataFrame

from .analysis.score_clustering import (
    add_cluster_labels,
    get_cluster_statistics,
    print_clustering_report,
)
from .config import AppConfig, load_config
from .data.converter import SPSSToParquetConverter
from .data.spark_manager import SparkSessionManager
from .utils.file_utils import ensure_directory_exists
from .utils.logger import setup_logger
from .visualization.score_clustering_viz import create_all_visualizations


def perform_score_clustering_analysis(
    student_df: DataFrame, config: AppConfig, logger: logging.Logger
) -> None:
    """
    Perform score-based clustering analysis on student data using math scores.

    This function performs a complete score-based clustering workflow:
    1. Adds cluster labels to students based on their PV1MATH scores
       (dividing them into low/medium/high performance groups)
    2. Computes weighted cluster statistics using PISA sample weights (W_FSTUWT)
    3. Prints a detailed clustering report to console
    4. Logs comprehensive statistics for each cluster
    5. Generates and saves visualizations to the artifacts directory

    Args:
        student_df: Spark DataFrame containing student data with PV1MATH
                   and W_FSTUWT columns
        config: Application configuration containing paths and settings
        logger: Logger instance for recording analysis steps and results
    """
    # Add cluster labels based on PV1MATH scores
    logger.info("\nPerforming score-based clustering...")
    clustered_df = add_cluster_labels(
        student_df, score_column="PV1MATH", cluster_column="score_cluster"
    )

    # Get cluster statistics
    logger.info("Computing cluster statistics with weights...")
    cluster_stats = get_cluster_statistics(
        clustered_df,
        score_column="PV1MATH",
        weight_column="W_FSTUWT",
        cluster_column="score_cluster",
    )

    # Print clustering report
    print_clustering_report(cluster_stats, verbose=True)

    # Log statistics
    logger.info("\nCluster Statistics Summary:")
    for level, stats in sorted(cluster_stats.items()):
        logger.info(f"\n{level.upper()}:")
        logger.info(f"  Sample Size: {stats['sample_count']:,}")
        logger.info(f"  Weighted Population: {stats['weighted_count']:,.0f}")
        logger.info(f"  Population %: {stats['population_percentage']:.2f}%")
        if stats["mean_score"] is not None:
            logger.info(f"  Mean Score: {stats['mean_score']:.2f}")
        if stats["weighted_mean_score"] is not None:
            logger.info(f"  Weighted Mean Score: {stats['weighted_mean_score']:.2f}")

    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    create_all_visualizations(
        cluster_stats, output_directory=config.get_artifact_path("visualizations")
    )


def main(target_column: str | None = None) -> None:
    """
    Main application entrypoint.

    Workflow:
    1. Initialize configuration and logging
    2. Convert SPSS files to Parquet (if not already converted)
    3. Start Spark session
    4. Load student data
    5. Calculate statistics for target column

    Args:
        target_column: Column name to analyze. Defaults to ST059Q01TA.
    """
    # Default target column
    if target_column is None:
        target_column = "ST059Q01TA"

    # Suppress Java/Spark warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("\n" + "=" * 70)
    print("PISA Behavior Analysis Application")
    print("=" * 70 + "\n", flush=True)

    # 1. Initialize configuration and logging
    config = load_config()
    logger = setup_logger(log_dir=config.data.LOG_DIR)

    logger.info("=" * 70)
    logger.info("Behavior Analysis Application Started")
    logger.info("=" * 70)

    try:
        # 2. Ensure output directory exists
        ensure_directory_exists(config.data.PARQUET_DIR)
        logger.info(f"Parquet output directory: {config.data.PARQUET_DIR}")

        # 3. Convert SPSS files to Parquet
        logger.info("\n" + "-" * 70)
        logger.info("PHASE 1: SPSS to Parquet Conversion")
        logger.info("-" * 70)

        converter = SPSSToParquetConverter(config.conversion)

        for dataset_name, _spss_filename in config.data.SPSS_FILES.items():
            spss_path = config.get_spss_path(dataset_name)
            parquet_path = config.get_parquet_path(dataset_name)

            logger.info(f"\nProcessing dataset: {dataset_name}")
            logger.info(f"  SPSS file: {spss_path}")
            logger.info(f"  Parquet file: {parquet_path}")

            # Check if already converted
            if converter.is_converted(parquet_path, spss_path):
                logger.info("  Status: Already converted, skipping...")
                print(f"✓ {dataset_name}: Already converted", flush=True)
            else:
                logger.info("  Status: Converting...")
                print(f"⟳ {dataset_name}: Converting...", flush=True)

                success = converter.convert_file(spss_path, parquet_path)

                if success:
                    logger.info("  Result: Conversion successful")
                    print(f"✓ {dataset_name}: Conversion completed", flush=True)
                else:
                    logger.error("  Result: Conversion failed")
                    print(f"✗ {dataset_name}: Conversion failed", flush=True)

        # 4. Spark Analysis Phase
        logger.info("\n" + "-" * 70)
        logger.info("PHASE 2: Score-based Clustering Analysis")
        logger.info("-" * 70)

        # Initialize Spark session with context manager for automatic cleanup
        with SparkSessionManager(config.spark) as spark:
            logger.info("Spark session initialized")

            # Load student data
            student_parquet_path = config.get_parquet_path("student")
            logger.info(f"\nLoading student data from: {student_parquet_path}")

            student_df = spark.read.parquet(student_parquet_path)
            logger.info(f"Student data loaded: {student_df.count():,} records")

            # Verify required columns exist
            required_columns = ["PV1MATH", "W_FSTUWT"]
            missing_columns = [col for col in required_columns if col not in student_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            logger.info(f"Required columns verified: {required_columns}")

            # Perform score-based clustering analysis
            perform_score_clustering_analysis(student_df, config, logger)

    except KeyboardInterrupt:
        logger.info("\nApplication interrupted by user")
        print("\n\n⚠ Application interrupted by user")
        sys.exit(1)

    except Exception as e:  # noqa: BLE001
        logger.error("Application failed with error: %s", e, exc_info=True)
        print(f"\n✗ Error: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    # Support command-line argument for column name
    column = sys.argv[1] if len(sys.argv) > 1 else None
    main(target_column=column)
