"""
Main application entry point for Behavior Analysis.

This module orchestrates the complete workflow:
1. Convert SPSS files to Parquet format
2. Load data using Spark
3. Perform statistical analysis
"""

import sys

from .analysis.basic_stats import (
    calculate_column_mean,
    get_column_statistics,
    print_statistics_report,
)
from .config import load_config
from .data.converter import SPSSToParquetConverter
from .data.spark_manager import SparkSessionManager
from .utils.file_utils import ensure_directory_exists
from .utils.logger import setup_logger


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

    print("\n" + "=" * 70)
    print("PISA Behavior Analysis Application")
    print("=" * 70 + "\n")

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
                print(f"✓ {dataset_name}: Already converted")
            else:
                logger.info("  Status: Converting...")
                print(f"⟳ {dataset_name}: Converting...")

                success = converter.convert_file(spss_path, parquet_path)

                if success:
                    logger.info("  Result: Conversion successful")
                    print(f"✓ {dataset_name}: Conversion completed")
                else:
                    logger.error("  Result: Conversion failed")
                    print(f"✗ {dataset_name}: Conversion failed")

        # 4. Spark Analysis Phase
        logger.info("\n" + "-" * 70)
        logger.info("PHASE 2: Spark Data Analysis")
        logger.info("-" * 70)

        print("\nStarting Spark analysis...")

        with SparkSessionManager(config.spark) as spark:
            # Load student data
            student_parquet_path = config.get_parquet_path("student")
            logger.info(f"\nLoading student data from: {student_parquet_path}")

            student_df = spark.read.parquet(student_parquet_path)
            row_count = student_df.count()

            logger.info(
                f"Student data loaded: {row_count:,} rows, {len(student_df.columns)} columns"
            )
            print(f"\n✓ Loaded student data: {row_count:,} rows")

            # Check if target column exists
            if target_column not in student_df.columns:
                # Try to find similar columns
                similar_cols = [col for col in student_df.columns if target_column[:6] in col]

                logger.error(f"Column '{target_column}' not found in dataset")

                if similar_cols:
                    logger.info(f"Similar columns found: {similar_cols[:5]}")
                    print(f"\n✗ Column '{target_column}' not found!")
                    print(f"  Similar columns: {', '.join(similar_cols[:5])}")
                else:
                    print(f"\n✗ Column '{target_column}' not found and no similar columns detected")

                return

            # Calculate mean
            logger.info(f"\nCalculating mean for column: {target_column}")
            print(f"\nAnalyzing column: {target_column}")

            mean_value = calculate_column_mean(student_df, target_column)

            if mean_value is not None:
                # Get detailed statistics
                stats = get_column_statistics(student_df, target_column)

                # Print results
                print_statistics_report(stats)

                # Log summary
                logger.info("\nAnalysis completed successfully:")
                logger.info(f"  Column: {target_column}")
                logger.info(f"  Mean: {mean_value:.4f}")
                logger.info(f"  Non-null count: {stats['count']:,}")
                logger.info(f"  Null count: {stats['null_count']:,}")
            else:
                logger.warning(f"No valid values found for column: {target_column}")
                print(f"\n⚠ No valid values found for column: {target_column}")

        logger.info("\n" + "=" * 70)
        logger.info("Application completed successfully")
        logger.info("=" * 70)

        print("\n" + "=" * 70)
        print("✓ Analysis completed successfully!")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        logger.info("\nApplication interrupted by user")
        print("\n\n⚠ Application interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\nApplication failed with error: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Support command-line argument for column name
    import sys

    column = sys.argv[1] if len(sys.argv) > 1 else None
    main(target_column=column)
