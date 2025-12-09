"""
Attitude clustering analysis module.

Provides functionality for attitude-based clustering analysis.
"""

import sys
from pathlib import Path

from ...config import load_config
from ...data.spark_manager import SparkSessionManager
from ...utils.logger import get_logger
from ...visualization.attitude_clustering_viz import (
    create_all_attitude_visualizations,
    create_missing_value_chart,
    create_sample_loss_chart,
    export_missing_value_table,
)
from ..attitude_clustering import (
    add_attitude_labels,
    create_attitude_features,
    get_attitude_statistics,
    perform_attitude_clustering,
    prepare_attitude_data,
    print_attitude_report,
    validate_attitude_columns,
)


def run_attitude_clustering() -> None:
    """
    Run attitude clustering analysis on student data.

    This function loads student data, performs attitude-based clustering,
    and displays the results.
    """
    logger = get_logger()

    print("\n" + "=" * 70)
    print("ATTITUDE CLUSTERING ANALYSIS")
    print("=" * 70 + "\n")

    try:
        # Initialize configuration
        config = load_config()

        # Use Spark session with context manager for automatic cleanup
        with SparkSessionManager(config.spark) as spark:
            logger.info("Spark session initialized for attitude clustering")

            # Load student data
            student_parquet_path = config.get_parquet_path("student")
            print(f"Loading student data from: {student_parquet_path}")

            student_df = spark.read.parquet(student_parquet_path)
            print(f"✓ Loaded: {student_df.count():,} records\n")

            # Validate attitude columns exist
            print("Validating attitude columns...")
            validate_attitude_columns(student_df)
            print("✓ All attitude columns found\n")

            # Display sample data
            print("Sample attitude data:")
            student_df.select("ST296Q01JA", "ST062Q01TA", "MATHMOT", "PERSEVAGR", "W_FSTUWT").show(
                5, truncate=False
            )
            print()

            # Prepare data (handle missing values)
            print("Preparing attitude data (removing rows with missing values)...")
            student_df_clean, missing_stats = prepare_attitude_data(
                student_df, weight_column="W_FSTUWT"
            )
            print("✓ Data prepared\n")

            # Display missing value summary
            print("Missing Value Summary:")
            print(
                f"  Sample Loss: {missing_stats['sample_loss']['removed_count']:,} rows "
                f"({missing_stats['sample_loss']['loss_rate']:.2f}%)"
            )
            print(
                f"  Weighted Population Loss: {missing_stats['weighted_loss']['removed_weighted']:,.0f} "
                f"({missing_stats['weighted_loss']['weighted_loss_rate']:.2f}%)"
            )
            print()

            # Create features
            print("Creating attitude features with Z-score standardization...")
            df_with_features = create_attitude_features(student_df_clean)
            print("✓ Features created\n")

            # Perform clustering
            print("Performing K-means clustering (k=3)...")
            df_clustered, label_mapping = perform_attitude_clustering(
                df_with_features, num_clusters=3
            )
            print("✓ Clustering completed\n")

            # Add labels
            print("Mapping clusters to attitude labels...")
            df_labeled = add_attitude_labels(df_clustered, label_mapping)
            print("✓ Labels assigned\n")

            # Display sample clusters
            print("Sample clustering results:")
            df_labeled.select(
                "ST296Q01JA",
                "ST062Q01TA",
                "MATHMOT",
                "PERSEVAGR",
                "attitude_cluster",
                "attitude_label",
            ).show(10, truncate=False)
            print()

            # Get statistics
            print("Computing weighted statistics...")
            stats = get_attitude_statistics(
                df_labeled, weight_column="W_FSTUWT", cluster_column="attitude_cluster"
            )
            print("✓ Statistics computed\n")

            # Print report
            print_attitude_report(stats, verbose=True)

            # Show distribution
            print("\nCluster distribution:")
            for label, stat in sorted(stats.items()):
                print(f"\n{label}:")
                print(f"  Sample Size: {stat['sample_count']:,}")
                print(f"  Weighted Population: {stat['weighted_count']:,.0f}")
                print(f"  Population %: {stat['population_percentage']:.2f}%")

            # Generate visualizations
            print("\n" + "=" * 70)
            print("Generating visualizations...")
            print("=" * 70 + "\n")

            artifact_path = config.get_artifact_path("visualizations")

            # Generate clustering visualizations
            visualizations = create_all_attitude_visualizations(stats, artifact_path)

            # Generate missing value visualizations
            missing_viz = {
                "missing_value_chart": create_missing_value_chart(
                    missing_stats["variable_missing_rates"],
                    str(Path(artifact_path) / "attitude_missing_values.png"),
                ),
                "sample_loss_chart": create_sample_loss_chart(
                    missing_stats["sample_loss"],
                    missing_stats["weighted_loss"],
                    str(Path(artifact_path) / "attitude_sample_loss.png"),
                ),
                "missing_value_table": export_missing_value_table(
                    missing_stats["variable_missing_rates"],
                    missing_stats["sample_loss"],
                    missing_stats["weighted_loss"],
                    str(Path(artifact_path) / "attitude_missing_values.csv"),
                ),
            }

            visualizations.update(missing_viz)

            print("✓ Visualizations generated:")
            for viz_name, viz_path in visualizations.items():
                print(f"  - {viz_name}: {viz_path}")

            print("\n" + "=" * 70)
            print("✓ ATTITUDE CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY")
            print("=" * 70 + "\n")

            logger.info("Attitude clustering analysis completed successfully")

        # Spark session is automatically closed here

    except Exception as e:  # noqa: BLE001
        logger.error("Attitude clustering analysis failed: %s", e, exc_info=True)
        print(f"\n✗ Error: {e}")
        sys.exit(1)
