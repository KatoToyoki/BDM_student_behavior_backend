"""
Attitude clustering test module.

Provides testing functionality for attitude-based clustering analysis.
"""

import sys
from pathlib import Path

from .analysis.attitude_clustering import (
    add_attitude_labels,
    create_attitude_features,
    get_attitude_statistics,
    perform_attitude_clustering,
    prepare_attitude_data,
    print_attitude_report,
    validate_attitude_columns,
)
from .config import load_config
from .data.spark_manager import SparkSessionManager
from .utils.logger import get_logger
from .visualization.attitude_clustering_viz import create_all_attitude_visualizations


def test_attitude_clustering() -> None:
    """
    Test attitude clustering on actual student data.

    This function loads student data, performs attitude-based clustering,
    and displays the results.
    """
    logger = get_logger()

    print("\n" + "=" * 70, flush=True)
    print("ATTITUDE CLUSTERING TEST", flush=True)
    print("=" * 70 + "\n", flush=True)

    try:
        # Initialize Spark session
        config = load_config()
        spark_manager = SparkSessionManager(config.spark)
        spark = spark_manager.get_session()
        logger.info("Spark session initialized for attitude clustering test")

        # Load student data
        student_parquet_path = config.get_parquet_path("student")
        print(f"Loading student data from: {student_parquet_path}", flush=True)

        student_df = spark.read.parquet(student_parquet_path)
        print(f"✓ Loaded: {student_df.count():,} records\n", flush=True)

        # Validate attitude columns exist
        print("Validating attitude columns...", flush=True)
        validate_attitude_columns(student_df)
        print("✓ All attitude columns found\n", flush=True)

        # Display sample data
        print("Sample attitude data:", flush=True)
        student_df.select("ST296Q01JA", "ST062Q01TA", "MATHMOT", "PERSEVAGR", "W_FSTUWT").show(
            5, truncate=False
        )
        print()

        # Prepare data (handle missing values)
        print("Preparing attitude data (removing rows with missing values)...", flush=True)
        student_df_clean, missing_stats = prepare_attitude_data(
            student_df, weight_column="W_FSTUWT"
        )
        print("✓ Data prepared\n", flush=True)

        # Display missing value summary
        print("Missing Value Summary:", flush=True)
        print(
            f"  Sample Loss: {missing_stats['sample_loss']['removed_count']:,} rows ({missing_stats['sample_loss']['loss_rate']:.2f}%)",
            flush=True,
        )
        print(
            f"  Weighted Population Loss: {missing_stats['weighted_loss']['removed_weighted']:,.0f} ({missing_stats['weighted_loss']['weighted_loss_rate']:.2f}%)",
            flush=True,
        )
        print()

        # Create features
        print("Creating attitude features with Z-score standardization...", flush=True)
        df_with_features = create_attitude_features(student_df_clean)
        print("✓ Features created\n", flush=True)

        # Perform clustering
        print("Performing K-means clustering (k=3)...", flush=True)
        df_clustered, label_mapping = perform_attitude_clustering(df_with_features, num_clusters=3)
        print("✓ Clustering completed\n", flush=True)

        # Add labels
        print("Mapping clusters to attitude labels...", flush=True)
        df_labeled = add_attitude_labels(df_clustered, label_mapping)
        print("✓ Labels assigned\n", flush=True)

        # Display sample clusters
        print("Sample clustering results:", flush=True)
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
        print("Computing weighted statistics...", flush=True)
        stats = get_attitude_statistics(
            df_labeled, weight_column="W_FSTUWT", cluster_column="attitude_cluster"
        )
        print("✓ Statistics computed\n", flush=True)

        # Print report
        print_attitude_report(stats, verbose=True)

        # Show distribution
        print("\nCluster distribution:", flush=True)
        for label, stat in sorted(stats.items()):
            print(f"\n{label}:")
            print(f"  Sample Size: {stat['sample_count']:,}")
            print(f"  Weighted Population: {stat['weighted_count']:,.0f}")
            print(f"  Population %: {stat['population_percentage']:.2f}%")

        # Generate visualizations
        print("\n" + "=" * 70, flush=True)
        print("Generating visualizations...", flush=True)
        print("=" * 70 + "\n", flush=True)

        artifact_path = config.get_artifact_path("visualizations")

        # Generate clustering visualizations
        visualizations = create_all_attitude_visualizations(stats, artifact_path)

        # Generate missing value visualizations
        from .visualization.attitude_clustering_viz import (
            create_missing_value_chart,
            create_sample_loss_chart,
            export_missing_value_table,
        )

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

        print("✓ Visualizations generated:", flush=True)
        for viz_name, viz_path in visualizations.items():
            print(f"  - {viz_name}: {viz_path}", flush=True)

        print("\n" + "=" * 70, flush=True)
        print("✓ ATTITUDE CLUSTERING TEST COMPLETED SUCCESSFULLY", flush=True)
        print("=" * 70 + "\n", flush=True)

        logger.info("Attitude clustering test completed successfully")

    except Exception as e:  # noqa: BLE001
        logger.error("Attitude clustering test failed: %s", e, exc_info=True)
        print(f"\n✗ Error: {e}", flush=True)
        sys.exit(1)
