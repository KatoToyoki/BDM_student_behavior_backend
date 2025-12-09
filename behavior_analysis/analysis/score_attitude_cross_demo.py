"""
Score-Attitude Cross-Dimensional Analysis Module.

Demonstrates the relationship analysis between attitude and score clustering.
"""

from ...config import load_config
from ...data.spark_manager import SparkSessionManager
from ...utils.logger import get_logger
from ...visualization.score_attitude_cross_analysis_viz import (
    create_all_score_attitude_visualizations,
)
from ..attitude_clustering import (
    add_attitude_labels,
    create_attitude_features,
    perform_attitude_clustering,
    prepare_attitude_data,
)
from ..score_attitude_cross_analysis import (
    create_cross_tabulation,
    export_cross_tabulation,
    perform_chi_square_test,
    print_chi_square_report,
    print_cross_tabulation,
)
from ..score_clustering import (
    add_cluster_labels as add_score_labels,
)
from ..score_clustering import (
    categorize_score,
)


def run_score_attitude_cross_analysis() -> None:
    """
    Run score-attitude cross-dimensional analysis.

    This function loads student data, performs both clustering analyses,
    creates cross-tabulation, performs chi-square test, and generates visualizations.
    """
    logger = get_logger()

    print("\n" + "=" * 70)
    print("SCORE-ATTITUDE CROSS-DIMENSIONAL ANALYSIS")
    print("=" * 70 + "\n")

    try:
        # Initialize configuration
        config = load_config()

        # Use Spark session with context manager for automatic cleanup
        with SparkSessionManager(config.spark) as spark:
            logger.info("Spark session initialized for cross-dimensional analysis")

            # Load student data
            student_parquet_path = config.get_parquet_path("student")
            print(f"Loading student data from: {student_parquet_path}")

            student_df = spark.read.parquet(student_parquet_path)
            print(f"✓ Loaded: {student_df.count():,} records\n")

            # Prepare score clustering
            print("Preparing score clustering...")

            df_with_score = student_df.withColumn(
                "score_cluster", categorize_score(score_column="PV1MATH")
            )
            df_with_score_labels = add_score_labels(
                df_with_score, score_column="PV1MATH", cluster_column="score_cluster"
            )
            print("✓ Score clustering completed\n")

            # Prepare attitude clustering
            print("Preparing attitude clustering...")

            df_clean, _ = prepare_attitude_data(df_with_score_labels, weight_column="W_FSTUWT")
            df_with_features = create_attitude_features(df_clean)
            df_clustered, label_mapping = perform_attitude_clustering(
                df_with_features, num_clusters=3
            )
            df_with_attitude_labels = add_attitude_labels(df_clustered, label_mapping)
            print("✓ Attitude clustering completed\n")

            # Create cross-tabulation
            print("Creating cross-tabulation table...")
            cross_tab = create_cross_tabulation(
                df_with_attitude_labels,
                score_column="score_cluster",
                attitude_column="attitude_cluster",
                weight_column="W_FSTUWT",
                score_label_column="score_cluster",
                attitude_label_column="attitude_label",
            )
            print("✓ Cross-tabulation created\n")

            # Display cross-tabulation
            print_cross_tabulation(cross_tab)

            # Export cross-tabulation
            print("Exporting cross-tabulation...")
            artifact_path = config.get_artifact_path("visualizations")
            export_cross_tabulation(
                cross_tab,
                str(artifact_path + "/attitude_score_crosstab.csv"),
            )
            print("✓ Cross-tabulation exported\n")

            # Perform chi-square test
            print("Performing chi-square test...")
            chi_results = perform_chi_square_test(cross_tab)
            print("✓ Chi-square test completed\n")

            # Display chi-square results
            print_chi_square_report(chi_results)

            # Generate visualizations
            print("=" * 70)
            print("Generating visualizations...")
            print("=" * 70 + "\n")

            visualizations = create_all_score_attitude_visualizations(cross_tab, artifact_path)

            print("✓ Visualizations generated:")
            for viz_name, viz_path in visualizations.items():
                print(f"  - {viz_name}: {viz_path}")

            print("\n" + "=" * 70)
            print("✓ SCORE-ATTITUDE CROSS-DIMENSIONAL ANALYSIS COMPLETED SUCCESSFULLY")
            print("=" * 70 + "\n")

            logger.info("Cross-dimensional analysis completed successfully")

        # Spark session is automatically closed here

    except Exception as e:
        logger.error("Score-attitude cross-dimensional analysis failed: %s", e, exc_info=True)
        print(f"\n✗ Error: {e}\n")
        raise
